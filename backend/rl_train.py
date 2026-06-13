import threading
import uuid
import queue
import time
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gymnasium as gym

# Save directory for checkpoints
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

# Global task manager
rl_tasks = {}
rl_tasks_lock = threading.Lock()

# ----------------------------------------------------------------------
# Q-Network & Utilities
# ----------------------------------------------------------------------

def get_activation(name):
    return {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
    }[name]

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(get_activation(activation))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def get_optimizer(name, params, lr, weight_decay):
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# ----------------------------------------------------------------------
# Reward shaping engine
# ----------------------------------------------------------------------

OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

def apply_reward_rules(rules, base_reward, state, action, terminated, truncated, fail_threshold):
    reward = base_reward
    for rule in rules:
        rtype = rule.get("type")
        if rtype == "step_penalty":
            reward += float(rule.get("value", 0))
        elif rtype == "state_threshold":
            idx = int(rule.get("index", 0))
            op = rule.get("op", ">")
            threshold = float(rule.get("value", 0))
            bonus = float(rule.get("reward", 0))
            if idx < len(state) and op in OPS:
                if OPS[op](state[idx], threshold):
                    reward += bonus
        elif rtype == "action_bonus":
            if action == int(rule.get("action", 0)):
                reward += float(rule.get("reward", 0))
        elif rtype == "terminal_bonus":
            if terminated and not truncated and base_reward >= fail_threshold:
                reward += float(rule.get("reward", 0))
        elif rtype == "terminal_penalty":
            if terminated and not truncated and base_reward < fail_threshold:
                reward += float(rule.get("reward", 0))
        elif rtype == "override_reward":
            reward = float(rule.get("value", reward))
    return reward

# ----------------------------------------------------------------------
# Background Training Task
# ----------------------------------------------------------------------

class RLTrainingTask:
    def __init__(self, task_id: str, config: dict):
        self.task_id = task_id
        self.config = config
        
        # Enforce server-side validation and limits
        self.env_name = config.get("env", "CartPole-v1")
        self.episodes = min(max(int(config.get("episodes", 200)), 1), 500)
        self.max_steps = min(max(int(config.get("max_steps", 500)), 1), 1000)
        self.gamma = min(max(float(config.get("gamma", 0.99)), 0.0), 1.0)
        self.lr = min(max(float(config.get("lr", 0.001)), 1e-6), 0.1)
        self.batch_size = min(max(int(config.get("batch_size", 64)), 1), 256)
        self.buffer_size = min(max(int(config.get("buffer_size", 10000)), 100), 50000)
        self.min_buffer_size = min(int(config.get("min_buffer_size", 1000)), self.buffer_size)
        self.target_update_freq = max(int(config.get("target_update_freq", 10)), 1)
        self.optimizer_name = config.get("optimizer", "adam")
        if self.optimizer_name not in ["adam", "adamw", "sgd", "rmsprop"]:
            self.optimizer_name = "adam"
        self.weight_decay = min(max(float(config.get("weight_decay", 0.0)), 0.0), 0.1)
        
        self.epsilon_start = min(max(float(config.get("epsilon_start", 1.0)), 0.0), 1.0)
        self.epsilon_end = min(max(float(config.get("epsilon_end", 0.05)), 0.0), 1.0)
        self.epsilon_decay = min(max(float(config.get("epsilon_decay", 0.995)), 0.0), 1.0)
        
        self.hidden_layers = config.get("hidden_layers", [128, 128])
        if not isinstance(self.hidden_layers, list):
            self.hidden_layers = [128, 128]
        if len(self.hidden_layers) > 4:
            self.hidden_layers = self.hidden_layers[:4]
        self.hidden_layers = [min(max(int(x), 1), 512) for x in self.hidden_layers]
        
        self.activation = config.get("activation", "relu")
        if self.activation not in ["relu", "tanh", "sigmoid", "leaky_relu", "gelu"]:
            self.activation = "relu"
            
        self.seed = int(config.get("seed", 42))
        self.reward_rules = config.get("reward_rules", [])
        self.fail_reward_threshold = float(config.get("fail_reward_threshold", 0.0))
        
        self.status = "running"  # running, completed, cancelled, failed
        self.error_message = None
        self.current_episode = 0
        self.episode_rewards = []
        self.history = []
        self.logs = []
        self.cancel_event = threading.Event()
        self.log_queue = queue.Queue()

    def add_log(self, text: str):
        log_line = f"[{time.strftime('%H:%M:%S')}] {text}"
        self.logs.append(log_line)
        self.log_queue.put(log_line)

    def cancel(self):
        self.cancel_event.set()
        self.status = "cancelled"
        self.add_log("Training process cancelled by user.")

    def run(self):
        try:
            import random
            
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.add_log(f"Device set to: {device}")
            self.add_log(f"Initializing Gymnasium environment: '{self.env_name}'...")
            
            try:
                env = gym.make(self.env_name)
            except Exception as e:
                raise ValueError(f"Could not load environment '{self.env_name}': {str(e)}")
                
            if not isinstance(env.action_space, gym.spaces.Discrete):
                env.close()
                raise ValueError("This wrapper only supports environments with discrete action spaces.")
                
            state_dim = int(np.prod(env.observation_space.shape))
            action_dim = env.action_space.n
            
            self.add_log(f"Environment Loaded | state_dim={state_dim} | action_dim={action_dim}")
            
            policy_net = QNetwork(state_dim, self.hidden_layers, action_dim, self.activation).to(device)
            target_net = QNetwork(state_dim, self.hidden_layers, action_dim, self.activation).to(device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            
            optimizer = get_optimizer(self.optimizer_name, policy_net.parameters(), self.lr, self.weight_decay)
            loss_fn = nn.MSELoss()
            buffer = ReplayBuffer(self.buffer_size)
            
            epsilon = self.epsilon_start
            self.add_log("Training loop started...")
            
            for episode in range(1, self.episodes + 1):
                if self.cancel_event.is_set():
                    env.close()
                    return
                    
                state, _ = env.reset(seed=self.seed + episode)
                state = np.array(state, dtype=np.float32).flatten()
                total_reward = 0.0
                episode_loss = 0.0
                train_steps = 0
                
                for step in range(self.max_steps):
                    if self.cancel_event.is_set():
                        env.close()
                        return
                        
                    # Epsilon-greedy action
                    if random.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            q_values = policy_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                            action = int(q_values.argmax(dim=1).item())
                            
                    next_state, base_reward, terminated, truncated, _ = env.step(action)
                    next_state = np.array(next_state, dtype=np.float32).flatten()
                    
                    # Reward shaping rules
                    reward = apply_reward_rules(
                        self.reward_rules, base_reward, state, action,
                        terminated, truncated, self.fail_reward_threshold
                    )
                    
                    done = terminated or truncated
                    buffer.push(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    
                    # Train model from replay buffer
                    if len(buffer) >= self.min_buffer_size:
                        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)
                        
                        states_t = torch.tensor(states, dtype=torch.float32, device=device)
                        actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
                        
                        q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)
                        with torch.no_grad():
                            next_q_values = target_net(next_states_t).max(dim=1)[0]
                            target_q = rewards_t + self.gamma * next_q_values * (1 - dones_t)
                            
                        loss = loss_fn(q_values, target_q)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        episode_loss += loss.item()
                        train_steps += 1
                        
                    if done:
                        break
                        
                self.episode_rewards.append(total_reward)
                epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
                
                if episode % self.target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    
                self.current_episode = episode
                avg_reward = float(np.mean(self.episode_rewards[-10:]))
                avg_loss = float(episode_loss / max(train_steps, 1))
                
                # Update history for plotting
                self.history.append({
                    "episode": episode,
                    "reward": float(total_reward),
                    "avg_reward": avg_reward,
                    "loss": avg_loss,
                    "epsilon": float(epsilon)
                })
                
                if episode % 5 == 0 or episode == self.episodes:
                    self.add_log(
                        f"Episode {episode}/{self.episodes} | Reward: {total_reward:.2f} | "
                        f"Avg Reward (last 10): {avg_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}"
                    )
                    
            env.close()
            
            # Save checkpoint matching the RLassistant.py output format
            model_path = os.path.join(MODELS_DIR, f"{self.task_id}.pt")
            
            dummy_args = {
                "env": self.env_name,
                "episodes": self.episodes,
                "max_steps": self.max_steps,
                "gamma": self.gamma,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "buffer_size": self.buffer_size,
                "min_buffer_size": self.min_buffer_size,
                "target_update_freq": self.target_update_freq,
                "optimizer": self.optimizer_name,
                "weight_decay": self.weight_decay,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "hidden_layers": self.hidden_layers,
                "activation": self.activation,
                "seed": self.seed,
                "device": device,
                "output_model": model_path
            }
            
            torch.save({
                "model_state_dict": policy_net.to("cpu").state_dict(),
                "args": dummy_args,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "reward_rules": self.reward_rules,
            }, model_path)
            
            self.status = "completed"
            self.add_log(f"Model successfully saved. Download available.")
            
        except Exception as e:
            self.status = "failed"
            self.error_message = str(e)
            self.add_log(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()

# ----------------------------------------------------------------------
# Manager API Functions
# ----------------------------------------------------------------------

def start_training_task(config: dict) -> str:
    task_id = str(uuid.uuid4())
    task = RLTrainingTask(task_id, config)
    
    with rl_tasks_lock:
        rl_tasks[task_id] = task
        
    thread = threading.Thread(target=task.run, name=f"RL-Train-{task_id}")
    thread.daemon = True
    thread.start()
    
    return task_id

def get_training_task(task_id: str) -> RLTrainingTask:
    with rl_tasks_lock:
        return rl_tasks.get(task_id)

def cancel_training_task(task_id: str) -> bool:
    task = get_training_task(task_id)
    if task:
        task.cancel()
        return True
    return False
