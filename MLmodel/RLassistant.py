#!/usr/bin/env python3
"""
Generalized Reinforcement Learning Trainer (DQN)

Designed for a no-code website wrapper: every hyperparameter, environment
choice, and reward shaping rule is set via command-line arguments (which can
be generated from a web form / JSON config).

----------------------------------------------------------------------------
ENVIRONMENT
----------------------------------------------------------------------------
By default, uses any Gymnasium environment by name (e.g. CartPole-v1,
MountainCar-v0, LunarLander-v2, FrozenLake-v1).

----------------------------------------------------------------------------
CUSTOM / MANUAL REWARDS
----------------------------------------------------------------------------
Students can shape the reward WITHOUT touching code, using simple rules
passed as JSON via --reward-rules. Each rule is checked every step and adds
(or overrides) to the environment's default reward.

Rule format (list of dicts), supported "type" values:
  - "step_penalty":   {"type": "step_penalty", "value": -0.01}
        Subtracts a fixed value every step (encourages efficiency).
  - "state_threshold": {"type": "state_threshold", "index": 0,
                         "op": ">", "value": 0.5, "reward": 10}
        If state[index] satisfies (state[index] op value), add `reward`.
  - "action_bonus":   {"type": "action_bonus", "action": 1, "reward": 1}
        Add `reward` whenever the agent takes a specific action.
  - "terminal_bonus": {"type": "terminal_bonus", "reward": 100}
        Add `reward` when the episode ends successfully
        (terminated=True and not truncated).
  - "terminal_penalty": {"type": "terminal_penalty", "reward": -100}
        Add `reward` when the episode ends in failure
        (terminated=True due to failure condition; treated same as
        terminal but applied when env signals termination AND
        reward from env is below --fail-reward-threshold).
  - "override_reward": {"type": "override_reward", "value": 1}
        Completely replaces the environment's reward with `value` every step
        (useful for sparse/manual reward experiments).

Example combining rules:
  --reward-rules '[
      {"type": "step_penalty", "value": -0.01},
      {"type": "state_threshold", "index": 2, "op": ">", "value": 0.1, "reward": 5},
      {"type": "terminal_bonus", "reward": 50}
  ]'

----------------------------------------------------------------------------
EXAMPLE
----------------------------------------------------------------------------
python rl_train.py \
    --env CartPole-v1 \
    --episodes 200 \
    --max-steps 500 \
    --gamma 0.99 \
    --lr 0.001 \
    --batch-size 64 \
    --hidden-layers 128 128 \
    --activation relu \
    --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.995 \
    --buffer-size 10000 \
    --target-update-freq 10 \
    --reward-rules '[{"type":"step_penalty","value":-0.01}]' \
    --output-model dqn_model.pt
"""

import argparse
import json
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
except ImportError:
    print("Error: gymnasium not installed. Run: pip install gymnasium", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generalized RL trainer (DQN)")

    # Environment
    p.add_argument("--env", type=str, required=True,
                    help="Gymnasium environment id, e.g. CartPole-v1")
    p.add_argument("--episodes", type=int, default=200,
                    help="Number of training episodes (default: 200)")
    p.add_argument("--max-steps", type=int, default=500,
                    help="Max steps per episode (default: 500)")

    # Reward shaping
    p.add_argument("--reward-rules", type=str, default="[]",
                    help="JSON list of manual reward shaping rules (see script docstring)")
    p.add_argument("--fail-reward-threshold", type=float, default=0.0,
                    help="Env reward below this on termination counts as failure (default: 0.0)")

    # Network architecture
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 128],
                    help="Hidden layer sizes for the Q-network (default: 128 128)")
    p.add_argument("--activation", type=str, default="relu",
                    choices=["relu", "tanh", "sigmoid", "leaky_relu", "gelu"],
                    help="Activation function (default: relu)")

    # Training hyperparameters
    p.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor (default: 0.99)")
    p.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate (default: 1e-3)")
    p.add_argument("--batch-size", type=int, default=64,
                    help="Replay batch size (default: 64)")
    p.add_argument("--buffer-size", type=int, default=10000,
                    help="Replay buffer size (default: 10000)")
    p.add_argument("--min-buffer-size", type=int, default=1000,
                    help="Minimum buffer size before training starts (default: 1000)")
    p.add_argument("--target-update-freq", type=int, default=10,
                    help="Episodes between target network syncs (default: 10)")
    p.add_argument("--optimizer", type=str, default="adam",
                    choices=["adam", "sgd", "rmsprop", "adamw"], help="Optimizer (default: adam)")
    p.add_argument("--weight-decay", type=float, default=0.0,
                    help="Weight decay / L2 regularization (default: 0.0)")

    # Exploration (epsilon-greedy)
    p.add_argument("--epsilon-start", type=float, default=1.0,
                    help="Initial exploration rate (default: 1.0)")
    p.add_argument("--epsilon-end", type=float, default=0.05,
                    help="Minimum exploration rate (default: 0.05)")
    p.add_argument("--epsilon-decay", type=float, default=0.995,
                    help="Multiplicative decay per episode (default: 0.995)")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"], help="Device to train on")
    p.add_argument("--output-model", type=str, default="dqn_model.pt",
                    help="Path to save the trained model (default: dqn_model.pt)")
    p.add_argument("--log-interval", type=int, default=10,
                    help="Print progress every N episodes (default: 10)")
    p.add_argument("--render", action="store_true",
                    help="Render the environment (only works locally with a display)")

    return p.parse_args()


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
    """Apply manually-defined reward shaping rules and return the final reward."""
    reward = base_reward

    for rule in rules:
        rtype = rule.get("type")

        if rtype == "step_penalty":
            reward += rule.get("value", 0)

        elif rtype == "state_threshold":
            idx = rule.get("index", 0)
            op = rule.get("op", ">")
            threshold = rule.get("value", 0)
            bonus = rule.get("reward", 0)
            if idx < len(state) and op in OPS:
                if OPS[op](state[idx], threshold):
                    reward += bonus

        elif rtype == "action_bonus":
            if action == rule.get("action"):
                reward += rule.get("reward", 0)

        elif rtype == "terminal_bonus":
            if terminated and not truncated and base_reward >= fail_threshold:
                reward += rule.get("reward", 0)

        elif rtype == "terminal_penalty":
            if terminated and not truncated and base_reward < fail_threshold:
                reward += rule.get("reward", 0)

        elif rtype == "override_reward":
            reward = rule.get("value", reward)

        else:
            print(f"Warning: unknown reward rule type '{rtype}', ignoring.", file=sys.stderr)

    return reward


# ----------------------------------------------------------------------
# Q-Network
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
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        reward_rules = json.loads(args.reward_rules)
        if not isinstance(reward_rules, list):
            raise ValueError("reward-rules must be a JSON list")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing --reward-rules: {e}", file=sys.stderr)
        sys.exit(1)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    render_mode = "human" if args.render else None
    try:
        env = gym.make(args.env, render_mode=render_mode)
    except Exception as e:
        print(f"Error creating environment '{args.env}': {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(env.action_space, gym.spaces.Discrete):
        print("Error: this trainer currently supports only discrete action spaces.", file=sys.stderr)
        sys.exit(1)

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n
    print(f"Environment: {args.env} | state_dim={state_dim} | action_dim={action_dim}")
    print(f"Reward rules: {reward_rules}")

    policy_net = QNetwork(state_dim, args.hidden_layers, action_dim, args.activation).to(device)
    target_net = QNetwork(state_dim, args.hidden_layers, action_dim, args.activation).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = get_optimizer(args.optimizer, policy_net.parameters(), args.lr, args.weight_decay)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(args.buffer_size)

    epsilon = args.epsilon_start
    episode_rewards = []

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        state = np.array(state, dtype=np.float32).flatten()
        total_reward = 0.0

        for step in range(args.max_steps):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                    action = int(q_values.argmax(dim=1).item())

            next_state, base_reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32).flatten()

            reward = apply_reward_rules(
                reward_rules, base_reward, state, action,
                terminated, truncated, args.fail_reward_threshold
            )

            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Train on a batch from the replay buffer
            if len(buffer) >= args.min_buffer_size:
                states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states_t).max(dim=1)[0]
                    target_q = rewards_t + args.gamma * next_q_values * (1 - dones_t)

                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if episode % args.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % args.log_interval == 0 or episode == args.episodes:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            print(f"Episode {episode}/{args.episodes} | "
                  f"reward: {total_reward:.2f} | avg_reward(last {args.log_interval}): {avg_reward:.2f} | "
                  f"epsilon: {epsilon:.3f}")

    env.close()

    torch.save({
        "model_state_dict": policy_net.state_dict(),
        "args": vars(args),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "reward_rules": reward_rules,
    }, args.output_model)
    print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    main()