#!/usr/bin/env python3
"""
Test / play a trained RL agent (from rl_train.py)

Example:
    python rl_test.py --model dqn_model.pt --episodes 10 --render
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym
except ImportError:
    print("Error: gymnasium not installed. Run: pip install gymnasium", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Test a trained RL agent")
    p.add_argument("--model", type=str, required=True, help="Path to trained model (.pt)")
    p.add_argument("--episodes", type=int, default=10, help="Number of test episodes (default: 10)")
    p.add_argument("--max-steps", type=int, default=None,
                    help="Max steps per episode (default: same as training)")
    p.add_argument("--render", action="store_true", help="Render the environment (needs a display)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--greedy", action="store_true", default=True,
                    help="Always pick the best action (default: True, no exploration)")
    return p.parse_args()


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


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]

    model = QNetwork(state_dim, train_args["hidden_layers"], action_dim, train_args["activation"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_steps = args.max_steps or train_args["max_steps"]

    render_mode = "human" if args.render else None
    env = gym.make(train_args["env"], render_mode=render_mode)

    rewards = []
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32).flatten()
        total_reward = 0.0

        for step in range(max_steps):
            with torch.no_grad():
                q_values = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(q_values.argmax(dim=1).item())

            state, reward, terminated, truncated, _ = env.step(action)
            state = np.array(state, dtype=np.float32).flatten()
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"Episode {ep}/{args.episodes} | reward: {total_reward:.2f} | steps: {step+1}")

    env.close()

    print(f"\nAverage reward over {args.episodes} episodes: {np.mean(rewards):.2f}")
    print(f"Min: {np.min(rewards):.2f} | Max: {np.max(rewards):.2f} | Std: {np.std(rewards):.2f}")


if __name__ == "__main__":
    main()