"""
Offline trainer: Advantage Actor-Critic (A2C) + Frame Stacking for OBELIX.
Run locally to create weights.pth.

Example:
  python train_ac.py --obelix_py ./obelix.py --out weights.pth --episodes 3000 --difficulty 1
"""

import argparse
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
STACK_SIZE = 4
OBS_DIM = 18

class ActorCritic(nn.Module):
    def __init__(self, in_dim=OBS_DIM * STACK_SIZE, n_actions=len(ACTIONS)):
        super().__init__()
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Actor head: Outputs logits for actions
        self.actor = nn.Linear(128, n_actions)
        # Critic head: Outputs estimated value of the state
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

class FrameStackEnv:
    """Wraps the OBELIX env to stack the last N frames."""
    def __init__(self, env, stack_size=4):
        self.env = env
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return np.concatenate(self.frames)

    def step(self, action, render=False):
        obs, reward, done = self.env.step(action, render=render)
        self.frames.append(obs)
        return np.concatenate(self.frames), reward, done

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights_RI.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for ep in tqdm(range(args.episodes), desc="Training Episodes"):
        base_env = OBELIX(
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            seed=args.seed + ep,
            scaling_factor=5,
            arena_size=500,
        )
        env = FrameStackEnv(base_env, stack_size=STACK_SIZE)
        
        state = env.reset(seed=args.seed + ep)
        
        log_probs = []
        values = []
        rewards = []
        entropies = []

        # Play one full episode
        for _ in range(args.max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            logits, value = model(state_tensor)
            dist = Categorical(logits=logits)
            
            action_idx = dist.sample()
            
            next_state, reward, done = env.step(ACTIONS[action_idx.item()])
            
            log_probs.append(dist.log_prob(action_idx))
            values.append(value)
            rewards.append(reward)
            entropies.append(dist.entropy())
            
            state = next_state
            if done:
                break

        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + args.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze(-1)
        entropies = torch.cat(entropies)
        
        # Calculate Advantages (Return - Baseline Value)
        advantages = returns - values.detach()
        
        # Calculate Losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -args.entropy_coef * entropies.mean() # Encourages exploration
        
        total_loss = actor_loss + critic_loss + entropy_loss
        
        # Backpropagate
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1} | Reward: {sum(rewards):.1f} | Actor Loss: {actor_loss.item():.3f} | Critic Loss: {critic_loss.item():.3f}")

    torch.save(model.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()