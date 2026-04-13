"""
Offline trainer: PPO + Frame Stacking for OBELIX.
Run locally to create weights.pth.

Example:
  python train_ppo.py --obelix_py ./obelix.py --out weights.pth --episodes 3000 --difficulty 1
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
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(), # Tanh is often preferred over ReLU for PPO
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.actor = nn.Linear(128, n_actions)
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
            self.frames.append(obs.copy())
        return np.concatenate(self.frames)

    def step(self, action, render=False):
        obs, reward, done = self.env.step(action, render=render)
        self.frames.append(obs.copy())
        return np.concatenate(self.frames), reward, done

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.is_terminals.clear()

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=1)
    
    # PPO Hyperparameters
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--update_timestep", type=int, default=4000) # Update network every N steps
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    buffer = RolloutBuffer()

    time_step = 0
    ep_reward_history = deque(maxlen=50)

    # We use a single continuous environment instance for PPO data collection
    base_env = OBELIX(max_steps=args.max_steps, difficulty=args.difficulty, seed=args.seed,scaling_factor=5,
            arena_size=500,wall_obstacles=True,box_speed=2)
    env = FrameStackEnv(base_env, stack_size=STACK_SIZE)
    state = env.reset(seed=args.seed)

    for ep in tqdm(range(args.episodes), desc="Training"):
        ep_reward = 0

        for _ in range(args.max_steps):
            time_step += 1

            # Select action
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, value = model(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            next_state, reward, done = env.step(ACTIONS[action.item()])
            ep_reward += reward

            # Save data to buffer
            buffer.states.append(state)
            buffer.actions.append(action.item())
            buffer.logprobs.append(logprob.item())
            buffer.rewards.append(reward)
            buffer.values.append(value.item())
            buffer.is_terminals.append(done)

            state = next_state

            # Perform PPO update if we have collected enough timesteps
            if time_step % args.update_timestep == 0:
                # Calculate GAE and Returns
                returns = []
                advantages = []
                discounted_reward = 0
                
                # Bootstrap value if not done
                with torch.no_grad():
                    _, next_val = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    next_val = next_val.item()

                adv = 0
                for r, v, is_term in zip(reversed(buffer.rewards), reversed(buffer.values), reversed(buffer.is_terminals)):
                    if is_term:
                        discounted_reward = 0
                        next_val = 0
                        adv = 0
                    
                    delta = r + args.gamma * next_val - v
                    adv = delta + args.gamma * args.gae_lambda * adv
                    advantages.insert(0, adv)
                    
                    discounted_reward = r + args.gamma * discounted_reward
                    returns.insert(0, discounted_reward)
                    next_val = v

                # Convert lists to tensors
                old_states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
                old_actions = torch.tensor(buffer.actions, dtype=torch.float32)
                old_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32)
                returns = torch.tensor(returns, dtype=torch.float32)
                advantages = torch.tensor(advantages, dtype=torch.float32)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Optimize policy for K epochs
                for _ in range(args.ppo_epochs):
                    logits, values = model(old_states)
                    dist = Categorical(logits=logits)
                    
                    logprobs = dist.log_prob(old_actions)
                    dist_entropy = dist.entropy()
                    
                    # Find ratio (pi_theta / pi_theta__old)
                    ratios = torch.exp(logprobs - old_logprobs)
                    
                    # Surrogate Loss
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - args.clip_eps, 1 + args.clip_eps) * advantages
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.functional.mse_loss(values.squeeze(-1), returns)
                    entropy_loss = -args.entropy_coef * dist_entropy.mean()
                    
                    loss = actor_loss + 0.5 * critic_loss + entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                buffer.clear()

            if done:
                state = env.reset()
                ep_reward_history.append(ep_reward)
                break

        if (ep + 1) % 2 == 0:
            avg_reward = np.mean(ep_reward_history) if ep_reward_history else 0
            print(f"Episode {ep+1} | Avg Return (last 50): {avg_reward:.1f}")

    torch.save(model.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()