# train_recurrent_ppo.py
from __future__ import annotations
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim=18, hidden_dim=128, n_actions=5):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, h=None):
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        x = self.encoder(x)
        out, h = self.gru(x, h)
        logits = self.actor(out)
        values = self.critic(out).squeeze(-1)
        return logits, values, h

    def step(self, obs, h=None):
        x = torch.as_tensor(obs, dtype=torch.float32)
        logits, values, h = self.forward(x, h)
        return logits[:, -1, :], values[:, -1], h


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = []
    gae = 0.0
    values = values + [0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv.insert(0, gae)
    return adv


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights_recurrent_ppo_improv_v2.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    model = RecurrentActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )

        obs = env.reset(seed=args.seed + ep)

        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        hidden = None

        for _ in range(args.max_steps):
            with torch.no_grad():
                logits, value, hidden = model.step(obs, hidden)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, done = env.step(ACTIONS[action.item()], render=False)

            states.append(np.asarray(obs, dtype=np.float32))
            actions.append(action.item())
            rewards.append(float(reward))
            dones.append(float(done))
            log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))

            obs = next_obs
            hidden = hidden.detach()

            if done:
                break

        if len(states) == 0:
            continue

        advantages = compute_gae(rewards, values, dones, args.gamma, args.lam)
        returns = [a + v for a, v in zip(advantages, values)]

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # states_t = torch.from_numpy(np.array(states, dtype=np.float32)).unsqueeze(0)
        # actions_t = torch.tensor(actions, dtype=torch.long)
        # old_log_probs_t = torch.tensor(log_probs, dtype=torch.float32)
        max_seq = 256
        states_trim = states[-max_seq:]
        states_t = torch.from_numpy(np.array(states_trim, dtype=np.float32)).unsqueeze(0)

        actions_t = torch.tensor(actions[-max_seq:], dtype=torch.long)
        old_log_probs_t = torch.tensor(log_probs[-max_seq:], dtype=torch.float32)
        advantages = advantages[-max_seq:]
        returns = returns[-max_seq:]     
        for _ in range(args.update_epochs):
            logits, values_pred, _ = model(states_t)
            logits = logits.squeeze(0)
            values_pred = values_pred.squeeze(0)

            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values_pred, returns)
            loss = actor_loss + args.vf_coef * critic_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        episode_reward = sum(rewards)    
        if ep == 0:
            running_reward = episode_reward
        else:
            running_reward = 0.9 * running_reward + 0.1 * episode_reward

        print(f"Episode {ep} | Reward: {episode_reward:.2f} | Running Avg: {running_reward:.2f}")
        # if ep % 50 == 0:
        # print(f"Episode {ep}")

    torch.save(model.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()