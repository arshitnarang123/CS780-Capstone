from __future__ import annotations
import argparse, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class DuelingDRQN(nn.Module):
    def __init__(self, obs_dim=18, hidden_dim=256, n_actions=5):
        super().__init__()
        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = torch.relu(self.fc(x))
        out, hidden = self.lstm(x, hidden)

        V = self.value(out)
        A = self.advantage(out)
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q, hidden


class Replay:
    def __init__(self, cap=50000):
        self.buf = deque(maxlen=cap)

    def add(self, episode):
        self.buf.append(episode)

    def sample(self, batch):
        idx = np.random.choice(len(self.buf), batch)
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--wall_obstacles", action="store_true")
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("env", args.obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    q = DuelingDRQN()
    tgt = DuelingDRQN()
    tgt.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=1e-4)

    replay = Replay()

    gamma = 0.99
    batch_size = 32
    seq_len = 50
    burn_in = 10
    target_sync = 1000

    steps = 0

    def eps(t):
        return max(0.1, 1.0 - t / 500000)

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            seed=ep,
        )

        obs = env.reset(seed=ep)
        hidden = None
        episode = []
        episode_reward = 0

        for _ in range(args.max_steps):
            if random.random() < eps(steps):
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    qvals, hidden = q(torch.tensor(obs).float().unsqueeze(0), hidden)
                    action = int(torch.argmax(qvals[0, -1]))

            next_obs, r, done = env.step(ACTIONS[action], render=False)

            episode.append((obs, action, r, next_obs, done))
            episode_reward += r

            obs = next_obs
            steps += 1

            if done:
                break

        replay.add(episode)

        if len(replay) > batch_size:
            batch = replay.sample(batch_size)

            for seq in batch:
                if len(seq) < seq_len:
                    continue

                start = random.randint(0, len(seq) - seq_len)
                sub = seq[start:start + seq_len]

                s = torch.tensor([x[0] for x in sub]).float()
                a = torch.tensor([x[1] for x in sub]).long()
                r = torch.tensor([x[2] for x in sub]).float()
                s2 = torch.tensor([x[3] for x in sub]).float()
                d = torch.tensor([x[4] for x in sub]).float()

                with torch.no_grad():
                    _, h = q(s[:burn_in].unsqueeze(0))

                q_vals, _ = q(s[burn_in:].unsqueeze(0), h)
                q_vals = q_vals.squeeze(0)

                with torch.no_grad():
                    next_q_online, _ = q(s2[burn_in:].unsqueeze(0), h)
                    next_a = torch.argmax(next_q_online, dim=-1)

                    next_q_target, _ = tgt(s2[burn_in:].unsqueeze(0), h)
                    next_q = next_q_target.gather(2, next_a.unsqueeze(-1)).squeeze(-1).squeeze(0)

                    y = r[burn_in:] + gamma * (1 - d[burn_in:]) * next_q

                pred = q_vals.gather(1, a[burn_in:].unsqueeze(1)).squeeze(1)

                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()

        if steps % target_sync == 0:
            tgt.load_state_dict(q.state_dict())

        print(f"Episode {ep} | Reward: {episode_reward:.2f}")

    torch.save(q.state_dict(), "weights_dueling_dqn_lstm.pth")


if __name__ == "__main__":
    main()