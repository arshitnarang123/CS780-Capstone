from __future__ import annotations
import argparse, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

STACK = 4
OBS_DIM = 18 * STACK


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, cap=100000):
        self.buf = deque(maxlen=cap)

    def add(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch):
        idx = np.random.choice(len(self.buf), batch, replace=False)
        batch = [self.buf[i] for i in idx]

        s = np.array([b[0] for b in batch], dtype=np.float32)
        a = np.array([b[1] for b in batch])
        r = np.array([b[2] for b in batch])
        s2 = np.array([b[3] for b in batch], dtype=np.float32)
        d = np.array([b[4] for b in batch])

        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


def stack_obs(buf):
    return np.concatenate(buf, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", args.obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    q = DQN()
    tgt = DQN()
    tgt.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=1e-3)
    replay = Replay()

    gamma = 0.99
    batch = 256
    warmup = 2000
    target_sync = 2000

    steps = 0

    def eps(t):
        return max(0.05, 1.0 - t / 200000)

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=ep,
        )

        s = env.reset(seed=ep)

        buf = deque(maxlen=STACK)
        for _ in range(STACK):
            buf.append(s)

        S = stack_obs(buf)

        for _ in range(args.max_steps):
            if random.random() < eps(steps):
                a = random.randint(0, 4)
            else:
                with torch.no_grad():
                    qvals = q(torch.tensor(S).float().unsqueeze(0)).squeeze(0)
                a = int(torch.argmax(qvals))

            s2, r, done = env.step(ACTIONS[a], render=False)

            buf.append(s2)
            S2 = stack_obs(buf)

            replay.add(S, a, r, S2, done)

            S = S2
            steps += 1

            if len(replay) > max(batch, warmup):
                sb, ab, rb, s2b, db = replay.sample(batch)

                sb = torch.tensor(sb)
                ab = torch.tensor(ab)
                rb = torch.tensor(rb)
                s2b = torch.tensor(s2b)
                db = torch.tensor(db).float()

                with torch.no_grad():
                    next_a = torch.argmax(q(s2b), dim=1)
                    next_q = tgt(s2b).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb + gamma * (1 - db) * next_q

                pred = q(sb).gather(1, ab.unsqueeze(1)).squeeze(1)

                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if steps % target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        # if ep % 50 == 0:
        print("Episode", ep)

    torch.save(q.state_dict(), "weights_new_naman.pth")


if __name__ == "__main__":
    main()