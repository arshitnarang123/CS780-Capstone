from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass
from typing import Deque

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

STACK = 4
OBS_DIM = 18 * STACK


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=OBS_DIM, n_actions=5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class Replay:
    def __init__(self, cap=100000):
        self.buf: Deque[Transition] = deque(maxlen=cap)

    def add(self, t):
        self.buf.append(t)

    def sample(self, batch):
        idx = np.random.choice(len(self.buf), batch, replace=False)
        items = [self.buf[i] for i in idx]
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items])
        r = np.array([it.r for it in items])
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items])
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def stack_obs(frames):
    return np.concatenate(frames, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=2000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q = DuelingDQN()
    tgt = DuelingDQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay()

    steps = 0

    def eps_fn(t):
        if t >= args.eps_decay:
            return args.eps_end
        return args.eps_start + (t / args.eps_decay) * (args.eps_end - args.eps_start)

    for ep in tqdm(range(args.episodes)):
        env = OBELIX(
            max_steps=args.max_steps,
            difficulty=args.difficulty,
            wall_obstacles=args.wall_obstacles,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )

        s = env.reset(seed=args.seed + ep)

        buffer = deque(maxlen=STACK)
        for _ in range(STACK):
            buffer.append(s)

        S = stack_obs(buffer)

        for _ in range(args.max_steps):
            eps = eps_fn(steps)

            if np.random.rand() < eps:
                a = np.random.randint(5)
            else:
                with torch.no_grad():
                    qvals = q(torch.tensor(S).float().unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qvals))

            s2, r, done = env.step(ACTIONS[a], render=False)

            buffer.append(s2)
            S2 = stack_obs(buffer)

            replay.add(Transition(S, a, r, S2, done))

            S = S2
            steps += 1

            if len(replay) >= max(args.batch, args.warmup):
                sb, ab, rb, s2b, db = replay.sample(args.batch)

                sb = torch.tensor(sb).float()
                ab = torch.tensor(ab).long()
                rb = torch.tensor(rb).float()
                s2b = torch.tensor(s2b).float()
                db = torch.tensor(db).float()

                with torch.no_grad():
                    next_q = q(s2b)
                    next_a = torch.argmax(next_q, dim=1)
                    next_q_tgt = tgt(s2b)
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                    target = rb + args.gamma * (1 - db) * next_val

                pred = q(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

    torch.save(q.state_dict(), args.out)


if __name__ == "__main__":
    main()