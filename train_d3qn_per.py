"""
Dueling DDQN with Prioritized Experience Replay for OBELIX.

Key ideas:
  - Dueling architecture: separates V(s) from A(s,a) — better value estimates
  - Double Q-learning: reduces overestimation bias  
  - Prioritized replay: rare +2000 episodes replayed much more often
  - Phase feature: inferred attach state fed as extra input

Run:
  python train_d3qn_per.py --obelix_py ./obelix.py \
      --out weights_d3qn.pth --episodes 1000 --difficulty 0
"""

import argparse, random, math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS  = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM  = 18


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """
    Stores transitions with priorities. High TD-error transitions
    are sampled more often — ensures rare successes (+2000) are
    replayed frequently instead of being drowned out.
    """
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha    = alpha          # priority exponent (0=uniform, 1=full)
        self.buffer   = []
        self.pos      = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        # New transitions get max priority so they're sampled at least once
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        n = len(self.buffer)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights — corrects for sampling bias
        total   = n
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones,   dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6  # small constant avoids zero

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Dueling Network Architecture
# ─────────────────────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """
    Dueling architecture splits into:
      Value stream    V(s)      — how good is this state?
      Advantage stream A(s,a)  — how much better is each action?
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    
    This is better than plain DQN because the agent can learn
    state values without needing to evaluate every action.
    Especially useful in OBELIX where many states have similar
    action values (e.g. when no sensors are active).
    """
    def __init__(self, in_dim, n_actions, hidden=128):
        super().__init__()

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Value stream — single scalar
        self.value_stream = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # Advantage stream — one value per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        feat = self.feature(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        # Combine: subtract mean advantage for stability
        Q    = V + A - A.mean(dim=-1, keepdim=True)
        return Q


# ─────────────────────────────────────────────────────────────────────────────
# Reward shaping (same as before)
# ─────────────────────────────────────────────────────────────────────────────
def shape_reward(reward, obs_prev, obs_next, action_idx, first_contact_given):
    shaped   = reward
    any_next = any(obs_next[:16])
    ir_next  = bool(obs_next[16])
    stk_next = bool(obs_next[17])

    if action_idx == 2 and any_next:               shaped += 2.0
    if any_next and not first_contact_given:        shaped += 5.0; first_contact_given = True
    if ir_next  and action_idx != 2:               shaped -= 2.0
    if not any_next and action_idx in [0,1,3,4]:   shaped -= 0.3
    if stk_next and not ir_next:                   shaped -= 5.0

    return shaped, first_contact_given


# ─────────────────────────────────────────────────────────────────────────────
# Import helper
# ─────────────────────────────────────────────────────────────────────────────
def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights_d3qn.pth")
    ap.add_argument("--load",           type=str,   default=None)
    ap.add_argument("--episodes",       type=int,   default=1000)
    ap.add_argument("--max_steps",      type=int,   default=500)
    ap.add_argument("--difficulty",     type=int,   default=0)
    ap.add_argument("--hidden",         type=int,   default=128)
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--batch_size",     type=int,   default=64)
    ap.add_argument("--buffer_size",    type=int,   default=50000)
    ap.add_argument("--target_update",  type=int,   default=500,
                    help="Update target network every N steps")
    ap.add_argument("--eps_start",      type=float, default=1.0)
    ap.add_argument("--eps_end",        type=float, default=0.05)
    ap.add_argument("--eps_decay",      type=int,   default=10000,
                    help="Steps over which epsilon decays")
    ap.add_argument("--per_alpha",      type=float, default=0.6)
    ap.add_argument("--per_beta_start", type=float, default=0.4)
    ap.add_argument("--per_beta_end",   type=float, default=1.0)
    ap.add_argument("--reward_clip",    type=float, default=10.0)
    ap.add_argument("--learn_start",    type=int,   default=1000,
                    help="Steps before learning starts")
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    # Online and target networks
    online_net = DuelingDQN(OBS_DIM, len(ACTIONS), args.hidden)
    target_net = DuelingDQN(OBS_DIM, len(ACTIONS), args.hidden)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    if args.load:
        online_net.load_state_dict(torch.load(args.load, map_location="cpu"))
        target_net.load_state_dict(online_net.state_dict())
        print(f"Loaded: {args.load}")

    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    buffer    = PrioritizedReplayBuffer(args.buffer_size, args.per_alpha)

    env = OBELIX(
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        seed=args.seed,
        scaling_factor=5,
        arena_size=500,
        wall_obstacles=True,
        box_speed=2
    )

    best_reward  = -float('inf')
    best_out     = args.out.replace(".pth", "_best.pth")
    total_steps  = 0
    recent_rewards = deque(maxlen=20)  # moving average window

    for ep in tqdm(range(args.episodes), desc="Training"):
        state = env.reset(seed=args.seed + ep)
        ep_reward   = 0.0
        ep_wall_hits = 0
        first_contact_given = False

        for step in range(args.max_steps):

            # ── Epsilon-greedy action selection ──────────────────────────────
            eps = args.eps_end + (args.eps_start - args.eps_end) * \
                  math.exp(-total_steps / args.eps_decay)

            if random.random() < eps:
                action_idx = random.randrange(len(ACTIONS))
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action_idx = int(online_net(s_t).argmax(dim=1).item())

            next_state, reward, done = env.step(ACTIONS[action_idx])
            if reward <= -200: ep_wall_hits += 1

            # ── Reward shaping ────────────────────────────────────────────────
            shaped_r, first_contact_given = shape_reward(
                reward, state, next_state, action_idx, first_contact_given)
            clipped_r = float(np.clip(shaped_r, -args.reward_clip, 2010.0))

            ep_reward   += reward
            total_steps += 1

            buffer.push(state, action_idx, clipped_r, next_state, float(done))
            state = next_state

            # ── Learning step ─────────────────────────────────────────────────
            if len(buffer) >= args.learn_start and len(buffer) >= args.batch_size:

                # Beta anneals from 0.4 → 1.0 over training
                beta = min(args.per_beta_end,
                           args.per_beta_start + total_steps *
                           (args.per_beta_end - args.per_beta_start) /
                           (args.episodes * args.max_steps))

                states_b, actions_b, rewards_b, \
                next_states_b, dones_b, indices, weights_b = \
                    buffer.sample(args.batch_size, beta)

                states_b      = torch.tensor(states_b,      dtype=torch.float32)
                actions_b     = torch.tensor(actions_b,     dtype=torch.long)
                rewards_b     = torch.tensor(rewards_b,     dtype=torch.float32)
                next_states_b = torch.tensor(next_states_b, dtype=torch.float32)
                dones_b       = torch.tensor(dones_b,       dtype=torch.float32)
                weights_b     = torch.tensor(weights_b,     dtype=torch.float32)

                # ── Double Q-learning target ──────────────────────────────────
                # Online net selects action, target net evaluates it
                with torch.no_grad():
                    next_actions = online_net(next_states_b).argmax(dim=1)
                    next_q       = target_net(next_states_b)\
                                   .gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    targets      = rewards_b + args.gamma * next_q * (1 - dones_b)

                # Current Q values
                current_q = online_net(states_b)\
                            .gather(1, actions_b.unsqueeze(1)).squeeze(1)

                # TD errors for priority update
                td_errors = (targets - current_q).detach().cpu().numpy()

                # Weighted Huber loss (less sensitive to outliers than MSE)
                loss = (weights_b * nn.functional.huber_loss(
                    current_q, targets, reduction='none')).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()

                # Update priorities in replay buffer
                buffer.update_priorities(indices, td_errors)

            # ── Target network update ─────────────────────────────────────────
            if total_steps % args.target_update == 0:
                target_net.load_state_dict(online_net.state_dict())

            if done:
                break

        # ── Logging and saving ────────────────────────────────────────────────
        recent_rewards.append(ep_reward)
        moving_avg = np.mean(recent_rewards)

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(online_net.state_dict(), best_out)
            tqdm.write(
                f"  ✓ Ep {ep+1:4d} | New best: {best_reward:.1f} — saved")

        if (ep + 1) % 1 == 0:
            tqdm.write(
                f"Ep {ep+1:4d} | Real: {ep_reward:.1f}"
                f" | Avg20: {moving_avg:.1f}"
                f" | WallHits: {ep_wall_hits}"
                f" | Eps: {eps:.3f}"
                f" | BufSize: {len(buffer)}"
                f" | Best: {best_reward:.1f}")

    torch.save(online_net.state_dict(), args.out)
    print(f"\nFinal: {args.out}")
    print(f"Best:  {best_out}  (reward: {best_reward:.1f})")


if __name__ == "__main__":
    main()