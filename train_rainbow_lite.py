"""
Rainbow-lite: Dueling DDQN + PER + N-step returns + Noisy Networks.

New vs previous D3QN:
  - N-step returns: faster reward propagation for sparse +2000
  - Noisy Networks: state-dependent exploration, no epsilon needed
  - Soft target update: smoother target network updates

Run:
  python train_rainbow_lite.py --obelix_py ./obelix.py \
      --out weights_rainbow.pth --episodes 1000 --difficulty 0
"""

import argparse, random, math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18


# ─────────────────────────────────────────────────────────────────────────────
# Noisy Linear Layer
# ─────────────────────────────────────────────────────────────────────────────
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma        = sigma

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def sample_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return nn.functional.linear(x, w, b)


# ─────────────────────────────────────────────────────────────────────────────
# Dueling Network with Noisy Streams
# ─────────────────────────────────────────────────────────────────────────────
class RainbowDQN(nn.Module):
    """
    Dueling architecture with NoisyLinear in the value/advantage heads.
    Shared feature extractor uses regular Linear (no noise needed there).
    """
    def __init__(self, in_dim, n_actions, hidden=128):
        super().__init__()

        # Shared feature layers — regular linear, no noise
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Value stream — noisy
        self.value_hidden = NoisyLinear(128, hidden)
        self.value_out    = NoisyLinear(hidden, 1)

        # Advantage stream — noisy
        self.adv_hidden = NoisyLinear(128, hidden)
        self.adv_out    = NoisyLinear(hidden, n_actions)

        self.n_actions = n_actions

    def forward(self, x):
        feat = self.feature(x)

        v = torch.relu(self.value_hidden(feat))
        v = self.value_out(v)

        a = torch.relu(self.adv_hidden(feat))
        a = self.adv_out(a)

        # Dueling combination
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q

    def sample_noise(self):
        """Resample noise in all noisy layers — call once per training step."""
        self.value_hidden.sample_noise()
        self.value_out.sample_noise()
        self.adv_hidden.sample_noise()
        self.adv_out.sample_noise()


# ─────────────────────────────────────────────────────────────────────────────
# N-Step Replay Buffer with Priorities
# ─────────────────────────────────────────────────────────────────────────────
class NStepPrioritizedBuffer:
    """
    Combines N-step return computation with prioritized replay.
    
    N-step: accumulates n transitions before storing, so the stored
    reward already contains n steps of discounted future reward.
    This means the +2000 terminal reward propagates n steps per update
    instead of just 1.
    """
    def __init__(self, capacity, alpha=0.6, n_step=5, gamma=0.99):
        self.capacity  = capacity
        self.alpha     = alpha
        self.n_step    = n_step
        self.gamma     = gamma

        self.buffer    = []
        self.pos       = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)

        # Temporary buffer holding the last n transitions
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_info(self):
        """
        Compute n-step return from the temporary buffer.
        Returns (state, action, n_step_reward, next_state, done)
        where n_step_reward = r0 + g*r1 + g^2*r2 + ... + g^(n-1)*r_{n-1}
        """
        reward     = 0.0
        next_state = self.n_step_buffer[-1][3]
        done       = self.n_step_buffer[-1][4]

        for transition in reversed(self.n_step_buffer):
            r, ns, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state = ns
                done       = d

        state  = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        return state, action, reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Only store once we have n steps accumulated
        if len(self.n_step_buffer) < self.n_step:
            return

        state, action, n_reward, next_state, done = self._get_n_step_info()

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, n_reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, n_reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def flush(self):
        """
        At episode end, flush remaining transitions in n_step_buffer.
        Important: don't lose the last n-1 transitions of each episode.
        """
        while len(self.n_step_buffer) > 0:
            state, action, n_reward, next_state, done = self._get_n_step_info()
            max_prio = self.priorities.max() if self.buffer else 1.0
            if len(self.buffer) < self.capacity:
                self.buffer.append((state, action, n_reward, next_state, done))
            else:
                self.buffer[self.pos] = (state, action, n_reward, next_state, done)
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity
            self.n_step_buffer.popleft()

    def sample(self, batch_size, beta=0.4):
        n     = len(self.buffer)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states),
            np.array(dones,       dtype=np.float32),
            indices,
            np.array(weights,     dtype=np.float32)
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Reward Shaping
# ─────────────────────────────────────────────────────────────────────────────
def shape_reward(reward, obs_prev, obs_next, action_idx, first_contact_given):
    shaped   = reward
    any_next = any(obs_next[:16])
    ir_next  = bool(obs_next[16])
    stk_next = bool(obs_next[17])

    if action_idx == 2 and any_next:              shaped += 2.0
    if any_next and not first_contact_given:      shaped += 5.0; first_contact_given = True
    if ir_next  and action_idx != 2:              shaped -= 2.0
    if not any_next and action_idx in [0,1,3,4]:  shaped -= 0.3
    if stk_next and not ir_next:                  shaped -= 5.0
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
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights_rainbow.pth")
    ap.add_argument("--load",            type=str,   default=None)
    ap.add_argument("--episodes",        type=int,   default=1000)
    ap.add_argument("--max_steps",       type=int,   default=500)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--hidden",          type=int,   default=128)
    ap.add_argument("--lr",              type=float, default=1e-4)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--batch_size",      type=int,   default=64)
    ap.add_argument("--buffer_size",     type=int,   default=50000)
    ap.add_argument("--n_step",          type=int,   default=5)
    ap.add_argument("--tau",             type=float, default=0.005,
                    help="Soft target update rate")
    ap.add_argument("--per_alpha",       type=float, default=0.6)
    ap.add_argument("--per_beta_start",  type=float, default=0.4)
    ap.add_argument("--per_beta_end",    type=float, default=1.0)
    ap.add_argument("--reward_clip",     type=float, default=10.0)
    ap.add_argument("--learn_start",     type=int,   default=1000)
    ap.add_argument("--noisy_sigma",     type=float, default=0.5)
    ap.add_argument("--seed",            type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    online_net = RainbowDQN(OBS_DIM, len(ACTIONS), args.hidden)
    target_net = RainbowDQN(OBS_DIM, len(ACTIONS), args.hidden)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    if args.load:
        online_net.load_state_dict(torch.load(args.load, map_location="cpu"))
        target_net.load_state_dict(online_net.state_dict())
        print(f"Loaded: {args.load}")

    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    buffer    = NStepPrioritizedBuffer(
        args.buffer_size, args.per_alpha, args.n_step, args.gamma)

    env = OBELIX(
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        seed=args.seed,
        scaling_factor=5,
        arena_size=500,
        wall_obstacles=True,
        box_speed=2
    )

    best_reward    = -float('inf')
    best_out       = args.out.replace(".pth", "_best.pth")
    total_steps    = 0
    recent_rewards = deque(maxlen=20)

    for ep in tqdm(range(args.episodes), desc="Training"):
        state = env.reset(seed=args.seed + ep)
        ep_reward   = 0.0
        ep_wall_hits = 0
        first_contact_given = False

        for step in range(args.max_steps):

            # ── Action selection — no epsilon needed with noisy nets ──────────
            online_net.sample_noise()       # resample noise each step
            with torch.no_grad():
                s_t        = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_idx = int(online_net(s_t).argmax(dim=1).item())

            next_state, reward, done = env.step(ACTIONS[action_idx])
            if reward <= -200: ep_wall_hits += 1

            shaped_r, first_contact_given = shape_reward(
                reward, state, next_state, action_idx, first_contact_given)
            clipped_r = float(np.clip(shaped_r, -args.reward_clip, 2010.0))

            ep_reward   += reward
            total_steps += 1

            buffer.push(state, action_idx, clipped_r, next_state, float(done))
            state = next_state

            # ── Learning step ────────────────────────────────────────────────
            if len(buffer) >= args.learn_start and \
               len(buffer) >= args.batch_size:

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

                # Resample noise for target net evaluation
                target_net.sample_noise()

                # Double Q: online selects, target evaluates
                with torch.no_grad():
                    next_actions = online_net(next_states_b).argmax(dim=1)
                    next_q       = target_net(next_states_b)\
                                   .gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    # N-step gamma: gamma^n instead of gamma
                    targets = rewards_b + \
                              (args.gamma ** args.n_step) * next_q * (1 - dones_b)

                current_q = online_net(states_b)\
                            .gather(1, actions_b.unsqueeze(1)).squeeze(1)

                td_errors = (targets - current_q).detach().cpu().numpy()

                loss = (weights_b * nn.functional.huber_loss(
                    current_q, targets, reduction='none')).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()

                buffer.update_priorities(indices, td_errors)

                # ── Soft target update ────────────────────────────────────────
                # Smoother than hard update every N steps
                for p_online, p_target in zip(
                        online_net.parameters(), target_net.parameters()):
                    p_target.data.copy_(
                        args.tau * p_online.data +
                        (1 - args.tau) * p_target.data)

            if done:
                break

        # Flush remaining n-step transitions at episode end
        buffer.flush()

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
                f" | BufSize: {len(buffer)}"
                f" | Best: {best_reward:.1f}")

    torch.save(online_net.state_dict(), args.out)
    print(f"\nFinal: {args.out}")
    print(f"Best:  {best_out}  (reward: {best_reward:.1f})")


if __name__ == "__main__":
    main()