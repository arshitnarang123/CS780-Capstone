"""
Offline trainer: Recurrent PPO (LSTM) for OBELIX — stabilized version.
Fixes: reward clipping, higher entropy, lower lr, grad norm logging.

Example:
  python train_ppo_lstm_stable.py --obelix_py ./obelix.py --out weights_stable.pth --episodes 500 --max_steps 500 --difficulty 3
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18

class RecurrentActorCritic(nn.Module):
    def __init__(self, in_dim=OBS_DIM, n_actions=len(ACTIONS), hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh()
        )
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        features = self.fc(x)
        lstm_out, new_hidden = self.lstm(features, hidden)
        logits = self.actor(lstm_out)
        value = self.critic(lstm_out)
        return logits, value, new_hidden


class RolloutBuffer:
    def __init__(self):
        self.states     = []
        self.actions    = []
        self.logprobs   = []
        self.rewards    = []
        self.values     = []
        self.is_terminals = []

    def clear(self):
        self.states.clear();     self.actions.clear()
        self.logprobs.clear();   self.rewards.clear()
        self.values.clear();     self.is_terminals.clear()


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def shape_reward(reward, obs_prev, obs_next, action_idx, first_contact_given):
    shaped = reward

    any_sensor_prev = any(obs_prev[:16])
    any_sensor_next = any(obs_next[:16])
    ir_next         = bool(obs_next[16])
    stuck_next      = bool(obs_next[17])

    # Moving forward while sensors active → approaching box
    if action_idx == 2 and any_sensor_next:
        shaped += 2.0

    # One-time bonus for first sensor contact in episode
    if any_sensor_next and not first_contact_given:
        shaped += 5.0
        first_contact_given = True

    # IR active but not moving forward → penalise rotating when aligned
    if ir_next and action_idx != 2:
        shaped -= 2.0

    # Aimless rotation when completely blind
    if not any_sensor_next and action_idx in [0, 1, 3, 4]:
        shaped -= 0.3

    # Stuck against wall without box contact
    if stuck_next and not ir_next:
        shaped -= 5.0

    return shaped, first_contact_given


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",    type=str,   required=True)
    ap.add_argument("--out",          type=str,   default="weights_stable_v2.pth")
    ap.add_argument("--load",         type=str,   default=None)
    ap.add_argument("--episodes",     type=int,   default=500)
    ap.add_argument("--max_steps",    type=int,   default=500)
    ap.add_argument("--difficulty",   type=int,   default=3)
    ap.add_argument("--hidden_size",  type=int,   default=128)
    ap.add_argument("--lr",           type=float, default=1e-4)       # ← lowered from 3e-4
    ap.add_argument("--gamma",        type=float, default=0.99)
    ap.add_argument("--gae_lambda",   type=float, default=0.95)
    ap.add_argument("--clip_eps",     type=float, default=0.2)
    ap.add_argument("--ppo_epochs",   type=int,   default=4)
    ap.add_argument("--entropy_coef", type=float, default=0.05)       # ← raised from 0.01
    ap.add_argument("--reward_clip",  type=float, default=10.0)       # ← new: clip magnitude
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    model  = RecurrentActorCritic(hidden_size=args.hidden_size)

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location="cpu"))
        print(f"Loaded weights from: {args.load}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    buffer    = RolloutBuffer()

    env = OBELIX(
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        seed=args.seed,
        scaling_factor=5,
        arena_size=500,
        wall_obstacles=True,
        box_speed=2
    )

    best_reward = -float('inf')
    best_out    = args.out.replace(".pth", "_best.pth")

    for ep in tqdm(range(args.episodes), desc="Training"):
        state            = env.reset(seed=args.seed + ep)
        ep_reward        = 0.0
        ep_shaped_reward = 0.0
        ep_wall_hits     = 0      # ← track how often agent hits walls
        first_contact_given = False

        hx = torch.zeros(1, 1, model.hidden_size)
        cx = torch.zeros(1, 1, model.hidden_size)
        hidden = (hx, cx)

        # ── 1. Rollout ───────────────────────────────────────────────────────
        for step in range(args.max_steps):
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state, dtype=torch.float32).view(1, 1, OBS_DIM)
                logits, value, hidden = model(state_tensor, hidden)
                dist    = Categorical(logits=logits.squeeze())
                action  = dist.sample()
                logprob = dist.log_prob(action)

            next_state, reward, done = env.step(ACTIONS[action.item()])

            # Track wall hits for diagnostics
            if reward <= -200:
                ep_wall_hits += 1

            # Compute shaped reward
            shaped_r, first_contact_given = shape_reward(
                reward, state, next_state,
                action.item(), first_contact_given
            )

            # ── KEY FIX: clip reward before storing ──────────────────────────
            # Prevents -200 wall penalties from causing massive gradient spikes.
            # Agent still learns walls are bad (gets -10 instead of -200),
            # but training stays stable.
            clipped_r = float(np.clip(shaped_r, -args.reward_clip, 2010.0))
            # ────────────────────────────────────────────────────────────────

            ep_reward        += reward      # track REAL unclipped reward
            ep_shaped_reward += shaped_r    # track shaped for monitoring

            buffer.states.append(state)
            buffer.actions.append(action.item())
            buffer.logprobs.append(logprob.item())
            buffer.rewards.append(clipped_r)   # ← train on CLIPPED shaped reward
            buffer.values.append(value.item())
            buffer.is_terminals.append(done)

            state = next_state
            if done:
                break

        # ── 2. GAE Advantages ────────────────────────────────────────────────
        returns    = []
        advantages = []
        disc_r     = 0
        next_val   = 0
        adv        = 0

        for r, v, is_term in zip(
                reversed(buffer.rewards),
                reversed(buffer.values),
                reversed(buffer.is_terminals)):
            if is_term:
                disc_r = 0; next_val = 0; adv = 0
            delta    = r + args.gamma * next_val - v
            adv      = delta + args.gamma * args.gae_lambda * adv
            advantages.insert(0, adv)
            disc_r   = r + args.gamma * disc_r
            returns.insert(0, disc_r)
            next_val = v

        old_states   = torch.tensor(
            np.array(buffer.states), dtype=torch.float32).unsqueeze(0)
        old_actions  = torch.tensor(buffer.actions,  dtype=torch.float32)
        old_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32)
        returns      = torch.tensor(returns,      dtype=torch.float32)
        advantages   = torch.tensor(advantages,   dtype=torch.float32)
        advantages   = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── 3. PPO Update ────────────────────────────────────────────────────
        hx_train = torch.zeros(1, 1, model.hidden_size)
        cx_train = torch.zeros(1, 1, model.hidden_size)
        last_grad_norm = 0.0

        for _ in range(args.ppo_epochs):
            logits, values, _ = model(old_states, (hx_train, cx_train))
            logits = logits.squeeze(0)
            values = values.squeeze(0).squeeze(-1)

            dist         = Categorical(logits=logits)
            logprobs     = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            ratios = torch.exp(logprobs - old_logprobs)
            surr1  = ratios * advantages
            surr2  = torch.clamp(
                ratios, 1 - args.clip_eps, 1 + args.clip_eps) * advantages

            actor_loss   = -torch.min(surr1, surr2).mean()
            critic_loss  = nn.functional.mse_loss(values, returns)
            entropy_loss = -args.entropy_coef * dist_entropy.mean()
            loss         = actor_loss + 0.5 * critic_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            last_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        buffer.clear()

        # ── Save best ────────────────────────────────────────────────────────
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(model.state_dict(), best_out)
            tqdm.write(
                f"  ✓ Ep {ep+1:4d} | New best: {best_reward:.1f} — saved")

        if (ep + 1) % 1 == 0:
            tqdm.write(
                f"Ep {ep+1:4d} | Real: {ep_reward:.1f}"
                f" | Shaped: {ep_shaped_reward:.1f}"
                f" | WallHits: {ep_wall_hits}"
                f" | GradNorm: {last_grad_norm:.3f}"
                f" | Best: {best_reward:.1f}")

    torch.save(model.state_dict(), args.out)
    print(f"\nFinal : {args.out}")
    print(f"Best  : {best_out}  (real reward: {best_reward:.1f})")


if __name__ == "__main__":
    main()