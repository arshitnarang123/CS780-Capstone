"""
Offline trainer: Recurrent PPO (LSTM) for OBELIX — with training logs and plots.
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
from plot_utils import TrainingLogger          # ← added

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18

class RecurrentActorCritic(nn.Module):
    def __init__(self, in_dim=OBS_DIM, n_actions=len(ACTIONS), hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh())
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.actor  = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        features = self.fc(x)
        lstm_out, new_hidden = self.lstm(features, hidden)
        return self.actor(lstm_out), self.critic(lstm_out), new_hidden

class RolloutBuffer:
    def __init__(self):
        self.states=[]; self.actions=[]; self.logprobs=[]
        self.rewards=[]; self.values=[]; self.is_terminals=[]
    def clear(self):
        self.states.clear(); self.actions.clear(); self.logprobs.clear()
        self.rewards.clear(); self.values.clear(); self.is_terminals.clear()

def import_obelix(obelix_py):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",    type=str,   required=True)
    ap.add_argument("--out",          type=str,   default="weights_ppo_lstm_plot.pth")
    ap.add_argument("--episodes",     type=int,   default=500)
    ap.add_argument("--max_steps",    type=int,   default=200)
    ap.add_argument("--difficulty",   type=int,   default=3)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--gamma",        type=float, default=0.99)
    ap.add_argument("--gae_lambda",   type=float, default=0.95)
    ap.add_argument("--clip_eps",     type=float, default=0.2)
    ap.add_argument("--ppo_epochs",   type=int,   default=4)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--run_name",     type=str,   default="ppo_lstm")  # ← added
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    OBELIX    = import_obelix(args.obelix_py)
    model     = RecurrentActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    buffer    = RolloutBuffer()
    logger    = TrainingLogger(run_name=args.run_name)   # ← added

    env = OBELIX(max_steps=args.max_steps, difficulty=args.difficulty,
                 seed=args.seed, scaling_factor=5, arena_size=500,
                 wall_obstacles=True, box_speed=2)

    best_reward = -float('inf')
    best_out    = args.out.replace(".pth", "_best.pth")

    for ep in tqdm(range(args.episodes), desc="Training"):
        state    = env.reset(seed=args.seed + ep)
        ep_reward = 0
        hx = torch.zeros(1, 1, model.hidden_size)
        cx = torch.zeros(1, 1, model.hidden_size)
        hidden = (hx, cx)

        for step in range(args.max_steps):
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state, dtype=torch.float32).view(1, 1, OBS_DIM)
                logits, value, hidden = model(state_tensor, hidden)
                dist    = Categorical(logits=logits.squeeze())
                action  = dist.sample()
                logprob = dist.log_prob(action)

            next_state, reward, done = env.step(ACTIONS[action.item()])
            ep_reward += reward
            buffer.states.append(state);         buffer.actions.append(action.item())
            buffer.logprobs.append(logprob.item()); buffer.rewards.append(reward)
            buffer.values.append(value.item());  buffer.is_terminals.append(done)
            state = next_state
            if done: break

        returns=[]; advantages=[]
        disc_r=0; next_val=0; adv=0
        for r,v,t in zip(reversed(buffer.rewards),
                         reversed(buffer.values),
                         reversed(buffer.is_terminals)):
            if t: disc_r=0; next_val=0; adv=0
            delta   = r + args.gamma*next_val - v
            adv     = delta + args.gamma*args.gae_lambda*adv
            advantages.insert(0, adv)
            disc_r  = r + args.gamma*disc_r
            returns.insert(0, disc_r)
            next_val = v

        old_states   = torch.tensor(
            np.array(buffer.states), dtype=torch.float32).unsqueeze(0)
        old_actions  = torch.tensor(buffer.actions,  dtype=torch.float32)
        old_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32)
        returns      = torch.tensor(returns,      dtype=torch.float32)
        advantages   = torch.tensor(advantages,   dtype=torch.float32)
        advantages   = (advantages-advantages.mean())/(advantages.std()+1e-8)

        hx_train = torch.zeros(1, 1, model.hidden_size)
        cx_train = torch.zeros(1, 1, model.hidden_size)

        for _ in range(args.ppo_epochs):
            logits, values, _ = model(old_states, (hx_train, cx_train))
            logits = logits.squeeze(0); values = values.squeeze(0).squeeze(-1)
            dist         = Categorical(logits=logits)
            logprobs     = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            ratios = torch.exp(logprobs - old_logprobs)
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1-args.clip_eps, 1+args.clip_eps)*advantages
            loss   = (-torch.min(surr1,surr2).mean()
                      + 0.5*nn.functional.mse_loss(values, returns)
                      - args.entropy_coef*dist_entropy.mean())
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        buffer.clear()

        # ── logging ──────────────────────────────────────────────────────────
        logger.log(ep + 1, ep_reward)                          # ← added
        # ─────────────────────────────────────────────────────────────────────

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(model.state_dict(), best_out)
            tqdm.write(f"  ✓ Ep {ep+1} | New best: {best_reward:.1f} — saved")

        if (ep+1) % 50 == 0:
            tqdm.write(f"Ep {ep+1} | Reward: {ep_reward:.1f} | Best: {best_reward:.1f}")

    torch.save(model.state_dict(), args.out)
    logger.save_csv()        # ← added
    logger.save_plots()      # ← added
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()