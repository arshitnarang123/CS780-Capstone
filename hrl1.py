"""
Offline trainer: Hierarchical RL (Hybrid DQN + Heuristics) for OBELIX.
"""

import argparse
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
SKILLS = ["SEARCH", "TRACK", "PUSH"]
SKILL_DURATION = 5  # The Manager waits 5 steps before picking a new skill

# --- THE WORKERS (HEURISTIC SKILLS) ---
def execute_skill(skill_idx, obs, step_in_skill):
    """Hardcoded logic for the 3 low-level workers."""
    ir_sensor = obs[16]
    stuck_flag = obs[17]
    sonar_sum = sum(obs[0:16])
    
    if skill_idx == 0:  # SEARCH Worker
        # Sweeping pattern: Turn a bit, then move forward
        if step_in_skill < 2:
            return 0  # "L45"
        else:
            return 2  # "FW"
            
    elif skill_idx == 1:  # TRACK Worker
        # If we see it, move forward. If we lose it, spin to find it.
        if sonar_sum > 0:
            return 2  # "FW"
        else:
            return 3  # "R22"
            
    elif skill_idx == 2:  # PUSH Worker
        # Just slam forward. If stuck against a wall, try to wiggle free.
        if stuck_flag == 1:
            return 0  # "L45" (Wiggle)
        return 2  # "FW"

# --- THE MANAGER (META-CONTROLLER) ---
class MetaController(nn.Module):
    def __init__(self, in_dim=18, n_skills=len(SKILLS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_skills) # Outputs Q-values for SKILLS, not actions
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, cap=50000):
        self.buf = deque(maxlen=cap)
    def add(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (torch.tensor(s, dtype=torch.float32), 
                torch.tensor(a, dtype=torch.int64), 
                torch.tensor(r, dtype=torch.float32), 
                torch.tensor(s2, dtype=torch.float32), 
                torch.tensor(d, dtype=torch.float32))
    def __len__(self): return len(self.buf)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights_hrl.pth")
    ap.add_argument("--episodes", type=int, default=1000)
    args = ap.parse_args()

    env = import_obelix(args.obelix_py)(max_steps=500, difficulty=3, seed=0, wall_obstacles=True, box_speed=2,scaling_factor=5,
            arena_size=500)
    
    q_net = MetaController()
    tgt_net = MetaController()
    tgt_net.load_state_dict(q_net.state_dict())
    
    opt = optim.Adam(q_net.parameters(), lr=1e-3)
    replay = ReplayBuffer()
    
    epsilon = 1.0
    eps_decay = 0.995
    steps = 0

    for ep in tqdm(range(args.episodes), desc="Training Manager"):
        state = env.reset()
        ep_ret = 0
        done = False
        
        while not done:
            # 1. Manager picks a skill
            if random.random() < epsilon:
                skill = random.randint(0, len(SKILLS) - 1)
            else:
                with torch.no_grad():
                    qs = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    skill = int(torch.argmax(qs).item())

            # 2. Worker executes the skill for N steps
            accumulated_reward = 0
            next_state = state
            
            for step_in_skill in range(SKILL_DURATION):
                action_idx = execute_skill(skill, next_state, step_in_skill)
                next_state, r, done = env.step(ACTIONS[action_idx])
                accumulated_reward += r
                if done:
                    break
            
            # 3. Manager learns from the result of the skill
            replay.add(state, skill, accumulated_reward, next_state, done)
            ep_ret += accumulated_reward
            state = next_state
            steps += 1

            # 4. Standard DQN Training Step
            if len(replay) > 256:
                s, a, r, s2, d = replay.sample(128)
                with torch.no_grad():
                    max_q_next = tgt_net(s2).max(1)[0]
                    target = r + 0.99 * max_q_next * (1 - d)
                
                current_q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(current_q, target)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if steps % 1000 == 0:
                    tgt_net.load_state_dict(q_net.state_dict())

        epsilon = max(0.05, epsilon * eps_decay)
        
        if (ep + 1) % 1 == 0:
            print(f"Episode {ep+1} | Return: {ep_ret:.1f} | Epsilon: {epsilon:.2f}")

    torch.save(q_net.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()