"""
PPO evaluation agent for OBELIX.
Includes built-in frame stacking and strict Codabench signature compliance.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
STACK_SIZE = 4
OBS_DIM = 18
MAX_STEPS_PER_EPISODE = 2000

class ActorCritic(nn.Module):
    def __init__(self, in_dim=OBS_DIM * STACK_SIZE, n_actions=len(ACTIONS)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
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

_model = None
_frame_buffer = None
_step_count = 0

def _load_once():
    global _model
    if _model is not None:
        return
        
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_ppo_stacking.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py.")
        
    model = ActorCritic()
    sd = torch.load(wpath, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    _model = model

@torch.no_grad()
def policy(obs, *args, **kwargs) -> str:
    global _frame_buffer, _step_count
    _load_once()

    # Reset buffer if we know we hit the environment's hard time limit
    if _step_count >= MAX_STEPS_PER_EPISODE:
        _frame_buffer = None
        _step_count = 0

    _step_count += 1

    # Initialize frame buffer using safe memory copies
    if _frame_buffer is None:
        _frame_buffer = deque([obs.copy() for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)
    else:
        _frame_buffer.append(obs.copy())

    # Flatten the 4 frames into a 72-bit vector
    stacked_obs = np.concatenate(_frame_buffer)
    
    x = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0)
    
    # Extract actor logits
    logits, _ = _model(x)
    
    # Greedy selection
    best_action = int(torch.argmax(logits, dim=1).item())
    
    return ACTIONS[best_action]