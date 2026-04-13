"""
Hierarchical RL evaluation agent for OBELIX.
Strict Codabench signature compliance.
"""

import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
SKILLS = ["SEARCH", "TRACK", "PUSH"]
SKILL_DURATION = 5

def execute_skill(skill_idx, obs, step_in_skill):
    ir_sensor = obs[16]
    stuck_flag = obs[17]
    sonar_sum = sum(obs[0:16])
    
    if skill_idx == 0:
        return 0 if step_in_skill < 2 else 2
    elif skill_idx == 1:
        return 2 if sonar_sum > 0 else 3
    elif skill_idx == 2:
        return 0 if stuck_flag == 1 else 2

class MetaController(nn.Module):
    def __init__(self, in_dim=18, n_skills=len(SKILLS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_skills)
        )
    def forward(self, x):
        return self.net(x)

_model = None
_current_skill = None
_skill_step = 0

def _load_once():
    global _model
    if _model is not None: return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_hrl_1000step.pth")
    m = MetaController()
    m.load_state_dict(torch.load(wpath, map_location="cpu"))
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs, *args, **kwargs) -> str:
    global _current_skill, _skill_step
    _load_once()

    # 1. Meta-Controller decides a skill (only every 5 steps)
    if _current_skill is None or _skill_step >= SKILL_DURATION:
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        qs = _model(x)
        _current_skill = int(torch.argmax(qs).item())
        _skill_step = 0

    # 2. Low-level Worker executes the skill
    action_idx = execute_skill(_current_skill, obs, _skill_step)
    _skill_step += 1
    
    return ACTIONS[action_idx]