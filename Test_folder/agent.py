"""
Agent for Dueling DDQN + PER.
policy(obs, rng=None) -> str
"""

import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS  = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM  = 18


class DuelingDQN(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),    nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        feat = self.feature(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        return V + A - A.mean(dim=-1, keepdim=True)


_model = None

def _load_once():
    global _model
    if _model is not None: return
    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights_d3qn_v2_best.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Weights not found: {wpath}")
    m = DuelingDQN(OBS_DIM, len(ACTIONS))
    m.load_state_dict(torch.load(wpath, map_location="cpu"))
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs, rng=None) -> str:
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return ACTIONS[int(_model(x).argmax(dim=1).item())]