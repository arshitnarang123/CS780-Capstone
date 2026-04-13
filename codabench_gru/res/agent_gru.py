"""
Agent file for GRU variant.
policy(obs, rng=None) -> str
"""

import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM   = 18
MAX_STEPS = 1000


class GRUActorCritic(nn.Module):
    def __init__(self, in_dim=OBS_DIM, n_actions=len(ACTIONS), hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc     = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh())
        self.gru    = nn.GRU(64, hidden_size, batch_first=True)
        self.actor  = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        features = self.fc(x)
        out, new_hidden = self.gru(features, hidden)
        return self.actor(out), self.critic(out), new_hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)


_model  = None
_hidden = None
_steps  = 0

def _load_once():
    global _model
    if _model is not None: return
    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights_ppo_gru_best.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Weights not found: {wpath}")
    m = GRUActorCritic()
    m.load_state_dict(torch.load(wpath, map_location="cpu"))
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs, rng=None) -> str:
    global _hidden, _steps
    _load_once()

    if _steps >= MAX_STEPS or _hidden is None:
        _hidden = _model.init_hidden()
        _steps  = 0

    _steps += 1
    x = torch.tensor(obs, dtype=torch.float32).view(1, 1, OBS_DIM)
    logits, _, _hidden = _model(x, _hidden)
    return ACTIONS[int(torch.argmax(logits.squeeze()).item())]