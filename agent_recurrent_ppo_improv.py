# agent.py
from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim=18, hidden_dim=128, n_actions=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, h=None):
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        x = self.encoder(x)
        out, h = self.gru(x, h)
        logits = self.actor(out)
        values = self.critic(out).squeeze(-1)
        return logits, values, h

    def step(self, obs, h=None):
        x = torch.as_tensor(obs, dtype=torch.float32)
        logits, values, h = self.forward(x, h)
        return logits[:, -1, :], values[:, -1], h


_model: Optional[RecurrentActorCritic] = None
_hidden = None
_step_count = 0
_MAX_STEPS_SAFETY = 2000


def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_recurrent_ppo_improv_v2.pth")

    model = RecurrentActorCritic()
    sd = torch.load(wpath, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    _model = model


def reset_state():
    global _hidden, _step_count
    _hidden = None
    _step_count = 0


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hidden, _step_count
    _load_once()

    if _hidden is None or _step_count >= _MAX_STEPS_SAFETY or np.allclose(obs, 0.0):
        _hidden = None
        _step_count = 0

    logits, _, _hidden = _model.step(obs, _hidden)
    action = int(torch.argmax(logits, dim=1).item())

    _step_count += 1
    return ACTIONS[action]