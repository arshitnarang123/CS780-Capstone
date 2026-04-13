from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class DuelingDRQN(nn.Module):
    def __init__(self, obs_dim=18, hidden_dim=256, n_actions=5):
        super().__init__()
        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x, hidden=None):
        if x.dim() == 1:
            x = x.view(1, 1, -1)

        x = torch.relu(self.fc(x))
        out, hidden = self.lstm(x, hidden)

        V = self.value(out)
        A = self.advantage(out)
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q, hidden


_model: Optional[DuelingDRQN] = None
_hidden = None


def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_dueling_dqn_lstm.pth")

    model = DuelingDRQN()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    _model = model


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hidden
    _load_once()

    q, _hidden = _model(torch.tensor(obs).float(), _hidden)
    action = int(torch.argmax(q[0, -1]))
    return ACTIONS[action]