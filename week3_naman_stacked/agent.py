from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

STACK = 4
OBS_DIM = 18 * STACK


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)


_model: Optional[DQN] = None
_buffer: Optional[deque] = None


def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    model = DQN()
    sd = torch.load(wpath, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    _model = model


def _stack():
    return np.concatenate(list(_buffer), axis=0)


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _buffer

    _load_once()

    if _buffer is None:
        _buffer = deque(maxlen=STACK)
        for _ in range(STACK):
            _buffer.append(obs)
    else:
        _buffer.append(obs)

    S = _stack()

    x = torch.tensor(S).float().unsqueeze(0)
    q = _model(x)

    a = int(torch.argmax(q))
    return ACTIONS[a]