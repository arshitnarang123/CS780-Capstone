"""
Agent file for LayerNorm 2-layer LSTM variant.
policy(obs, rng=None) -> str
"""

import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS  = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM  = 18
MAX_STEPS = 1000


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear   = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.ln_gates = nn.LayerNorm(4 * hidden_size)
        self.ln_cell  = nn.LayerNorm(hidden_size)

    def forward(self, x, hx, cx):
        combined = torch.cat([x, hx], dim=-1)
        gates    = self.ln_gates(self.linear(combined))
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        g = torch.tanh(g);    o = torch.sigmoid(o)
        cx_new = f * cx + i * g
        hx_new = o * torch.tanh(self.ln_cell(cx_new))
        return hx_new, cx_new


class RecurrentActorCritic(nn.Module):
    def __init__(self, in_dim=OBS_DIM, n_actions=len(ACTIONS), hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc    = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh())
        self.lstm1 = LayerNormLSTMCell(64,          hidden_size)
        self.lstm2 = LayerNormLSTMCell(hidden_size, hidden_size)
        self.actor  = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        (hx1, cx1), (hx2, cx2) = hidden
        hx1=hx1.squeeze(0); cx1=cx1.squeeze(0)
        hx2=hx2.squeeze(0); cx2=cx2.squeeze(0)
        outputs = []
        for t in range(x.shape[1]):
            inp = self.fc(x[:, t, :])
            hx1, cx1 = self.lstm1(inp, hx1, cx1)
            hx2, cx2 = self.lstm2(hx1, hx2, cx2)
            outputs.append(hx2.unsqueeze(1))
        out    = torch.cat(outputs, dim=1)
        logits = self.actor(out)
        value  = self.critic(out)
        new_hidden = (
            (hx1.unsqueeze(0), cx1.unsqueeze(0)),
            (hx2.unsqueeze(0), cx2.unsqueeze(0))
        )
        return logits, value, new_hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_size)
        c = torch.zeros(1, batch_size, self.hidden_size)
        return (h, c), (h.clone(), c.clone())


_model  = None
_hidden = None
_steps  = 0

def _load_once():
    global _model
    if _model is not None: return
    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights_ln_best.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Weights not found: {wpath}")
    m = RecurrentActorCritic()
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