"""
Agent for Rainbow-lite (Dueling DDQN + PER + N-step + Noisy Networks).
policy(obs, rng=None) -> str
"""

import os, math
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma        = sigma
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def sample_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        # At eval time — use mean weights only, no noise
        w = self.weight_mu
        b = self.bias_mu
        return nn.functional.linear(x, w, b)


class RainbowDQN(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),    nn.ReLU()
        )
        self.value_hidden = NoisyLinear(128, hidden)
        self.value_out    = NoisyLinear(hidden, 1)
        self.adv_hidden   = NoisyLinear(128, hidden)
        self.adv_out      = NoisyLinear(hidden, n_actions)

    def forward(self, x):
        feat = self.feature(x)
        v = torch.relu(self.value_hidden(feat))
        v = self.value_out(v)
        a = torch.relu(self.adv_hidden(feat))
        a = self.adv_out(a)
        return v + a - a.mean(dim=-1, keepdim=True)


_model = None

def _load_once():
    global _model
    if _model is not None: return
    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights_rainbow_v2_final_best.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Weights not found: {wpath}")
    m = RainbowDQN(OBS_DIM, len(ACTIONS))
    m.load_state_dict(torch.load(wpath, map_location="cpu"))
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs, rng=None) -> str:
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return ACTIONS[int(_model(x).argmax(dim=1).item())]