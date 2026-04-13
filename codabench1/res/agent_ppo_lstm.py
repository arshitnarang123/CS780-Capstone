"""
Recurrent PPO (LSTM) evaluation agent for OBELIX.
Maintains persistent internal memory state for partial observability.
"""

import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18
MAX_STEPS_PER_EPISODE = 500

class RecurrentActorCritic(nn.Module):
    def __init__(self, in_dim=OBS_DIM, n_actions=len(ACTIONS), hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh()
        )
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        features = self.fc(x)
        lstm_out, new_hidden = self.lstm(features, hidden)
        logits = self.actor(lstm_out)
        value = self.critic(lstm_out)
        return logits, value, new_hidden

_model = None
_hidden_state = None
_step_count = 0

def _load_once():
    global _model
    if _model is not None:
        return
        
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_ppo_lstm_shaped_v3_best.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py.")
        
    model = RecurrentActorCritic()
    sd = torch.load(wpath, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    _model = model

@torch.no_grad()
def policy(obs, *args, **kwargs) -> str:
    global _hidden_state, _step_count
    _load_once()

    # Reset hidden state if we hit the episode limit
    if _step_count >= MAX_STEPS_PER_EPISODE or _hidden_state is None:
        hx = torch.zeros(1, 1, _model.hidden_size)
        cx = torch.zeros(1, 1, _model.hidden_size)
        _hidden_state = (hx, cx)
        _step_count = 0

    _step_count += 1

    # Format input: (batch_size=1, sequence_length=1, features=18)
    x = torch.tensor(obs, dtype=torch.float32).view(1, 1, OBS_DIM)
    
    # Pass through network and update our global hidden state
    logits, _, _hidden_state = _model(x, _hidden_state)
    
    # Greedy selection for evaluation
    best_action = int(torch.argmax(logits.squeeze()).item())
    
    return ACTIONS[best_action]