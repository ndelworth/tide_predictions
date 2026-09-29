import numpy as np
import torch
import torch.nn as nn

from config import PAST_HOURS, WX_VARS
from features import TIME_FEATURES, WINDOW

INPUT_FEATURES = WX_VARS + ["surge", "is_future"] + TIME_FEATURES
D_MODEL  = 64
N_HEADS  = 4
N_LAYERS = 2
DIM_FF   = 256
DROPOUT  = 0.25

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def sinusoidal_positional_encoding(length, d_model):
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


def persistence(surge):
    held_last_surge = surge[:, PAST_HOURS:]
    return torch.nan_to_num(held_last_surge)


class Standardize(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("std", torch.ones(num_features))

    def fit(self, x):
        self.mean.copy_(torch.nanmean(x, dim=0))
        self.std.copy_(torch.nanmean((x - self.mean) ** 2, dim=0).sqrt().clamp(min=1e-5))

    def forward(self, x):
        return torch.nan_to_num((x - self.mean) / self.std)


class SurgeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weather_scale = Standardize(len(WX_VARS))
        self.surge_scale = Standardize(1)
        self.input_proj = nn.Linear(len(INPUT_FEATURES), D_MODEL)
        self.register_buffer("pos", sinusoidal_positional_encoding(WINDOW, D_MODEL), persistent=False)
        self.register_buffer("is_future", (torch.arange(WINDOW) >= PAST_HOURS).float(), persistent=False)
        # No causal mask: the future surge is already hidden in the inputs.
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, DIM_FF, DROPOUT,
                                           activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL // 2), nn.GELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(D_MODEL // 2, 1))

    def fit_standardization(self, df):
        self.weather_scale.fit(torch.tensor(df[WX_VARS].to_numpy(np.float32)))
        self.surge_scale.fit(torch.tensor(df[["surge"]].to_numpy(np.float32)))

    def forward(self, weather, surge, time):
        x = torch.cat([self.weather_scale(weather), self.surge_scale(surge).unsqueeze(-1),
                       self.is_future.expand(len(surge), -1).unsqueeze(-1), time], dim=-1)
        encoded = self.encoder(self.input_proj(x) + self.pos)
        correction = self.head(encoded[:, PAST_HOURS:]).squeeze(-1)
        return persistence(surge) + correction
