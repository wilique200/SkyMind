# src/model_loader.py
# Loads all 10 trained PyTorch models once at Streamlit startup.
# Uses st.cache_resource so models stay in memory between requests.

import os
import json
import torch
import torch.nn as nn
import numpy as np
import joblib
import streamlit as st


# ── Architecture (must match training exactly) ───────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.block(x) + self.skip(x))


class WeatherNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        layers = []
        in_d = input_dim
        for out_d in hidden_dims:
            layers.append(ResidualBlock(in_d, out_d, dropout))
            in_d = out_d
        self.backbone       = nn.Sequential(*layers)
        self.final_dropout  = nn.Dropout(dropout)
        self.head           = nn.Linear(in_d, output_dim)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.backbone(x)
        x = self.final_dropout(x)
        return self.head(x)


MODEL_NAMES = [
    "temperature", "rain_probability", "weather_condition",
    "wind_speed", "uv_index", "thunderstorm",
    "visibility", "outdoor_score", "heat_stress", "soil_score",
]


@st.cache_resource(show_spinner="Loading SkyMind AI models...")
def load_all_models(models_dir="models"):
    """
    Load all 10 models into memory once.
    Returns a dict: model_name → {model, scaler, features, config}
    """
    device = torch.device("cpu")  # Streamlit Cloud uses CPU
    loaded = {}

    for name in MODEL_NAMES:
        model_dir = os.path.join(models_dir, name)

        try:
            # Load config
            with open(os.path.join(model_dir, f"{name}_config.json")) as f:
                cfg = json.load(f)

            # Load feature list
            with open(os.path.join(model_dir, f"{name}_features.json")) as f:
                features = json.load(f)

            # Rebuild architecture
            net = WeatherNet(
                input_dim   = cfg["input_dim"],
                hidden_dims = cfg["hidden_dims"],
                output_dim  = cfg["output_dim"],
                dropout     = cfg["dropout"],
            ).to(device)

            # Load weights
            weights_path = os.path.join(model_dir, f"{name}.pt")
            state = torch.load(weights_path, map_location=device)
            net.load_state_dict(state)
            net.eval()

            # Load scaler
            scaler = joblib.load(
                os.path.join(model_dir, f"{name}_scaler.pkl")
            )

            loaded[name] = {
                "model":    net,
                "scaler":   scaler,
                "features": features,
                "config":   cfg,
                "device":   device,
            }

        except Exception as e:
            st.warning(f"Could not load model '{name}': {e}")
            loaded[name] = None

    n_loaded = sum(1 for v in loaded.values() if v is not None)
    print(f"SkyMind: {n_loaded}/{len(MODEL_NAMES)} models loaded.")
    return loaded
