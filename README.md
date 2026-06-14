## 🌤️ SkyMind — AI Weather Intelligence

SkyMind is a deep learning-powered weather application that goes far beyond
standard forecasting. Built on 10 specialist neural networks trained on 2M+
hourly observations across 20 global cities.

## 🚀 Live Demo
[Deploy link here]

## ✨ Features

### Standard Forecasts
- 🌡️ **Temperature** — Deep ResNet, R²=0.9998
- 🌧️ **Rain Probability** — AUC=0.9495
- ⛅ **Weather Condition** — 6-class, Acc=100%
- 💨 **Wind Speed** — R²=0.71
- ☀️ **UV Index** — R²=0.9999

### Exclusive SkyMind Insights
- ⛈️ **Thunderstorm Risk Score** — AUC=1.00
- 👁️ **Visibility Forecast** — km-level prediction
- 🏃 **Outdoor Activity Score** — 0–100 composite
- 🥵 **Heat Stress Index** — WHO 4-level classification
- 🌱 **Soil & Agricultural Score** — unique to SkyMind

### Smart Features
- 🎨 Dynamic UI that changes with weather conditions
- 🤖 Rule-based intelligent weather descriptions
- 📍 GPS location support
- 🔄 Automatic fallback for unavailable locations
- ⚠️ Confidence warnings for remote locations
- 📅 24-hour + 7-day ML-powered forecasts

## 🏗️ Architecture

```
Open-Meteo Forecast API → Feature Engineering → 10 Deep ResNets → UI
```

Each model is a deep residual network (ResNet-style) trained with:
- PyTorch + GPU acceleration
- OneCycleLR scheduler
- Mixed precision (AMP)
- Time-aware train/val/test split (2015–2022 / 2023 / 2024)

## 📁 Project Structure

```
skymind/
├── app.py                 # Streamlit UI
├── src/
│   ├── model_loader.py    # Load all 10 models
│   ├── inference.py       # Run predictions
│   ├── feature_engineering.py
│   ├── weather_api.py     # Open-Meteo integration
│   ├── geocoding.py       # Location search
│   ├── edge_cases.py      # Robustness handlers
│   └── descriptions.py    # Smart text generation
├── models/                # 10 trained model packages
│   ├── temperature/
│   ├── rain_probability/
│   └── ... (8 more)
├── utils/
│   └── constants.py
└── requirements.txt
```

## 🛠️ Local Setup

```bash
git clone https://github.com/yourusername/skymind
cd skymind
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Training Data
- Source: Open-Meteo Historical Archive API
- Period: 2015–2024
- Locations: 20 cities across 6 continents
- Size: 2,035,978 hourly observations
- Features: 62 variables + 40 engineered features

## 🌍 Training Cities
Berlin • New York • Itakpe • Tokyo • Riyadh • Antarctica •
Sydney • Moscow • Mexico City • New Delhi • Singapore • Dubai •
Rio de Janeiro • Stockholm • London • Cairo • Beijing • Paris •
Nairobi • Toronto

## 📄 License
MIT License — free to use and modify.
