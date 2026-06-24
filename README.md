# 🌞 Solar Flare Nowcasting & Forecasting Pipeline

**Aditya-L1 Mission — SoLEXS + HEL1OS Combined X-ray Analysis**

An end-to-end automated pipeline for detecting (nowcasting) and predicting (forecasting) solar flares using combined soft and hard X-ray data from ISRO's Aditya-L1 mission.

## Features

- **Nowcasting**: Real-time flare detection using adaptive derivative-threshold algorithms on both soft (SoLEXS, 1-30 keV) and hard (HEL1OS, 10-150 keV) X-ray channels
- **Master Catalogue**: Combined flare catalogue with temporal coincidence matching and confidence scoring
- **GOES Classification**: Automatic assignment of A/B/C/M/X flare classes
- **Forecasting**: XGBoost model trained on 33 engineered features to predict flare probability within the next N minutes
- **Premium Dashboard**: Real-time web visualization with live light curves, forecast gauge, and alert system
- **Evaluation**: TPR, FAR, TSS, HSS, Brier Skill Score, and lead-time analysis

## Quick Start

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Run the pipeline (simulation mode — works out of the box)
python backend/app.py

# 3. Open the dashboard
# Navigate to http://localhost:8000
```

## Architecture

```
┌─────────────────┐   ┌─────────────────┐
│  SoLEXS (Soft)  │   │  HEL1OS (Hard)  │
│  1-30 keV FITS  │   │  10-150 keV     │
└───────┬─────────┘   └───────┬─────────┘
        │                     │
        ▼                     ▼
┌─────────────────────────────────────────┐
│  Preprocessing                          │
│  • Resampling • Background subtraction  │
│  • MAD normalization • Derivatives      │
└───────────────────┬─────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  Nowcasting   │       │  Forecasting  │
│  Detector     │       │  XGBoost      │
│  (soft+hard)  │       │  33 features  │
└───────┬───────┘       └───────┬───────┘
        │                       │
        ▼                       ▼
┌─────────────────────────────────────────┐
│  Web Dashboard (Chart.js + WebSocket)   │
│  Light curves • Alerts • Gauge • Table  │
└─────────────────────────────────────────┘
```

## Data Modes

### Simulation Mode (Default)
Generates realistic synthetic X-ray light curves with embedded flares (Neupert effect modelling). 

### FITS Mode
Load real SoLEXS/HEL1OS Level-1 data from the PRADAN portal:
```bash
# Set environment variables
set DATA_MODE=fits
set FITS_DIR=I:\ISRO ADITYA
python backend/app.py
```

## Project Structure

```
solar-flare-pipeline/
├── backend/
│   ├── app.py              # FastAPI server (REST + WebSocket)
│   ├── data/
│   │   ├── loader_fits.py   # FITS file reader
│   │   ├── loader_sim.py    # Synthetic data generator
│   │   └── preprocessor.py  # Signal processing pipeline
│   ├── nowcast/
│   │   ├── detector_soft.py # Soft X-ray flare detector
│   │   ├── detector_hard.py # Hard X-ray flare detector
│   │   ├── combiner.py      # Master catalogue combiner
│   │   └── classifier.py    # GOES class assignment
│   ├── forecast/
│   │   ├── features.py      # Feature engineering (33 features)
│   │   └── model.py         # XGBoost forecasting model
│   ├── catalogue/
│   │   └── catalogue.py     # SQLite event database
│   └── evaluation/
│       └── metrics.py       # TPR, FAR, TSS, HSS, lead time
├── frontend/
│   ├── index.html           # Dashboard
│   ├── index.css            # Dark space theme
│   └── index.js             # Chart.js + WebSocket client
└── models/                  # Saved model weights
```

## Evaluation Metrics

| Metric | Description |
|--------|------------|
| TPR (Recall) | True Positive Rate — fraction of real flares detected |
| FAR | False Alarm Rate — fraction of alerts that are false |
| TSS | True Skill Statistic = TPR - FPR |
| HSS | Heidke Skill Score |
| AUC-ROC | Area under ROC curve |
| Lead Time | Minutes before flare peak that the model triggers an alert |

## License

Open source — built for the ISRO Aditya-L1 Space Weather Challenge.
