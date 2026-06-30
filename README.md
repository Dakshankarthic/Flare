# 🌞 Solar Flare Nowcasting & Forecasting Pipeline

**Aditya-L1 Mission — SoLEXS + HEL1OS Combined X-ray Analysis**
**ISRO BAH 2026 — Team TEJAS**

An end-to-end automated pipeline for detecting (nowcasting) and predicting (forecasting) solar flares using combined soft and hard X-ray data from ISRO's Aditya-L1 mission.

## Features

- **Nowcasting**: Real-time flare detection using adaptive derivative-threshold algorithms on both soft (SoLEXS, 1-30 keV) and hard (HEL1OS, 10-150 keV) X-ray channels
- **Master Catalogue**: Combined flare catalogue with temporal coincidence matching and confidence scoring
- **GOES Classification**: Automatic assignment of A/B/C/M/X flare classes
- **Dual Forecasting Models**:
  - **XGBoost**: Gradient-boosted trees on 43 engineered features (33 original + 10 advanced)
  - **CNN-BiLSTM + NeupertAttention**: Physics-embedded deep learning with Neupert Effect constraint
- **Conformal Prediction**: Mathematically guaranteed FAR ≤ 10% with coverage intervals
- **Focal Loss**: Handles severe class imbalance (A/B flares 100× more common than X-class)
- **Multi-Horizon Forecasting**: Simultaneous prediction at 5, 10, 15, 30, 60 minute horizons
- **Advanced Features**: Wavelet coefficients, Transfer Entropy, Hurst exponent, FRED synthetic data
- **Premium Dashboard**: Real-time web visualization with:
  - Live light curves with flare annotations
  - Forecast gauge with conformal prediction intervals
  - NeupertAttention heatmap (what the model sees)
  - Per-class confusion matrix (A/B/C/M/X)
  - Lead time distribution histogram
  - Neupert Effect physics check (∫HXR vs SXR correlation)
  - Model toggle (XGBoost ↔ CNN-BiLSTM)
- **Evaluation**: TPR, FAR, TSS, HSS, Brier Skill Score, lead-time analysis, bootstrap CIs

## Quick Start

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Run the pipeline (simulation mode — works out of the box)
python backend/app.py

# 3. Open the dashboard
# Navigate to http://localhost:8000

# 4. Run tests (optional)
python test_pipeline.py
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
┌───────────────┐       ┌───────────────────────────────┐
│  Nowcasting   │       │  Forecasting                  │
│  Detector     │       │  ┌─────────────────────┐      │
│  (soft+hard)  │       │  │ XGBoost (43 feats)  │      │
│               │       │  └─────────────────────┘      │
│               │       │  ┌─────────────────────┐      │
│               │       │  │ CNN-BiLSTM +         │      │
│               │       │  │ NeupertAttention     │      │
│               │       │  └─────────────────────┘      │
│               │       │  ┌─────────────────────┐      │
│               │       │  │ Conformal Prediction │      │
│               │       │  │ (FAR ≤ 10% guarantee)│      │
│               │       │  └─────────────────────┘      │
└───────┬───────┘       └───────────────┬───────────────┘
        │                               │
        ▼                               ▼
┌─────────────────────────────────────────────────────┐
│  Web Dashboard (Chart.js + WebSocket)               │
│  Light curves • Alerts • Gauge • Attention Heatmap  │
│  Confusion Matrix • Lead Time • Neupert Check       │
└─────────────────────────────────────────────────────┘
```

## Data Modes

### Simulation Mode (Default)
Generates realistic synthetic X-ray light curves with embedded flares (Neupert effect modelling + FRED multi-thermal cooling). 

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
│   │   ├── loader_fits.py   # FITS file reader (SoLEXS + HEL1OS)
│   │   ├── loader_sim.py    # Synthetic data generator (+ FRED physics)
│   │   └── preprocessor.py  # Signal processing pipeline
│   ├── nowcast/
│   │   ├── detector_soft.py # Soft X-ray flare detector
│   │   ├── detector_hard.py # Hard X-ray flare detector
│   │   ├── combiner.py      # Master catalogue combiner
│   │   └── classifier.py    # GOES class assignment
│   ├── forecast/
│   │   ├── features.py      # Feature engineering (43 features)
│   │   ├── model.py         # XGBoost forecasting model
│   │   ├── neural_model.py  # CNN-BiLSTM + NeupertAttention
│   │   └── conformal.py     # Conformal Prediction (FAR guarantee)
│   ├── catalogue/
│   │   └── catalogue.py     # SQLite event database
│   └── evaluation/
│       └── metrics.py       # TPR, FAR, TSS, HSS, lead time, bootstrap CIs
├── frontend/
│   ├── index.html           # Dashboard (12 panels)
│   ├── index.css            # Dark space theme + new panel styles
│   └── index.js             # Chart.js + WebSocket + new charts
├── models/                  # Saved model weights
└── test_pipeline.py         # 10-test end-to-end validation
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
| ECE | Expected Calibration Error (reliability diagram) |
| Bootstrap CI | 95% confidence intervals on AUC and TSS |

## Key Innovations

1. **NeupertAttention**: Physics-embedded attention that enforces ∫HXR ∝ SXR (Neupert Effect)
2. **Conformal Prediction**: Formal guarantee of P(y_true ∈ set) ≥ 90%
3. **Focal Loss**: α-balanced focal loss handles 100× class imbalance
4. **FRED Synthetic Data**: Multi-thermal Kopp-Poletto cooling for pre-training
5. **43 Physics-Informed Features**: Wavelets, Transfer Entropy, Hurst exponent, flare memory

## License

Open source — built for the ISRO Aditya-L1 Space Weather Challenge.

