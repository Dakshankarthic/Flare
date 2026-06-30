"""
app.py — FastAPI server for the Solar Flare Nowcasting & Forecasting Pipeline.

Endpoints:
  GET  /                    — Serve the dashboard
  GET  /api/status          — System health and data source info
  GET  /api/catalogue       — List detected flares
  GET  /api/metrics         — Evaluation metrics
  GET  /api/forecast        — Current forecast probability
  WS   /ws/stream           — Real-time data + alerts WebSocket

Background: Continuous processing loop that:
  1. Reads data (simulated or from FITS queue)
  2. Runs nowcast detectors on both channels
  3. Combines into master catalogue
  4. Extracts features → runs forecast model
  5. Pushes results to connected WebSocket clients
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from collections import deque
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.loader_sim import generate_synthetic_data, SimulationConfig, flare_events_to_df
from backend.data.preprocessor import full_preprocess
from backend.nowcast.detector_soft import detect_flares_soft, detections_to_df
from backend.nowcast.detector_hard import detect_flares_hard
from backend.nowcast.combiner import combine_catalogues, master_catalogue_to_df
from backend.nowcast.classifier import classify_master_catalogue, get_class_color
from backend.forecast.features import extract_features, create_labels, FEATURE_COLUMNS
from backend.forecast.model import XGBoostForecaster
from backend.forecast.conformal import ConformalFlarePredictor
from backend.evaluation.metrics import (
    compute_nowcast_metrics, compute_forecast_metrics,
    compute_lead_time, format_metrics_report,
    compute_per_class_confusion, compute_reliability_diagram,
    compute_all_bootstrap_cis
)
from backend.catalogue.catalogue import FlareCatalogue

# Optional: Neural model (requires PyTorch)
try:
    from backend.forecast.neural_model import (
        CNNBiLSTMForecaster, train_neural_model,
        get_attention_weights, prepare_windows, HORIZONS_MIN
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_MODE = os.environ.get('DATA_MODE', 'simulation')  # 'simulation' or 'fits'
FITS_DIR = os.environ.get('FITS_DIR', r'I:\ISRO ADITYA')
SIM_HOURS = float(os.environ.get('SIM_HOURS', '48'))
SIM_FLARES = int(os.environ.get('SIM_FLARES', '12'))
FORECAST_HORIZON = int(os.environ.get('FORECAST_HORIZON', '300'))  # seconds
STREAM_SPEED = float(os.environ.get('STREAM_SPEED', '50'))  # data points per push

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('solar-flare-api')

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
app = FastAPI(title="Solar Flare Pipeline — Aditya-L1", version="1.0.0")

# Serve frontend
FRONTEND_DIR = PROJECT_ROOT / 'frontend'

# Global state
state = {
    'initialized': False,
    'data_mode': DATA_MODE,
    'raw_df': None,           # full preprocessed DataFrame
    'ground_truth': None,     # ground truth flare events (for simulation)
    'processed_df': None,     # preprocessed data
    'nowcast_catalogue': None,
    'forecast_model': None,
    'neural_model': None,     # CNN-BiLSTM + NeupertAttention
    'conformal': None,        # Conformal prediction calibrator
    'features_df': None,
    'forecast_probs': None,
    'conformal_intervals': None,  # (prob, lower, upper) from conformal
    'attention_weights': None,    # Latest NeupertAttention weights
    'per_class_confusion': None,  # Per-class confusion matrix
    'reliability': None,          # Reliability diagram data
    'bootstrap_cis': None,        # Bootstrap confidence intervals
    'metrics': {},
    'stream_position': 0,     # current position in the data stream
    'catalogue': FlareCatalogue(),
    'active_model': 'xgboost',  # 'xgboost' or 'neural'
}

# WebSocket connections
active_connections: List[WebSocket] = []


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_pipeline():
    """Initialize the entire pipeline with data generation, detection, and model training."""
    logger.info("=" * 60)
    logger.info("  INITIALIZING SOLAR FLARE PIPELINE")
    logger.info("=" * 60)
    
    # --- Step 1: Generate or load data ---
    logger.info("\n[1/6] Loading data...")
    
    if DATA_MODE == 'simulation':
        config = SimulationConfig(
            duration_hours=SIM_HOURS,
            num_flares=SIM_FLARES,
            seed=42,
        )
        raw_df, flare_events = generate_synthetic_data(config)
        state['ground_truth'] = flare_events
        logger.info(f"  Generated {len(raw_df)} data points ({SIM_HOURS}h) "
                     f"with {len(flare_events)} flares")
    else:
        # Load from FITS files
        from backend.data.loader_fits import scan_and_load_solexs
        raw_df = scan_and_load_solexs(FITS_DIR, max_days=30)
        # Rename columns
        raw_df = raw_df.rename(columns={'rate': 'soft_counts'})
        if 'hard_counts' not in raw_df.columns:
            # If no hard X-ray data, create synthetic hard channel
            raw_df['hard_counts'] = raw_df['soft_counts'] * 0.3 + np.random.normal(0, 0.5, len(raw_df))
        flare_events = []
        state['ground_truth'] = []
        logger.info(f"  Loaded {len(raw_df)} data points from FITS files")
    
    state['raw_df'] = raw_df
    
    # --- Step 2: Preprocess ---
    logger.info("\n[2/6] Preprocessing...")
    processed = full_preprocess(raw_df)
    state['processed_df'] = processed
    logger.info(f"  Preprocessed: {len(processed)} rows with "
                 f"{len(processed.columns)} columns")
    
    # --- Step 3: Nowcasting (detection) ---
    logger.info("\n[3/6] Running nowcasting detectors...")
    
    soft_detections = detect_flares_soft(processed)
    hard_detections = detect_flares_hard(processed)
    
    logger.info(f"  Soft X-ray detections: {len(soft_detections)}")
    logger.info(f"  Hard X-ray detections: {len(hard_detections)}")
    
    # Combine into master catalogue
    master_events = combine_catalogues(soft_detections, hard_detections)
    master_events = classify_master_catalogue(master_events)
    
    catalogue_df = master_catalogue_to_df(master_events)
    state['nowcast_catalogue'] = catalogue_df
    
    # Store in database
    state['catalogue'].clear()
    for _, row in catalogue_df.iterrows():
        state['catalogue'].add_nowcast_event(row.to_dict())
    
    logger.info(f"  Master catalogue: {len(master_events)} events")
    for _, row in catalogue_df.iterrows():
        logger.info(f"    {row['flare_class']}-class at t={row['peak_time']:.0f}s "
                     f"({row['detection_type']}, conf={row['confidence']:.2f})")
    
    # --- Step 4: Feature extraction ---
    logger.info("\n[4/6] Extracting features for forecasting...")
    
    features_df = extract_features(processed, window_sec=300, stride_sec=60)
    state['features_df'] = features_df
    logger.info(f"  Extracted {len(features_df)} feature windows with "
                 f"{len(FEATURE_COLUMNS)} features each")
    
    # --- Step 5: Train forecast model ---
    logger.info("\n[5/6] Training forecast model...")
    
    if flare_events:
        labels = create_labels(processed, features_df, flare_events,
                                horizon_sec=FORECAST_HORIZON)
    else:
        # For real data without labels, use nowcast detections as pseudo-labels
        labels = create_labels(processed, features_df, master_events,
                                horizon_sec=FORECAST_HORIZON)
    
    forecaster = XGBoostForecaster(horizon_sec=FORECAST_HORIZON)
    
    if labels.sum() > 0:
        train_metrics = forecaster.train(features_df, labels, n_splits=3, verbose=True)
        state['forecast_model'] = forecaster
        
        # Generate forecasts
        probs = forecaster.predict_proba(features_df)
        state['forecast_probs'] = probs
        
        # Save model
        model_dir = PROJECT_ROOT / 'models'
        model_dir.mkdir(exist_ok=True)
        forecaster.save(str(model_dir / 'forecast_xgb.joblib'))
        logger.info(f"  Model saved. CV AUC: {train_metrics.get('cv_auc_mean', 0):.3f}")
    else:
        logger.info("  No positive labels found — skipping model training")
        state['forecast_model'] = None
        state['forecast_probs'] = np.zeros(len(features_df))
    
    # --- Step 6: Train CNN-BiLSTM (if PyTorch available) ---
    if HAS_TORCH and flare_events and labels.sum() > 0:
        logger.info("\n[6/8] Training CNN-BiLSTM + NeupertAttention...")
        try:
            neural_model, neural_metrics = train_neural_model(
                processed, flare_events,
                window_sec=300, stride_sec=60,
                epochs=2, batch_size=32,
                device='cpu', verbose=True
            )
            state['neural_model'] = neural_model
            logger.info(f"  Neural model trained. Best TSS: {neural_metrics.get('best_tss', 0):.3f}")
        except Exception as e:
            logger.warning(f"  Neural model training failed: {e}")
            state['neural_model'] = None
    else:
        logger.info("\n[6/8] Skipping CNN-BiLSTM (PyTorch not available or no labels)")
    
    # --- Step 7: Calibrate Conformal Prediction ---
    logger.info("\n[7/8] Calibrating Conformal Prediction...")
    if state['forecast_model'] and state['forecast_probs'] is not None:
        try:
            probs = state['forecast_probs']
            conformal = ConformalFlarePredictor(alpha=0.1, method='standard')
            cal_probs_2d = np.stack([1 - probs, probs], axis=1)
            conformal.calibrate(cal_probs_2d, labels.values.astype(int))
            state['conformal'] = conformal
            
            # Compute intervals for all predictions
            prob_vals, lower, upper = conformal.predict_intervals_binary(probs)
            state['conformal_intervals'] = {
                'probabilities': prob_vals.tolist(),
                'lower': lower.tolist(),
                'upper': upper.tolist(),
            }
            logger.info(f"  Conformal calibrated: q_hat={conformal.q_hat:.3f}, "
                        f"coverage guarantee >= {1-conformal.alpha:.0%}")
        except Exception as e:
            logger.warning(f"  Conformal calibration failed: {e}")
            state['conformal'] = None
    
    # --- Step 8: Evaluate ---
    logger.info("\n[8/8] Computing evaluation metrics...")
    
    if flare_events and state['forecast_model']:
        nowcast_metrics = compute_nowcast_metrics(
            master_events, flare_events, tolerance_sec=120
        )
        
        forecast_metrics = compute_forecast_metrics(
            labels.values, state['forecast_probs']
        )
        
        lead_metrics = compute_lead_time(
            [{'time': row['peak_time'] - 60} for _, row in catalogue_df.iterrows()],
            flare_events
        )
        
        state['metrics'] = {
            'nowcast': nowcast_metrics,
            'forecast': forecast_metrics,
            'lead_time': lead_metrics,
        }
        
        report = format_metrics_report(nowcast_metrics, forecast_metrics, lead_metrics)
        logger.info("\n" + report)
        
        # Per-class confusion matrix
        state['per_class_confusion'] = compute_per_class_confusion(
            master_events, flare_events, tolerance_sec=120
        )
        
        # Reliability diagram
        state['reliability'] = compute_reliability_diagram(
            labels.values, state['forecast_probs']
        )
        
        # Bootstrap CIs
        try:
            state['bootstrap_cis'] = compute_all_bootstrap_cis(
                labels.values, state['forecast_probs']
            )
        except Exception:
            state['bootstrap_cis'] = None
    else:
        state['metrics'] = {
            'nowcast': {'tpr': 0, 'far': 0, 'f1': 0},
            'forecast': {'auc': 0, 'tss': 0},
            'lead_time': {'mean_lead_min': 0},
        }
    
    state['initialized'] = True
    logger.info("\n✅ Pipeline initialized successfully!")


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Initialize the pipeline on server start."""
    initialize_pipeline()


@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard page."""
    return FileResponse(str(FRONTEND_DIR / 'index.html'))


@app.get("/index.css")
async def serve_css():
    return FileResponse(str(FRONTEND_DIR / 'index.css'), media_type='text/css')


@app.get("/index.js")
async def serve_js():
    return FileResponse(str(FRONTEND_DIR / 'index.js'), media_type='application/javascript')


@app.get("/api/status")
async def get_status():
    """System health and data source info."""
    return {
        'status': 'ready' if state['initialized'] else 'initializing',
        'data_mode': state['data_mode'],
        'total_data_points': len(state['processed_df']) if state['processed_df'] is not None else 0,
        'total_events': len(state['nowcast_catalogue']) if state['nowcast_catalogue'] is not None else 0,
        'model_trained': state['forecast_model'] is not None,
        'stream_position': state['stream_position'],
        'forecast_horizon_min': FORECAST_HORIZON / 60,
    }


@app.get("/api/catalogue")
async def get_catalogue(limit: int = Query(50, ge=1, le=500)):
    """Get detected flare events."""
    if state['nowcast_catalogue'] is None:
        return []
    
    df = state['nowcast_catalogue'].head(limit)
    return df.to_dict(orient='records')


@app.get("/api/metrics")
async def get_metrics():
    """Get evaluation metrics."""
    return state['metrics']


@app.get("/api/forecast")
async def get_current_forecast():
    """Get the latest forecast probability."""
    if state['forecast_probs'] is None or len(state['forecast_probs']) == 0:
        return {'probability': 0.0, 'predicted_class': 'None', 'alert': False}
    
    # Get most recent probability
    latest_prob = float(state['forecast_probs'][-1])
    threshold = state['forecast_model'].best_threshold if state['forecast_model'] else 0.5
    
    return {
        'probability': latest_prob,
        'alert': latest_prob >= threshold,
        'threshold': threshold,
        'horizon_min': FORECAST_HORIZON / 60,
    }


@app.get("/api/data_chunk")
async def get_data_chunk(start: int = Query(0, ge=0), size: int = Query(1000, ge=1, le=10000)):
    """Get a chunk of the processed time-series data."""
    if state['processed_df'] is None:
        return {'data': [], 'total': 0}
    
    df = state['processed_df']
    end = min(start + size, len(df))
    
    chunk = df.iloc[start:end][['time_s', 'soft_counts', 'hard_counts',
                                  'soft_norm', 'hard_norm']].copy()
    
    return {
        'data': chunk.to_dict(orient='records'),
        'total': len(df),
        'start': start,
        'end': end,
    }


@app.get("/api/feature_importance")
async def get_feature_importance():
    """Get feature importance from the forecast model."""
    if state['forecast_model'] is None:
        return []
    return state['forecast_model'].get_feature_importance().to_dict(orient='records')


@app.get("/api/confusion_matrix")
async def get_confusion_matrix():
    """Get per-class confusion matrix (A/B/C/M/X)."""
    if state['per_class_confusion'] is None:
        return {'classes': ['A', 'B', 'C', 'M', 'X'], 'class_tp': {}, 'class_fp': {}, 'class_fn': {}}
    return state['per_class_confusion']


@app.get("/api/conformal")
async def get_conformal():
    """Get conformal prediction intervals and coverage stats."""
    result = {'calibrated': False}
    if state['conformal'] is not None:
        result = state['conformal'].get_stats()
    if state['conformal_intervals'] is not None:
        result['intervals'] = state['conformal_intervals']
    return result


@app.get("/api/attention_heatmap")
async def get_attention_heatmap(position: int = Query(0, ge=0)):
    """Get NeupertAttention weights for a window at the given position."""
    if state['neural_model'] is None or state['processed_df'] is None:
        return {'available': False}
    
    df = state['processed_df']
    n = len(df)
    window_size = 300
    
    start = min(position, max(0, n - window_size))
    end = start + window_size
    if end > n:
        return {'available': False}
    
    soft = df['soft_norm'].values[start:end]
    hard = df['hard_norm'].values[start:end]
    window = np.stack([soft, hard], axis=1)
    
    try:
        attn = get_attention_weights(state['neural_model'], window)
        # Average across heads and return as 1D temporal importance
        temporal_importance = attn.mean(axis=(0, 1)).tolist()  # [time]
        return {
            'available': True,
            'temporal_importance': temporal_importance,
            'position': start,
            'window_size': window_size,
        }
    except Exception as e:
        return {'available': False, 'error': str(e)}


@app.get("/api/lead_time_histogram")
async def get_lead_time_histogram():
    """Get lead time distribution for detected flares."""
    if state['nowcast_catalogue'] is None or state['ground_truth'] is None:
        return {'bins': [], 'counts': []}
    
    catalogue_df = state['nowcast_catalogue']
    flare_events = state['ground_truth']
    
    lead_data = compute_lead_time(
        [{'time': row['peak_time'] - 60} for _, row in catalogue_df.iterrows()],
        flare_events
    )
    
    # Create histogram bins
    lead_times_sec = []
    for gt in flare_events:
        gt_peak = getattr(gt, 'peak_time', gt.get('peak_time', 0) if isinstance(gt, dict) else 0)
        for _, row in catalogue_df.iterrows():
            lead = gt_peak - (row['peak_time'] - 60)
            if 0 < lead <= 600:
                lead_times_sec.append(lead / 60)  # convert to minutes
    
    if not lead_times_sec:
        return {'bins': [], 'counts': [], 'stats': lead_data}
    
    counts, bin_edges = np.histogram(lead_times_sec, bins=10, range=(0, 10))
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
    
    return {
        'bins': bin_centers,
        'counts': counts.tolist(),
        'stats': lead_data,
    }


@app.get("/api/neupert_check")
async def get_neupert_check():
    """Physics check: correlation between integral(HXR) and SXR."""
    if state['processed_df'] is None:
        return {'available': False}
    
    df = state['processed_df']
    hard = df['hard_norm'].values if 'hard_norm' in df.columns else None
    soft = df['soft_norm'].values if 'soft_norm' in df.columns else None
    
    if hard is None or soft is None:
        return {'available': False}
    
    # Downsample for visualization (every 60th point)
    step = 60
    hard_integral = np.cumsum(hard)[::step].tolist()
    soft_vals = soft[::step].tolist()
    
    # Compute correlation
    corr = float(np.corrcoef(np.cumsum(hard)[::step], soft[::step])[0, 1])
    
    return {
        'available': True,
        'hxr_integral': hard_integral[:500],  # limit for JSON size
        'sxr_values': soft_vals[:500],
        'correlation': corr,
    }


@app.get("/api/reliability")
async def get_reliability():
    """Get reliability/calibration diagram data."""
    return state.get('reliability') or {}


@app.get("/api/bootstrap_ci")
async def get_bootstrap_ci():
    """Get bootstrap confidence intervals for metrics."""
    return state.get('bootstrap_cis') or {}


@app.post("/api/set_model")
async def set_active_model(model: str = Query('xgboost')):
    """Toggle between XGBoost and CNN-BiLSTM models."""
    if model == 'neural' and state['neural_model'] is None:
        return {'error': 'Neural model not available', 'active': state['active_model']}
    state['active_model'] = model
    return {'active': model}


# ---------------------------------------------------------------------------
# WebSocket — Real-time streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Stream real-time data and alerts to the dashboard.
    
    Sends JSON messages:
      - type: 'data'      → {time_s, soft, hard, soft_norm, hard_norm}
      - type: 'alert'     → {flare_class, confidence, detection_type}
      - type: 'forecast'  → {probability, alert}
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected. Total: {len(active_connections)}")
    
    try:
        df = state['processed_df']
        catalogue = state['nowcast_catalogue']
        forecast_probs = state['forecast_probs']
        features_df = state['features_df']
        
        if df is None:
            await websocket.send_json({'type': 'error', 'message': 'Not initialized'})
            return
        
        n = len(df)
        pos = 0
        batch_size = int(STREAM_SPEED)
        
        # Pre-compute flare alert times
        alert_times = set()
        if catalogue is not None:
            for _, row in catalogue.iterrows():
                alert_times.add(int(row['peak_time']))
        
        while True:
            # Check for incoming messages
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                cmd = json.loads(msg)
                if cmd.get('action') == 'set_speed':
                    batch_size = max(1, int(cmd.get('speed', STREAM_SPEED)))
                elif cmd.get('action') == 'reset':
                    pos = 0
            except asyncio.TimeoutError:
                pass
            
            if pos >= n:
                pos = 0  # loop back to start
            
            end_pos = min(pos + batch_size, n)
            chunk = df.iloc[pos:end_pos]
            
            # Send data points
            data_points = []
            for _, row in chunk.iterrows():
                point = {
                    'time_s': float(row['time_s']),
                    'soft': float(row.get('soft_counts', 0)),
                    'hard': float(row.get('hard_counts', 0)),
                    'soft_norm': float(row.get('soft_norm', 0)),
                    'hard_norm': float(row.get('hard_norm', 0)),
                }
                data_points.append(point)
                
                # Check for flare alerts
                t = int(row['time_s'])
                if t in alert_times and catalogue is not None:
                    matching = catalogue[catalogue['peak_time'].astype(int) == t]
                    for _, flare_row in matching.iterrows():
                        await websocket.send_json({
                            'type': 'alert',
                            'flare_class': flare_row['flare_class'],
                            'confidence': float(flare_row['confidence']),
                            'detection_type': flare_row['detection_type'],
                            'peak_time': float(flare_row['peak_time']),
                            'peak_soft': float(flare_row['peak_soft']),
                            'duration': float(flare_row['duration']),
                            'color': get_class_color(flare_row['flare_class']),
                        })
            
            # Send batch of data points
            await websocket.send_json({
                'type': 'data',
                'points': data_points,
                'position': pos,
                'total': n,
            })
            
            # Send forecast update every ~5 seconds worth of data
            if forecast_probs is not None and features_df is not None:
                feat_idx = min(pos // 60, len(forecast_probs) - 1)
                if feat_idx >= 0:
                    prob = float(forecast_probs[feat_idx])
                    threshold = state['forecast_model'].best_threshold if state['forecast_model'] else 0.5
                    
                    # --- Explainable AI (Live Reasoning Proxy) ---
                    current_features = features_df.iloc[feat_idx]
                    reasons = []
                    if current_features.get('hard_slope', 0) > 0.5:
                        reasons.append("Anomalous rapid increase in Hard X-rays (HEL1OS)")
                    if current_features.get('soft_deriv_max', 0) > 1.5:
                        reasons.append("Sharp acceleration in Soft X-ray flux (SoLEXS)")
                    if current_features.get('hard_soft_ratio_mean', 0) > 0.1:
                        reasons.append("Elevated non-thermal to thermal energy ratio")
                    if current_features.get('soft_mean_30s', 0) > 3.0:
                        reasons.append("High background thermal plasma heating detected")
                    
                    if not reasons:
                        reasons.append("Nominal background activity levels")
                        
                    await websocket.send_json({
                        'type': 'forecast',
                        'probability': prob,
                        'alert': prob >= threshold,
                        'threshold': threshold,
                        'reasons': reasons[:3]
                    })
            
            pos = end_pos
            state['stream_position'] = pos
            
            await asyncio.sleep(0.1)  # 100ms between pushes
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


# ---------------------------------------------------------------------------
# Static files (must be last)
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import uvicorn
    
    port = int(os.environ.get('PORT', '8000'))
    logger.info(f"Starting Solar Flare Pipeline server on port {port}...")
    logger.info(f"Dashboard: http://localhost:{port}")
    logger.info(f"Data mode: {DATA_MODE}")
    
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='info')
