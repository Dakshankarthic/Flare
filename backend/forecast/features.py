"""
features.py — Feature engineering for solar flare forecasting.

Extracts sliding-window features from soft and hard X-ray time-series
for use in the predictive (forecasting) model.

Features capture:
  - Statistical moments (mean, std, skew, kurtosis)
  - Rate of change (derivatives)
  - Spectral properties (hardness ratio, trend)
  - Recent activity history (flare count, time since last)
  - Cross-channel correlations
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def extract_features(df: pd.DataFrame,
                      window_sec: int = 300,
                      stride_sec: int = 60,
                      time_col: str = 'time_s') -> pd.DataFrame:
    """
    Extract features from sliding windows over the time-series.
    
    Args:
        df: Preprocessed DataFrame with soft_norm, hard_norm, derivatives, etc.
        window_sec: Look-back window size in seconds (default 5 min)
        stride_sec: Step between consecutive windows (default 1 min)
        time_col: Column with time in seconds
    
    Returns:
        DataFrame with one row per window, containing all features
    """
    n = len(df)
    features_list = []
    
    # Get column arrays
    soft = df['soft_norm'].values if 'soft_norm' in df.columns else np.zeros(n)
    hard = df['hard_norm'].values if 'hard_norm' in df.columns else np.zeros(n)
    
    soft_d1 = df.get('soft_deriv1', pd.Series(np.gradient(soft))).values
    hard_d1 = df.get('hard_deriv1', pd.Series(np.gradient(hard))).values
    
    hardness = df.get('hardness_ratio', pd.Series(
        hard / np.maximum(np.abs(soft), 1e-10)
    )).values
    
    times = df[time_col].values
    
    for start in range(0, n - window_sec, stride_sec):
        end = start + window_sec
        if end > n:
            break
        
        s_win = soft[start:end]
        h_win = hard[start:end]
        sd_win = soft_d1[start:end]
        hd_win = hard_d1[start:end]
        hr_win = hardness[start:end]
        
        # Center time of window
        center_time = times[start + window_sec // 2]
        
        feat = {
            'time_s': center_time,
            'window_start': times[start],
            'window_end': times[end - 1],
            
            # --- Soft X-ray features ---
            'soft_mean': np.nanmean(s_win),
            'soft_std': np.nanstd(s_win),
            'soft_max': np.nanmax(s_win),
            'soft_min': np.nanmin(s_win),
            'soft_range': np.nanmax(s_win) - np.nanmin(s_win),
            'soft_skew': float(sp_stats.skew(s_win, nan_policy='omit')),
            'soft_kurtosis': float(sp_stats.kurtosis(s_win, nan_policy='omit')),
            
            # --- Hard X-ray features ---
            'hard_mean': np.nanmean(h_win),
            'hard_std': np.nanstd(h_win),
            'hard_max': np.nanmax(h_win),
            'hard_min': np.nanmin(h_win),
            'hard_range': np.nanmax(h_win) - np.nanmin(h_win),
            'hard_skew': float(sp_stats.skew(h_win, nan_policy='omit')),
            'hard_kurtosis': float(sp_stats.kurtosis(h_win, nan_policy='omit')),
            
            # --- Derivative features ---
            'soft_slope': _safe_slope(s_win),
            'hard_slope': _safe_slope(h_win),
            'soft_deriv_mean': np.nanmean(sd_win),
            'soft_deriv_max': np.nanmax(sd_win),
            'hard_deriv_mean': np.nanmean(hd_win),
            'hard_deriv_max': np.nanmax(hd_win),
            
            # --- Acceleration (2nd derivative) ---
            'soft_accel': _safe_slope(sd_win),
            'hard_accel': _safe_slope(hd_win),
            
            # --- Hardness ratio features ---
            'hardness_mean': np.nanmean(hr_win),
            'hardness_std': np.nanstd(hr_win),
            'hardness_trend': _safe_slope(hr_win),
            
            # --- Cross-channel features ---
            'hard_soft_ratio_mean': np.nanmean(h_win) / max(abs(np.nanmean(s_win)), 1e-10),
            'cross_corr': _safe_correlation(s_win, h_win),
            
            # --- Short-window features (last 30s of window) ---
            'soft_mean_30s': np.nanmean(s_win[-30:]),
            'hard_mean_30s': np.nanmean(h_win[-30:]),
            'soft_slope_30s': _safe_slope(s_win[-30:]),
            'hard_slope_30s': _safe_slope(h_win[-30:]),
            
            # --- Energy ratio (last 60s vs first 60s) ---
            'soft_energy_ratio': (
                np.nansum(s_win[-60:]) / max(np.nansum(s_win[:60]), 1e-10)
            ),
            'hard_energy_ratio': (
                np.nansum(h_win[-60:]) / max(np.nansum(h_win[:60]), 1e-10)
            ),
        }
        
        features_list.append(feat)
    
    features_df = pd.DataFrame(features_list)
    
    # Replace infinities and extreme values
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)
    
    return features_df


def _safe_slope(x: np.ndarray) -> float:
    """Compute linear slope (least-squares) safely."""
    x_clean = x[~np.isnan(x)]
    if len(x_clean) < 2:
        return 0.0
    try:
        return np.polyfit(np.arange(len(x_clean)), x_clean, 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation safely."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return 0.0
    try:
        r, _ = sp_stats.pearsonr(x[mask], y[mask])
        return r if np.isfinite(r) else 0.0
    except (ValueError, RuntimeWarning):
        return 0.0


def create_labels(df: pd.DataFrame, features_df: pd.DataFrame,
                   flare_events: list, horizon_sec: int = 300) -> pd.Series:
    """
    Create binary labels: 1 if a flare starts within horizon_sec after the window.
    
    Args:
        df: Original time-series DataFrame
        features_df: Features DataFrame (must have 'window_end' column)
        flare_events: List of flare events with onset_time attribute
        horizon_sec: Forecast horizon in seconds (default 5 min)
    
    Returns:
        Series of binary labels aligned with features_df
    """
    labels = np.zeros(len(features_df), dtype=int)
    
    for i, row in features_df.iterrows():
        window_end = row['window_end']
        
        for event in flare_events:
            # Get onset time
            onset = getattr(event, 'onset_time', getattr(event, 'start_time', None))
            if onset is None:
                continue
            
            # Check if flare starts within horizon after window end
            if window_end < onset <= window_end + horizon_sec:
                labels[i] = 1
                break
    
    return pd.Series(labels, index=features_df.index, name='future_flare')


# ---------------------------------------------------------------------------
# Feature names for model training (exclude metadata columns)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    'soft_mean', 'soft_std', 'soft_max', 'soft_min', 'soft_range',
    'soft_skew', 'soft_kurtosis',
    'hard_mean', 'hard_std', 'hard_max', 'hard_min', 'hard_range',
    'hard_skew', 'hard_kurtosis',
    'soft_slope', 'hard_slope',
    'soft_deriv_mean', 'soft_deriv_max',
    'hard_deriv_mean', 'hard_deriv_max',
    'soft_accel', 'hard_accel',
    'hardness_mean', 'hardness_std', 'hardness_trend',
    'hard_soft_ratio_mean', 'cross_corr',
    'soft_mean_30s', 'hard_mean_30s',
    'soft_slope_30s', 'hard_slope_30s',
    'soft_energy_ratio', 'hard_energy_ratio',
]
