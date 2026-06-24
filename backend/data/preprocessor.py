"""
preprocessor.py — Clean, align, and normalize X-ray time-series data.

Steps:
  1. Resample to uniform 1-second cadence
  2. Interpolate short gaps (≤30s)
  3. Compute rolling background baseline (10-minute median)
  4. Subtract background → detrended signal
  5. Normalize using Median Absolute Deviation (MAD) scaling
  6. Compute derived quantities (hardness ratio, derivatives)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def resample_uniform(df: pd.DataFrame, cadence_sec: float = 1.0,
                      time_col: str = 'time_s') -> pd.DataFrame:
    """
    Resample a DataFrame to a uniform time grid.
    Fills gaps ≤30s by linear interpolation; larger gaps become NaN.
    """
    t0 = df[time_col].min()
    t1 = df[time_col].max()
    
    n_points = int((t1 - t0) / cadence_sec) + 1
    uniform_time = np.linspace(t0, t0 + (n_points - 1) * cadence_sec, n_points)
    
    # Set time as index for reindexing
    df_indexed = df.set_index(time_col).sort_index()
    df_indexed = df_indexed[~df_indexed.index.duplicated(keep='first')]
    
    # Reindex to uniform grid
    new_index = pd.Index(uniform_time, name=time_col)
    df_resampled = df_indexed.reindex(new_index, method=None)
    
    # Interpolate short gaps (≤30 samples at 1s cadence)
    numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_resampled[col] = df_resampled[col].interpolate(
            method='linear', limit=30, limit_direction='both'
        )
    
    df_resampled = df_resampled.reset_index()
    return df_resampled


def subtract_background(df: pd.DataFrame, window_sec: int = 600,
                         soft_col: str = 'soft_counts',
                         hard_col: str = 'hard_counts') -> pd.DataFrame:
    """
    Subtract a slowly-varying background using a rolling median.
    
    Args:
        window_sec: Background window size in seconds (default 10 min)
        soft_col: Column name for soft X-ray counts
        hard_col: Column name for hard X-ray counts
    """
    df = df.copy()
    
    # Rolling median background
    df['soft_bg'] = df[soft_col].rolling(
        window=window_sec, min_periods=1, center=True
    ).median()
    df['hard_bg'] = df[hard_col].rolling(
        window=window_sec, min_periods=1, center=True
    ).median()
    
    # Detrended = signal - background
    df['soft_det'] = df[soft_col] - df['soft_bg']
    df['hard_det'] = df[hard_col] - df['hard_bg']
    
    return df


def normalize_mad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize detrended signals using Median Absolute Deviation (MAD).
    Result: ~0 for quiet periods, >3-5 sigma during flares.
    """
    df = df.copy()
    
    for col, out_col in [('soft_det', 'soft_norm'), ('hard_det', 'hard_norm')]:
        if col in df.columns:
            median_val = np.nanmedian(df[col])
            mad = np.nanmedian(np.abs(df[col] - median_val))
            if mad > 0:
                df[out_col] = (df[col] - median_val) / (1.4826 * mad)  # 1.4826 = consistency constant
            else:
                df[out_col] = df[col] - median_val
    
    return df


def smooth_signal(df: pd.DataFrame, window: int = 11, polyorder: int = 3,
                   cols: list = None) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing to reduce high-frequency noise."""
    df = df.copy()
    if cols is None:
        cols = ['soft_counts', 'hard_counts']
    
    for col in cols:
        if col in df.columns:
            valid = df[col].notna()
            if valid.sum() > window:
                smoothed = savgol_filter(
                    df.loc[valid, col].values, window, polyorder
                )
                df.loc[valid, f'{col}_smooth'] = smoothed
    
    return df


def compute_derivatives(df: pd.DataFrame, dt: float = 1.0) -> pd.DataFrame:
    """
    Compute first and second derivatives of the flux signals.
    
    These are critical for flare detection (rise rate) and
    forecasting (acceleration precursors).
    """
    df = df.copy()
    
    for prefix in ['soft', 'hard']:
        col = f'{prefix}_norm'
        if col in df.columns:
            # First derivative (rate of change per second)
            df[f'{prefix}_deriv1'] = df[col].diff() / dt
            
            # Second derivative (acceleration)
            df[f'{prefix}_deriv2'] = df[f'{prefix}_deriv1'].diff() / dt
            
            # Smoothed derivatives (5-point window)
            df[f'{prefix}_deriv1_smooth'] = df[f'{prefix}_deriv1'].rolling(
                window=5, center=True, min_periods=1
            ).mean()
    
    return df


def compute_hardness_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the hard/soft flux ratio (hardness ratio).
    
    This is a key precursor indicator:
    - Rising hardness ratio often precedes flare onset
    - High hardness = non-thermal emission dominance
    """
    df = df.copy()
    
    soft = df.get('soft_norm', df.get('soft_counts', None))
    hard = df.get('hard_norm', df.get('hard_counts', None))
    
    if soft is not None and hard is not None:
        # Avoid division by zero
        denominator = np.maximum(np.abs(soft), 1e-10)
        df['hardness_ratio'] = hard / denominator
        
        # Rolling hardness ratio trend
        df['hardness_trend'] = df['hardness_ratio'].rolling(
            window=30, center=False, min_periods=1
        ).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0)
    
    return df


def full_preprocess(df: pd.DataFrame, 
                     bg_window: int = 600,
                     smooth_window: int = 11) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.
    
    Input: raw DataFrame with time_s, soft_counts, hard_counts
    Output: DataFrame with all derived features added
    """
    # Step 1: Resample to uniform grid
    if 'time_s' in df.columns:
        df = resample_uniform(df, cadence_sec=1.0, time_col='time_s')
    
    # Step 2: Smooth raw signals
    df = smooth_signal(df, window=smooth_window)
    
    # Step 3: Background subtraction
    df = subtract_background(df, window_sec=bg_window)
    
    # Step 4: MAD normalization
    df = normalize_mad(df)
    
    # Step 5: Derivatives
    df = compute_derivatives(df)
    
    # Step 6: Hardness ratio
    df = compute_hardness_ratio(df)
    
    # Drop rows with NaN in critical columns
    critical_cols = ['soft_norm', 'hard_norm']
    existing_critical = [c for c in critical_cols if c in df.columns]
    if existing_critical:
        df = df.dropna(subset=existing_critical).reset_index(drop=True)
    
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    
    print("Preprocessor module. Import and use full_preprocess(df).")
    print("Or run loader_sim.py first, then preprocess the output.")
