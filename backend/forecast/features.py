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

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


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

    # Build a simple flare history from large peaks for "time since last flare"
    flare_history = _detect_flare_times_simple(soft, times, threshold=5.0)
    
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

            # === ADVANCED FEATURES (TEJAS Improvement Plan) ===

            # --- Wavelet coefficients (db4, level 3) ---
            # Captures transient precursors at multiple time scales
            **_wavelet_features(s_win, prefix='soft'),
            **_wavelet_features(h_win, prefix='hard'),

            # --- Transfer Entropy (HXR → SXR) ---
            # Quantifies directional information flow between channels
            'transfer_entropy_h2s': _transfer_entropy(h_win, s_win),

            # --- Rolling Hurst exponent ---
            # Detects regime changes: H>0.5 = trending, H<0.5 = mean-reverting
            'hurst_soft': _hurst_exponent(s_win),
            'hurst_hard': _hurst_exponent(h_win),

            # --- Peak-to-background ratio ---
            'peak_bg_ratio_soft': np.nanmax(s_win) / max(abs(np.nanmedian(s_win)), 1e-10),
            'peak_bg_ratio_hard': np.nanmax(h_win) / max(abs(np.nanmedian(h_win)), 1e-10),

            # --- Time since last flare (flare memory effect) ---
            'time_since_last_flare': _time_since_last_flare(
                center_time, flare_history
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


def _wavelet_features(x: np.ndarray, prefix: str = 'soft',
                      wavelet: str = 'db4', level: int = 3) -> dict:
    """Extract wavelet coefficient statistics (captures transient precursors)."""
    feats = {}
    if not HAS_PYWT or len(x) < 16:
        feats[f'{prefix}_wavelet_energy'] = 0.0
        feats[f'{prefix}_wavelet_entropy'] = 0.0
        return feats

    try:
        x_clean = np.nan_to_num(x, nan=0.0)
        max_level = pywt.dwt_max_level(len(x_clean), wavelet)
        actual_level = min(level, max_level)
        if actual_level < 1:
            feats[f'{prefix}_wavelet_energy'] = 0.0
            feats[f'{prefix}_wavelet_entropy'] = 0.0
            return feats

        coeffs = pywt.wavedec(x_clean, wavelet, level=actual_level)
        # Energy in detail coefficients (high-frequency transients)
        detail_energy = sum(np.sum(c ** 2) for c in coeffs[1:])
        total_energy = sum(np.sum(c ** 2) for c in coeffs) + 1e-10
        feats[f'{prefix}_wavelet_energy'] = detail_energy / total_energy

        # Wavelet entropy (complexity measure)
        energies = np.array([np.sum(c ** 2) for c in coeffs])
        probs = energies / (energies.sum() + 1e-10)
        probs = probs[probs > 0]
        feats[f'{prefix}_wavelet_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
    except Exception:
        feats[f'{prefix}_wavelet_energy'] = 0.0
        feats[f'{prefix}_wavelet_entropy'] = 0.0

    return feats


def _transfer_entropy(source: np.ndarray, target: np.ndarray,
                      lag: int = 1, n_bins: int = 8) -> float:
    """Estimate transfer entropy from source to target (directional info flow)."""
    try:
        s = np.nan_to_num(source, nan=0.0)
        t = np.nan_to_num(target, nan=0.0)
        n = len(s)
        if n < lag + 2:
            return 0.0

        # Discretize into bins
        s_bins = np.digitize(s, np.linspace(s.min() - 1e-10, s.max() + 1e-10, n_bins))
        t_bins = np.digitize(t, np.linspace(t.min() - 1e-10, t.max() + 1e-10, n_bins))

        # Build joint distributions
        t_future = t_bins[lag:]
        t_past = t_bins[:-lag]
        s_past = s_bins[:-lag]

        m = len(t_future)
        # P(t_future | t_past, s_past) vs P(t_future | t_past)
        # Simplified: use contingency table approach
        joint_3 = np.zeros((n_bins + 1, n_bins + 1, n_bins + 1))
        joint_2 = np.zeros((n_bins + 1, n_bins + 1))

        for i in range(m):
            joint_3[t_future[i], t_past[i], s_past[i]] += 1
            joint_2[t_future[i], t_past[i]] += 1

        joint_3 /= m + 1e-10
        joint_2 /= m + 1e-10

        te = 0.0
        for tf in range(n_bins + 1):
            for tp in range(n_bins + 1):
                for sp in range(n_bins + 1):
                    p_3 = joint_3[tf, tp, sp]
                    if p_3 > 1e-10:
                        p_2 = joint_2[tf, tp]
                        p_tp_sp = joint_3[:, tp, sp].sum()
                        p_tp = joint_2[:, tp].sum()
                        if p_tp > 1e-10 and p_tp_sp > 1e-10 and p_2 > 1e-10:
                            te += p_3 * np.log2(
                                (p_3 * p_tp) / (p_2 * p_tp_sp + 1e-10) + 1e-10
                            )
        return max(te, 0.0)
    except Exception:
        return 0.0


def _hurst_exponent(x: np.ndarray) -> float:
    """Estimate Hurst exponent using R/S analysis. H>0.5=trending, H<0.5=mean-reverting."""
    try:
        x_clean = np.nan_to_num(x, nan=0.0)
        n = len(x_clean)
        if n < 20:
            return 0.5

        max_k = min(n // 2, 128)
        rs_values = []
        ns = []

        for k in [int(n / d) for d in [2, 4, 8, 16] if n / d >= 8]:
            rs_list = []
            for start in range(0, n - k + 1, k):
                segment = x_clean[start:start + k]
                mean_seg = np.mean(segment)
                deviations = np.cumsum(segment - mean_seg)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(segment)
                if s > 1e-10:
                    rs_list.append(r / s)
            if rs_list:
                rs_values.append(np.mean(rs_list))
                ns.append(k)

        if len(rs_values) < 2:
            return 0.5

        log_n = np.log(ns)
        log_rs = np.log(np.array(rs_values) + 1e-10)
        slope = np.polyfit(log_n, log_rs, 1)[0]
        return np.clip(slope, 0.0, 1.0)
    except Exception:
        return 0.5


def _detect_flare_times_simple(flux: np.ndarray, times: np.ndarray,
                                threshold: float = 5.0) -> list:
    """Simple peak detection for building flare history (for time-since-last feature)."""
    peaks = []
    n = len(flux)
    i = 0
    while i < n:
        if flux[i] > threshold:
            # Find peak in this region
            peak_idx = i
            while i < n and flux[i] > threshold * 0.5:
                if flux[i] > flux[peak_idx]:
                    peak_idx = i
                i += 1
            peaks.append(times[peak_idx])
        i += 1
    return peaks


def _time_since_last_flare(current_time: float, flare_times: list) -> float:
    """Compute time (seconds) since the most recent flare before current_time."""
    past_flares = [t for t in flare_times if t < current_time]
    if not past_flares:
        return 99999.0  # large value = no recent flare
    return current_time - max(past_flares)


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
    # Original 33 features
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
    # Advanced features (TEJAS Improvement Plan)
    'soft_wavelet_energy', 'soft_wavelet_entropy',
    'hard_wavelet_energy', 'hard_wavelet_entropy',
    'transfer_entropy_h2s',
    'hurst_soft', 'hurst_hard',
    'peak_bg_ratio_soft', 'peak_bg_ratio_hard',
    'time_since_last_flare',
]
