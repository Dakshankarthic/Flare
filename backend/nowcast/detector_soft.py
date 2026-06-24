"""
detector_soft.py — Soft X-ray (SoLEXS) flare detection algorithm.

Algorithm (derivative-threshold based, adapted from NOAA SWPC):
  1. Compute 1st derivative of smoothed soft X-ray flux
  2. Identify onset: N consecutive seconds where derivative exceeds adaptive threshold
  3. Adaptive threshold = k × σ(quiet-period derivative)
  4. Track peak: maximum flux before derivative turns negative
  5. Identify end: flux decays to 50% of (peak - pre-flare baseline)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DetectedFlare:
    """A single detected flare event."""
    start_idx: int
    peak_idx: int
    end_idx: int
    start_time: float
    peak_time: float
    end_time: float
    peak_flux: float        # peak normalized flux
    pre_flare_bg: float     # background level before flare
    rise_rate: float        # max derivative during rise
    duration: float         # seconds
    channel: str            # 'soft' or 'hard'
    confidence: float       # 0-1 detection confidence


def detect_flares_soft(df: pd.DataFrame,
                        flux_col: str = 'soft_norm',
                        deriv_col: str = 'soft_deriv1_smooth',
                        time_col: str = 'time_s',
                        sigma_threshold: float = 2.5,
                        min_rise_duration: int = 3,
                        min_peak_flux: float = 2.0,
                        min_flare_duration: int = 15,
                        bg_window: int = 300) -> List[DetectedFlare]:
    """
    Detect solar flares in the soft X-ray channel.
    
    Args:
        df: Preprocessed DataFrame with normalized flux and derivatives
        flux_col: Column with normalized soft X-ray flux
        deriv_col: Column with smoothed 1st derivative
        sigma_threshold: Detection threshold in units of derivative σ
        min_rise_duration: Minimum consecutive seconds of rising flux
        min_peak_flux: Minimum normalized peak flux to qualify
        min_flare_duration: Minimum event duration in seconds
        bg_window: Window for pre-flare background estimation
    
    Returns:
        List of DetectedFlare objects
    """
    flux = df[flux_col].values
    time = df[time_col].values
    
    # Compute derivative if not present
    if deriv_col in df.columns:
        deriv = df[deriv_col].values
    else:
        deriv = np.gradient(flux)
        # Smooth with 5-point moving average
        kernel = np.ones(5) / 5
        deriv = np.convolve(deriv, kernel, mode='same')
    
    n = len(flux)
    
    # --- Adaptive threshold ---
    # Use the MAD of the derivative during "quiet" periods
    # Quiet = flux below 2σ (likely no flare)
    quiet_mask = flux < 2.0
    if quiet_mask.sum() < 100:
        quiet_mask = np.ones(n, dtype=bool)  # fallback: use everything
    
    quiet_deriv = deriv[quiet_mask]
    deriv_median = np.nanmedian(quiet_deriv)
    deriv_mad = np.nanmedian(np.abs(quiet_deriv - deriv_median))
    deriv_sigma = 1.4826 * deriv_mad  # MAD to σ conversion
    
    if deriv_sigma <= 0:
        deriv_sigma = np.nanstd(quiet_deriv)
    
    threshold = deriv_median + sigma_threshold * deriv_sigma
    
    # --- Find candidate onset points ---
    # Where derivative exceeds threshold for min_rise_duration consecutive seconds
    above_threshold = deriv > threshold
    
    # Count consecutive True values
    candidates = []
    streak_start = None
    streak_count = 0
    
    for i in range(n):
        if above_threshold[i]:
            if streak_start is None:
                streak_start = i
            streak_count += 1
        else:
            if streak_count >= min_rise_duration:
                candidates.append(streak_start)
            streak_start = None
            streak_count = 0
    
    # Handle final streak
    if streak_count >= min_rise_duration and streak_start is not None:
        candidates.append(streak_start)
    
    # --- For each candidate onset, find peak and end ---
    detected = []
    used_ranges = []  # prevent overlapping detections
    
    for onset_idx in candidates:
        # Check for overlap with existing detections
        overlaps = False
        for (s, e) in used_ranges:
            if s <= onset_idx <= e:
                overlaps = True
                break
        if overlaps:
            continue
        
        # Pre-flare background (median of bg_window seconds before onset)
        bg_start = max(0, onset_idx - bg_window)
        pre_flare_bg = np.nanmedian(flux[bg_start:onset_idx])
        
        # Find peak: maximum flux after onset
        # Search forward up to 600s (10 min) or until flux starts declining steadily
        search_end = min(n, onset_idx + 600)
        peak_region = flux[onset_idx:search_end]
        
        if len(peak_region) == 0:
            continue
        
        peak_local_idx = np.argmax(peak_region)
        peak_idx = onset_idx + peak_local_idx
        peak_flux = flux[peak_idx]
        
        # Check minimum peak flux
        if peak_flux < min_peak_flux:
            continue
        
        # Find end: flux decays to 50% of (peak - background)
        half_level = pre_flare_bg + 0.5 * (peak_flux - pre_flare_bg)
        end_search = min(n, peak_idx + 3600)  # search up to 1 hour after peak
        
        end_idx = peak_idx
        for j in range(peak_idx + 1, end_search):
            if flux[j] <= half_level:
                end_idx = j
                break
        else:
            end_idx = end_search - 1
        
        # Duration check
        duration = time[end_idx] - time[onset_idx]
        if duration < min_flare_duration:
            continue
        
        # Rise rate
        rise_rate = np.max(deriv[onset_idx:peak_idx+1]) if peak_idx > onset_idx else 0
        
        # Confidence score (0-1)
        # Based on: peak_flux strength, rise duration, derivative significance
        conf_flux = min(1.0, (peak_flux - min_peak_flux) / (10 * min_peak_flux))
        conf_deriv = min(1.0, rise_rate / (3 * threshold)) if threshold > 0 else 0.5
        conf_duration = min(1.0, duration / 120)  # 2-min events get full score
        confidence = 0.4 * conf_flux + 0.4 * conf_deriv + 0.2 * conf_duration
        
        flare = DetectedFlare(
            start_idx=onset_idx,
            peak_idx=peak_idx,
            end_idx=end_idx,
            start_time=time[onset_idx],
            peak_time=time[peak_idx],
            end_time=time[end_idx],
            peak_flux=peak_flux,
            pre_flare_bg=pre_flare_bg,
            rise_rate=rise_rate,
            duration=duration,
            channel='soft',
            confidence=confidence,
        )
        
        detected.append(flare)
        used_ranges.append((onset_idx, end_idx))
    
    return detected


def detections_to_df(detections: List[DetectedFlare]) -> pd.DataFrame:
    """Convert list of DetectedFlare to a DataFrame."""
    records = []
    for d in detections:
        records.append({
            'start_time': d.start_time,
            'peak_time': d.peak_time,
            'end_time': d.end_time,
            'peak_flux': d.peak_flux,
            'pre_flare_bg': d.pre_flare_bg,
            'rise_rate': d.rise_rate,
            'duration': d.duration,
            'channel': d.channel,
            'confidence': d.confidence,
        })
    return pd.DataFrame(records)
