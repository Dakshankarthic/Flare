"""
detector_hard.py — Hard X-ray (HEL1OS) flare detection algorithm.

Hard X-rays are more impulsive than soft X-rays:
  - Shorter onset windows (2 seconds vs 4 for soft)
  - More impulsive profiles (faster rise and decay)
  - Higher noise floor → energy-dependent thresholds
  - Detects non-thermal bursts that precede soft X-ray flares
"""

import numpy as np
import pandas as pd
from typing import List
from backend.nowcast.detector_soft import DetectedFlare, detect_flares_soft


def detect_flares_hard(df: pd.DataFrame,
                        flux_col: str = 'hard_norm',
                        deriv_col: str = 'hard_deriv1_smooth',
                        time_col: str = 'time_s',
                        sigma_threshold: float = 3.5,
                        min_rise_duration: int = 2,
                        min_peak_flux: float = 2.5,
                        min_flare_duration: int = 10,
                        bg_window: int = 300) -> List[DetectedFlare]:
    """
    Detect solar flares in the hard X-ray channel.
    
    Uses the same algorithm as soft X-ray but with parameters tuned
    for the more impulsive hard X-ray characteristics:
      - Lower min_rise_duration (2s vs 4s) — hard X-rays are sharper
      - Lower sigma_threshold (3.5 vs 4.0) — detect weaker hard events
      - Lower min_flare_duration (10s vs 30s) — hard bursts can be brief
      - Lower min_peak_flux (2.5 vs 3.0) — hard X-ray amplitudes are lower
    """
    detections = detect_flares_soft(
        df=df,
        flux_col=flux_col,
        deriv_col=deriv_col,
        time_col=time_col,
        sigma_threshold=sigma_threshold,
        min_rise_duration=min_rise_duration,
        min_peak_flux=min_peak_flux,
        min_flare_duration=min_flare_duration,
        bg_window=bg_window,
    )
    
    # Override channel label
    for d in detections:
        d.channel = 'hard'
    
    return detections
