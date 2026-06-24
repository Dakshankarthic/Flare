"""
classifier.py — Assign GOES flare class (A/B/C/M/X) to detected events.

GOES classification is based on peak soft X-ray flux (W/m² in 1-8 Å):
  A: < 10⁻⁷    B: 10⁻⁷ to 10⁻⁶    C: 10⁻⁶ to 10⁻⁵
  M: 10⁻⁵ to 10⁻⁴    X: ≥ 10⁻⁴

For normalized/simulated data, we use peak_soft (σ above background):
  B: 3-10σ    C: 10-30σ    M: 30-80σ    X: ≥ 80σ
"""

import numpy as np
from typing import List

# Thresholds for normalized (MAD-scaled) soft X-ray flux
NORM_THRESHOLDS = {
    'B': (3.0, 10.0),
    'C': (10.0, 30.0),
    'M': (30.0, 80.0),
    'X': (80.0, float('inf')),
}

# GOES flux thresholds (W/m²) for calibrated data
GOES_THRESHOLDS = {
    'A': (0.0, 1e-7),
    'B': (1e-7, 1e-6),
    'C': (1e-6, 1e-5),
    'M': (1e-5, 1e-4),
    'X': (1e-4, float('inf')),
}


def classify_flare_normalized(peak_soft: float) -> str:
    """
    Classify a flare based on normalized peak soft X-ray flux.
    For use with MAD-normalized data (simulation or uncalibrated data).
    """
    for cls, (lo, hi) in NORM_THRESHOLDS.items():
        if lo <= peak_soft < hi:
            return cls
    if peak_soft >= NORM_THRESHOLDS['X'][0]:
        return 'X'
    return 'A'  # below detection threshold


def classify_flare_goes(peak_flux_wm2: float) -> str:
    """
    Classify a flare using GOES flux thresholds (calibrated data).
    """
    for cls, (lo, hi) in GOES_THRESHOLDS.items():
        if lo <= peak_flux_wm2 < hi:
            return cls
    return 'X' if peak_flux_wm2 >= 1e-4 else 'A'


def classify_master_catalogue(events: list, use_normalized: bool = True) -> list:
    """
    Assign GOES class to all events in the master catalogue.
    
    Args:
        events: List of MasterFlareEvent objects
        use_normalized: If True, use normalized thresholds; else use GOES flux
    
    Returns:
        Same list with flare_class field updated
    """
    classify_fn = classify_flare_normalized if use_normalized else classify_flare_goes
    
    for event in events:
        event.flare_class = classify_fn(event.peak_soft)
    
    return events


def get_class_color(flare_class: str) -> str:
    """Return a color for each GOES class (for visualization)."""
    colors = {
        'A': '#4CAF50',   # green
        'B': '#2196F3',   # blue
        'C': '#FF9800',   # orange
        'M': '#FF5722',   # deep orange
        'X': '#F44336',   # red
    }
    return colors.get(flare_class, '#9E9E9E')


def get_class_severity(flare_class: str) -> int:
    """Return numeric severity (0-4) for a GOES class."""
    severity = {'A': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}
    return severity.get(flare_class, 0)
