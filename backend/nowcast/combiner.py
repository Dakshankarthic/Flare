"""
combiner.py — Merge soft and hard X-ray flare catalogues into a master catalogue.

Temporal coincidence matching:
  - Confirmed flare: Detected in BOTH soft and hard channels (±60s window)
  - Soft-only: Thermal event (low-energy gradual flare)
  - Hard-only: Non-thermal burst (microflare or particle event)

Combined confidence = weighted average of individual confidences + overlap bonus.
"""

import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass


@dataclass
class MasterFlareEvent:
    """A flare event in the combined master catalogue."""
    event_id: int
    start_time: float
    peak_time: float
    end_time: float
    duration: float
    peak_soft: float
    peak_hard: float
    soft_detected: bool
    hard_detected: bool
    detection_type: str       # 'confirmed', 'soft_only', 'hard_only'
    confidence: float
    flare_class: str          # GOES class
    hard_lead_time: float     # seconds: soft_peak - hard_peak (positive = hard first)


def combine_catalogues(soft_detections: list, hard_detections: list,
                        coincidence_window: float = 60.0) -> List[MasterFlareEvent]:
    """
    Merge independent soft and hard X-ray detection catalogues.
    
    Args:
        soft_detections: List of DetectedFlare from soft detector
        hard_detections: List of DetectedFlare from hard detector
        coincidence_window: Maximum time difference (seconds) for matching
    
    Returns:
        List of MasterFlareEvent objects (the master catalogue)
    """
    matched_hard = set()
    master_events = []
    event_id = 0
    
    # --- Match soft detections with hard detections ---
    for soft in soft_detections:
        best_hard = None
        best_dt = float('inf')
        best_hard_idx = -1
        
        for h_idx, hard in enumerate(hard_detections):
            if h_idx in matched_hard:
                continue
            
            # Check temporal coincidence based on peak times
            dt = abs(soft.peak_time - hard.peak_time)
            if dt <= coincidence_window and dt < best_dt:
                best_hard = hard
                best_dt = dt
                best_hard_idx = h_idx
        
        if best_hard is not None:
            # Confirmed: detected in both channels
            matched_hard.add(best_hard_idx)
            
            hard_lead = soft.peak_time - best_hard.peak_time
            combined_conf = (
                0.35 * soft.confidence +
                0.35 * best_hard.confidence +
                0.30  # bonus for dual-channel detection
            )
            
            event = MasterFlareEvent(
                event_id=event_id,
                start_time=min(soft.start_time, best_hard.start_time),
                peak_time=soft.peak_time,  # use soft peak as canonical
                end_time=max(soft.end_time, best_hard.end_time),
                duration=max(soft.end_time, best_hard.end_time) - min(soft.start_time, best_hard.start_time),
                peak_soft=soft.peak_flux,
                peak_hard=best_hard.peak_flux,
                soft_detected=True,
                hard_detected=True,
                detection_type='confirmed',
                confidence=min(1.0, combined_conf),
                flare_class='',  # will be assigned by classifier
                hard_lead_time=hard_lead,
            )
        else:
            # Soft-only: thermal event
            event = MasterFlareEvent(
                event_id=event_id,
                start_time=soft.start_time,
                peak_time=soft.peak_time,
                end_time=soft.end_time,
                duration=soft.duration,
                peak_soft=soft.peak_flux,
                peak_hard=0.0,
                soft_detected=True,
                hard_detected=False,
                detection_type='soft_only',
                confidence=soft.confidence * 0.7,  # lower confidence for single-channel
                flare_class='',
                hard_lead_time=0.0,
            )
        
        master_events.append(event)
        event_id += 1
    
    # --- Add unmatched hard-only detections ---
    for h_idx, hard in enumerate(hard_detections):
        if h_idx in matched_hard:
            continue
        
        event = MasterFlareEvent(
            event_id=event_id,
            start_time=hard.start_time,
            peak_time=hard.peak_time,
            end_time=hard.end_time,
            duration=hard.duration,
            peak_soft=0.0,
            peak_hard=hard.peak_flux,
            soft_detected=False,
            hard_detected=True,
            detection_type='hard_only',
            confidence=hard.confidence * 0.6,
            flare_class='',
            hard_lead_time=0.0,
        )
        master_events.append(event)
        event_id += 1
    
    # Sort by peak time
    master_events.sort(key=lambda e: e.peak_time)
    
    return master_events


def master_catalogue_to_df(events: List[MasterFlareEvent]) -> pd.DataFrame:
    """Convert master catalogue to DataFrame."""
    records = []
    for e in events:
        records.append({
            'event_id': e.event_id,
            'start_time': e.start_time,
            'peak_time': e.peak_time,
            'end_time': e.end_time,
            'duration': e.duration,
            'peak_soft': e.peak_soft,
            'peak_hard': e.peak_hard,
            'soft_detected': e.soft_detected,
            'hard_detected': e.hard_detected,
            'detection_type': e.detection_type,
            'confidence': e.confidence,
            'flare_class': e.flare_class,
            'hard_lead_time': e.hard_lead_time,
        })
    return pd.DataFrame(records)
