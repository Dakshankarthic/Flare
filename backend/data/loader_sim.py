"""
loader_sim.py — Generate realistic synthetic solar X-ray light curves.

Produces soft (SoLEXS-like, 1-30 keV) and hard (HEL1OS-like, 10-150 keV)
X-ray time-series with embedded solar flares of all GOES classes.

Physics modelling:
  - Quiet Sun: slowly varying background with Poisson noise
  - Flare rise: exponential ramp (fast for impulsive, slow for gradual)
  - Flare peak: brief plateau
  - Flare decay: exponential decay (longer than rise, following Kopp-Poletto cooling)
  - Neupert effect: hard X-ray burst precedes soft X-ray peak
  - Hard X-ray is impulsive; soft X-ray is gradual
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# GOES Flare class flux thresholds (W/m² in 1-8 Å band)
# We map these to normalized count rates for simulation
# ---------------------------------------------------------------------------
FLARE_CLASSES = {
    'B': {'peak_min': 5.0,   'peak_max': 15.0,   'weight': 0.40},
    'C': {'peak_min': 15.0,  'peak_max': 50.0,    'weight': 0.30},
    'M': {'peak_min': 50.0,  'peak_max': 150.0,   'weight': 0.20},
    'X': {'peak_min': 150.0, 'peak_max': 500.0,   'weight': 0.10},
}


@dataclass
class FlareEvent:
    """Represents a single simulated solar flare."""
    onset_time: float        # seconds from start
    peak_time: float
    end_time: float
    flare_class: str         # B, C, M, X
    peak_soft: float         # peak soft X-ray amplitude (normalized)
    peak_hard: float         # peak hard X-ray amplitude
    rise_time: float         # seconds
    decay_time: float        # seconds
    hard_lead_time: float    # seconds (hard X-ray precursor before soft peak)


@dataclass
class SimulationConfig:
    """Configuration for the synthetic light-curve generator."""
    duration_hours: float = 24.0
    cadence_sec: float = 1.0
    
    # Background parameters
    bg_soft_mean: float = 10.0        # baseline soft count rate
    bg_hard_mean: float = 3.0         # baseline hard count rate
    bg_drift_period_hr: float = 6.0   # slow sinusoidal drift period
    bg_drift_amplitude: float = 0.15  # fractional amplitude of drift
    noise_poisson: bool = True        # add Poisson noise
    noise_gaussian_frac: float = 0.02 # additional Gaussian noise (fraction of signal)
    
    # Flare parameters
    num_flares: int = 8               # number of flares to inject
    flare_classes: dict = field(default_factory=lambda: FLARE_CLASSES.copy())
    
    # Neupert effect: hard X-ray peaks before soft
    hard_lead_min_sec: float = 10.0
    hard_lead_max_sec: float = 120.0
    hard_impulsive_frac: float = 0.3  # hard X-ray rise/decay ratio vs soft
    
    # Random seed for reproducibility
    seed: int = 42


def generate_flare_profile(t: np.ndarray, onset: float, peak_time: float,
                            end_time: float, peak_amplitude: float,
                            rise_tau: float, decay_tau: float) -> np.ndarray:
    """
    Generate a single flare temporal profile.
    
    Uses exponential rise + exponential decay model:
      F(t) = A * exp(-(t_peak - t) / rise_tau)   for t < t_peak
      F(t) = A * exp(-(t - t_peak) / decay_tau)   for t >= t_peak
    """
    profile = np.zeros_like(t, dtype=np.float64)
    
    # Rising phase
    rise_mask = (t >= onset) & (t < peak_time)
    if np.any(rise_mask):
        profile[rise_mask] = peak_amplitude * np.exp(
            -(peak_time - t[rise_mask]) / rise_tau
        )
    
    # Decay phase
    decay_mask = (t >= peak_time) & (t <= end_time)
    if np.any(decay_mask):
        profile[decay_mask] = peak_amplitude * np.exp(
            -(t[decay_mask] - peak_time) / decay_tau
        )
    
    return profile


def generate_fred_flare(t: np.ndarray, onset: float, peak_time: float,
                         end_time: float, peak_amplitude: float,
                         rise_tau: float, decay_tau: float,
                         n_thermal_components: int = 3,
                         rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a FRED (Fast Rise Exponential Decay) flare profile with
    multi-thermal cooling components.

    Physics:
      - Rise: exponential (same as basic model)
      - Decay: sum of N thermal cooling components with different time-scales
               (Kopp-Poletto radiative cooling from coronal loops)
      - Each component represents a different temperature plasma
      - Produces more realistic, multi-peaked decay profiles

    This is a DIFFERENTIATOR: generates physically-constrained synthetic
    data for pre-training before fine-tuning on real Aditya-L1 data.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    profile = np.zeros_like(t, dtype=np.float64)

    # Rising phase (same as basic)
    rise_mask = (t >= onset) & (t < peak_time)
    if np.any(rise_mask):
        profile[rise_mask] = peak_amplitude * np.exp(
            -(peak_time - t[rise_mask]) / rise_tau
        )

    # Multi-thermal decay: sum of N exponential components
    decay_mask = (t >= peak_time) & (t <= end_time)
    if np.any(decay_mask):
        t_decay = t[decay_mask] - peak_time

        for i in range(n_thermal_components):
            # Each component has a different decay timescale
            # Hotter components cool faster
            component_tau = decay_tau * (0.3 + 0.7 * i / max(n_thermal_components - 1, 1))
            component_amp = peak_amplitude * rng.uniform(0.2, 0.6)

            # Add slight delay for each thermal component (cascading cooling)
            delay = i * decay_tau * 0.1
            delayed_t = np.maximum(t_decay - delay, 0)

            profile[decay_mask] += component_amp * np.exp(-delayed_t / component_tau)

        # Normalize so peak matches peak_amplitude
        peak_val = profile[decay_mask].max() if len(profile[decay_mask]) > 0 else 1.0
        if peak_val > 0:
            profile[decay_mask] *= peak_amplitude / peak_val

    return profile


def generate_synthetic_data(config: Optional[SimulationConfig] = None) -> tuple:
    """
    Generate synthetic soft and hard X-ray light curves with embedded flares.
    
    Returns:
        (df, flare_events) where:
            df: DataFrame with columns [time_s, soft_counts, hard_counts, 
                                         soft_bg, hard_bg]
            flare_events: list of FlareEvent objects (ground truth)
    """
    if config is None:
        config = SimulationConfig()
    
    rng = np.random.default_rng(config.seed)
    
    # Time array
    n_samples = int(config.duration_hours * 3600 / config.cadence_sec)
    t = np.arange(n_samples) * config.cadence_sec
    
    # ---- Background ----
    drift_phase = rng.uniform(0, 2 * np.pi)
    drift_soft = 1.0 + config.bg_drift_amplitude * np.sin(
        2 * np.pi * t / (config.bg_drift_period_hr * 3600) + drift_phase
    )
    drift_hard = 1.0 + config.bg_drift_amplitude * 0.5 * np.sin(
        2 * np.pi * t / (config.bg_drift_period_hr * 3600) + drift_phase + 0.3
    )
    
    soft_bg = config.bg_soft_mean * drift_soft
    hard_bg = config.bg_hard_mean * drift_hard
    
    soft_signal = soft_bg.copy()
    hard_signal = hard_bg.copy()
    
    # ---- Generate flare events ----
    flare_events = []
    
    # Choose flare classes weighted by frequency
    classes = list(config.flare_classes.keys())
    weights = [config.flare_classes[c]['weight'] for c in classes]
    chosen_classes = rng.choice(classes, size=config.num_flares, p=weights)
    
    # Space flares at least 20 minutes apart
    min_spacing = 1200  # seconds
    available_time = config.duration_hours * 3600 - 2 * min_spacing
    flare_times = np.sort(
        rng.uniform(min_spacing, available_time, size=config.num_flares)
    )
    
    # Ensure minimum spacing
    for i in range(1, len(flare_times)):
        if flare_times[i] - flare_times[i-1] < min_spacing:
            flare_times[i] = flare_times[i-1] + min_spacing
    
    for i, (flare_class, onset_time) in enumerate(zip(chosen_classes, flare_times)):
        cls_info = config.flare_classes[flare_class]
        
        # Peak amplitude
        peak_soft = rng.uniform(cls_info['peak_min'], cls_info['peak_max'])
        
        # Hard X-ray peak: typically 0.3-0.8x of soft peak but more impulsive
        hard_ratio = rng.uniform(0.3, 0.8)
        peak_hard = peak_soft * hard_ratio
        
        # Timing
        rise_time = rng.uniform(30, 300)  # 30s to 5min rise
        decay_time = rise_time * rng.uniform(2.0, 5.0)  # decay is 2-5x longer
        
        peak_time = onset_time + rise_time
        end_time = peak_time + decay_time
        
        # Neupert effect: hard X-ray arrives earlier
        hard_lead_time = rng.uniform(config.hard_lead_min_sec, config.hard_lead_max_sec)
        hard_peak_time = peak_time - hard_lead_time
        hard_onset = max(0, hard_peak_time - rise_time * config.hard_impulsive_frac)
        hard_end = hard_peak_time + decay_time * config.hard_impulsive_frac
        
        # Generate profiles
        soft_flare = generate_flare_profile(
            t, onset_time, peak_time, end_time, peak_soft,
            rise_tau=rise_time / 3, decay_tau=decay_time / 3
        )
        hard_flare = generate_flare_profile(
            t, hard_onset, hard_peak_time, hard_end, peak_hard,
            rise_tau=rise_time * config.hard_impulsive_frac / 3,
            decay_tau=decay_time * config.hard_impulsive_frac / 3
        )
        
        soft_signal += soft_flare
        hard_signal += hard_flare
        
        flare_events.append(FlareEvent(
            onset_time=onset_time,
            peak_time=peak_time,
            end_time=end_time,
            flare_class=flare_class,
            peak_soft=peak_soft,
            peak_hard=peak_hard,
            rise_time=rise_time,
            decay_time=decay_time,
            hard_lead_time=hard_lead_time,
        ))
    
    # ---- Add noise ----
    if config.noise_poisson:
        # Poisson noise: variance = signal level
        soft_noisy = rng.poisson(np.maximum(soft_signal, 0.1)).astype(np.float64)
        hard_noisy = rng.poisson(np.maximum(hard_signal, 0.1)).astype(np.float64)
    else:
        soft_noisy = soft_signal.copy()
        hard_noisy = hard_signal.copy()
    
    # Additional Gaussian noise
    if config.noise_gaussian_frac > 0:
        soft_noisy += rng.normal(0, config.noise_gaussian_frac * soft_bg, n_samples)
        hard_noisy += rng.normal(0, config.noise_gaussian_frac * hard_bg, n_samples)
    
    # ---- Build DataFrame ----
    df = pd.DataFrame({
        'time_s': t,
        'soft_counts': np.maximum(soft_noisy, 0),
        'hard_counts': np.maximum(hard_noisy, 0),
        'soft_clean': soft_signal,
        'hard_clean': hard_signal,
        'soft_bg': soft_bg,
        'hard_bg': hard_bg,
    })
    
    return df, flare_events


def flare_events_to_df(events: List[FlareEvent]) -> pd.DataFrame:
    """Convert flare events to a DataFrame (ground truth catalogue)."""
    records = []
    for e in events:
        records.append({
            'onset_time': e.onset_time,
            'peak_time': e.peak_time,
            'end_time': e.end_time,
            'flare_class': e.flare_class,
            'peak_soft': e.peak_soft,
            'peak_hard': e.peak_hard,
            'rise_time': e.rise_time,
            'decay_time': e.decay_time,
            'hard_lead_time': e.hard_lead_time,
            'duration': e.end_time - e.onset_time,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI — generate and save simulation data
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic solar X-ray data')
    parser.add_argument('--hours', type=float, default=48, help='Duration in hours')
    parser.add_argument('--flares', type=int, default=12, help='Number of flares')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='data/sim', help='Output directory')
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output, exist_ok=True)
    
    config = SimulationConfig(
        duration_hours=args.hours,
        num_flares=args.flares,
        seed=args.seed,
    )
    
    df, events = generate_synthetic_data(config)
    events_df = flare_events_to_df(events)
    
    # Save
    df.to_parquet(os.path.join(args.output, 'lightcurves.parquet'), compression='gzip')
    events_df.to_csv(os.path.join(args.output, 'ground_truth_flares.csv'), index=False)
    
    print(f"Generated {len(df)} data points ({args.hours} hours)")
    print(f"Injected {len(events)} flares:")
    for e in events:
        print(f"  {e.flare_class}-class at t={e.peak_time:.0f}s "
              f"(peak_soft={e.peak_soft:.1f}, lead={e.hard_lead_time:.0f}s)")
