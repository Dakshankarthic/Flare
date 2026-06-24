"""
loader_fits.py — Read Aditya-L1 SoLEXS and HEL1OS Level-1 FITS files.

SoLEXS structure (from real data):
  ZIP → date_folder/SDD1/ and SDD2/ directories
  Each contains:
    *.lc.gz   — Light curve FITS (binary table with TIME, RATE, ERROR columns)
    *.gti.gz  — Good Time Interval FITS
    *.pi.gz   — Spectral PHA-II FITS

HEL1OS structure (similar):
  ZIP → date_folder/CdTe/ and CZT/ directories
  Light curves at 1-s cadence for each detector.
"""

import os
import gzip
import shutil
import zipfile
import pathlib
import tempfile
import numpy as np
import pandas as pd
from astropy.io import fits


# ---------------------------------------------------------------------------
# SoLEXS Loader
# ---------------------------------------------------------------------------

def read_solexs_lc(fits_path: str) -> pd.DataFrame:
    """
    Read a single SoLEXS light-curve FITS file (.lc or .lc.gz).
    
    Returns DataFrame with columns:
        time_s   — time in seconds (from FITS TIME column, usually MET or UNIX)
        rate     — count rate (counts/s)
        error    — statistical error on rate
    """
    # Handle gzipped files by decompressing to a temp file
    if fits_path.endswith('.gz'):
        with tempfile.NamedTemporaryFile(suffix='.lc', delete=False) as tmp:
            tmp_path = tmp.name
        with gzip.open(fits_path, 'rb') as f_in:
            with open(tmp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        result = _parse_lc_fits(tmp_path)
        os.unlink(tmp_path)
        return result
    else:
        return _parse_lc_fits(fits_path)


def _parse_lc_fits(fits_path: str) -> pd.DataFrame:
    """Parse a FITS light-curve file into a DataFrame."""
    with fits.open(fits_path) as hdul:
        # Print info for debugging on first call
        # hdul.info()
        
        # Light curve data is typically in extension 1 (RATE table)
        # Common column names: TIME, RATE (or COUNT_RATE), ERROR (or COUNT_ERR)
        data = hdul[1].data
        header = hdul[1].header
        
        # Try various column name conventions
        time_col = _find_column(data, ['TIME', 'time', 'Time'])
        rate_col = _find_column(data, ['RATE', 'COUNT_RATE', 'COUNTS', 'rate', 'count_rate'])
        err_col = _find_column(data, ['ERROR', 'COUNT_ERR', 'STAT_ERR', 'error'])
        
        times = np.array(data[time_col], dtype=np.float64)
        rates = np.array(data[rate_col], dtype=np.float64)
        
        if err_col:
            errors = np.array(data[err_col], dtype=np.float64)
        else:
            # Poisson error estimate
            errors = np.sqrt(np.maximum(rates, 0))
        
        # Get reference time from header (MJDREF, TIMEZERO, etc.)
        mjdref = header.get('MJDREFI', 0) + header.get('MJDREFF', 0.0)
        timezero = header.get('TIMEZERO', 0.0)
        timeunit = header.get('TIMEUNIT', 's')
        
        df = pd.DataFrame({
            'time_s': times + timezero,
            'rate': rates,
            'error': errors,
        })
        
        # Store metadata
        df.attrs['mjdref'] = mjdref
        df.attrs['timeunit'] = timeunit
        df.attrs['source'] = fits_path
        df.attrs['tstart'] = header.get('TSTART', times[0])
        df.attrs['tstop'] = header.get('TSTOP', times[-1])
        
        return df


def _find_column(data, candidates):
    """Find the first matching column name from candidates."""
    available = [c.upper() for c in data.columns.names]
    for c in candidates:
        if c.upper() in available:
            # Return the actual column name (preserve case)
            idx = available.index(c.upper())
            return data.columns.names[idx]
    return candidates[0]  # fallback


def read_solexs_gti(gti_path: str) -> pd.DataFrame:
    """Read Good Time Interval file."""
    if gti_path.endswith('.gz'):
        with tempfile.NamedTemporaryFile(suffix='.gti', delete=False) as tmp:
            tmp_path = tmp.name
        with gzip.open(gti_path, 'rb') as f_in:
            with open(tmp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        result = _parse_gti_fits(tmp_path)
        os.unlink(tmp_path)
        return result
    else:
        return _parse_gti_fits(gti_path)


def _parse_gti_fits(fits_path: str) -> pd.DataFrame:
    """Parse GTI FITS into DataFrame with START, STOP columns."""
    with fits.open(fits_path) as hdul:
        # GTI is usually in extension 1 or named 'GTI'
        for ext in hdul:
            if hasattr(ext, 'data') and ext.data is not None:
                names = [n.upper() for n in ext.columns.names] if hasattr(ext, 'columns') else []
                if 'START' in names or 'TSTART' in names:
                    start_col = 'START' if 'START' in names else 'TSTART'
                    stop_col = 'STOP' if 'STOP' in names else 'TSTOP'
                    return pd.DataFrame({
                        'start': ext.data[start_col],
                        'stop': ext.data[stop_col]
                    })
    return pd.DataFrame(columns=['start', 'stop'])


# ---------------------------------------------------------------------------
# Load a full day of SoLEXS data from the extracted directory structure
# ---------------------------------------------------------------------------

def load_solexs_day(day_dir: str, detector: str = 'SDD2') -> pd.DataFrame:
    """
    Load all SoLEXS light-curve data for a single day.
    
    Args:
        day_dir: Path to the extracted day directory 
                 (e.g., 'AL1_SLX_L1_20240701_v1.1/AL1_SLX_L1_20240701_v1.1/SDD2/')
        detector: 'SDD1' or 'SDD2' (SDD2 is the primary science detector)
    
    Returns:
        DataFrame with time_s, rate, error columns
    """
    day_path = pathlib.Path(day_dir)
    
    # Find the detector subdirectory
    det_dir = None
    for candidate in [day_path / detector, day_path]:
        if candidate.exists():
            det_dir = candidate
            break
    
    if det_dir is None:
        raise FileNotFoundError(f"No detector directory found at {day_dir}")
    
    # Find light-curve files
    lc_files = list(det_dir.glob('*.lc.gz')) + list(det_dir.glob('*.lc'))
    if not lc_files:
        # Try recursing one level deeper
        lc_files = list(det_dir.rglob('*.lc.gz')) + list(det_dir.rglob('*.lc'))
    
    if not lc_files:
        raise FileNotFoundError(f"No light-curve files found in {det_dir}")
    
    frames = []
    for lc_file in sorted(lc_files):
        try:
            df = read_solexs_lc(str(lc_file))
            frames.append(df)
        except Exception as e:
            print(f"Warning: Could not read {lc_file}: {e}")
    
    if not frames:
        raise RuntimeError(f"No readable light-curve files in {det_dir}")
    
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('time_s').reset_index(drop=True)
    return combined


def scan_and_load_solexs(root_dir: str, detector: str = 'SDD2', 
                          max_days: int = None) -> pd.DataFrame:
    """
    Scan a directory tree for SoLEXS day directories and load all of them.
    
    The PRADAN data is typically organized as:
        root/
          solexs_download/
            AL1_SLX_L1_YYYYMMDD_v1.1/
              AL1_SLX_L1_YYYYMMDD_v1.1/
                SDD1/
                SDD2/
                  *.lc.gz
    
    This function recursively searches for .lc.gz files and loads them.
    """
    root = pathlib.Path(root_dir)
    
    # Find all .lc.gz and .lc files recursively
    lc_files = sorted(root.rglob(f'*{detector}*.lc.gz')) + sorted(root.rglob(f'*{detector}*.lc'))
    
    if not lc_files:
        # Try without detector filter
        lc_files = sorted(root.rglob('*.lc.gz')) + sorted(root.rglob('*.lc'))
    
    if max_days and len(lc_files) > max_days:
        lc_files = lc_files[:max_days]
    
    print(f"Found {len(lc_files)} light-curve files in {root_dir}")
    
    frames = []
    for lc_file in lc_files:
        try:
            df = read_solexs_lc(str(lc_file))
            df['source_file'] = lc_file.name
            frames.append(df)
            print(f"  Loaded {lc_file.name}: {len(df)} rows, "
                  f"t=[{df['time_s'].min():.1f}, {df['time_s'].max():.1f}]")
        except Exception as e:
            print(f"  Warning: Could not read {lc_file}: {e}")
    
    if not frames:
        raise RuntimeError(f"No readable light-curve files in {root_dir}")
    
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('time_s').reset_index(drop=True)
    combined = combined.drop_duplicates(subset='time_s', keep='first')
    
    print(f"\nTotal: {len(combined)} data points, "
          f"time span: {(combined['time_s'].max() - combined['time_s'].min())/86400:.1f} days")
    
    return combined


# ---------------------------------------------------------------------------
# HEL1OS Loader (same structure, different column conventions)
# ---------------------------------------------------------------------------

def read_hel1os_lc(fits_path: str) -> pd.DataFrame:
    """Read a HEL1OS light-curve FITS file. Same format as SoLEXS."""
    return read_solexs_lc(fits_path)  # Same FITS structure


def scan_and_load_hel1os(root_dir: str, detector: str = 'CdTe',
                          max_days: int = None) -> pd.DataFrame:
    """Scan and load HEL1OS data. CdTe covers 10-40 keV, CZT covers 20-150 keV."""
    return scan_and_load_solexs(root_dir, detector=detector, max_days=max_days)


# ---------------------------------------------------------------------------
# CLI interface for quick testing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python loader_fits.py <path_to_solexs_data_dir>")
        print("  Scans for .lc.gz files and prints a summary.")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    df = scan_and_load_solexs(data_dir)
    print(f"\n--- Summary ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rate stats:\n{df['rate'].describe()}")
