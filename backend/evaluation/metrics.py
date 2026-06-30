"""
metrics.py — Evaluation metrics for flare detection and forecasting.

Metrics:
  - True Positive Rate (TPR / Recall)
  - False Alarm Rate (FAR)
  - True Skill Statistic (TSS)
  - Heidke Skill Score (HSS)
  - Mean Lead Time
  - Brier Skill Score
  - Per-class breakdown
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_fscore_support,
    brier_score_loss, classification_report
)


def compute_nowcast_metrics(detected_events: list, ground_truth_events: list,
                             tolerance_sec: float = 60.0) -> dict:
    """
    Evaluate nowcasting (detection) performance.
    
    A detection is a True Positive if its peak time is within tolerance_sec
    of a ground truth flare peak.
    
    Args:
        detected_events: List of detected flare events (with peak_time attribute)
        ground_truth_events: List of ground truth flares (with peak_time attribute)
        tolerance_sec: Matching tolerance in seconds
    
    Returns:
        dict with TP, FP, FN, TPR, FAR, precision, F1
    """
    gt_matched = set()
    tp = 0
    fp = 0
    
    for det in detected_events:
        det_peak = getattr(det, 'peak_time', det.get('peak_time', 0) if isinstance(det, dict) else 0)
        
        matched = False
        for i, gt in enumerate(ground_truth_events):
            if i in gt_matched:
                continue
            gt_peak = getattr(gt, 'peak_time', gt.get('peak_time', 0) if isinstance(gt, dict) else 0)
            
            if abs(det_peak - gt_peak) <= tolerance_sec:
                tp += 1
                gt_matched.add(i)
                matched = True
                break
        
        if not matched:
            fp += 1
    
    fn = len(ground_truth_events) - len(gt_matched)
    
    tpr = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * tpr / max(precision + tpr, 1e-10)
    far = fp / max(fp + tp, 1)
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tpr': tpr,           # Recall / True Positive Rate
        'precision': precision,
        'f1': f1,
        'far': far,            # False Alarm Ratio
        'n_detected': len(detected_events),
        'n_truth': len(ground_truth_events),
    }


def compute_forecast_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                               threshold: float = 0.5) -> dict:
    """
    Evaluate forecasting model performance.
    
    Args:
        y_true: Binary ground truth labels
        y_prob: Predicted probabilities
        threshold: Decision threshold
    
    Returns:
        dict with comprehensive metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion matrix
    if len(np.unique(y_true)) < 2:
        # All same class — limited metrics
        return {
            'auc': 0.5, 'tpr': 0.0, 'far': 0.0, 'tss': 0.0,
            'hss': 0.0, 'brier': 1.0, 'f1': 0.0,
            'note': 'Single class in ground truth'
        }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    far = fp / max(fp + tp, 1)
    
    # True Skill Statistic (TSS = TPR - FPR)
    tss = tpr - fpr
    
    # Heidke Skill Score
    n = tp + tn + fp + fn
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / max(n, 1)
    hss = (tp + tn - expected) / max(n - expected, 1e-10)
    
    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    
    # Brier Score
    brier = brier_score_loss(y_true, y_prob)
    
    # Climatological Brier score (baseline: predict mean frequency)
    climatology = y_true.mean()
    brier_clim = brier_score_loss(y_true, np.full_like(y_prob, climatology))
    brier_skill = 1 - brier / max(brier_clim, 1e-10)
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    return {
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'tpr': tpr,
        'fpr': fpr,
        'far': far,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tss': tss,
        'hss': hss,
        'auc': auc,
        'brier': brier,
        'brier_skill': brier_skill,
        'threshold': threshold,
    }


def compute_lead_time(forecast_alerts: list, ground_truth_events: list,
                       tolerance_sec: float = 600) -> dict:
    """
    Compute the lead time of forecast alerts.
    
    Lead time = time between forecast alert and actual flare peak.
    
    Args:
        forecast_alerts: List of dicts with {'time': alert_time, 'prob': probability}
        ground_truth_events: List of flare events with peak_time
        tolerance_sec: Maximum time before flare to consider as valid lead
    
    Returns:
        dict with mean, median, min, max lead times in seconds
    """
    lead_times = []
    
    for gt in ground_truth_events:
        gt_peak = getattr(gt, 'peak_time', gt.get('peak_time', 0) if isinstance(gt, dict) else 0)
        
        # Find the earliest alert within tolerance window before this flare
        earliest_alert = None
        for alert in forecast_alerts:
            alert_time = alert.get('time', 0)
            lead = gt_peak - alert_time
            
            if 0 < lead <= tolerance_sec:
                if earliest_alert is None or alert_time < earliest_alert:
                    earliest_alert = alert_time
        
        if earliest_alert is not None:
            lead_times.append(gt_peak - earliest_alert)
    
    if not lead_times:
        return {
            'mean_lead_sec': 0, 'median_lead_sec': 0,
            'min_lead_sec': 0, 'max_lead_sec': 0,
            'n_forecasted': 0,
        }
    
    lead_arr = np.array(lead_times)
    return {
        'mean_lead_sec': float(np.mean(lead_arr)),
        'median_lead_sec': float(np.median(lead_arr)),
        'min_lead_sec': float(np.min(lead_arr)),
        'max_lead_sec': float(np.max(lead_arr)),
        'mean_lead_min': float(np.mean(lead_arr) / 60),
        'n_forecasted': len(lead_times),
    }


def format_metrics_report(nowcast_metrics: dict, forecast_metrics: dict,
                            lead_metrics: dict) -> str:
    """Format a human-readable metrics report."""
    lines = [
        "=" * 60,
        "  SOLAR FLARE PIPELINE — EVALUATION REPORT",
        "=" * 60,
        "",
        "--- NOWCASTING (Detection) ---",
        f"  True Positives:   {nowcast_metrics.get('tp', 0)}",
        f"  False Positives:  {nowcast_metrics.get('fp', 0)}",
        f"  Missed (FN):      {nowcast_metrics.get('fn', 0)}",
        f"  TPR (Recall):     {nowcast_metrics.get('tpr', 0):.3f}",
        f"  Precision:        {nowcast_metrics.get('precision', 0):.3f}",
        f"  F1 Score:         {nowcast_metrics.get('f1', 0):.3f}",
        f"  False Alarm Rate: {nowcast_metrics.get('far', 0):.3f}",
        "",
        "--- FORECASTING (Prediction) ---",
        f"  AUC-ROC:          {forecast_metrics.get('auc', 0):.3f}",
        f"  TSS:              {forecast_metrics.get('tss', 0):.3f}",
        f"  HSS:              {forecast_metrics.get('hss', 0):.3f}",
        f"  Brier Skill:      {forecast_metrics.get('brier_skill', 0):.3f}",
        f"  F1 Score:         {forecast_metrics.get('f1', 0):.3f}",
        f"  TPR:              {forecast_metrics.get('tpr', 0):.3f}",
        f"  FAR:              {forecast_metrics.get('far', 0):.3f}",
        "",
        "--- LEAD TIME ---",
        f"  Mean:             {lead_metrics.get('mean_lead_min', 0):.1f} min",
        f"  Median:           {lead_metrics.get('median_lead_sec', 0)/60:.1f} min",
        f"  Min:              {lead_metrics.get('min_lead_sec', 0)/60:.1f} min",
        f"  Max:              {lead_metrics.get('max_lead_sec', 0)/60:.1f} min",
        f"  Flares forecast:  {lead_metrics.get('n_forecasted', 0)}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-class confusion matrix (A/B/C/M/X)
# ---------------------------------------------------------------------------

GOES_CLASSES = ['A', 'B', 'C', 'M', 'X']


def compute_per_class_confusion(detected_events: list, ground_truth_events: list,
                                 tolerance_sec: float = 60.0) -> dict:
    """
    Compute a per-class confusion matrix for flare detection.

    Returns:
        dict with:
            'matrix': {true_class: {pred_class: count}}
            'per_class_tpr': {class: tpr}
            'per_class_far': {class: far}
            'total': confusion matrix as 2D list
    """
    # Build class-level matching
    class_tp = {c: 0 for c in GOES_CLASSES}
    class_fp = {c: 0 for c in GOES_CLASSES}
    class_fn = {c: 0 for c in GOES_CLASSES}

    gt_matched = set()

    for det in detected_events:
        det_peak = getattr(det, 'peak_time', det.get('peak_time', 0) if isinstance(det, dict) else 0)
        det_class = getattr(det, 'flare_class', det.get('flare_class', '?') if isinstance(det, dict) else '?')

        matched = False
        for i, gt in enumerate(ground_truth_events):
            if i in gt_matched:
                continue
            gt_peak = getattr(gt, 'peak_time', gt.get('peak_time', 0) if isinstance(gt, dict) else 0)
            if abs(det_peak - gt_peak) <= tolerance_sec:
                gt_class = getattr(gt, 'flare_class', gt.get('flare_class', '?') if isinstance(gt, dict) else '?')
                if det_class == gt_class:
                    class_tp[det_class] = class_tp.get(det_class, 0) + 1
                else:
                    class_fp[det_class] = class_fp.get(det_class, 0) + 1
                    class_fn[gt_class] = class_fn.get(gt_class, 0) + 1
                gt_matched.add(i)
                matched = True
                break
        if not matched:
            class_fp[det_class] = class_fp.get(det_class, 0) + 1

    for i, gt in enumerate(ground_truth_events):
        if i not in gt_matched:
            gt_class = getattr(gt, 'flare_class', gt.get('flare_class', '?') if isinstance(gt, dict) else '?')
            class_fn[gt_class] = class_fn.get(gt_class, 0) + 1

    per_class_tpr = {}
    per_class_far = {}
    for c in GOES_CLASSES:
        tp_c = class_tp.get(c, 0)
        fp_c = class_fp.get(c, 0)
        fn_c = class_fn.get(c, 0)
        per_class_tpr[c] = tp_c / max(tp_c + fn_c, 1)
        per_class_far[c] = fp_c / max(fp_c + tp_c, 1)

    return {
        'class_tp': class_tp,
        'class_fp': class_fp,
        'class_fn': class_fn,
        'per_class_tpr': per_class_tpr,
        'per_class_far': per_class_far,
        'classes': GOES_CLASSES,
    }


# ---------------------------------------------------------------------------
# Reliability Diagram (calibration curve)
# ---------------------------------------------------------------------------

def compute_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray,
                                 n_bins: int = 10) -> dict:
    """
    Compute reliability/calibration curve data.

    A well-calibrated model has: predicted probability ≈ observed frequency.

    Returns:
        dict with 'bin_centers', 'observed_freq', 'bin_counts', 'ece'
        (Expected Calibration Error)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    observed_freq = np.zeros(n_bins)
    avg_predicted = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge for last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            observed_freq[i] = y_true[mask].mean()
            avg_predicted[i] = y_prob[mask].mean()

    # Expected Calibration Error
    total = max(y_prob.shape[0], 1)
    ece = np.sum(bin_counts * np.abs(observed_freq - avg_predicted)) / total

    return {
        'bin_centers': bin_centers.tolist(),
        'observed_freq': observed_freq.tolist(),
        'avg_predicted': avg_predicted.tolist(),
        'bin_counts': bin_counts.astype(int).tolist(),
        'ece': float(ece),
        'n_bins': n_bins,
    }


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_metric(y_true: np.ndarray, y_prob: np.ndarray,
                      metric_fn, n_bootstrap: int = 200,
                      ci: float = 0.95, seed: int = 42) -> dict:
    """
    Compute bootstrap confidence interval for any metric function.

    Args:
        y_true: ground truth labels
        y_prob: predicted probabilities or labels
        metric_fn: function(y_true, y_prob) -> float
        n_bootstrap: number of bootstrap iterations
        ci: confidence level (0.95 = 95% CI)

    Returns:
        dict with 'mean', 'lower', 'upper', 'std'
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            if np.isfinite(score):
                scores.append(score)
        except (ValueError, ZeroDivisionError):
            continue

    if not scores:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0, 'std': 0.0}

    scores = np.array(scores)
    alpha = 1 - ci
    return {
        'mean': float(np.mean(scores)),
        'lower': float(np.percentile(scores, 100 * alpha / 2)),
        'upper': float(np.percentile(scores, 100 * (1 - alpha / 2))),
        'std': float(np.std(scores)),
    }


def compute_all_bootstrap_cis(y_true: np.ndarray, y_prob: np.ndarray,
                                threshold: float = 0.5) -> dict:
    """Compute bootstrap CIs for all major forecast metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    def _tss(yt, yp):
        yp_bin = (yp >= threshold).astype(int) if yp.max() <= 1.0 else yp
        tp = ((yp_bin == 1) & (yt == 1)).sum()
        tn = ((yp_bin == 0) & (yt == 0)).sum()
        fp = ((yp_bin == 1) & (yt == 0)).sum()
        fn = ((yp_bin == 0) & (yt == 1)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return tpr - fpr

    def _auc(yt, yp):
        if len(np.unique(yt)) < 2:
            return 0.5
        return roc_auc_score(yt, yp)

    return {
        'auc_ci': bootstrap_metric(y_true, y_prob, _auc),
        'tss_ci': bootstrap_metric(y_true, y_prob, _tss),
    }
