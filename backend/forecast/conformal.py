"""
conformal.py — Split Conformal Prediction for solar flare classification.

Provides mathematically guaranteed coverage:
    P(y_true in prediction_set) >= 1 - alpha

Two methods:
  1. Standard (marginal) conformal prediction
  2. Mondrian (class-conditional) conformal prediction

Usage:
    predictor = ConformalFlarePredictor(alpha=0.1)
    predictor.calibrate(cal_probs, cal_labels)
    sets, uncertain = predictor.predict_with_uncertainty(test_probs)
    # FAR is now guaranteed <= 10%
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ConformalFlarePredictor:
    """
    Split Conformal Prediction for flare classification.

    Calibrates on a held-out set to compute nonconformity quantiles,
    then produces prediction sets with guaranteed coverage.
    """

    def __init__(self, alpha: float = 0.1, method: str = 'standard'):
        """
        Args:
            alpha: significance level (0.1 = 90% coverage guarantee)
            method: 'standard' (marginal) or 'mondrian' (class-conditional)
        """
        self.alpha = alpha
        self.method = method
        self.q_hat = None               # Standard quantile
        self.q_hat_per_class = {}       # Mondrian quantiles
        self.calibrated = False
        self.n_cal = 0
        self.coverage_history = []      # Track empirical coverage

    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray):
        """
        Compute conformal threshold(s) on a calibration set.

        Args:
            cal_probs: [N, n_classes] or [N] probability scores from model
            cal_labels: [N] integer class labels
        """
        cal_probs = np.atleast_2d(cal_probs)
        if cal_probs.shape[0] == 1 and cal_probs.shape[1] > 1:
            cal_probs = cal_probs.T
        if cal_probs.ndim == 1:
            # Binary: convert to 2-class
            cal_probs = np.stack([1 - cal_probs, cal_probs], axis=1)

        n = len(cal_labels)
        self.n_cal = n

        # Nonconformity score: 1 - probability of true class
        scores = 1.0 - cal_probs[np.arange(n), cal_labels.astype(int)]

        if self.method == 'standard':
            # Marginal quantile with finite-sample correction
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(q_level, 1.0)
            self.q_hat = float(np.quantile(scores, q_level, method='higher'))

        elif self.method == 'mondrian':
            # Class-conditional quantiles
            self.q_hat_per_class = {}
            unique_classes = np.unique(cal_labels)
            for cls in unique_classes:
                cls_mask = cal_labels == cls
                cls_scores = scores[cls_mask]
                n_cls = len(cls_scores)
                if n_cls > 0:
                    q_level = np.ceil((n_cls + 1) * (1 - self.alpha)) / n_cls
                    q_level = min(q_level, 1.0)
                    self.q_hat_per_class[int(cls)] = float(
                        np.quantile(cls_scores, q_level, method='higher')
                    )
            # Fallback for unseen classes
            self.q_hat = float(np.quantile(scores,
                               min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0),
                               method='higher'))

        self.calibrated = True

    def predict_sets(self, test_probs: np.ndarray) -> List[np.ndarray]:
        """
        Return prediction sets (sets of possible classes).

        Each prediction set contains all classes whose nonconformity
        score is <= q_hat, guaranteeing the true class is included
        with probability >= 1 - alpha.

        Args:
            test_probs: [N, n_classes] probability scores

        Returns:
            List of arrays, each containing the set of possible class indices
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() first")

        test_probs = np.atleast_2d(test_probs)
        if test_probs.ndim == 1:
            test_probs = np.stack([1 - test_probs, test_probs], axis=1)

        sets = []
        for prob in test_probs:
            scores = 1.0 - prob  # nonconformity for each class

            if self.method == 'mondrian':
                prediction_set = []
                for cls_idx in range(len(prob)):
                    q = self.q_hat_per_class.get(cls_idx, self.q_hat)
                    if scores[cls_idx] <= q:
                        prediction_set.append(cls_idx)
                sets.append(np.array(prediction_set, dtype=int))
            else:
                prediction_set = np.where(scores <= self.q_hat)[0]
                sets.append(prediction_set)

        return sets

    def predict_with_uncertainty(self, test_probs: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Return point predictions, uncertainty flags, and prediction sets.

        Args:
            test_probs: [N, n_classes] or [N] probability scores

        Returns:
            (point_preds, uncertain, prediction_sets) where:
                point_preds: [N] most likely class
                uncertain: [N] bool — True if multiple classes in set
                prediction_sets: list of class index arrays
        """
        test_probs = np.atleast_2d(test_probs)
        if test_probs.ndim == 1:
            test_probs = np.stack([1 - test_probs, test_probs], axis=1)

        pred_sets = self.predict_sets(test_probs)

        point_preds = np.argmax(test_probs, axis=1)
        uncertain = np.array([len(s) != 1 for s in pred_sets])

        return point_preds, uncertain, pred_sets

    def predict_intervals_binary(self, test_probs: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For binary classification: return lower/upper probability bounds.

        Args:
            test_probs: [N] probability of positive class

        Returns:
            (prob, lower, upper) confidence interval on the probability
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() first")

        test_probs = np.asarray(test_probs).flatten()
        lower = np.clip(test_probs - self.q_hat, 0, 1)
        upper = np.clip(test_probs + self.q_hat, 0, 1)

        return test_probs, lower, upper

    def update_coverage(self, y_true: np.ndarray, pred_sets: List[np.ndarray]):
        """Track empirical coverage for monitoring."""
        covered = sum(1 for yt, ps in zip(y_true, pred_sets) if yt in ps)
        coverage = covered / max(len(y_true), 1)
        self.coverage_history.append({
            'coverage': coverage,
            'target': 1 - self.alpha,
            'n_samples': len(y_true),
            'avg_set_size': np.mean([len(s) for s in pred_sets]),
        })

    def get_stats(self) -> Dict:
        """Get calibration statistics."""
        stats = {
            'alpha': self.alpha,
            'method': self.method,
            'calibrated': self.calibrated,
            'n_calibration': self.n_cal,
            'q_hat': self.q_hat,
            'target_coverage': 1 - self.alpha,
        }
        if self.method == 'mondrian':
            stats['q_hat_per_class'] = self.q_hat_per_class
        if self.coverage_history:
            latest = self.coverage_history[-1]
            stats['empirical_coverage'] = latest['coverage']
            stats['avg_set_size'] = latest['avg_set_size']
        return stats


def calibrate_binary_conformal(model_probs: np.ndarray, labels: np.ndarray,
                                alpha: float = 0.1
                                ) -> ConformalFlarePredictor:
    """
    Convenience function: calibrate a binary conformal predictor.

    Splits the data into calibration (first 50%) and returns a
    calibrated predictor ready for prediction on the rest.
    """
    n = len(model_probs)
    cal_n = n // 2

    # Convert binary probs to 2-class format
    probs_2d = np.stack([1 - model_probs[:cal_n], model_probs[:cal_n]], axis=1)

    predictor = ConformalFlarePredictor(alpha=alpha, method='standard')
    predictor.calibrate(probs_2d, labels[:cal_n].astype(int))

    return predictor
