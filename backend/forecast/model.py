"""
model.py — Forecasting models for solar flare prediction.

Two models:
  1. XGBoost (primary) — gradient-boosted trees on engineered features
  2. Simple LSTM baseline — raw time-series input (optional, requires PyTorch)

Both output: P(flare in next N minutes) ∈ [0, 1]
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    average_precision_score, brier_score_loss
)
import xgboost as xgb

from backend.forecast.features import FEATURE_COLUMNS


class XGBoostForecaster:
    """
    XGBoost-based solar flare forecaster.
    
    Trained on sliding-window features to predict the probability
    of a flare occurring in the next N minutes.
    """
    
    def __init__(self, horizon_sec: int = 300):
        self.horizon_sec = horizon_sec
        self.model = None
        self.feature_names = FEATURE_COLUMNS
        self.best_threshold = 0.5
        self.metrics = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              n_splits: int = 5, verbose: bool = True) -> dict:
        """
        Train the model with time-series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Binary labels (1 = flare within horizon)
            n_splits: Number of CV folds
            verbose: Print progress
        
        Returns:
            dict of evaluation metrics
        """
        # Ensure we only use defined feature columns
        available_features = [f for f in self.feature_names if f in X.columns]
        X_train = X[available_features].values
        y_train = y.values
        
        # Class imbalance ratio
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)
        
        if verbose:
            print(f"Training set: {len(y_train)} samples, "
                  f"{pos_count} positive ({100*pos_count/len(y_train):.1f}%)")
            print(f"Scale pos weight: {scale_pos_weight:.1f}")
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 8,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'min_child_weight': 5,
            'gamma': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'seed': 42,
        }
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            dtrain = xgb.DMatrix(X_train[train_idx], label=y_train[train_idx],
                                  feature_names=available_features)
            dval = xgb.DMatrix(X_train[val_idx], label=y_train[val_idx],
                                feature_names=available_features)
            
            model = xgb.train(
                params, dtrain,
                num_boost_round=300,
                evals=[(dval, 'val')],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            
            prob = model.predict(dval)
            
            try:
                auc = roc_auc_score(y_train[val_idx], prob)
            except ValueError:
                auc = 0.5
            
            try:
                ap = average_precision_score(y_train[val_idx], prob)
            except ValueError:
                ap = 0.0
            
            pred = (prob > 0.5).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_train[val_idx], pred, average='binary', zero_division=0
            )
            
            fold_metrics = {
                'fold': fold, 'auc': auc, 'ap': ap,
                'precision': prec, 'recall': rec, 'f1': f1,
                'n_val': len(val_idx), 'n_pos': y_train[val_idx].sum()
            }
            cv_metrics.append(fold_metrics)
            
            if verbose:
                print(f"  Fold {fold}: AUC={auc:.3f} AP={ap:.3f} "
                      f"F1={f1:.3f} Recall={rec:.3f}")
        
        # Final model on all data
        dtrain_all = xgb.DMatrix(X_train, label=y_train,
                                  feature_names=available_features)
        self.model = xgb.train(
            params, dtrain_all,
            num_boost_round=200,
            verbose_eval=False
        )
        
        # Find optimal threshold
        prob_all = self.model.predict(dtrain_all)
        self.best_threshold = _find_optimal_threshold(y_train, prob_all)
        
        # Summary metrics (filter out NaN values from single-class folds)
        valid_aucs = [m['auc'] for m in cv_metrics if np.isfinite(m['auc'])]
        valid_f1s = [m['f1'] for m in cv_metrics if np.isfinite(m['f1'])]
        valid_recalls = [m['recall'] for m in cv_metrics if np.isfinite(m['recall'])]
        
        self.metrics = {
            'cv_auc_mean': np.mean(valid_aucs) if valid_aucs else 0.5,
            'cv_auc_std': np.std(valid_aucs) if valid_aucs else 0.0,
            'cv_f1_mean': np.mean(valid_f1s) if valid_f1s else 0.0,
            'cv_recall_mean': np.mean(valid_recalls) if valid_recalls else 0.0,
            'best_threshold': self.best_threshold,
            'n_features': len(available_features),
            'horizon_sec': self.horizon_sec,
        }
        
        if verbose:
            print(f"\n  Mean AUC: {self.metrics['cv_auc_mean']:.3f} "
                  f"± {self.metrics['cv_auc_std']:.3f}")
            print(f"  Best threshold: {self.best_threshold:.3f}")
        
        return self.metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict flare probability for each sample."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        available = [f for f in self.feature_names if f in X.columns]
        dmat = xgb.DMatrix(X[available].values, feature_names=available)
        return self.model.predict(dmat)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary flare/no-flare using optimal threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.best_threshold).astype(int)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model is None:
            return pd.DataFrame()
        
        importance = self.model.get_score(importance_type='gain')
        return pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'best_threshold': self.best_threshold,
            'metrics': self.metrics,
            'horizon_sec': self.horizon_sec,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'XGBoostForecaster':
        """Load model from disk."""
        data = joblib.load(path)
        forecaster = cls(horizon_sec=data['horizon_sec'])
        forecaster.model = data['model']
        forecaster.feature_names = data['feature_names']
        forecaster.best_threshold = data['best_threshold']
        forecaster.metrics = data['metrics']
        return forecaster


def _find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find the threshold that maximizes the F1 score."""
    best_f1 = 0
    best_thr = 0.5
    
    for thr in np.arange(0.1, 0.9, 0.05):
        pred = (y_prob >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, pred, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    
    return best_thr
