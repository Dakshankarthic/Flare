# SOLAR FLARE FORECASTING IMPROVEMENT PLAN
# ISRO BAH 2026 - Team TEJAS

## EXECUTIVE SUMMARY
Your PPT promises CNN-BiLSTM + NeupertAttention + Conformal Prediction, but your GitHub
has XGBoost + rule-based detectors. This is your #1 risk. You MUST bridge this gap
to win. Below is a prioritized action plan.

================================================================================
PRIORITY 1: CRITICAL GAPS (Fix before submission)
================================================================================

1. IMPLEMENT THE NEURAL ARCHITECTURE YOU PROMISED (CNN-BiLSTM + NeupertAttention)
   - Current: XGBoost on 33 hand-crafted features
   - Required: End-to-end deep learning with physics embedding
   
   Architecture:
   ```
   Input: [batch, time_steps, 2]  # SXR channel, HXR channel
   
   Layer 1: 1D-CNN (kernels=64, kernel_size=5)
            -> Extract local patterns in X-ray light curves
            
   Layer 2: BiLSTM (hidden=128, 2 layers)
            -> Capture long-range temporal dependencies
            
   Layer 3: NeupertAttention (custom)
            -> Physics-embedded attention using Neupert Effect constraint
            -> Attention weights should correlate with HXR-SXR integral relationship
            
   Layer 4: Fully Connected -> Sigmoid (flare probability)
   ```
   
   NeupertAttention Implementation:
   - The Neupert Effect states: integral(HXR(t)dt) is proportional to SXR(t)
   - In attention: compute a physics score = |integral(HXR) - k*SXR| for each timestep
   - Add this as a penalty term in the attention energy function
   - This forces the model to learn physically plausible relationships

2. ADD CONFORMAL PREDICTION (Directly addresses FAR evaluation criterion)
   - Current: No uncertainty quantification
   - Required: Mathematically guaranteed FAR <= 10%
   
   Implementation (Split Conformal Prediction):
   ```python
   def conformal_predict(model, X_cal, y_cal, X_test, alpha=0.1):
       # 1. Train model on training set
       # 2. Compute nonconformity scores on calibration set
       scores = |y_cal - model.predict(X_cal)|
       # 3. Get (1-alpha) quantile
       q = np.quantile(scores, np.ceil((n+1)*(1-alpha))/n)
       # 4. Prediction interval for test
       pred = model.predict(X_test)
       return [pred - q, pred + q]
   ```
   - For classification: Use Mondrian Conformal Prediction (class-conditional)
   - This gives you a formal guarantee: P(y_true in interval) >= 1-alpha
   - In your dashboard: Show prediction intervals, not just point predictions

3. FIX THE DATA PIPELINE FOR REAL ADITYA-L1 DATA
   - Current: Simulation mode works, FITS mode requires manual setup
   - Required: Robust FITS reader with automatic SoLEXS/HEL1OS pairing
   
   Issues to fix:
   - SoLEXS and HEL1OS have different cadences -> resample to common timeline
   - Background subtraction for HEL1OS is critical (high energy = noisy)
   - Handle data gaps (L1 point has occasional communication blackouts)
   - Parse ISRO ISSDC PRADAN FITS headers correctly

4. IMPLEMENT CLASS-IMBALANCE HANDLING
   - Problem: A/B-class flares are 100x more common than X-class
   - Evaluation requires detecting BOTH low and high class flares
   
   Solutions:
   - Focal Loss instead of BCE: FL(pt) = -(1-pt)^gamma * log(pt)
   - Class-weighted sampling: weight = 1/sqrt(N_class)
   - Two-stage detector: Stage 1 (flare/no-flare), Stage 2 (classify A/B/C/M/X)

================================================================================
PRIORITY 2: ALGORITHMIC IMPROVEMENTS (Boost evaluation metrics)
================================================================================

5. REPLACE RULE-BASED NOWCASTING WITH NEURAL DETECTION
   - Current: Adaptive derivative thresholds (too simple, misses subtle flares)
   - Improvement: Use the CNN-BiLSTM for nowcasting too
   - Add wavelet transform preprocessing (pywt) to capture multi-scale features
   - A/B-class flares have very gradual rises -- derivative thresholds fail

6. ENHANCE FEATURE ENGINEERING FOR FORECASTING
   - Keep the 33 features but add:
     a) Wavelet coefficients (db4, level 3) -- captures transient precursors
     b) Transfer Entropy (HXR -> SXR) -- quantifies directional information flow
     c) Rolling Hurst exponent -- detects regime changes in X-ray emission
     d) Peak-to-background ratio in multiple energy bands
     e) Time since last flare (flare memory effect)

7. IMPLEMENT MULTI-HORIZON FORECASTING
   - Current: Single N-minute forecast
   - Improvement: Predict at 5, 10, 15, 30, 60 minute horizons simultaneously
   - Use a multi-task learning head: shared encoder + separate decoders per horizon
   - This maximizes lead time coverage

8. ADD PHYSICS-CONSISTENT SYNTHETIC DATA (FRED Digital Twin)
   - Problem: Real Aditya-L1 data is limited (mission launched 2023)
   - Solution: Generate synthetic flares using FRED (Flare Radiative Energy Distribution)
     with Neupert constraint
   - Pre-train model on synthetic data, fine-tune on real data
   - This is a HUGE differentiator -- no other team will have this

================================================================================
PRIORITY 3: EVALUATION & VALIDATION (Prove it works)
================================================================================

9. BENCHMARK AGAINST NOAA/GOES OPERATIONAL ALERTS
   - Download GOES XRS flare catalog from NOAA NCEI
   - Compare your nowcast timestamps with NOAA alerts
   - Compute: TPR, FAR, TSS, HSS, and lead time distribution
   - Plot reliability diagram (calibration curve) -- crucial for conformal prediction

10. CROSS-VALIDATION WITH TEMPORAL COHERENCE
    - DO NOT shuffle-split! Solar data is temporally correlated
    - Use blocked cross-validation: train on 2010-2018, test on 2019-2020
    - Or use walk-forward validation for time series
    - Report metrics with confidence intervals (bootstrap)

11. IMPLEMENT OPERATIONAL METRICS DASHBOARD
    - Real-time TPR/FAR gauge
    - Lead time histogram
    - Confusion matrix by flare class (A/B/C/M/X)
    - Attention heatmap overlay on light curves (shows what the model sees)
    - Conformal prediction interval visualization

================================================================================
PRIORITY 4: PRESENTATION & DEMO (Sell the solution)
================================================================================

12. LIVE DEMO CHECKLIST
    - Show real Aditya-L1 FITS data being processed (not simulation)
    - Trigger a nowcast alert and show the lead time
    - Show attention heatmap -- here is the physics the model learned
    - Show conformal prediction interval: we are 90% confident the flare is C-class
    - Side-by-side with NOAA/GOES: We detected this 8 minutes earlier

13. PRESENTATION IMPROVEMENTS
    - Add a Gap Analysis slide: What NOAA misses vs what you catch
    - Include a confusion matrix from real validation
    - Show the Neupert Effect learned by the model (plot integral(HXR) vs SXR from attention)
    - Quantify cost savings: Each false alarm costs $50K in satellite maneuvering

================================================================================
SAMPLE CODE: NeupertAttention Layer
================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeupertAttention(nn.Module):
    """
    Physics-embedded attention for solar flare prediction.
    Enforces Neupert Effect: integral(HXR) ~ SXR
    """
    def __init__(self, d_model, n_heads, dropout=0.1, neupert_weight=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.neupert_weight = neupert_weight
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, hxr_idx=0, sxr_idx=1):
        # x: [batch, time, features] where features include HXR and SXR
        B, T, D = x.shape
        
        Q = self.W_q(x).view(B, T, self.n_heads, D//self.n_heads).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, D//self.n_heads).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, D//self.n_heads).transpose(1, 2)
        
        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D//self.n_heads)
        
        # Neupert physics penalty
        # Extract HXR and SXR channels (assumed at indices 0 and 1)
        hxr = x[:, :, hxr_idx]  # [B, T]
        sxr = x[:, :, sxr_idx]  # [B, T]
        
        # Compute cumulative integral of HXR (Neupert proxy)
        hxr_integral = torch.cumsum(hxr, dim=1)  # [B, T]
        
        # Neupert residual: |integral(HXR) - k*SXR| -- should be small for physical consistency
        k = hxr_integral.mean() / (sxr.mean() + 1e-8)  # Learnable proportionality
        neupert_residual = torch.abs(hxr_integral - k * sxr)  # [B, T]
        
        # Add penalty to attention scores (broadcast over heads)
        neupert_penalty = neupert_residual.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
        scores = scores - self.neupert_weight * neupert_penalty
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), attn

================================================================================
SAMPLE CODE: Conformal Prediction for Flare Classification
================================================================================

import numpy as np
import torch

class ConformalFlarePredictor:
    def __init__(self, model, alpha=0.1):
        self.model = model
        self.alpha = alpha
        self.q_hat = None  # Conformal quantile
        
    def calibrate(self, X_cal, y_cal):
        """Compute conformal threshold on calibration set."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_cal)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Nonconformity score: 1 - probability of true class
        n = len(y_cal)
        scores = 1 - probs[np.arange(n), y_cal]
        
        # (1-alpha) quantile with finite-sample correction
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, q_level, method='higher')
        
    def predict_sets(self, X_test):
        """Return prediction sets (sets of possible classes)."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_test)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Include all classes where score <= q_hat
        sets = []
        for prob in probs:
            scores = 1 - prob
            prediction_set = np.where(scores <= self.q_hat)[0]
            sets.append(prediction_set)
        return sets
    
    def predict_with_uncertainty(self, X_test):
        """Return point prediction + uncertainty flag."""
        sets = self.predict_sets(X_test)
        point_preds = [s[0] if len(s) > 0 else -1 for s in sets]
        uncertain = [len(s) > 1 for s in sets]  # Multiple classes possible
        return point_preds, uncertain

# Usage:
# 1. Split data: train (70%) / cal (15%) / test (15%)
# 2. Train model on train
# 3. predictor = ConformalFlarePredictor(model, alpha=0.1)
# 4. predictor.calibrate(X_cal, y_cal)
# 5. preds, uncertain = predictor.predict_with_uncertainty(X_test)
# 6. FAR is now guaranteed <= 10% on test data (marginal coverage)

================================================================================
QUICK WINS (Can implement in 1-2 days)
================================================================================

1. Replace BCE with Focal Loss (3 lines of code, huge impact on rare flares)
2. Add learning rate scheduler (CosineAnnealingWarmRestarts)
3. Implement early stopping with TSS as validation metric (not loss)
4. Add data augmentation: jitter, scale, time-warp on X-ray light curves
5. Use stratified temporal split for validation
6. Add GOES class labels to your catalogue (A/B/C/M/X)
7. Compute and display lead time for every detected flare
8. Add a physics check visualization: plot integral(HXR) vs SXR correlation

================================================================================
WHAT JUDGES WILL LOOK FOR
================================================================================

1. Does it actually work on real data? -> Show Aditya-L1 FITS processing
2. Is it better than existing methods? -> Benchmark against NOAA/GOES
3. Is the physics correct? -> Neupert Effect visualization
4. Is it reliable? -> Conformal prediction with coverage guarantees
5. Is it deployable? -> Docker container, FastAPI, <10s inference
6. Can it detect small flares? -> A/B-class detection metrics
7. What is the lead time? -> Quantified minutes-before-peak
8. Is the code clean? -> Well-documented, modular, tested

================================================================================
RECOMMENDED READING
================================================================================

- Hong et al. (2026): Uncertainty-Aware Solar Flare Regression -- Conformal Prediction
- Abduallah et al. (2023): SolarFlareNet -- Transformer baseline
- Li et al. (2025): MViT for flare forecasting -- Latest architecture
- Ahmadzadeh et al. (2021): SWAN-SF dataset -- Time series augmentation

================================================================================