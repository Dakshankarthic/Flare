"""
neural_model.py — CNN-BiLSTM + NeupertAttention for solar flare forecasting.

Architecture:
  Input: [batch, time_steps, 2]  (SXR + HXR channels)
  Layer 1: 1D-CNN (extract local patterns in X-ray light curves)
  Layer 2: BiLSTM (capture long-range temporal dependencies)
  Layer 3: NeupertAttention (physics-embedded attention using Neupert Effect)
  Layer 4: Multi-horizon FC heads → Sigmoid (flare probability at 5/10/15/30/60 min)

Also includes:
  - FocalLoss for class-imbalance handling
  - Data augmentation (jitter, scale, time-warp)
  - CosineAnnealingWarmRestarts LR scheduler
  - Early stopping on TSS metric
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# NeupertAttention — physics-embedded attention
# ---------------------------------------------------------------------------

class NeupertAttention(nn.Module):
    """
    Physics-embedded attention for solar flare prediction.
    Enforces Neupert Effect: integral(HXR) ~ SXR

    The Neupert Effect states that the time integral of the hard X-ray
    flux is proportional to the instantaneous soft X-ray flux. This
    layer adds a physics penalty to the standard scaled-dot-product
    attention, biasing the model toward physically plausible relationships.
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 dropout: float = 0.1, neupert_weight: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.neupert_weight = neupert_weight

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                hxr_channel: Optional[torch.Tensor] = None,
                sxr_channel: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, time, d_model] — hidden states from BiLSTM
            hxr_channel: [batch, time] — raw hard X-ray values (for physics)
            sxr_channel: [batch, time] — raw soft X-ray values (for physics)

        Returns:
            (output, attention_weights) where:
                output: [batch, time, d_model]
                attention_weights: [batch, n_heads, time, time]
        """
        residual = x
        B, T, D = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Standard scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # --- Neupert physics penalty ---
        if hxr_channel is not None and sxr_channel is not None:
            # Cumulative integral of HXR (Neupert proxy)
            hxr_integral = torch.cumsum(hxr_channel, dim=1)  # [B, T]

            # Estimate proportionality constant k
            k = (hxr_integral.mean(dim=1, keepdim=True) /
                 (sxr_channel.mean(dim=1, keepdim=True) + 1e-8))

            # Neupert residual: should be small for physical consistency
            neupert_residual = torch.abs(hxr_integral - k * sxr_channel)  # [B, T]

            # Normalize residual to [0, 1] range
            res_max = neupert_residual.max(dim=1, keepdim=True)[0] + 1e-8
            neupert_norm = neupert_residual / res_max  # [B, T]

            # Add penalty — broadcast over heads and query dimension
            # Shape: [B, 1, 1, T] — penalizes attending to non-physical keys
            penalty = neupert_norm.unsqueeze(1).unsqueeze(2)
            scores = scores - self.neupert_weight * penalty

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        # Residual + layer norm
        out = self.layer_norm(out + residual)

        return out, attn


# ---------------------------------------------------------------------------
# Focal Loss — class imbalance handling
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in flare detection.

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    When gamma > 0, reduces the loss for well-classified examples,
    focusing training on hard/rare examples (e.g., M/X-class flares).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, ...] raw model outputs (before sigmoid)
            targets: [batch, ...] binary labels (0 or 1)
        """
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        return loss.mean()


# ---------------------------------------------------------------------------
# CNN-BiLSTM Forecaster
# ---------------------------------------------------------------------------

HORIZONS_MIN = [5, 10, 15, 30, 60]  # multi-horizon forecast windows


class CNNBiLSTMForecaster(nn.Module):
    """
    End-to-end deep learning model for solar flare forecasting.

    Architecture:
        1D-CNN → BiLSTM → NeupertAttention → Multi-horizon FC heads

    Outputs a probability for each forecast horizon (5/10/15/30/60 min).
    """

    def __init__(self, input_channels: int = 2,
                 cnn_filters: int = 64, cnn_kernel: int = 5,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.2,
                 neupert_weight: float = 0.1,
                 n_horizons: int = 5):
        super().__init__()

        self.input_channels = input_channels
        self.n_horizons = n_horizons

        # Layer 1: 1D-CNN for local pattern extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_filters, kernel_size=cnn_kernel,
                      padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_filters),
            nn.GELU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=cnn_kernel,
                      padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_filters),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Layer 2: BiLSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Layer 3: NeupertAttention
        self.attention = NeupertAttention(
            d_model=lstm_out_dim,
            n_heads=n_heads,
            dropout=dropout,
            neupert_weight=neupert_weight,
        )

        # Layer 4: Multi-horizon output heads (shared encoder, separate decoders)
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_out_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor,
                hxr: Optional[torch.Tensor] = None,
                sxr: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, time_steps, input_channels] — raw SXR + HXR
            hxr: [batch, time_steps] — hard X-ray for physics attention
            sxr: [batch, time_steps] — soft X-ray for physics attention

        Returns:
            (logits, attention_weights) where:
                logits: [batch, n_horizons] — raw scores per horizon
                attention_weights: [batch, n_heads, time, time]
        """
        B, T, C = x.shape

        # CNN expects [batch, channels, time]
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # → [B, T, cnn_filters]

        # BiLSTM
        h, _ = self.lstm(h)  # → [B, T, lstm_hidden*2]

        # Physics-embedded attention
        h, attn_weights = self.attention(h, hxr_channel=hxr, sxr_channel=sxr)

        # Global temporal pooling: mean over time
        h_pool = h.mean(dim=1)  # [B, lstm_out_dim]

        # Multi-horizon prediction
        logits = torch.cat([
            head(h_pool) for head in self.horizon_heads
        ], dim=1)  # [B, n_horizons]

        return logits, attn_weights

    def predict_proba(self, x: torch.Tensor,
                      hxr: Optional[torch.Tensor] = None,
                      sxr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return sigmoid probabilities for each horizon."""
        logits, _ = self.forward(x, hxr, sxr)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Data augmentation for X-ray light curves
# ---------------------------------------------------------------------------

def augment_lightcurve(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply random augmentations to a light curve window.

    Args:
        x: [time_steps, channels] array
        rng: numpy random generator

    Augmentations:
        - Jitter: add small Gaussian noise
        - Scale: multiply by random factor ~N(1.0, 0.1)
        - Time-warp: slight temporal distortion
    """
    aug = x.copy()

    # Jitter
    if rng.random() > 0.3:
        noise_scale = 0.02 * np.std(aug, axis=0, keepdims=True)
        aug += rng.normal(0, noise_scale + 1e-8, size=aug.shape)

    # Scale
    if rng.random() > 0.3:
        scale = rng.normal(1.0, 0.1, size=(1, aug.shape[1]))
        aug *= np.maximum(scale, 0.5)

    # Time-warp (simple linear interpolation stretch/compress)
    if rng.random() > 0.5:
        T = aug.shape[0]
        warp_factor = rng.uniform(0.9, 1.1)
        new_len = int(T * warp_factor)
        if new_len > 2:
            indices = np.linspace(0, T - 1, new_len)
            warped = np.zeros((new_len, aug.shape[1]))
            for c in range(aug.shape[1]):
                warped[:, c] = np.interp(indices, np.arange(T), aug[:, c])
            # Resize back to original length
            final_indices = np.linspace(0, new_len - 1, T)
            result = np.zeros_like(aug)
            for c in range(aug.shape[1]):
                result[:, c] = np.interp(final_indices, np.arange(new_len), warped[:, c])
            aug = result

    return aug


# ---------------------------------------------------------------------------
# Dataset for training
# ---------------------------------------------------------------------------

class FlareDataset(Dataset):
    """
    PyTorch dataset for solar flare time-series windows.

    Each sample is a (window, labels) pair where:
        window: [time_steps, 2] — SXR + HXR
        labels: [n_horizons] — binary labels per horizon
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray,
                 augment: bool = False, seed: int = 42):
        """
        Args:
            windows: [N, time_steps, 2] — stacked windows
            labels: [N, n_horizons] — multi-horizon labels
            augment: whether to apply data augmentation
        """
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = self.labels[idx]

        if self.augment:
            x = augment_lightcurve(x, self.rng)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_tss(y_true: np.ndarray, y_prob: np.ndarray,
                threshold: float = 0.5) -> float:
    """Compute True Skill Statistic = TPR - FPR."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return tpr - fpr


def prepare_windows(df: pd.DataFrame, flare_events: list,
                    window_sec: int = 300, stride_sec: int = 60,
                    horizons_min: List[int] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract time-series windows and multi-horizon labels from preprocessed data.

    Args:
        df: preprocessed DataFrame with soft_norm, hard_norm columns
        flare_events: list of flare events with onset_time
        window_sec: window size in seconds
        stride_sec: stride between windows
        horizons_min: list of forecast horizons in minutes

    Returns:
        (windows, labels) where:
            windows: [N, window_sec, 2] — SXR + HXR
            labels: [N, n_horizons] — binary labels per horizon
    """
    if horizons_min is None:
        horizons_min = HORIZONS_MIN

    soft = df['soft_norm'].values if 'soft_norm' in df.columns else np.zeros(len(df))
    hard = df['hard_norm'].values if 'hard_norm' in df.columns else np.zeros(len(df))
    times = df['time_s'].values

    windows = []
    labels = []
    n = len(df)

    # Get onset times from events
    onset_times = []
    for ev in flare_events:
        onset = getattr(ev, 'onset_time', None)
        if onset is None and isinstance(ev, dict):
            onset = ev.get('onset_time', ev.get('start_time', None))
        if onset is not None:
            onset_times.append(onset)

    for start in range(0, n - window_sec, stride_sec):
        end = start + window_sec
        if end > n:
            break

        win = np.stack([soft[start:end], hard[start:end]], axis=1)  # [T, 2]
        window_end_time = times[end - 1]

        # Multi-horizon labels
        lbl = np.zeros(len(horizons_min), dtype=np.float32)
        for h_idx, h_min in enumerate(horizons_min):
            h_sec = h_min * 60
            for onset in onset_times:
                if window_end_time < onset <= window_end_time + h_sec:
                    lbl[h_idx] = 1.0
                    break

        windows.append(win)
        labels.append(lbl)

    return np.array(windows), np.array(labels)


def train_neural_model(df: pd.DataFrame, flare_events: list,
                       window_sec: int = 300, stride_sec: int = 60,
                       epochs: int = 50, batch_size: int = 32,
                       lr: float = 1e-3, patience: int = 10,
                       device: str = 'cpu', verbose: bool = True
                       ) -> Tuple['CNNBiLSTMForecaster', Dict]:
    """
    Train the CNN-BiLSTM model end-to-end.

    Uses:
        - Focal Loss for class imbalance
        - CosineAnnealingWarmRestarts LR scheduler
        - Early stopping on validation TSS (not loss)
        - Data augmentation on training set

    Returns:
        (trained_model, metrics_dict)
    """
    # Prepare data
    windows, labels = prepare_windows(df, flare_events, window_sec, stride_sec)

    if len(windows) < 10:
        raise ValueError(f"Too few windows ({len(windows)}) for training")

    # Temporal split: 70% train, 15% val, 15% test (no shuffling!)
    n = len(windows)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_ds = FlareDataset(windows[:train_end], labels[:train_end], augment=True)
    val_ds = FlareDataset(windows[train_end:val_end], labels[train_end:val_end])
    # test windows: windows[val_end:]  (reserved for conformal calibration)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = CNNBiLSTMForecaster(
        input_channels=2,
        cnn_filters=64,
        cnn_kernel=5,
        lstm_hidden=128,
        lstm_layers=2,
        n_heads=4,
        dropout=0.2,
        neupert_weight=0.1,
        n_horizons=len(HORIZONS_MIN),
    ).to(device)

    # Loss, optimizer, scheduler
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # Early stopping state
    best_tss = -float('inf')
    best_state = None
    wait = 0

    history = {'train_loss': [], 'val_loss': [], 'val_tss': []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        n_batches = 0

        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Extract HXR and SXR for physics attention
            hxr = x_batch[:, :, 1]  # hard channel
            sxr = x_batch[:, :, 0]  # soft channel

            logits, _ = model(x_batch, hxr=hxr, sxr=sxr)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []
        n_val = 0

        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                hxr = x_batch[:, :, 1]
                sxr = x_batch[:, :, 0]

                logits, _ = model(x_batch, hxr=hxr, sxr=sxr)
                loss = criterion(logits, y_batch)

                val_loss += loss.item()
                n_val += 1

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y_batch.cpu().numpy())

        val_loss /= max(n_val, 1)
        all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

        # TSS on primary horizon (5 min = index 0)
        val_tss = 0.0
        if len(all_probs) > 0 and all_labels[:, 0].sum() > 0:
            val_tss = compute_tss(all_labels[:, 0], all_probs[:, 0])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_tss'].append(val_tss)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_TSS={val_tss:.3f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping on TSS
        if val_tss > best_tss:
            best_tss = val_tss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (best TSS={best_tss:.3f})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        'best_tss': best_tss,
        'final_epoch': epoch,
        'history': history,
        'horizons_min': HORIZONS_MIN,
    }

    return model, metrics


def get_attention_weights(model: CNNBiLSTMForecaster,
                          window: np.ndarray,
                          device: str = 'cpu') -> np.ndarray:
    """
    Get NeupertAttention weights for a single window (for visualization).

    Args:
        model: trained model
        window: [time_steps, 2] array

    Returns:
        attention_weights: [n_heads, time, time] numpy array
    """
    model.eval()
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    hxr = x[:, :, 1]
    sxr = x[:, :, 0]

    with torch.no_grad():
        _, attn = model(x, hxr=hxr, sxr=sxr)

    return attn.squeeze(0).cpu().numpy()
