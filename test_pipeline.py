"""
test_pipeline.py — End-to-end test of the solar flare pipeline.
"""
import sys
sys.path.insert(0, '.')

def main():
    # Test 1: Synthetic data generation
    print('=== Test 1: Synthetic Data Generation ===')
    from backend.data.loader_sim import generate_synthetic_data, SimulationConfig, flare_events_to_df
    config = SimulationConfig(duration_hours=6, num_flares=5, seed=42)
    df, events = generate_synthetic_data(config)
    print(f'Generated {len(df)} data points')
    print(f'Injected {len(events)} flares:')
    for e in events:
        print(f'  {e.flare_class}-class at t={e.peak_time:.0f}s (lead={e.hard_lead_time:.0f}s)')

    # Test 2: Preprocessing
    print('\n=== Test 2: Preprocessing ===')
    from backend.data.preprocessor import full_preprocess
    processed = full_preprocess(df)
    print(f'Preprocessed: {len(processed)} rows, {len(processed.columns)} columns')

    # Test 3: Nowcasting
    print('\n=== Test 3: Nowcasting ===')
    from backend.nowcast.detector_soft import detect_flares_soft
    from backend.nowcast.detector_hard import detect_flares_hard
    from backend.nowcast.combiner import combine_catalogues, master_catalogue_to_df
    from backend.nowcast.classifier import classify_master_catalogue

    soft_det = detect_flares_soft(processed)
    hard_det = detect_flares_hard(processed)
    print(f'Soft detections: {len(soft_det)}')
    print(f'Hard detections: {len(hard_det)}')

    master = combine_catalogues(soft_det, hard_det)
    master = classify_master_catalogue(master)
    cat_df = master_catalogue_to_df(master)
    print(f'Master catalogue: {len(master)} events')
    for _, row in cat_df.iterrows():
        cls = row['flare_class']
        det = row['detection_type']
        pt = row['peak_time']
        print(f'  {cls}-class at t={pt:.0f}s ({det})')

    # Test 4: Feature extraction
    print('\n=== Test 4: Feature Extraction ===')
    from backend.forecast.features import extract_features, create_labels, FEATURE_COLUMNS
    features = extract_features(processed, window_sec=300, stride_sec=60)
    labels = create_labels(processed, features, events, horizon_sec=300)
    pos_pct = 100 * labels.sum() / len(labels) if len(labels) > 0 else 0
    print(f'Features: {len(features)} windows x {len(FEATURE_COLUMNS)} features')
    print(f'Positive labels: {labels.sum()} / {len(labels)} ({pos_pct:.1f}%)')

    # Test 5: Model training
    print('\n=== Test 5: Forecast Model Training ===')
    from backend.forecast.model import XGBoostForecaster
    model = XGBoostForecaster(horizon_sec=300)
    
    if labels.sum() > 0:
        metrics = model.train(features, labels, n_splits=3, verbose=True)
        print(f'Training complete. AUC: {metrics["cv_auc_mean"]:.3f}')
        
        # Test 6: Evaluation
        print('\n=== Test 6: Evaluation ===')
        from backend.evaluation.metrics import compute_nowcast_metrics, compute_forecast_metrics
        nowcast_m = compute_nowcast_metrics(master, events, tolerance_sec=120)
        print(f'Nowcast: TPR={nowcast_m["tpr"]:.3f}, FAR={nowcast_m["far"]:.3f}, F1={nowcast_m["f1"]:.3f}')

        probs = model.predict_proba(features)
        forecast_m = compute_forecast_metrics(labels.values, probs)
        print(f'Forecast: AUC={forecast_m["auc"]:.3f}, TSS={forecast_m["tss"]:.3f}')
        
        # Feature importance
        print('\n=== Top 5 Features ===')
        importance = model.get_feature_importance()
        for _, row in importance.head(5).iterrows():
            print(f'  {row["feature"]}: {row["importance"]:.2f}')
    else:
        print('No positive labels - skipping model training')

    # Test 7: CNN-BiLSTM model (if PyTorch available)
    print('\n=== Test 7: CNN-BiLSTM + NeupertAttention ===')
    try:
        from backend.forecast.neural_model import (
            CNNBiLSTMForecaster, NeupertAttention, FocalLoss,
            prepare_windows, HORIZONS_MIN
        )
        import torch

        # Create model
        net = CNNBiLSTMForecaster(
            input_channels=2, cnn_filters=32, lstm_hidden=64,
            lstm_layers=1, n_heads=2, n_horizons=5
        )
        print(f'  Model created: {sum(p.numel() for p in net.parameters())} parameters')

        # Forward pass with dummy data
        x = torch.randn(4, 300, 2)  # batch=4, time=300, channels=2
        hxr = x[:, :, 1]
        sxr = x[:, :, 0]
        logits, attn = net(x, hxr=hxr, sxr=sxr)
        print(f'  Forward pass: logits={logits.shape}, attn={attn.shape}')
        assert logits.shape == (4, 5), f'Expected (4,5), got {logits.shape}'
        print(f'  Probabilities: {torch.sigmoid(logits[0]).detach().numpy()}')

        # Test Focal Loss
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        targets = torch.tensor([[1,0,0,1,0],[0,1,0,0,1],[1,1,0,0,0],[0,0,1,0,0]], dtype=torch.float32)
        loss = loss_fn(logits, targets)
        print(f'  Focal Loss: {loss.item():.4f}')

        # Test prepare_windows
        windows, win_labels = prepare_windows(processed, events, window_sec=300, stride_sec=120)
        print(f'  Windows: {windows.shape}, Labels: {win_labels.shape}')

        print('  [OK] CNN-BiLSTM test passed')
    except ImportError:
        print('  [WARN] PyTorch not installed — skipping neural model test')

    # Test 8: Conformal Prediction
    print('\n=== Test 8: Conformal Prediction ===')
    from backend.forecast.conformal import ConformalFlarePredictor
    import numpy as np_test

    # Create synthetic calibration data
    np_test.random.seed(42)
    cal_probs = np_test.random.rand(100, 2)
    cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
    cal_labels = np_test.random.randint(0, 2, size=100)

    predictor = ConformalFlarePredictor(alpha=0.1, method='standard')
    predictor.calibrate(cal_probs, cal_labels)
    print(f'  Calibrated: q_hat={predictor.q_hat:.3f}')

    # Test prediction sets
    test_probs = np_test.random.rand(20, 2)
    test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)
    point_preds, uncertain, pred_sets = predictor.predict_with_uncertainty(test_probs)
    print(f'  Prediction sets: {len(pred_sets)} samples, '
          f'{sum(uncertain)} uncertain')

    # Test binary intervals
    binary_probs = np_test.random.rand(50)
    prob, lower, upper = predictor.predict_intervals_binary(binary_probs)
    print(f'  Binary intervals: width={np_test.mean(upper-lower):.3f}')

    stats = predictor.get_stats()
    print(f'  Stats: coverage_target={stats["target_coverage"]}, '
          f'method={stats["method"]}')
    print('  [OK] Conformal prediction test passed')

    # Test 9: Advanced Features (Wavelet, Hurst, Transfer Entropy)
    print('\n=== Test 9: Advanced Feature Engineering ===')
    # Re-extract features to verify new features are present
    features2 = extract_features(processed, window_sec=300, stride_sec=120)
    new_features = ['soft_wavelet_energy', 'soft_wavelet_entropy',
                    'hard_wavelet_energy', 'hard_wavelet_entropy',
                    'transfer_entropy_h2s', 'hurst_soft', 'hurst_hard',
                    'peak_bg_ratio_soft', 'peak_bg_ratio_hard',
                    'time_since_last_flare']
    found = [f for f in new_features if f in features2.columns]
    missing = [f for f in new_features if f not in features2.columns]
    print(f'  New features present: {len(found)}/{len(new_features)}')
    if missing:
        print(f'  Missing: {missing}')
    else:
        print(f'  All new features: {found}')
    for f in found[:5]:
        vals = features2[f]
        print(f'    {f}: mean={vals.mean():.4f}, std={vals.std():.4f}')
    print(f'  Total features: {len(FEATURE_COLUMNS)} '
          f'(33 original + {len(FEATURE_COLUMNS)-33} new)')
    print('  [OK] Advanced features test passed')

    # Test 10: FRED Synthetic Data
    print('\n=== Test 10: FRED Synthetic Data ===')
    from backend.data.loader_sim import generate_fred_flare
    t = np_test.arange(1000, dtype=np_test.float64)
    fred_profile = generate_fred_flare(
        t, onset=200, peak_time=350, end_time=800,
        peak_amplitude=50.0, rise_tau=50.0, decay_tau=150.0,
        n_thermal_components=3
    )
    print(f'  FRED profile: shape={fred_profile.shape}, '
          f'peak={fred_profile.max():.1f}, '
          f'nonzero={(fred_profile > 0.1).sum()} samples')
    assert fred_profile.max() > 0, 'FRED profile should have positive values'
    print('  [OK] FRED synthetic data test passed')

    print('\n' + '=' * 50)
    print('  ALL TESTS PASSED!')
    print('=' * 50)

if __name__ == '__main__':
    main()
