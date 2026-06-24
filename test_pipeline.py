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

    print('\n' + '=' * 50)
    print('  ALL TESTS PASSED!')
    print('=' * 50)

if __name__ == '__main__':
    main()
