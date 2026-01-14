#!/usr/bin/env python
"""
Standalone test script for Profiles Random Forest classifier.
Tests trained RF models on profile data for each FPS level.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_DIR))

from notebooks.config import RAW_DATA_DIR, EXP_DIR_ROOT, SEED, TEST_RATIO, FPS_LIST

# Config
MODEL_DIR = EXP_DIR_ROOT / "profiles_random_forest"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "profiles_rf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_profiles_for_fps(fps_label, data_dir=RAW_DATA_DIR):
    """Load all profile CSV files for a given FPS level."""
    df_list = []
    
    for f in data_dir.glob(f"**/{fps_label}/*.csv"):
        if "profiles" in f.name.lower():
            df = pd.read_csv(f, dtype='float32')
            mineral = f.parts[-3]
            df['mineral'] = mineral
            df_list.append(df)
    
    if not df_list:
        return None
    
    data = pd.concat(df_list, ignore_index=True)
    return data


def test_profiles_rf(fps_label):
    """Test Random Forest model for given FPS level."""
    print(f"\n{'='*70}")
    print(f"Testing Profiles RF: {fps_label}")
    print(f"{'='*70}")
    
    # Check if model exists
    model_path = MODEL_DIR / fps_label / "rf_model.pkl"
    scaler_path = MODEL_DIR / fps_label / "scaler.pkl"
    metrics_path = MODEL_DIR / fps_label / "metrics.json"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print(f"   Please run the training notebook first.")
        return None
    
    # Load data
    print(f"Loading test data...")
    df = load_profiles_for_fps(fps_label)
    
    if df is None:
        print(f"❌ No profiles found for {fps_label}")
        return None
    
    print(f"✓ Loaded {len(df)} samples")
    
    # Split data (use same seed for reproducibility)
    feature_cols = [col for col in df.columns if col != 'mineral']
    X = df[feature_cols].values.astype('float32')
    y = df['mineral'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=SEED, stratify=y
    )
    
    # Load model and scaler
    print(f"Loading model and scaler...")
    rf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load training metrics
    with open(metrics_path) as f:
        train_metrics = json.load(f)
    
    print(f"✓ Training accuracy: {train_metrics['accuracy']:.4f}")
    
    # Transform test data
    print(f"Transforming test data...")
    X_test = scaler.transform(X_test).astype('float32')
    
    # Evaluate
    print(f"Evaluating...")
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Test Results for {fps_label}:")
    print(f"{'='*70}")
    print(f"✓ Test Accuracy: {accuracy:.4f}")
    print(f"✓ Test Samples: {len(X_test)}")
    print(f"✓ Classes: {sorted(np.unique(y_test))}")
    
    print(f"\n📄 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion Matrix
    print(f"Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(np.unique(y_test))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'},
        square=True
    )
    plt.title(f'Test Confusion Matrix - Profiles RF ({fps_label})', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = RESULTS_DIR / f"confusion_matrix_{fps_label}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {cm_path}")
    
    plt.show()
    
    # Save test results
    test_results = {
        'fps': fps_label,
        'test_accuracy': float(accuracy),
        'train_accuracy': float(train_metrics['accuracy']),
        'n_test_samples': int(len(X_test)),
        'n_test_errors': int(np.sum(y_pred != y_test)),
        'classes': [str(c) for c in classes]
    }
    
    results_path = RESULTS_DIR / f"test_results_{fps_label}.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"✓ Saved test results to {results_path}")
    
    return test_results


def main():
    """Test all FPS levels."""
    print(f"\n{'='*70}")
    print("PROFILES RANDOM FOREST - TEST SCRIPT")
    print(f"{'='*70}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    
    all_results = {}
    
    for fps_label in FPS_LIST:
        results = test_profiles_rf(fps_label)
        if results:
            all_results[fps_label] = results
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if all_results:
        print("\nTest Results:")
        for fps, results in all_results.items():
            print(f"  {fps}:")
            print(f"    Train Accuracy: {results['train_accuracy']:.4f}")
            print(f"    Test Accuracy:  {results['test_accuracy']:.4f}")
            print(f"    Test Samples:   {results['n_test_samples']}")
            print(f"    Errors:         {results['n_test_errors']}")
        
        # Save summary
        summary_path = RESULTS_DIR / "test_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved summary to {summary_path}")
    else:
        print("❌ No results to summarize")
    
    print(f"\n✅ Testing complete!\n")


if __name__ == "__main__":
    main()
