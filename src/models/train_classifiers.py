"""
Train and evaluate machine learning classifiers for FDIA detection.

This script trains four classifiers on the MAL-ICS++ dataset:
- Support Vector Machine (RBF kernel)
- Random Forest
- Gradient Boosting
- Logistic Regression

Author: MAL-ICS++ Research Team
Date: 2026-02-12
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import json
from pathlib import Path


def load_dataset(data_path='../../data/processed/malics_dataset_complete.csv'):
    """Load the MAL-ICS++ dataset."""
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Features: {df.shape[1]}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """Split dataset into train/test sets."""
    # Separate features and labels
    X = df.drop(['label', 'timestamp'], axis=1, errors='ignore')
    y = df['label']
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def create_pipeline(classifier):
    """Create preprocessing + classifier pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])


def train_models(X_train, y_train):
    """Train all four classifiers."""
    models = {
        'svm_rbf': create_pipeline(SVC(kernel='rbf', probability=True, random_state=42)),
        'random_forest': create_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)),
        'gradient_boost': create_pipeline(GradientBoostingClassifier(n_estimators=100, random_state=42)),
        'log_reg': create_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✓ {name} trained successfully")
    
    return trained_models


def evaluate_model(model, X_test, y_test):
    """Evaluate a single model."""
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    # AUC for binary classification
    if len(np.unique(y_test)) == 2 and y_proba is not None:
        metrics['auc'] = roc_auc_score(y_test, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # False positive/negative rates
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics, cm


def save_results(results, output_dir='../../results/metrics'):
    """Save metrics to CSV and JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    df_results = pd.DataFrame(results).T
    csv_path = Path(output_dir) / 'model_results.csv'
    df_results.to_csv(csv_path)
    print(f"\n✓ Results saved to {csv_path}")
    
    # Save as JSON
    json_path = Path(output_dir) / 'model_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {json_path}")


def save_models(models, output_dir='../../results/models'):
    """Save trained models."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        model_path = Path(output_dir) / f'{name}.joblib'
        joblib.dump(model, model_path)
        print(f"✓ Model saved: {model_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("MAL-ICS++ Classifier Training")
    print("=" * 60)
    
    # Load dataset
    df = load_dataset()
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate all models
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    results = {}
    for name, model in models.items():
        print(f"\n{name.upper()}:")
        metrics, cm = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        if 'auc' in metrics:
            print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  FPR:       {metrics['fpr']:.4f}")
        print(f"  FNR:       {metrics['fnr']:.4f}")
    
    # Save results
    save_results(results)
    save_models(models)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
