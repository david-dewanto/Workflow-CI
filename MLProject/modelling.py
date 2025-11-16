"""
Basic Model Training with MLflow Autolog
Author: David Dewanto

This script trains a Random Forest Classifier on the Transactions dataset
using MLflow autolog for automatic tracking.

Target: Basic (2 pts) - Kriteria 2
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(file_path='transactions_preprocessing.csv'):
    """Load the preprocessed dataset"""
    print(f"Loading preprocessed data from: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Data loaded. Shape: {df.shape}")
    return df


def prepare_features_target(df):
    """Prepare features and target from preprocessed data"""
    # Feature columns (numerical features only, excluding categorical and identifiers)
    feature_columns = [
        'account_age_days', 'total_transactions_user', 'avg_amount_user',
        'amount', 'promo_used', 'avs_match', 'cvv_result', 'three_ds_flag',
        'shipping_distance_km', 'amount_transactions_product',
        'amount_avg_product', 'amount_avg_ratio', 'shipping_age_ratio'
    ]

    X = df[feature_columns]
    y = df['target_encoded']

    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Classes: {y.nunique()}")

    return X, y


def train_model_with_autolog(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=10, min_samples_split=2):
    """
    Train Random Forest model with MLflow autolog

    MLflow autolog automatically logs:
    - Parameters (n_estimators, max_depth, etc.)
    - Metrics (accuracy, f1_score, etc.)
    - Model artifact
    - Feature importances
    """
    print("\nTraining model with MLflow autolog...")

    # Enable MLflow autolog for scikit-learn
    mlflow.sklearn.autolog()

    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )

        print("Training Random Forest Classifier...")
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics (autolog will capture these automatically)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Log additional information
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("dataset", "Transactions")

        print("\nModel training completed successfully.")

        return model


def main():
    """Main function with argument parsing for MLflow Project"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Transactions classification model')
    parser.add_argument('--data-path', type=str, default='transactions_preprocessing.csv',
                       help='Path to preprocessed dataset')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in random forest')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth of trees')
    parser.add_argument('--min-samples-split', type=int, default=2,
                       help='Minimum samples required to split')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0.0 to 1.0)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')

    args = parser.parse_args()

    # Set MLflow experiment
    mlflow.set_experiment("Transactions_CI_CD_Training")

    print(f"Experiment: Transactions_CI_CD_Training")
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, min_samples_split={args.min_samples_split}")

    # Load data
    df = load_preprocessed_data(args.data_path)

    # Prepare features and target
    X, y = prepare_features_target(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # Train model with autolog
    model = train_model_with_autolog(
        X_train, X_test, y_train, y_test,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split
    )


if __name__ == "__main__":
    main()
