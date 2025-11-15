"""
Basic Model Training with MLflow Autolog
Author: David Dewanto

This script trains a Random Forest Classifier on the Iris dataset
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
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(file_path='iris_preprocessing.csv'):
    """Load the preprocessed dataset"""
    print(f"Loading preprocessed data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def prepare_features_target(df):
    """Prepare features and target from preprocessed data"""
    # Feature columns
    feature_columns = [
        'sepal length (cm)', 'sepal width (cm)',
        'petal length (cm)', 'petal width (cm)',
        'sepal_area', 'petal_area',
        'sepal_ratio', 'petal_ratio'
    ]

    X = df[feature_columns]
    y = df['target_encoded']

    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Classes: {y.nunique()}")

    return X, y


def train_model_with_autolog(X_train, X_test, y_train, y_test):
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
            n_estimators=100,
            max_depth=10,
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
        mlflow.log_param("dataset", "Iris")

        print("\nModel training completed successfully.")

        return model


def main():
    """Main function"""
    # Set MLflow tracking URI (local)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Iris_Classification_Basic")

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: Iris_Classification_Basic")

    # Load data
    df = load_preprocessed_data('iris_preprocessing.csv')

    # Prepare features and target
    X, y = prepare_features_target(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # Train model with autolog
    model = train_model_with_autolog(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
