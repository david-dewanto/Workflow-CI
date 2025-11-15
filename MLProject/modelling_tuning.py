"""
Advanced Model Training with Hyperparameter Tuning and DagsHub Integration
Author: David Dewanto

This script trains multiple models with hyperparameter tuning and logs
everything to DagsHub using manual MLflow logging.

Target: Advanced (4 pts) - Kriteria 2

Features:
- Hyperparameter tuning with GridSearchCV
- Manual logging (not autolog)
- DagsHub integration for online storage
- Additional metrics beyond autolog
- Confusion matrix and classification report
- Feature importance visualization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    log_loss, matthews_corrcoef
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import os
warnings.filterwarnings('ignore')

# DagsHub integration
try:
    import dagshub
    DAGSHUB_AVAILABLE = True
except ImportError:
    DAGSHUB_AVAILABLE = False
    print("Warning: dagshub package not installed. Install with: pip install dagshub")


def setup_dagshub(repo_owner, repo_name):
    """Setup DagsHub for MLflow tracking"""
    if not DAGSHUB_AVAILABLE:
        print("⚠ DagsHub not available, using local MLflow")
        return False

    try:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        print(f"DagsHub initialized: {repo_owner}/{repo_name}")
        return True
    except Exception as e:
        print(f"DagsHub initialization failed: {e}")
        print("Falling back to local MLflow")
        return False


def load_preprocessed_data(file_path='iris_preprocessing.csv'):
    """Load the preprocessed dataset"""
    print(f"Loading preprocessed data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def prepare_features_target(df):
    """Prepare features and target from preprocessed data"""
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

    return X, y, feature_columns


def plot_confusion_matrix(y_true, y_pred, classes, run_id):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Save figure
    filename = f'confusion_matrix_{run_id}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return filename


def plot_feature_importance(model, feature_names, run_id):
    """Plot and save feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title('Feature Importances', fontsize=16, fontweight='bold')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)),
                   [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.tight_layout()

        # Save figure
        filename = f'feature_importance_{run_id}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

        return filename, importances
    return None, None


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive metrics

    Standard metrics (like autolog):
    - accuracy, precision, recall, f1_score

    Additional metrics (beyond autolog):
    - ROC AUC score (multiclass OVR)
    - Log Loss
    - Matthews Correlation Coefficient
    - Cohen's Kappa
    """
    metrics = {}

    # Standard metrics (like autolog)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

    # Macro averages
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')

    # ADDITIONAL METRICS (beyond autolog)
    # 1. ROC AUC Score (multiclass)
    if y_pred_proba is not None:
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(
                y_true, y_pred_proba, multi_class='ovr', average='weighted'
            )
            metrics['roc_auc_ovo'] = roc_auc_score(
                y_true, y_pred_proba, multi_class='ovo', average='weighted'
            )
        except:
            pass

    # 2. Log Loss
    if y_pred_proba is not None:
        try:
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except:
            pass

    # 3. Matthews Correlation Coefficient
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

    # 4. Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    for i in range(len(precision_per_class)):
        metrics[f'precision_class_{i}'] = precision_per_class[i]
        metrics[f'recall_class_{i}'] = recall_per_class[i]
        metrics[f'f1_class_{i}'] = f1_per_class[i]

    return metrics


def train_with_grid_search(X_train, y_train, model_name='RandomForest'):
    """Train model with hyperparameter tuning using GridSearchCV"""
    print(f"\nTraining {model_name} with GridSearchCV...")

    if model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif model_name == 'SVC':
        model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'poly']
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='f1_weighted',
        n_jobs=-1, verbose=1
    )

    print("Running grid search...")
    grid_search.fit(X_train, y_train)

    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_and_log_model(X_train, X_test, y_train, y_test,
                        feature_names, model_name='RandomForest'):
    """
    Train model with hyperparameter tuning and log to MLflow (manual logging)
    """
    print(f"\nTraining {model_name} with hyperparameter tuning...")

    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_name}_Tuned"):
        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")

        # Train with grid search
        best_model, best_params, best_cv_score = train_with_grid_search(
            X_train, y_train, model_name
        )

        # MANUAL LOGGING - Parameters
        print("\nLogging parameters...")
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("dataset", "Iris")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        mlflow.log_param("cv_folds", 5)

        # Log best hyperparameters
        for param, value in best_params.items():
            mlflow.log_param(f"best_{param}", value)

        # Make predictions
        print("Making predictions...")
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None

        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        # MANUAL LOGGING - Metrics
        print("Logging metrics...")
        mlflow.log_metric("cv_score_best", best_cv_score)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_weighted')
        mlflow.log_metric("cv_score_mean", cv_scores.mean())
        mlflow.log_metric("cv_score_std", cv_scores.std())

        # Print key metrics
        print(f"\nAccuracy:             {metrics['accuracy']:.4f}")
        print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (weighted):    {metrics['recall_weighted']:.4f}")
        print(f"F1 Score (weighted):  {metrics['f1_weighted']:.4f}")
        if 'roc_auc_ovr' in metrics:
            print(f"ROC AUC (OVR):        {metrics['roc_auc_ovr']:.4f}")
        if 'log_loss' in metrics:
            print(f"Log Loss:             {metrics['log_loss']:.4f}")
        print(f"Matthews Corr Coef:   {metrics['matthews_corrcoef']:.4f}")
        print(f"CV Score (mean±std):  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['setosa', 'versicolor', 'virginica']))

        # MANUAL LOGGING - Artifacts

        # 1. Log model (creates model/ folder with MLmodel, conda.yaml, model.pkl, etc.)
        print("\nLogging model...")
        import pickle
        import sklearn
        import cloudpickle

        # Create model directory structure manually for DagsHub compatibility
        model_dir = "model_temp"
        os.makedirs(model_dir, exist_ok=True)

        # Save model.pkl
        model_pkl_path = os.path.join(model_dir, "model.pkl")
        with open(model_pkl_path, 'wb') as f:
            pickle.dump(best_model, f)

        # Create MLmodel file
        mlmodel_content = f"""artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.12.7
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: {sklearn.__version__}
mlflow_version: {mlflow.__version__}
model_size_bytes: {os.path.getsize(model_pkl_path)}
model_uuid: {run_id}
run_id: {run_id}
utc_time_created: '2025-01-16 00:00:00.000000'
"""
        with open(os.path.join(model_dir, "MLmodel"), 'w') as f:
            f.write(mlmodel_content)

        # Create conda.yaml
        conda_yaml = f"""channels:
- conda-forge
dependencies:
- python=3.12.7
- pip<=24.0
- pip:
  - mlflow=={mlflow.__version__}
  - cloudpickle=={cloudpickle.__version__}
  - scikit-learn=={sklearn.__version__}
name: mlflow-env
"""
        with open(os.path.join(model_dir, "conda.yaml"), 'w') as f:
            f.write(conda_yaml)

        # Create python_env.yaml
        python_env_yaml = f"""python: 3.12.7
build_dependencies:
- pip==24.0
- setuptools
- wheel
dependencies:
- -r requirements.txt
"""
        with open(os.path.join(model_dir, "python_env.yaml"), 'w') as f:
            f.write(python_env_yaml)

        # Create requirements.txt
        requirements_txt = f"""mlflow=={mlflow.__version__}
cloudpickle=={cloudpickle.__version__}
scikit-learn=={sklearn.__version__}
"""
        with open(os.path.join(model_dir, "requirements.txt"), 'w') as f:
            f.write(requirements_txt)

        # Log the entire model directory
        mlflow.log_artifacts(model_dir, "model")

        # Clean up temporary directory
        import shutil
        shutil.rmtree(model_dir)
        print("Model logged successfully with full MLflow structure")

        # 2. Save and log confusion matrix as training_confusion_matrix.png
        print("Generating confusion matrix...")
        cm_file = plot_confusion_matrix(y_test, y_pred,
                                        ['setosa', 'versicolor', 'virginica'], run_id)
        # Rename to training_confusion_matrix.png
        training_cm_file = "training_confusion_matrix.png"
        if os.path.exists(cm_file):
            os.rename(cm_file, training_cm_file)
            mlflow.log_artifact(training_cm_file)
            os.remove(training_cm_file)

        # 3. Save and log feature importance
        print("Generating feature importance...")
        fi_file, importances = plot_feature_importance(best_model, feature_names, run_id)
        if fi_file:
            mlflow.log_artifact(fi_file)
            os.remove(fi_file)

            # Log feature importances as dict
            importance_dict = {name: float(imp)
                              for name, imp in zip(feature_names, importances)}
            mlflow.log_dict(importance_dict, "feature_importances.json")

        # 4. Create and log metric_info.json
        print("Saving metric info...")
        metric_info = {
            "metrics": {
                "accuracy": float(metrics['accuracy']),
                "precision_weighted": float(metrics['precision_weighted']),
                "recall_weighted": float(metrics['recall_weighted']),
                "f1_weighted": float(metrics['f1_weighted']),
                "roc_auc_ovr": float(metrics.get('roc_auc_ovr', 0)),
                "roc_auc_ovo": float(metrics.get('roc_auc_ovo', 0)),
                "log_loss": float(metrics.get('log_loss', 0)),
                "matthews_corrcoef": float(metrics['matthews_corrcoef']),
                "cv_score_mean": float(cv_scores.mean()),
                "cv_score_std": float(cv_scores.std())
            },
            "model_name": model_name,
            "best_cv_score": float(best_cv_score),
            "dataset": "Iris",
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0],
            "n_test_samples": X_test.shape[0]
        }
        mlflow.log_dict(metric_info, "metric_info.json")

        # 5. Save classification report
        print("Saving classification report...")
        report = classification_report(y_test, y_pred,
                                      target_names=['setosa', 'versicolor', 'virginica'],
                                      output_dict=True)
        mlflow.log_dict(report, "classification_report.json")

        # 6. Create and log estimator.html (model summary)
        print("Generating estimator HTML...")
        estimator_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{model_name} Estimator Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .param {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{model_name} Model Summary</h1>
    <h2>Best Hyperparameters</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        {''.join(f'<tr class="param"><td>{k}</td><td>{v}</td></tr>' for k, v in best_params.items())}
    </table>
    <h2>Performance Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Accuracy</td><td>{metrics['accuracy']:.4f}</td></tr>
        <tr><td>Precision (weighted)</td><td>{metrics['precision_weighted']:.4f}</td></tr>
        <tr><td>Recall (weighted)</td><td>{metrics['recall_weighted']:.4f}</td></tr>
        <tr><td>F1 Score (weighted)</td><td>{metrics['f1_weighted']:.4f}</td></tr>
        <tr><td>ROC AUC (OVR)</td><td>{metrics.get('roc_auc_ovr', 0):.4f}</td></tr>
        <tr><td>Matthews Correlation</td><td>{metrics['matthews_corrcoef']:.4f}</td></tr>
        <tr><td>CV Score</td><td>{cv_scores.mean():.4f} ± {cv_scores.std():.4f}</td></tr>
    </table>
    <h2>Dataset Information</h2>
    <table>
        <tr><td>Training Samples</td><td>{X_train.shape[0]}</td></tr>
        <tr><td>Testing Samples</td><td>{X_test.shape[0]}</td></tr>
        <tr><td>Features</td><td>{X_train.shape[1]}</td></tr>
        <tr><td>Classes</td><td>3 (setosa, versicolor, virginica)</td></tr>
    </table>
</body>
</html>
"""
        estimator_file = "estimator.html"
        with open(estimator_file, 'w') as f:
            f.write(estimator_html)
        mlflow.log_artifact(estimator_file)
        os.remove(estimator_file)

        # 7. Log best parameters as JSON
        mlflow.log_dict(best_params, "best_parameters.json")

        # 8. Additional artifact: Training summary text file
        print("Saving training summary...")
        summary_text = f"""Model Training Summary
{'='*50}
Model: {model_name}
Run ID: {run_id}
Dataset: Iris Classification

Best Hyperparameters:
{json.dumps(best_params, indent=2)}

Performance Metrics:
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision_weighted']:.4f}
- Recall: {metrics['recall_weighted']:.4f}
- F1 Score: {metrics['f1_weighted']:.4f}
- ROC AUC (OVR): {metrics.get('roc_auc_ovr', 0):.4f}
- Log Loss: {metrics.get('log_loss', 0):.4f}
- Matthews Correlation: {metrics['matthews_corrcoef']:.4f}
- CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

Training Configuration:
- Training Samples: {X_train.shape[0]}
- Testing Samples: {X_test.shape[0]}
- Features: {X_train.shape[1]}
- Cross-Validation Folds: 5
"""
        summary_file = "training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        mlflow.log_artifact(summary_file)
        os.remove(summary_file)

        print("\nTraining and logging completed!")
        print(f"All parameters manually logged")
        print(f"All metrics manually logged (including additional metrics)")
        print(f"All artifacts saved")

        return best_model, metrics


def main():
    """Main function"""
    # DagsHub configuration
    DAGSHUB_REPO_OWNER = "david-dewanto"
    DAGSHUB_REPO_NAME = "MLOps-Dicoding_DavidDewanto"
    USE_DAGSHUB = True  # Using DagsHub for online artifact storage

    if USE_DAGSHUB:
        dagshub_success = setup_dagshub(DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME)
        if dagshub_success:
            print(f"Using DagsHub: https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
    else:
        # Use local MLflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print(f"\nMLflow Tracking URI: {mlflow.get_tracking_uri()}")

    # Set experiment
    mlflow.set_experiment("Iris_Classification_Advanced")
    print(f"MLflow Experiment: Iris_Classification_Advanced")

    # Load data
    df = load_preprocessed_data('iris_preprocessing.csv')

    # Prepare features and target
    X, y, feature_names = prepare_features_target(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # Train multiple models
    models_to_train = ['RandomForest', 'GradientBoosting']  # 'SVC' can be added but slower

    best_model = None
    best_f1 = 0

    for model_name in models_to_train:
        model, metrics = train_and_log_model(
            X_train, X_test, y_train, y_test,
            feature_names, model_name
        )

        # Track best model
        if metrics['f1_weighted'] > best_f1:
            best_f1 = metrics['f1_weighted']
            best_model = model_name

    print(f"\nBest model: {best_model}")
    print(f"Best F1 Score: {best_f1:.4f}")


if __name__ == "__main__":
    main()
