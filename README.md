# Workflow-CI

Repository for Kriteria 3: MLflow Project with CI/CD Pipeline

## Overview

This repository contains an MLflow Project with automated CI/CD pipeline using GitHub Actions for training machine learning models on the Iris dataset.

## Project Structure

```
Workflow-CI/
├── .github/workflows/
│   └── mlflow-training.yml    # CI/CD pipeline configuration
├── MLProject/
│   ├── MLProject              # MLflow project definition
│   ├── conda.yaml             # Environment dependencies
│   ├── modelling.py           # Training script
│   ├── iris_preprocessing.csv # Preprocessed dataset
│   └── DockerHub.txt          # Docker Hub repository link
└── README.md
```

## Features

- Automated model training using MLflow Projects
- CI/CD pipeline with GitHub Actions
- Artifact storage in GitHub repository
- Docker image generation and deployment to Docker Hub
- Parameterized training with customizable hyperparameters

## Requirements

- Python 3.12.7
- MLflow 2.19.0
- scikit-learn 1.5.0
- pandas 2.3.3
- numpy 2.3.4
- matplotlib 3.9.0
- seaborn 0.13.2

## Local Usage

### Run MLflow Project

```bash
cd MLProject
mlflow run . --experiment-name "Iris_Training"
```

### Run with Custom Parameters

```bash
mlflow run . \
  --experiment-name "Iris_Training" \
  -P n_estimators=200 \
  -P max_depth=15
```

### Available Entry Points

**main** (default):
- Fully customizable parameters
- data_path, n_estimators, max_depth, min_samples_split, test_size, random_state

**train_default**:
- Uses default parameters
- Only requires data_path

**train_custom**:
- Pre-configured high-performance settings
- n_estimators=200, max_depth=15

## CI/CD Pipeline

The GitHub Actions workflow automatically triggers on:
- Push to main branch (MLProject/** files)
- Pull requests to main branch
- Manual workflow dispatch

### Pipeline Steps

1. Model Training: Runs MLflow Project with specified parameters
2. Artifact Collection: Gathers trained model and metrics
3. Artifact Storage: Commits artifacts to repository
4. Docker Build: Creates Docker image using MLflow
5. Docker Push: Publishes image to Docker Hub

### GitHub Secrets Required

Configure these secrets in repository settings:
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password or access token

## Model Details

- Algorithm: Random Forest Classifier
- Dataset: Iris (preprocessed with 8 features)
- Training: 119 samples
- Testing: 30 samples
- Classes: 3 (setosa, versicolor, virginica)

## Docker Usage

Pull the Docker image:

```bash
docker pull [username]/iris-mlops-ci:latest
```

Run the container:

```bash
docker run -p 5000:5000 [username]/iris-mlops-ci:latest
```

## Artifacts

Each training run produces:
- Trained model (pickle format)
- Performance metrics (accuracy, precision, recall, F1)
- Model metadata and parameters
- MLflow tracking information

## License

This project is created for Dicoding "Membangun Sistem Machine Learning" submission.

## Author

David Dewanto
