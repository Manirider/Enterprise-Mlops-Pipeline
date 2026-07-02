# Enterprise MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat-square&logo=python&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![License](https://img.shields.io/github/license/Manirider/Enterprise-Mlops-Pipeline?style=flat-square)

A modular, production-grade Machine Learning Operations (MLOps) pipeline establishing end-to-end versioning, experiment tracking, artifact control, registry mapping, and low-latency API serving.

`mlops` `mlflow` `fastapi` `machine-learning` `python` `docker` `model-serving` `experiment-tracking`

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Motivation & Objectives](#motivation--objectives)
- [Core Features](#core-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Folder Structure](#folder-structure)
- [Installation & Local Setup](#installation--local-setup)
- [Environment Variables](#environment-variables)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Performance & Scale Considerations](#performance--scale-considerations)
- [Testing Requirements](#testing-requirements)
- [Deployment Options](#deployment-options)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

## Overview

Enterprise MLOps Pipeline bridges the operational gap between experimental notebooks and stable production environments. In typical data science setups, model updates lack reproducibility, performance shifts go undetected, and deployment environments lack strict input contracts. This project addresses these instabilities by wrapping model ingestion, hyperparameter runs, artifact logging, and containerized serving endpoints into a unified workflow.

## Problem Statement

Transitioning models from development to active software integrations often introduces failures:
- **Reproducibility Deficits:** Models trained without versioned code and features cannot be reliably retrained or audited.
- **Serving Inconsistencies:** Custom pickling configurations mismatch production API environments.
- **Monitoring Blindspots:** Underperforming model models are deployed without standardized latency checks or input validation guards.

## Motivation & Objectives

This framework is built around three operational requirements:
- **Comprehensive Lineage:** Every registered model must trace back to its input feature sets, configurations, and code commits.
- **Deterministic Serving Contracts:** APIs must perform validation checks on inputs, intercepting anomalies before they reach model buffers.
- **Isolation of Concerns:** Decouple training compute pipelines from high-availability serving microservices.

## Core Features

- **Centralized Registry Tracking:** Complete parameter logging and artifact collection utilizing local or remote MLflow servers.
- **Validated Input Contracts:** FastAPI request wrappers parsing inputs via Pydantic schemas.
- **Automated Preprocessing Pipelines:** Scalable tokenizers, feature scaling, and categorical encoders versioned alongside model files.
- **Docker Compose Configurations:** Container orchestration setting up database backends, MLflow metrics servers, and API runners in isolation.
- **System Verification Suites:** CI workflows executing parameter validation checks and unit tests.

## Tech Stack

- **Core Languages:** Python 3.10
- **Libraries:** scikit-learn, Pandas, NumPy, Pydantic
- **MLOps Middleware:** MLflow tracking backend
- **Serving Engines:** FastAPI, Uvicorn
- **Container Infrastructure:** Docker, Docker Compose

## System Architecture

The workflow consists of the following components:

1. Raw Data Ingestion:
   Processes input data streams, executing cleaning, normalization, and split operations.
2. Experiment Tracking Layer:
   Trains candidate models, logging metric runs and serialization graphs directly to MLflow.
3. Registered Artifact Database:
   Catalogs champion weights into production registry queues.
4. Production API Service:
   FastAPI pulls registered files from the registry bucket, serving client requests.
5. Monitoring:
   Tracks model metrics and prediction latency values.

## Folder Structure

```
Enterprise-Mlops-Pipeline/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── ARCHITECTURE.md
├── API.md
├── ROADMAP.md
├── DEPLOYMENT.md
├── docker-compose.yml
├── requirements.txt
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── serve.py
│   └── monitor.py
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   └── test_serve.py
└── .github/
    └── workflows/
        └── ci.yml
```

## Installation & Local Setup

```bash
# Clone this repository
git clone https://github.com/Manirider/Enterprise-Mlops-Pipeline.git
cd Enterprise-Mlops-Pipeline

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

## Environment Variables

The pipeline reads the following configuration variables:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MODEL_REGISTRY_PATH="models:/EnterpriseModel/Production"
export API_PORT=8000
```

## Usage Guide

### 1. Launch MLOps Infrastructure

Start Postgres, MLflow, and local storage buckets using Docker Compose:

```bash
docker-compose up -d
```

Access the MLflow UI by navigating to `http://localhost:5000` in your browser.

### 2. Train and Register Model

Execute the training script to pre-process datasets, train classification classifiers, and register the best-performing model:

```bash
python src/train.py --config config.yaml
```

### 3. Start Prediction API

Launch the FastAPI microservice to load the active model version and serve predictions:

```bash
python src/serve.py
```

## API Documentation

The FastAPI service exposes the following main endpoints:

- **GET `/health`** — Basic health monitoring status.
- **POST `/predict`** — Submits features to the registered model, returning predictions.
  - Request Body:
    ```json
    {
      "features": [5.1, 3.5, 1.4, 0.2]
    }
    ```
  - Response:
    ```json
    {
      "prediction": 0,
      "probability": [0.98, 0.01, 0.01],
      "model_version": "v1.2.0"
    }
    ```

## Performance & Scale Considerations

- **Dynamic Loading:** The serving API utilizes a background thread query to reload production model versions without system restarts.
- **Batch Processing:** The prediction endpoint handles single and batch predictions efficiently.

## Testing Requirements

Run unit tests verifying the preprocessing pipeline and model predictions:

```bash
pytest tests/
```

## Deployment Options

For cloud deployments, configure remote storage backends (e.g., Google Cloud Storage or AWS S3) for the MLflow artifact store, and deploy the serving container to scalable environments like Kubernetes or Cloud Run.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)