# Architecture & Pipeline Layout

This document provides a detailed breakdown of the Enterprise MLOps Pipeline architecture.

## Architecture Diagram

The system components interact as follows:
- Data Prep ingestion pipeline normalizes incoming raw datasets.
- Training scripts coordinate metric updates with the MLflow tracking service.
- The MLflow Artifact Store saves serialized model configurations.
- The FastAPI serving container queries the Registry database to load and serve model weights.

## Core Modules

### Preprocessing Layer (src/preprocess.py)
Encapsulates data cleaning, normalizers, and encoders to ensure raw inputs are transformed consistently during both training and inference.

### Serving Service (src/serve.py)
A lightweight FastAPI microservice designed for high throughput, loading serialized artifacts from the registry to serve client predictions.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)