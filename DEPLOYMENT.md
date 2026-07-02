# Deployment Configurations

This document explains how to deploy the pipeline containers in production.

## Docker Compose Production

Configure your production environment variables in `.env` and start services:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Cloud Ingestion Setup

For Google Cloud or AWS deployment:
1. Map MLflow tracking logs to managed databases (e.g., Cloud SQL).
2. Configure artifact buckets (GCS/S3) as the default storage storage path.
3. Deploy the FastAPI serving image to container orchestration platforms like Google Cloud Run or AWS ECS.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)