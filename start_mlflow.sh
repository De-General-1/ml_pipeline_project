#!/bin/bash

echo "🚀 Starting MLflow with S3 artifact storage..."

# Set AWS profile
export AWS_PROFILE=degen-mlops

# Start MLflow server with S3 artifacts
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri file://./mlruns \
    --default-artifact-root s3://phase3-mlops-source-bucket-degen-1/mlflow-artifacts

echo "✅ MLflow server started!"
echo "🌐 Access MLflow UI at: http://localhost:5000"