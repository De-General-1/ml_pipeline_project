#!/bin/bash

# Setup script for Airflow with MLflow

echo "🚀 Setting up Airflow with MLflow for ML Pipeline..."

# Create necessary directories
mkdir -p logs plugins mlruns

# Set Airflow UID (required for Docker)
echo "AIRFLOW_UID=$(id -u)" > .env
echo "AIRFLOW_PROJ_DIR=$(pwd)" >> .env

# Initialize Airflow database
echo "📦 Initializing Airflow..."
docker compose up airflow-init

# Start services
echo "🔄 Starting Airflow and MLflow services..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker compose ps

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Access URLs:"
echo "   - Airflow UI: http://localhost:8080 (admin/admin)"
echo "   - MLflow UI: http://localhost:5000"
echo ""