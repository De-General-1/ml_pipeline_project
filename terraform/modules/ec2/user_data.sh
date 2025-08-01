#!/bin/bash

# Update system
apt update -y
apt upgrade -y

# Install Docker
apt install -y docker.io
systemctl start docker
systemctl enable docker
usermod -a -G docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Git
apt install -y git

# Create project directory
mkdir -p /home/ubuntu/ml_pipeline_project
cd /home/ubuntu/ml_pipeline_project

# Clone repository
git clone https://github.com/De-General-1/ml_pipeline_project.git .

# Install AWS CLI
apt install -y awscli

# Create AWS credentials directory and configure
mkdir -p /home/ubuntu/.aws
cat > /home/ubuntu/.aws/config << 'AWSEOF'
[default]
region = eu-west-1
output = json
AWSEOF

# Set up IAM role credentials (will be used automatically)
chown -R ubuntu:ubuntu /home/ubuntu/.aws

# Update docker-compose with proper configuration
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 10s
      retries: 5
      start_period: 5s
    restart: always

  airflow-init:
    image: apache/airflow:2.8.0-python3.10
    container_name: ml_pipeline_project_airflow_init
    entrypoint: /bin/bash -c "pip install --no-cache-dir mlflow boto3 scikit-learn pandas numpy && airflow db migrate && airflow users create --username airflow --firstname Airflow --lastname Admin --role Admin --email admin@example.com --password airflow"
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AWS_DEFAULT_REGION: eu-west-1
      S3_BUCKET: ${s3_bucket}
    volumes:
      - ./dags:/opt/airflow/dags
    depends_on:
      postgres:
        condition: service_healthy

  airflow-webserver:
    image: apache/airflow:2.8.0-python3.10
    container_name: ml_pipeline_project_airflow_webserver
    command: ["airflow", "webserver"]
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AWS_DEFAULT_REGION: eu-west-1
      S3_BUCKET: ${s3_bucket}
    ports:
      - "8080:8080"
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - ./src:/opt/airflow/src
      - ./models:/opt/airflow/models
    depends_on:
      airflow-init:
        condition: service_completed_successfully
      postgres:
        condition: service_healthy
    restart: always

  airflow-scheduler:
    image: apache/airflow:2.8.0-python3.10
    container_name: ml_pipeline_project_airflow_scheduler
    command: ["airflow", "scheduler"]
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AWS_DEFAULT_REGION: eu-west-1
      S3_BUCKET: ${s3_bucket}
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - ./src:/opt/airflow/src
      - ./models:/opt/airflow/models
    depends_on:
      airflow-init:
        condition: service_completed_successfully
      postgres:
        condition: service_healthy
    restart: always

  mlflow:
    image: python:3.10-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
    working_dir: /mlflow
    environment:
      AWS_DEFAULT_REGION: eu-west-1
    command: >
      bash -c "
        pip install mlflow boto3 &&
        mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:///mlflow/mlruns --default-artifact-root s3://${s3_bucket}/mlflow-artifacts
      "
    restart: always

  inference-api:
    image: python:3.10-slim
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
    working_dir: /app
    environment:
      S3_BUCKET: ${s3_bucket}
      MODEL_PATH: models/random_forest
    command: >
      bash -c "
        pip install fastapi uvicorn boto3 scikit-learn pandas numpy pickle5 &&
        python -m uvicorn src.api.inference_api:app --host 0.0.0.0 --port 8000 --reload
      "
    restart: always

volumes:
  postgres-db-volume:
EOF

# Create directories
mkdir -p dags logs plugins src/api src/ml_pipeline

# Set permissions
chown -R ubuntu:ubuntu /home/ubuntu/ml_pipeline_project

# Create .env file
cat > .env << 'EOF'
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=/home/ubuntu/ml_pipeline_project
EOF

# Copy the EC2 DAG to dags directory
cp dags/ml_pipeline_ec2.py dags/

# Start services
cd /home/ubuntu/ml_pipeline_project
docker-compose up -d

# Wait for services to be ready
sleep 60

# Check service status
docker-compose ps

echo "ML Pipeline deployment completed!"
echo "Access Airflow at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
echo "Access MLflow at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
echo "Access Inference API at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"