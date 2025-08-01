# ML Pipeline Deployment Guide

This guide covers deploying the ML pipeline with Airflow orchestration and MLflow tracking, using S3 for model storage.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Airflow DAG   │───▶│  ML Pipeline    │───▶│   S3 Storage    │
│  (Orchestrator) │    │   (Training)    │    │   (Models)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     MLflow      │    │  Inference API  │
│   (Metadata)    │    │   (Tracking)    │    │   (Future)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

1. **AWS Account** with S3 bucket: `phase3-mlops-source-bucket-degen-1`
2. **AWS CLI** configured with appropriate permissions
3. **Docker & Docker Compose** installed
4. **Python 3.10+** with virtual environment

## S3 Bucket Setup

### Required S3 Permissions

Your AWS user needs the following permissions for the bucket:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::phase3-mlops-source-bucket-degen-1",
                "arn:aws:s3:::phase3-mlops-source-bucket-degen-1/*"
            ]
        }
    ]
}
```

### Bucket Structure

```
phase3-mlops-source-bucket-degen-1/
├── data/
│   └── ml-100k/
│       ├── u.data
│       ├── u.item
│       ├── u.user
│       └── ...
└── models/
    ├── logistic_regression/
    │   ├── model.pkl
    │   ├── scaler.pkl
    │   └── ...
    └── random_forest/
        ├── model.pkl
        ├── scaler.pkl
        └── ...
```

## Local Development Setup

### 1. Test S3 Integration

```bash
# Test S3 access
python scripts/test_s3_integration.py

# Upload dataset to S3
python scripts/upload_data_to_s3.py
```

### 2. Run Training with S3

```bash
# Activate virtual environment
source venv/bin/activate

# Run training (will save models to S3)
python scripts/train_modular.py
```

## Airflow Deployment

### 1. Local Airflow Setup

```bash
# Make setup script executable
chmod +x setup_airflow.sh

# Run setup
./setup_airflow.sh
```

### 2. Access Services

- **Airflow UI**: http://localhost:8080
  - Username: `airflow`
  - Password: `airflow`

- **MLflow UI**: http://localhost:5000

### 3. Configure Airflow Connections

In Airflow UI, go to Admin > Connections and add:

**AWS Connection:**
- Connection Id: `aws_default`
- Connection Type: `Amazon Web Services`
- Extra: `{"region_name": "us-east-1"}`

### 4. Trigger DAG

1. Go to Airflow UI
2. Find `ml_pipeline_training` DAG
3. Toggle it ON
4. Click "Trigger DAG"

## EC2 Deployment

### 1. EC2 Instance Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone <your-repo-url>
cd ml_pipeline_project
```

### 2. Configure Environment

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Create .env file
echo "AIRFLOW_UID=$(id -u)" > .env
echo "AIRFLOW_PROJ_DIR=$(pwd)" >> .env
```

### 3. Deploy Services

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f airflow-scheduler
```

### 4. Security Configuration

```bash
# Configure firewall (if needed)
sudo ufw allow 8080  # Airflow
sudo ufw allow 5000  # MLflow
sudo ufw enable
```

## Monitoring & Troubleshooting

### Check Service Health

```bash
# Check all services
docker-compose ps

# Check specific service logs
docker-compose logs airflow-webserver
docker-compose logs mlflow

# Check Airflow task logs
docker-compose exec airflow-scheduler airflow tasks list ml_pipeline_training
```

### Common Issues

1. **S3 Access Denied**
   - Verify AWS credentials
   - Check bucket permissions
   - Ensure bucket exists

2. **Airflow DAG Not Appearing**
   - Check DAG syntax: `python dags/ml_pipeline_dag.py`
   - Verify file permissions
   - Check Airflow logs

3. **MLflow Tracking Issues**
   - Verify MLflow server is running
   - Check tracking URI configuration
   - Ensure mlruns directory permissions

### Performance Tuning

```bash
# Increase Airflow worker resources
export AIRFLOW__CELERY__WORKER_CONCURRENCY=4

# Optimize Docker resources
docker-compose up -d --scale airflow-worker=2
```

## Production Considerations

1. **Security**
   - Use IAM roles instead of access keys
   - Enable HTTPS for web interfaces
   - Configure proper network security groups

2. **Scalability**
   - Use RDS for Airflow metadata database
   - Consider ECS/EKS for container orchestration
   - Implement auto-scaling for workers

3. **Monitoring**
   - Set up CloudWatch logging
   - Configure alerts for failed tasks
   - Monitor S3 costs and usage

4. **Backup & Recovery**
   - Regular database backups
   - S3 versioning for models
   - Disaster recovery procedures

## Next Steps

1. **Inference API**: Create FastAPI service to serve models from S3
2. **CI/CD Pipeline**: Automate deployment with GitHub Actions
3. **Model Registry**: Implement model versioning and promotion
4. **Data Validation**: Add data quality checks to pipeline