# 🎬 Movie Engagement Prediction ML Pipeline

A complete MLOps pipeline that predicts user engagement with movies using machine learning, featuring automated training with Airflow, experiment tracking with MLflow, and a beautiful web interface for testing predictions.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   Airflow       │
│  (Web Interface)│    │ (Inference API) │    │ (Orchestration) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   S3 Storage    │◄───┤     MLflow      │◄───┤   PostgreSQL    │
│   (Models)      │    │   (Tracking)    │    │   (Metadata)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Features

- **🤖 ML Pipeline**: Automated training with Logistic Regression and Random Forest
- **📊 Experiment Tracking**: MLflow integration with S3 artifact storage
- **🔄 Orchestration**: Airflow DAGs for automated model training
- **🌐 Web Interface**: Interactive Streamlit app for testing predictions
- **🔗 REST API**: FastAPI service for programmatic access
- **☁️ Cloud Integration**: AWS S3 for model storage and artifacts
- **🐳 Containerization**: Docker support for easy deployment
- **🔧 CI/CD**: GitHub Actions for automated testing and deployment

## 📋 Prerequisites

- **Python 3.10+**
- **AWS Account** with S3 access
- **AWS CLI** configured with `degen-mlops` profile
- **Docker & Docker Compose** (for deployment)
- **Git**

## 🛠️ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/De-General-1/ml_pipeline_project.git
cd ml_pipeline_project
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure AWS
```bash
# Configure AWS CLI with your profile
aws configure --profile degen-mlops
# Enter your AWS Access Key ID, Secret Key, and region (eu-west-1)
```

### 4. Test S3 Integration
```bash
python scripts/test_s3_integration.py
```

## 🎬 Using the Web Application

### Start the Interactive Web Interface
```bash
cd ml_pipeline_project
source venv/bin/activate
export AWS_PROFILE=degen-mlops
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Then open: **http://localhost:8501**

### Web App Features

#### 🎯 **User Profile Settings**
- **Age**: 7-73 years (slider)
- **Gender**: Male/Female
- **Occupation**: 21 options (student, engineer, teacher, etc.)

#### 🎬 **Movie Details**
- **Release Year**: 1920-2000 (slider)
- **Genres**: Multi-select from 19 genres (Action, Comedy, Drama, etc.)

#### 📊 **Prediction Results**
- **Engagement Probability**: Percentage likelihood of user engagement
- **Prediction**: Will Engage / Won't Engage
- **Confidence Score**: Model confidence in prediction
- **Interactive Gauge**: Visual probability display
- **Feature Importance**: Key factors influencing prediction

#### 📈 **Analytics**
- **Prediction History**: Track all your test predictions
- **Probability Distribution**: Histogram of engagement scores
- **Model Information**: Technical details about the loaded model

## 🔧 Local Development

### Train Models Locally
```bash
# Upload dataset to S3
python scripts/upload_data_to_s3.py

# Train models and save to S3
python scripts/train_modular.py
```

### Test Model Directly
```bash
# Test model without web interface
python test_model_direct.py
```

### Run FastAPI Server
```bash
export AWS_PROFILE=degen-mlops
uvicorn src.api.inference_api:app --host 0.0.0.0 --port 8000 --reload
```

## ☁️ Cloud Deployment

### Deploy to EC2
```bash
cd terraform
chmod +x deploy.sh
./deploy.sh
```

### Access Services
- **Airflow**: http://EC2_IP:8080 (airflow/airflow)
- **MLflow**: http://EC2_IP:5000
- **Inference API**: http://EC2_IP:8000

### Run Training Pipeline
1. Access Airflow UI
2. Enable `ml_pipeline_ec2_training` DAG
3. Trigger DAG execution
4. Monitor progress in MLflow

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/test_inference_api.py -v
```

### Test API Endpoints
```bash
python test_inference_api.py http://localhost:8000
```

## 📁 Project Structure

```
ml_pipeline_project/
├── src/
│   ├── api/
│   │   └── inference_api.py      # FastAPI inference service
│   └── ml_pipeline/
│       ├── data_loader.py        # Data loading utilities
│       ├── preprocessor.py       # Data preprocessing
│       └── model_trainer.py      # Model training logic
├── dags/
│   ├── ml_pipeline_dag.py        # Local Airflow DAG
│   └── ml_pipeline_ec2.py        # EC2 Airflow DAG
├── scripts/
│   ├── train_modular.py          # Training script
│   ├── upload_data_to_s3.py      # Data upload utility
│   └── test_s3_integration.py    # S3 connectivity test
├── terraform/
│   ├── main.tf                   # Infrastructure as code
│   ├── modules/ec2/              # EC2 module
│   └── deploy.sh                 # Deployment script
├── tests/
│   └── test_inference_api.py     # API tests
├── .github/workflows/
│   └── inference-api-ci.yml      # CI/CD pipeline
├── streamlit_app.py              # Web interface
├── docker-compose.yml            # Local services
├── Dockerfile                    # Container definition
└── requirements.txt              # Python dependencies
```

## 🎯 Model Details

### Dataset
- **Source**: MovieLens 100K dataset
- **Users**: 943 users with demographics
- **Movies**: 1,682 movies with genres and release years
- **Ratings**: 100,000 ratings (1-5 scale)

### Features
- **User Features**: Age, gender, occupation (84 encoded features)
- **Movie Features**: Release year, genres (19 genre categories)
- **Target**: Binary engagement (rating ≥ 4)

### Models
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble model (default for predictions)
- **Evaluation**: Accuracy, F1-score, precision, recall

### Performance
- **Random Forest**: ~64% accuracy, ~63% F1-score
- **Logistic Regression**: ~62% accuracy, ~61% F1-score

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
- **Triggers**: Push/PR to `feat/inference-api` or `main`
- **Tests**: Automated pytest execution
- **Build**: Docker image creation
- **Deploy**: Push to Amazon ECR

### Required Secrets
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=eu-west-1
```

## 🛡️ Security & Best Practices

- **AWS IAM Roles**: Used in production instead of access keys
- **Environment Variables**: Sensitive data stored securely
- **Input Validation**: Pydantic models for API validation
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging throughout the pipeline

## 🚨 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Check S3 access
aws s3 ls s3://phase3-mlops-source-bucket-degen-1/ --profile degen-mlops
```

**Streamlit Connection Issues**
```bash
# Check if model artifacts exist
python test_model_direct.py
```

**Docker Issues**
```bash
# Restart services
docker-compose down && docker-compose up -d
```

## 📚 API Documentation

### Endpoints

**GET /** - Root endpoint
**GET /health** - Health check
**POST /predict** - Make prediction
**POST /reload-model** - Reload model from S3
**GET /model-info** - Model information

### Example Request
```json
{
  "user_id": 123,
  "movie_id": 456,
  "age": 25,
  "gender": "M",
  "occupation": "student",
  "genres": ["Action", "Comedy"],
  "release_year": 1995
}
```

### Example Response
```json
{
  "user_id": 123,
  "movie_id": 456,
  "engagement_probability": 0.463,
  "engagement_prediction": 0,
  "model_version": "random_forest_v1",
  "prediction_timestamp": "2024-01-01T12:00:00"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feat/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feat/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MovieLens dataset by GroupLens Research
- FastAPI and Streamlit communities
- AWS for cloud infrastructure
- MLflow for experiment tracking