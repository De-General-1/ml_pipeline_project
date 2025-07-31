#!/bin/bash

echo "🚀 Deploying ML Pipeline to EC2..."

# Check if SSH key exists
if [ ! -f ~/.ssh/ml_pipeline.pub ]; then
    echo "❌ SSH key not found. Generating new key pair..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/ml_pipeline -C "${USER}@ml-pipeline" -N ""
fi

# Initialize Terraform
echo "📦 Initializing Terraform..."
terraform init

# Plan deployment
echo "📋 Planning deployment..."
terraform plan

# Apply deployment
echo "🔄 Applying deployment..."
terraform apply -auto-approve

# Get outputs
echo "📊 Getting deployment information..."
EC2_IP=$(terraform output -raw ec2_public_ip)
AIRFLOW_URL=$(terraform output -raw airflow_url)
MLFLOW_URL=$(terraform output -raw mlflow_url)
INFERENCE_URL=$(terraform output -raw inference_api_url)

echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "🌐 Access URLs:"
echo "   - EC2 Instance: ssh -i ~/.ssh/ml_pipeline ubuntu@$EC2_IP"
echo "   - Airflow UI: $AIRFLOW_URL (airflow/airflow)"
echo "   - MLflow UI: $MLFLOW_URL"
echo "   - Inference API: $INFERENCE_URL"
echo ""
echo "📝 Next steps:"
echo "   1. Wait 5-10 minutes for services to start"
echo "   2. Upload your code to the EC2 instance"
echo "   3. Test the inference API"
echo ""