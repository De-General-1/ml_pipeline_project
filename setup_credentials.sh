#!/bin/bash

echo "🔐 Setting up AWS Credentials for ML Pipeline"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Installing..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
fi

# Option 1: Use AWS CLI configure
echo ""
echo "Option 1: Configure AWS CLI (Recommended)"
echo "This will store credentials in ~/.aws/credentials"
read -p "Do you want to configure AWS CLI? (y/n): " configure_cli

if [[ $configure_cli == "y" ]]; then
    aws configure
    echo "✅ AWS CLI configured"
fi

# Option 2: Set environment variables
echo ""
echo "Option 2: Set environment variables for this session"
read -p "Do you want to set environment variables? (y/n): " set_env

if [[ $set_env == "y" ]]; then
    read -p "Enter AWS Access Key ID: " access_key
    read -s -p "Enter AWS Secret Access Key: " secret_key
    echo ""
    read -p "Enter AWS Region (default: us-east-1): " region
    region=${region:-us-east-1}
    
    export AWS_ACCESS_KEY_ID=$access_key
    export AWS_SECRET_ACCESS_KEY=$secret_key
    export AWS_DEFAULT_REGION=$region
    
    # Save to .env file for Docker Compose
    echo "AWS_ACCESS_KEY_ID=$access_key" >> .env
    echo "AWS_SECRET_ACCESS_KEY=$secret_key" >> .env
    echo "AWS_DEFAULT_REGION=$region" >> .env
    
    echo "✅ Environment variables set and saved to .env file"
fi

# Test S3 access
echo ""
echo "🧪 Testing S3 access..."
python scripts/test_s3_integration.py

echo ""
echo "✅ Credentials setup complete!"
echo "You can now run: python scripts/upload_data_to_s3.py"