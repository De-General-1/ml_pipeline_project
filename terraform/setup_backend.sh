#!/bin/bash

echo "🔧 Setting up Terraform S3 backend..."

export AWS_PROFILE=degen-mlops

# Create S3 bucket for state
echo "📦 Creating S3 bucket for Terraform state..."
aws s3 mb s3://phase3-mlops-terraform-state-degen-1 --region eu-west-1

# Enable versioning
echo "🔄 Enabling versioning..."
aws s3api put-bucket-versioning \
    --bucket phase3-mlops-terraform-state-degen-1 \
    --versioning-configuration Status=Enabled

# Create DynamoDB table for state locking
echo "🔒 Creating DynamoDB table for state locking..."
aws dynamodb create-table \
    --table-name terraform-state-lock \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
    --region eu-west-1

echo "✅ Backend setup complete!"
echo "You can now run: terraform init"