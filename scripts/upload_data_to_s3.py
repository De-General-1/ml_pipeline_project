#!/usr/bin/env python3
"""
Script to upload MovieLens dataset to S3 bucket
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_pipeline import DataLoader, PipelineConfig

def upload_file_to_s3(file_path, bucket_name, s3_key):
    """Upload a file to S3"""
    session = boto3.Session(profile_name='degen-mlops')
    s3_client = session.client('s3')
    
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"✅ Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        print(f"❌ Error uploading {file_path}: {e}")
        return False

def upload_dataset_to_s3():
    """Upload MovieLens dataset to S3"""
    
    bucket_name = "phase3-mlops-source-bucket-degen-1"
    s3_prefix = "data/ml-100k"
    
    print(f"Uploading MovieLens dataset to S3 bucket: {bucket_name}")
    
    # Initialize data loader to ensure data is downloaded
    data_loader = DataLoader()
    
    # Load data to ensure it's downloaded locally
    try:
        ratings_df, movies_df, users_df = data_loader.load_data()
        print(f"Data loaded successfully - Ratings: {ratings_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Files to upload
    data_files = [
        'u.data',      # ratings
        'u.item',      # movies
        'u.user',      # users
        'u.info',      # dataset info
        'u.genre',     # genres
        'u.occupation' # occupations
    ]
    
    # Upload each file
    success_count = 0
    for file_name in data_files:
        local_path = os.path.join(data_loader.data_root_dir, file_name)
        s3_key = f"{s3_prefix}/{file_name}"
        
        if os.path.exists(local_path):
            if upload_file_to_s3(local_path, bucket_name, s3_key):
                success_count += 1
        else:
            print(f"⚠️  File not found: {local_path}")
    
    print(f"\n📊 Upload Summary:")
    print(f"   - Files uploaded: {success_count}/{len(data_files)}")
    print(f"   - S3 bucket: {bucket_name}")
    print(f"   - S3 prefix: {s3_prefix}")
    
    if success_count == len(data_files):
        print("✅ All files uploaded successfully!")
        return True
    else:
        print("⚠️  Some files failed to upload")
        return False

def verify_s3_upload():
    """Verify files were uploaded to S3"""
    bucket_name = "phase3-mlops-source-bucket-degen-1"
    s3_prefix = "data/ml-100k"
    
    session = boto3.Session(profile_name='degen-mlops')
    s3_client = session.client('s3')
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=s3_prefix
        )
        
        if 'Contents' in response:
            print(f"\n📁 Files in S3 bucket {bucket_name}/{s3_prefix}:")
            for obj in response['Contents']:
                size_mb = obj['Size'] / (1024 * 1024)
                print(f"   - {obj['Key']} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"No files found in S3 bucket {bucket_name}/{s3_prefix}")
            return False
            
    except ClientError as e:
        print(f"Error verifying S3 upload: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting dataset upload to S3...")
    
    # Upload dataset
    if upload_dataset_to_s3():
        # Verify upload
        verify_s3_upload()
        print("\n✅ Dataset upload completed successfully!")
    else:
        print("\n❌ Dataset upload failed!")
        sys.exit(1)