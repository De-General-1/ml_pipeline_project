#!/usr/bin/env python3
"""
Test S3 integration for ML pipeline
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_s3_access():
    """Test S3 bucket access"""
    bucket_name = "phase3-mlops-source-bucket-degen-1"
    
    print(f"🔍 Testing S3 access to bucket: {bucket_name}")
    
    try:
        session = boto3.Session(profile_name='degen-mlops')
        s3_client = session.client('s3')
        
        # Test bucket access
        response = s3_client.head_bucket(Bucket=bucket_name)
        print("✅ Bucket access successful")
        
        # Test list objects
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=5)
        if 'Contents' in response:
            print(f"📁 Found {len(response['Contents'])} objects in bucket")
            for obj in response['Contents'][:3]:
                print(f"   - {obj['Key']}")
        else:
            print("📁 Bucket is empty")
        
        # Test write permission with a small test file
        test_key = "test/access_test.txt"
        test_content = "This is a test file to verify S3 write access"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content.encode('utf-8')
        )
        print("✅ Write access successful")
        
        # Clean up test file
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
        print("✅ Test file cleaned up")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            print("❌ Access denied. Please check your AWS credentials and bucket permissions.")
            print("   Required permissions: s3:GetObject, s3:PutObject, s3:ListBucket")
        elif error_code == 'NoSuchBucket':
            print(f"❌ Bucket {bucket_name} does not exist")
        else:
            print(f"❌ S3 error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def show_aws_config():
    """Show current AWS configuration"""
    print("\n🔧 AWS Configuration:")
    
    # Check AWS credentials
    try:
        session = boto3.Session(profile_name='degen-mlops')
        credentials = session.get_credentials()
        if credentials:
            print(f"   - Access Key: {credentials.access_key[:8]}...")
            print(f"   - Region: {session.region_name or 'Not set'}")
        else:
            print("   - No AWS credentials found")
    except Exception as e:
        print(f"   - Error getting credentials: {e}")
    
    # Check environment variables
    aws_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
    for var in aws_env_vars:
        value = os.getenv(var)
        if value:
            if 'KEY' in var:
                print(f"   - {var}: {value[:8]}...")
            else:
                print(f"   - {var}: {value}")
        else:
            print(f"   - {var}: Not set")

if __name__ == "__main__":
    print("🚀 Testing S3 Integration for ML Pipeline\n")
    
    show_aws_config()
    
    if test_s3_access():
        print("\n✅ S3 integration test passed!")
        print("\n📝 Next steps:")
        print("   1. Run: python scripts/upload_data_to_s3.py")
        print("   2. Run: python scripts/train_modular.py")
        print("   3. Check S3 bucket for uploaded models")
    else:
        print("\n❌ S3 integration test failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Check AWS credentials: aws configure")
        print("   2. Verify bucket permissions")
        print("   3. Ensure bucket exists and is accessible")