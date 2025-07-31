# Security Group for ML Pipeline
resource "aws_security_group" "ml_pipeline" {
  name_prefix = "${var.project_name}-ml-pipeline"
  vpc_id      = var.vpc_id

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Airflow
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # MLflow
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Inference API
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-ml-pipeline-sg"
    Environment = var.environment
  }
}

# IAM Role for EC2
resource "aws_iam_role" "ec2_ml_pipeline" {
  name = "${var.project_name}-ec2-ml-pipeline-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Policy for S3 access
resource "aws_iam_role_policy" "s3_access" {
  name = "${var.project_name}-s3-access"
  role = aws_iam_role.ec2_ml_pipeline.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket}",
          "arn:aws:s3:::${var.s3_bucket}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_ml_pipeline" {
  name = "${var.project_name}-ec2-ml-pipeline-profile"
  role = aws_iam_role.ec2_ml_pipeline.name
}

# Key Pair
resource "aws_key_pair" "ml_pipeline" {
  key_name   = "${var.project_name}-key"
  public_key = file("/home/fafa/.ssh/ml_pipeline.pub")
}
# EC2 Instance
resource "aws_instance" "ml_pipeline" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.ml_pipeline.key_name
  vpc_security_group_ids = [aws_security_group.ml_pipeline.id]
  subnet_id              = var.subnet_id
  iam_instance_profile   = aws_iam_instance_profile.ec2_ml_pipeline.name

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    s3_bucket = var.s3_bucket
  }))

  tags = {
    Name        = "${var.project_name}-ml-pipeline"
    Environment = var.environment
  }
}