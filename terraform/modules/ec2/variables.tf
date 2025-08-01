variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for EC2 instance"
  type        = string
}

variable "ami_id" {
  description = "AMI ID for EC2 instance"
  type        = string
  default     = "ami-0c1c30571d2dae5c9" # Ubuntu 22.04 LTS eu-west-1
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "m5.2xlarge"
}

variable "s3_bucket" {
  description = "S3 bucket name for ML artifacts"
  type        = string
  default     = "phase3-mlops-source-bucket-degen-1"
}