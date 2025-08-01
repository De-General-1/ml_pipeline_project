# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.project_name}-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "${var.project_name}-igw"
    Environment = var.environment
  }
}

# Public Subnet
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[0]
  availability_zone       = var.availability_zones[0]
  map_public_ip_on_launch = true

  tags = {
    Name        = "${var.project_name}-public-subnet"
    Environment = var.environment
  }
}

# Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "${var.project_name}-public-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# EC2 Module
module "ec2" {
  source = "./modules/ec2"

  project_name = var.project_name
  environment  = var.environment
  vpc_id       = aws_vpc.main.id
  subnet_id    = aws_subnet.public.id
}

module "inference_api_repo" {
  source = "./modules/ecr"
  providers = {
    aws = aws.primary
  }
  repository_name = var.inference_api_repo_name
}

module "inference_api_ecs" {
  source = "./modules/ecs"
  providers = {
    aws = aws.primary
  }
  ecs_cluster_name           = "inference_api_cluster"
  container_insights_enabled = "enabled"

  ecs_td_family          = "inference"
  assign_public_ip       = true
  container_port         = var.inference_container_port
  cpu_size               = 1024
  desired_count          = 1
  ecs_service_name       = var.inference_ecs_service_name
  target_group_arn       = module.inference_target_group.target_group_arn
  ecs_service_sg         = [aws_security_group.inference_service_sg.id]
  ecs_service_subnets    = module.project_vpc.public_subnet_ids
  host_port              = var.inference_container_port
  image_uri              = "664418998745.dkr.ecr.eu-west-1.amazonaws.com/mlops/inference:latest"
  mem_size               = 4096
  task_name              = "inference-api"
  aws_region             = "eu-west-1"
  log_group_name         = "inference-api"
  alb_http_listener_arn  = module.inference_api_load_balancer.http_alb_listener_arn
  alb_https_listener_arn = module.inference_api_load_balancer.https_alb_listener_arn
  elb_name               = var.inference_api_alb_name
  environment_variables = [
    {
      name  = "MFLOW_SERVER_IP"
      value = module.ec2.public_ip
    },
  ]
}

# # Outputs
output "ec2_public_ip" {
  value = module.ec2.public_ip
}

# output "airflow_url" {
#   value = module.ec2.airflow_url
# }

# output "mlflow_url" {
#   value = module.ec2.mlflow_url
# }

# output "inference_api_url" {
#   value = module.ec2.inference_api_url
# }
