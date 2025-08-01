output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.ml_pipeline.id
}

output "public_ip" {
  description = "EC2 instance public IP"
  value       = aws_instance.ml_pipeline.public_ip
}

output "private_ip" {
  description = "EC2 instance private IP"
  value       = aws_instance.ml_pipeline.private_ip
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.ml_pipeline.id
}

output "airflow_url" {
  description = "Airflow web UI URL"
  value       = "http://${aws_instance.ml_pipeline.public_ip}:8080"
}

output "mlflow_url" {
  description = "MLflow web UI URL"
  value       = "http://${aws_instance.ml_pipeline.public_ip}:5000"
}

output "inference_api_url" {
  description = "Inference API URL"
  value       = "http://${aws_instance.ml_pipeline.public_ip}:8000"
}