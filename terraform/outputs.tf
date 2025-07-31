# Outputs
output "ec2_public_ip" {
  value = module.ec2.public_ip
}

output "airflow_url" {
  value = module.ec2.airflow_url
}

output "mlflow_url" {
  value = module.ec2.mlflow_url
}

output "inference_api_url" {
  value = module.ec2.inference_api_url
}