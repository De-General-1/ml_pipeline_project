terraform {
  backend "s3" {
    bucket         = "phase3-mlops-terraform-state-degen-1"
    key            = "ml-pipeline/terraform.tfstate"
    region         = "eu-west-1"
    profile        = "degen-mlops"
    encrypt      = true
    use_lockfile = true
  }
}