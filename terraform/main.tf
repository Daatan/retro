terraform {
  required_version = ">= 1.5"

  backend "s3" {
    bucket       = "daatan-terraform-state"
    key          = "retro/terraform.tfstate"
    region       = "eu-central-1"
    use_lockfile = true # S3-native state locking (replaces deprecated dynamodb_table)
    encrypt      = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "eu-central-1"
  # No default_tags here on purpose: this stack imports a pre-existing,
  # out-of-band box and the first PR must be a literal no-op plan. Tag
  # standardisation (e.g. ManagedBy) is a deliberate later change.
}
