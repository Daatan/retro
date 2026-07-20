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

# AWS/Bedrock metrics for the models this pipeline calls (Nova Micro/Lite, the Haiku
# extractor override) all publish in us-east-1 — confirmed via live get-metric-data,
# not assumed — regardless of the eu-central-1 host running the calls (tm/config.py's
# aws_region default is itself "us-east-1"). Same pattern as daatan's and
# news-indexer's main.tf, which both already needed this alias for their billing alarms.
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}
