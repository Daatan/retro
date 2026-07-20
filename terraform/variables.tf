# The Bedrock model ID the live oracle-api service uses for article extraction —
# today an env-var override (systemd drop-in, see infra/iam/README.md §4) on top of
# tm/config.py's Nova Lite default. Declared here, not just on the host, so
# monitoring.tf's alarm dimensions can't silently drift from what's actually deployed
# the way the systemd drop-in and the IAM allowlist already did once (2026-07-12).
variable "extractor_model_id" {
  description = "Bedrock ModelId (as reported in CloudWatch's AWS/Bedrock dimensions) the live oracle-api extractor is currently pinned to."
  type        = string
  default     = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}
