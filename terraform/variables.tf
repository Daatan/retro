# The Bedrock model ID the live oracle-api service uses for article extraction —
# pinned by the systemd drop-in infra/oracle-api.service.d/extractor-model.conf (see
# infra/iam/README.md §4) and matching tm/config.py's `extractor_model` default.
# Declared here, not just on the host, so monitoring.tf's alarm dimensions can't
# silently drift from what's actually deployed the way the systemd drop-in and the
# IAM allowlist already did once (2026-07-12) — and did again after the Haiku 5.5
# switch (retro#921 → retro#1008). **Any extractor model switch must update this
# default and run a targeted apply of the two monitoring.tf extractor alarms.**
# Value = the ModelId dimension CloudWatch reports (the model string without the
# `bedrock/` prefix); confirm with `aws cloudwatch list-metrics --namespace AWS/Bedrock
# --region us-east-1`.
variable "extractor_model_id" {
  description = "Bedrock ModelId (as reported in CloudWatch's AWS/Bedrock dimensions) the live oracle-api extractor is currently pinned to."
  type        = string
  default     = "us.anthropic.claude-haiku-5-5"
}
