# Minimal Bedrock-specific safety net. Before this, nothing watched Bedrock spend or
# invocation health for retro at all — only the blunt, whole-account $50/$150/$200
# billing alarms in daatan/terraform/monitoring.tf, at 24h granularity. That's what
# let the 2026-07-12 IAM AccessDenied incident (a missing model ARN on the
# truthmachine-ec2-role policy — see infra/iam/README.md §4) run silently until
# someone happened to notice extractions failing.
#
# Deliberately NOT doing here: reconstructing a dollar estimate from
# InputTokenCount/OutputTokenCount × a hardcoded price constant — that's fragile and
# silently wrong the moment Bedrock repricing happens. If per-service $ visibility is
# wanted later, the right primitive is an aws_budgets_budget with a
# Service = ["Amazon Bedrock"] cost filter, not more CloudWatch math.

# Reuse daatan's existing SNS topic — no duplicate subscription, same pattern as
# news-indexer/terraform/monitoring.tf.
#
# It must be the us-east-1 topic, and the provider alias below is the whole point:
# a CloudWatch alarm can only publish to a topic in its OWN region, and the alarms
# here are in us-east-1 because that is the only region AWS/Bedrock metrics exist
# in. Pointing them at daatan-infra-alerts (eu-central-1) is what kept both alarms
# from ever being created — PutMetricAlarm fails with "Invalid region eu-central-1
# specified. Only us-east-1 is supported", and terraform plan shows a clean
# "2 to add" right up until the apply. See #669.
data "aws_sns_topic" "infra_alerts" {
  provider = aws.us_east_1
  name     = "daatan-infra-alerts-us-east-1"
}

# Client-side errors on the live extractor model — the exact failure signature of a
# missing IAM model-ARN grant (AccessDenied surfaces as InvocationClientErrors, not a
# cost spike). Any non-zero count here in a 5-minute window is worth paging on: a
# healthy pipeline should see essentially none.
resource "aws_cloudwatch_metric_alarm" "extractor_invocation_client_errors" {
  provider            = aws.us_east_1
  alarm_name          = "retro-extractor-invocation-client-errors"
  alarm_description   = "Bedrock InvocationClientErrors on the live extractor model — likely a missing IAM model-ARN grant (see infra/iam/README.md §4) or a bad model ID"
  metric_name         = "InvocationClientErrors"
  namespace           = "AWS/Bedrock"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [data.aws_sns_topic.infra_alerts.arn]
  ok_actions          = [data.aws_sns_topic.infra_alerts.arn]

  dimensions = { ModelId = var.extractor_model_id }
}

# Dead-pipeline canary: the same 2026-07-12 incident could equally have surfaced as
# zero invocations (every call failing before it's even attempted) rather than a
# client-error count — this catches that shape too. A long window + "breaching" on
# missing data, since a genuinely idle pipeline (no candidate articles for hours) is
# not expected in normal operation on either the live oracle-api or batch
# truthmachine.service paths.
resource "aws_cloudwatch_metric_alarm" "extractor_invocations_zero" {
  provider            = aws.us_east_1
  alarm_name          = "retro-extractor-invocations-zero"
  alarm_description   = "No Bedrock invocations at all for the extractor model in 3h — pipeline may be stalled or silently failing upstream of the LLM call"
  metric_name         = "Invocations"
  namespace           = "AWS/Bedrock"
  statistic           = "Sum"
  period              = 10800
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching"
  alarm_actions       = [data.aws_sns_topic.infra_alerts.arn]
  ok_actions          = [data.aws_sns_topic.infra_alerts.arn]

  dimensions = { ModelId = var.extractor_model_id }
}
