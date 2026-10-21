# Daily EBS snapshots of the Oracle box (Daatan/platform#42 item 2).
#
# Until this file the box's root volume (vol-069ea3eed75d8efc8, 30 GB gp3, encrypted with
# the AWS-managed alias/aws/ebs key) had zero snapshots, and a lot of what lives on it is
# not rebuildable from git: oracle_log.txt (the only copy of the Oracle's application log,
# deliberately not rotated), the paper-bot ledgers (retro#620/#918), the diskcache stores
# (jev_values, extraction_memo, settlement_verdict_cache, ...), the oracle-series node log
# and hand-made systemd drop-ins. The platform#42 inventory comment lists them all.
#
# Targeting: the root volume carries NO tags, so a VOLUME-type policy would match nothing
# and silently take zero snapshots. This policy targets the INSTANCE by its existing Name
# tag instead (multi-volume, crash-consistent snapshot of every attached volume incl. the
# boot volume), which also avoids touching aws_instance.oracle at all. If the box is ever
# rebuilt, keep the Name tag or this policy stops matching.
#
# KMS: snapshots of an alias/aws/ebs volume stay encrypted with that key — no grants
# needed for DLM, but they cannot be shared or copied to another account. An off-account
# DR copy (daatan#1957) would need a customer-managed key first.
#
# Cost (eu-central-1, $0.05/GB-month, incremental): first snapshot ~ used blocks (~21 GB,
# ~$1.05/mo), daily deltas ~0.3-1 GB → roughly $1.3-1.6/month at 7-day retention.
#
# Apply with -target only (all three addresses), never a blanket apply:
#   terraform plan  -target=aws_iam_role.dlm -target=aws_iam_role_policy_attachment.dlm \
#                   -target=aws_dlm_lifecycle_policy.oracle_daily
#   terraform apply -target=... (same three) — plan must read "3 to add, 0 to change, 0 to destroy".

resource "aws_iam_role" "dlm" {
  name        = "truthmachine-dlm-lifecycle-role"
  description = "Execution role for the Oracle box DLM snapshot policy (platform#42)"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "dlm.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Project = "truthmachine"
  }
}

# AWS-managed policy for EBS snapshot management; includes ec2:CreateSnapshots, which
# INSTANCE-type (multi-volume) policies need.
resource "aws_iam_role_policy_attachment" "dlm" {
  role       = aws_iam_role.dlm.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole"
}

resource "aws_dlm_lifecycle_policy" "oracle_daily" {
  description        = "Oracle box truthmachine-pipeline daily snapshots keep 7" # DLM allows only [0-9A-Za-z _-]
  execution_role_arn = aws_iam_role.dlm.arn
  state              = "ENABLED"

  policy_details {
    policy_type    = "EBS_SNAPSHOT_MANAGEMENT"
    resource_types = ["INSTANCE"]

    target_tags = {
      Name = "truthmachine-pipeline"
    }

    parameters {
      exclude_boot_volume = false
      no_reboot           = true
    }

    schedule {
      name = "daily-0300utc-keep7"

      # 03:00 UTC: quiet window — after certbot/logrotate (00:00) and before the 06:30
      # oracle-series cron and the morning forecast burst.
      create_rule {
        interval      = 24
        interval_unit = "HOURS"
        times         = ["03:00"]
      }

      retain_rule {
        count = 7
      }

      copy_tags = true

      tags_to_add = {
        Source  = "dlm"
        Project = "truthmachine"
        Policy  = "oracle-daily"
      }
    }
  }

  tags = {
    Name    = "truthmachine-oracle-daily-snapshots"
    Project = "truthmachine"
  }
}
