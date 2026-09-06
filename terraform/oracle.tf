# The retro/TruthMachine Oracle box — serves oracle.daatan.com (Oracle FastAPI)
# + bayes.daatan.com (BayesOracle) + the truthmachine.service batch pipeline.
# Hand-provisioned originally; imported here so it is rebuildable and auditable.
#
# Config generated from the live instance via -generate-config-out, then cleaned.
# The security group (sg-0c9c7cee5ebcf853d, "openclaw-sg") is referenced by id but
# intentionally NOT managed by this stack — it predates this terraform.
resource "aws_instance" "oracle" {
  ami                         = "ami-023c2be60b92b00a3"
  instance_type               = "t4g.small"
  availability_zone           = "eu-central-1c"
  subnet_id                   = "subnet-0eadf1fc3ba9d9339"
  iam_instance_profile        = "truthmachine-instance-profile"
  vpc_security_group_ids      = ["sg-0c9c7cee5ebcf853d"]
  associate_public_ip_address = true
  private_ip                  = "172.31.13.202"
  source_dest_check           = true
  ebs_optimized               = false
  monitoring                  = false

  tags = {
    Name    = "truthmachine-pipeline"
    Project = "truthmachine"
  }

  credit_specification {
    cpu_credits = "unlimited"
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
    instance_metadata_tags      = "disabled"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 30
    iops                  = 3000
    throughput            = 125
    encrypted             = true
    delete_on_termination = false
  }

  # Hard guard: this is a load-bearing box with no automated rebuild path yet.
  # Refuse to destroy it, and don't let an AMI/user_data drift trigger a replace.
  lifecycle {
    prevent_destroy = true
    ignore_changes  = [ami, user_data]
  }
}

# Note: a 2 GiB swapfile (/swapfile) was added on the running instance via SSM on
# 2026-07-16 — a truthmachine.service OOM-kill mitigation. That's OS-level state
# this resource doesn't (and, per ignore_changes above, can't) model or enforce; it
# only exists on this specific box, not in any image or bootstrap script. See the
# "Swap" note in docs/ARCHITECTURE.md § Infrastructure for the incident + rationale,
# and re-provision it by hand if this instance is ever rebuilt.
