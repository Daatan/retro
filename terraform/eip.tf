# Static public IP for the Oracle box.
#
# Until 2026-08-27 this box had only an *ephemeral* public IP (3.120.185.111): AWS
# hands those back to the pool on stop, so any stop/start returned a different
# address and left oracle.daatan.com / bayes.daatan.com pointing at nothing until
# Route53 was updated by hand. That turned every maintenance window on this box into an
# avoidable outage — including the retro#436 encrypted-root-volume swap, which
# requires a stop by construction. An EIP survives stop/start, so the address is
# stable and those windows become DNS-neutral.
#
# Cost is neutral: AWS bills $0.005/hr for every public IPv4 address, in-use EIPs
# included, and this box was already paying that for the auto-assigned one.

resource "aws_eip" "oracle" {
  domain = "vpc"

  tags = {
    Name    = "truthmachine-oracle"
    Project = "truthmachine"
  }

  # Same reasoning as the instance's guard: releasing this allocation would break
  # both hostnames, and the address could not be reclaimed afterwards.
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_eip_association" "oracle" {
  instance_id   = aws_instance.oracle.id
  allocation_id = aws_eip.oracle.id
}

# The daatan.com zone is owned by platform/foundation — consumed here, never redefined.
data "aws_route53_zone" "daatan" {
  name         = "daatan.com."
  private_zone = false
}

# Both records predate this stack and are imported, not created. TTL was lowered
# from 300 to 60 ahead of the cutover so the one-time address change drained
# quickly; 60 is kept because it also bounds any future re-point.
resource "aws_route53_record" "oracle" {
  zone_id = data.aws_route53_zone.daatan.zone_id
  name    = "oracle.daatan.com"
  type    = "A"
  ttl     = 60
  records = [aws_eip.oracle.public_ip]
}

resource "aws_route53_record" "bayes" {
  zone_id = data.aws_route53_zone.daatan.zone_id
  name    = "bayes.daatan.com"
  type    = "A"
  ttl     = 60
  records = [aws_eip.oracle.public_ip]
}
