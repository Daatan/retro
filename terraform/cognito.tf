# Cognito user pool — Authorization Server for the Oracle MCP endpoint (/mcp).
#
# The Oracle is an OAuth 2.1 Resource Server (see api/src/forecast_api/mcp_auth.py
# and docs/ORACLE_MCP.md); this pool mints the JWT access tokens it verifies.
# Two app clients: the Claude MCP client (Authorization Code + PKCE, human login
# via the DCR façade in api/src/forecast_api/mcp_dcr.py) and daatan's M2M client
# (client_credentials).
#
# Login: NATIVE Cognito accounts (admin-invite only — `admin_create_user_config`
# below) PLUS Google federation (self-service — see the note below).
# `allow_admin_create_user_only` only gates the native username/password sign-up
# form; it does NOT restrict federated IdP logins. Anyone with a Google account
# can sign in via the Claude client once the Google IdP is active — that is
# intentional (self-service trader onboarding), not a gap. Native admin-created
# accounts (e.g. the initial mark@daatan.com) keep working unaffected.
#
# ── NOT a blanket apply ──────────────────────────────────────────────────────
# These are NEW resources. Create them with -target, one at a time, per the
# workspace terraform rule:
#   terraform apply -target=aws_cognito_user_pool.oracle_mcp
#   terraform apply -target=aws_cognito_resource_server.oracle_mcp
#   terraform apply -target=aws_cognito_user_pool_domain.oracle_mcp
#   terraform apply -target=aws_cognito_identity_provider.google
#   terraform apply -target=aws_cognito_user_pool_client.claude
#   terraform apply -target=aws_cognito_user_pool_client.m2m
# (The pool, resource server, domain, and m2m client are already applied. This
#  change adds the Google IdP and updates the claude client's
#  supported_identity_providers, so apply just those two:
#  -target=aws_cognito_identity_provider.google and
#  -target=aws_cognito_user_pool_client.claude — the IdP must apply first, the
#  client's depends_on enforces that ordering.)
#
# ── Manual prerequisite (Google federation) ───────────────────────────────────
# Requires a Google OAuth 2.0 client (Cloud Console — outside this repo's/any
# CLI's reach, a human has to click through it) whose authorized redirect URI is
# https://<domain>.auth.eu-central-1.amazoncognito.com/oauth2/idpresponse. Store
# {client_id, client_secret} as JSON in Secrets Manager under
# var.google_oauth_secret_id BEFORE running `terraform plan` — the data source
# below fails the plan if the secret doesn't exist yet. The secret is read via a
# data source so it never lands in the tf files or state as a literal.

variable "cognito_domain_prefix" {
  description = "Globally-unique prefix for the Cognito hosted-UI domain."
  type        = string
  default     = "daatan-oracle"
}

variable "google_oauth_secret_id" {
  description = "Secrets Manager secret id holding {client_id, client_secret} JSON for the Google IdP."
  type        = string
  default     = "daatan/cognito-google-oauth"
}

variable "claude_callback_urls" {
  description = "Exact OAuth redirect URIs for the Claude MCP client (Cognito matches these exactly — no port wildcards)."
  type        = list(string)
  # https://claude.ai/api/mcp/auth_callback covers claude.ai web + Claude Desktop
  # (both use this single fixed callback). The localhost entry is for Claude Code
  # CLI run with a FIXED port — Cognito can't match Claude Code's default
  # ephemeral loopback port, so pin it (complete login on :8080).
  default = [
    "https://claude.ai/api/mcp/auth_callback",
    "http://localhost:8080/callback",
  ]
}

# ── Google federation ──────────────────────────────────────────────────────────
# Requires var.google_oauth_secret_id to exist in Secrets Manager first (see the
# header note) — this data source fails the plan otherwise.
data "aws_secretsmanager_secret_version" "google_oauth" {
  secret_id = var.google_oauth_secret_id
}

locals {
  google_oauth = jsondecode(data.aws_secretsmanager_secret_version.google_oauth.secret_string)
}

resource "aws_cognito_user_pool" "oracle_mcp" {
  name = "oracle-mcp"

  # Email is the account key. Native accounts only.
  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]

  # Admin-invite only: no public self-signup. Approved traders are created with
  # `cognito-idp admin-create-user`; a /mcp token spends LLM + paid-search
  # credits, so open registration is not acceptable.
  admin_create_user_config {
    allow_admin_create_user_only = true
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  tags = {
    Project = "truthmachine"
    Name    = "oracle-mcp"
  }
}

# Hosted UI domain — hosts the login/consent flow and the OAuth2 token endpoint.
resource "aws_cognito_user_pool_domain" "oracle_mcp" {
  domain       = var.cognito_domain_prefix
  user_pool_id = aws_cognito_user_pool.oracle_mcp.id
}

# Custom scopes → tools. identifier "oracle-mcp" makes the full scope strings
# "oracle-mcp/read" and "oracle-mcp/forecast" (matched in mcp_auth.py).
resource "aws_cognito_resource_server" "oracle_mcp" {
  identifier   = "oracle-mcp"
  name         = "oracle-mcp"
  user_pool_id = aws_cognito_user_pool.oracle_mcp.id

  scope {
    scope_name        = "read"
    scope_description = "Read-only Oracle tools (market lookup, search, leaderboard, bayes)."
  }
  scope {
    scope_name        = "forecast"
    scope_description = "Expensive Oracle tools (forecast, polymarket_edge — LLM + search spend)."
  }
}

# Google as the upstream identity provider.
resource "aws_cognito_identity_provider" "google" {
  user_pool_id  = aws_cognito_user_pool.oracle_mcp.id
  provider_name = "Google"
  provider_type = "Google"

  provider_details = {
    client_id        = local.google_oauth.client_id
    client_secret    = local.google_oauth.client_secret
    authorize_scopes = "openid email profile"
  }

  attribute_mapping = {
    email    = "email"
    username = "sub"
  }
}

# Public client for the Claude MCP connector: Authorization Code + PKCE, no secret.
# The DCR façade (api/src/forecast_api/mcp_dcr.py) hands this client's id back to
# Claude on /register; its id goes in COGNITO_CLAUDE_CLIENT_ID + the allow-list.
resource "aws_cognito_user_pool_client" "claude" {
  name         = "oracle-mcp-claude"
  user_pool_id = aws_cognito_user_pool.oracle_mcp.id

  generate_secret = false # public client — PKCE, no client secret

  allowed_oauth_flows                  = ["code"]
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_scopes = [
    "openid",
    "email",
    "oracle-mcp/read",
    "oracle-mcp/forecast",
  ]
  callback_urls                = var.claude_callback_urls
  supported_identity_providers = ["COGNITO", "Google"] # native admin-created accounts + self-service Google sign-in

  # Short-lived access tokens; refresh handles longevity.
  access_token_validity  = 60 # minutes
  id_token_validity      = 60
  refresh_token_validity = 30 # days
  token_validity_units {
    access_token  = "minutes"
    id_token      = "minutes"
    refresh_token = "days"
  }

  explicit_auth_flows = ["ALLOW_REFRESH_TOKEN_AUTH", "ALLOW_USER_SRP_AUTH"]

  depends_on = [aws_cognito_resource_server.oracle_mcp, aws_cognito_identity_provider.google]
}

# Machine-to-machine client for daatan's own agents: client_credentials only.
# M2M tokens may carry custom scopes only (no openid/email). Both scopes are
# granted because the Resource Server enforces a global `oracle-mcp/read` floor on
# EVERY /mcp request (AuthSettings.required_scopes in mcp_server.py) — a
# forecast-only token would be 403'd before reaching any tool. The M2M client must
# therefore request both at token time: scope=oracle-mcp/read oracle-mcp/forecast.
resource "aws_cognito_user_pool_client" "m2m" {
  name         = "oracle-mcp-m2m"
  user_pool_id = aws_cognito_user_pool.oracle_mcp.id

  generate_secret = true # confidential client

  allowed_oauth_flows                  = ["client_credentials"]
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_scopes                 = ["oracle-mcp/read", "oracle-mcp/forecast"]

  access_token_validity = 60
  token_validity_units {
    access_token = "minutes"
  }

  depends_on = [aws_cognito_resource_server.oracle_mcp]
}

# ── Outputs — plug these into the Oracle box's /home/ubuntu/truthmachine/.env ──
output "cognito_user_pool_id" {
  value = aws_cognito_user_pool.oracle_mcp.id
}

output "cognito_issuer" {
  value = "https://cognito-idp.eu-central-1.amazonaws.com/${aws_cognito_user_pool.oracle_mcp.id}"
}

output "cognito_allowed_client_ids" {
  description = "Set COGNITO_ALLOWED_CLIENT_IDS to this comma-joined value."
  value       = "${aws_cognito_user_pool_client.claude.id},${aws_cognito_user_pool_client.m2m.id}"
}

output "cognito_claude_client_id" {
  description = "Set COGNITO_CLAUDE_CLIENT_ID to this — the public client the DCR façade returns."
  value       = aws_cognito_user_pool_client.claude.id
}

output "cognito_hosted_ui_domain" {
  description = "Set COGNITO_HOSTED_UI_DOMAIN to this — base URL of the /oauth2/authorize + /oauth2/token endpoints."
  value       = "https://${aws_cognito_user_pool_domain.oracle_mcp.domain}.auth.eu-central-1.amazoncognito.com"
}

output "cognito_m2m_client_secret" {
  value     = aws_cognito_user_pool_client.m2m.client_secret
  sensitive = true
}
