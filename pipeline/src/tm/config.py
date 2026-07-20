from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    brave_api_key: str = ""

    gatekeeper_model: str = "bedrock/amazon.nova-micro-v1:0"
    extractor_model: str = "bedrock/amazon.nova-lite-v1:0"
    ground_truth_model: str = "bedrock/amazon.nova-lite-v1:0"

    # Kill-switch for Bedrock/Anthropic prompt caching (llm.py::complete_structured's
    # cached_prefix). Verified ON: smoke_test_prompt_cache.py confirmed reliable
    # cache_read/cache_creation token accounting against live Bedrock for Nova Micro
    # (gatekeeper) and the live Haiku extractor override, using the real prompts and
    # schemas (not synthetic ones) — 4/4 clean runs each, ~90-93% of input tokens
    # landing in the cached prefix. See docs/PROMPT_CACHING.md for the full results
    # and the one unrelated finding (a pre-existing Nova Lite JSON-formatting quirk,
    # independent of caching) surfaced while verifying this.
    enable_prompt_cache: bool = True

    # Optional: override API base/key (for Ollama or other OpenAI-compatible backends)
    model_api_base: str = ""
    model_api_key: str = ""

    aws_region: str = "us-east-1"

    data_dir: Path = Path("/app/data")
    vault_dir: Path = Path("")  # empty = data_dir/vault2 (avoids root-owned vault/)

    @property
    def atlas_dir(self) -> Path:
        return self.data_dir / "atlas"

    @property
    def events_dir(self) -> Path:
        return self.data_dir / "events"

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    @property
    def progress_file(self) -> Path:
        return self.data_dir / "progress.json"


settings = Settings()
