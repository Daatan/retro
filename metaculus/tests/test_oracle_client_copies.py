"""oracle_client.py is deliberately copied into three standalone uv projects (retro#977):
each is deployed and run from its own directory with its own lockfile, so a shared module
would mean repackaging all three. Copies must stay byte-identical — change all three together."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COPIES = ["metaculus", "polymarket_paper", "manifold_paper"]


def test_oracle_client_copies_are_identical():
    contents = {d: (ROOT / d / "oracle_client.py").read_bytes() for d in COPIES}
    reference = contents[COPIES[0]]
    drifted = [d for d, body in contents.items() if body != reference]
    assert not drifted, f"oracle_client.py differs from metaculus/ in: {drifted}"
