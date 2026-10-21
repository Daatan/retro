Check the status of the Factum Atlas pipeline running on the EC2 server (truthmachine-pipeline, i-00ac444b94c5ff9b2, eu-central-1).

Connect via AWS SSM (no SSH key on this instance) and run this single command block:

```bash
python3 - <<'PY'
import json, os, subprocess, collections
from pathlib import Path

# 1. systemd status
svc = subprocess.run(
    ["sudo", "systemctl", "is-active", "truthmachine.service"],
    capture_output=True, text=True
).stdout.strip()

# 2. cell counts from progress.json
p = Path("/home/ubuntu/truthmachine/data/progress.json")
cells = {}
try:
    cells = json.loads(p.read_text()).get("cells", {})
except Exception:
    pass

counts = collections.Counter(v.get("status") for v in cells.values())
done   = counts.get("done", 0)
nopred = counts.get("no_predictions", 0)
failed = counts.get("failed", 0)
pend   = counts.get("pending", 0)
total  = len(cells)

# 3. snapshot file to track changes since last /progress call
snap_path = Path("/tmp/progress_snapshot.json")
prev = {}
try:
    prev = json.loads(snap_path.read_text())
except Exception:
    pass

new_done   = done   - prev.get("done", 0)
new_nopred = nopred - prev.get("nopred", 0)
new_failed = failed - prev.get("failed", 0)

# Save current snapshot
snap_path.write_text(json.dumps({"done": done, "nopred": nopred, "failed": failed, "total": total}))

# 4. last 10 log lines + detect current cell
log_lines = []
current_cell = "(unknown)"
try:
    raw = Path("/home/ubuntu/truthmachine/pipeline_log.txt").read_text()
    log_lines = raw.strip().splitlines()[-10:]
    # Look for most recent cell pattern like C05/toi or B09:toi
    import re
    for line in reversed(log_lines):
        m = re.search(r'([A-Z]\d+)[:/](\w+)', line)
        if m:
            current_cell = f"{m.group(1)}/{m.group(2)}"
            break
except Exception:
    pass

# 5. running process
procs = subprocess.run(
    ["pgrep", "-a", "-f", "tm\\.gnews_ingest|tm\\.orchestrator|tm\\.render_atlas"],
    capture_output=True, text=True
).stdout.strip()
current_step = "idle"
if "gnews_ingest" in procs:
    current_step = "gnews_ingest (scraping articles)"
elif "orchestrator" in procs:
    current_step = "orchestrator (LLM extraction)"
elif "render_atlas" in procs:
    current_step = "render_atlas (building HTML)"

print(f"Status:       {svc} | Step: {current_step}")
print(f"Current cell: {current_cell}")
print(f"Cells:        {done}/{total} done | {nopred} no_pred | {failed} failed | {pend} pending")
print(f"Since last /progress: +{new_done} done | +{new_nopred} no_pred | +{new_failed} failed")
print(f"Last log lines:")
for l in log_lines[-5:]:
    print(f"  {l}")
PY
```

Report back in this format:
- **Status:** (service state | current step)
- **Current cell:** (last cell seen in log)
- **Cells:** X/N done | Y no_pred | Z failed | W pending
- **Since last /progress:** +X done | +Y no_pred | +Z failed
- **Last log lines:** (last 5 lines)

Loop liveness without SSM: `curl -s https://oracle.daatan.com/health | jq .batch_loop` shows the
heartbeat written at the end of every cycle (retro#996). Use `last_cycle_ok_at` to tell a dead
loop from an idle one; an idle loop with a drained queue makes no atlas commits. A `failed`
count that stays flat between calls may be cells parked after 3 failed passes (retro#959), not
new failures.
