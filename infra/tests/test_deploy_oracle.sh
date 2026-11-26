#!/bin/bash
# Behavioural tests for infra/deploy_oracle.sh (retro#1110).
#
# On 2026-11-26 a deploy's `uv sync` died halfway on two root-owned files, after
# it had already removed litellm and fastapi from the live venv. The next worker
# recycle took the API down, and re-running the deploy did nothing because HEAD
# was already at the target. These tests drive the real script against a
# throwaway git origin, with fake uv / systemctl / curl, and pin each of those
# behaviours:
#   - a failed sync or `uv pip check` rolls the checkout and the venv back and
#     does not reload the service; a failed rollback leaves no marker behind;
#   - a dist-info dir with no METADATA (a half-done uninstall) is removed;
#   - "nothing to do" means the target is the last *healthy* deploy, not HEAD;
#   - the marker is written only after the health check passes.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="$DIR/deploy_oracle.sh"

pass=0; fail=0
ok()   { echo "  ok: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
check(){ if eval "$2"; then ok "$1"; else bad "$1"; fi; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@example.invalid GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@example.invalid
git init -q --bare "$W/origin.git"
git -C "$W/origin.git" symbolic-ref HEAD refs/heads/main
git clone -q "$W/origin.git" "$W/seed" 2>/dev/null
git -C "$W/seed" checkout -q -b main
mkdir -p "$W/seed/api/src/forecast_api"   # the script writes _build_info.json here
touch "$W/seed/api/src/forecast_api/__init__.py"

commit() {  # commit <n>: a new api/uv.lock revision on origin/main; prints its SHA
  echo "lock $1" > "$W/seed/api/uv.lock"
  git -C "$W/seed" add -A && git -C "$W/seed" commit -q -m "c$1" && git -C "$W/seed" push -q origin main
  git -C "$W/seed" rev-parse HEAD
}

C1=$(commit 1)
git clone -q "$W/origin.git" "$W/box"
API="$W/box"

mkdir -p "$W/bin"
cat > "$W/bin/uv" <<'EOF'
#!/bin/bash
head=$(git -C "$FAKE_API" rev-parse HEAD)
echo "$1 $head" >> "$FAKE_LOG/uv.log"
[[ $1 == sync && " ${FAKE_UV_FAIL_AT:-} " == *" $head "* ]] && { echo "error: failed to remove file: Permission denied" >&2; exit 2; }
[[ $1 == pip && " ${FAKE_CHECK_FAIL_AT:-} " == *" $head "* ]] && { echo "forecast-api is missing requirement fastapi" >&2; exit 1; }
exit 0
EOF
cat > "$W/bin/systemctl" <<'EOF'
#!/bin/bash
echo "$*" >> "$FAKE_LOG/systemctl.log"
EOF
cat > "$W/bin/sudo" <<'EOF'
#!/bin/bash
echo "$*" >> "$FAKE_LOG/sudo.log"
EOF
cat > "$W/bin/curl" <<'EOF'
#!/bin/bash
printf '%s' "${FAKE_HEALTH:-200}"
EOF
chmod +x "$W/bin/"*
touch "$W/uv.log" "$W/systemctl.log" "$W/sudo.log"

run_deploy() {
  FAKE_API="$API" FAKE_LOG="$W" PATH="$W/bin:$PATH" \
  DEPLOY_API_DIR="$API" DEPLOY_OWNER="${FAKE_OWNER:-$(id -un)}" DEPLOY_SUDO="$W/bin/sudo" \
  DEPLOY_AS_UBUNTU="" DEPLOY_UV="$W/bin/uv" DEPLOY_SYSTEMCTL="$W/bin/systemctl" \
  DEPLOY_HEALTH_URL="http://test.invalid/health" DEPLOY_HEALTH_SLEEP=0 DEPLOY_GRACEFUL_DRAIN=0 \
    bash "$DEPLOY" "$@" > "$W/out.log" 2>&1
}
head_of() { git -C "$API" rev-parse HEAD; }
marker()  { cat "$API/.deployed_sha" 2>/dev/null || echo none; }
lines()   { wc -l < "$1"; }

echo "== deploy_oracle.sh =="

# --- 1. A normal deploy syncs, reloads and records the commit ----------------
C2=$(commit 2)
run_deploy; rc=$?
check "first deploy exits 0"                        "[[ $rc -eq 0 ]]"
check "first deploy records the deployed commit"    "[[ \$(marker) == '$C2' ]]"
check "first deploy syncs at the new commit"        "grep -q '^sync $C2' '$W/uv.log'"
check "first deploy reloads the service"            "grep -q '^reload-or-restart oracle-api' '$W/systemctl.log'"

# --- 2. Re-running with nothing new does nothing -----------------------------
uv_before=$(lines "$W/uv.log"); sc_before=$(lines "$W/systemctl.log")
run_deploy; rc=$?
check "no-op run exits 0"                           "[[ $rc -eq 0 ]]"
check "no-op run says so"                           "grep -q 'no change' '$W/out.log'"
check "no-op run neither syncs nor reloads"         "[[ \$(lines '$W/uv.log') -eq $uv_before && \$(lines '$W/systemctl.log') -eq $sc_before ]]"
check "a tree owned by the owner is left alone"     "[[ \$(lines '$W/sudo.log') -eq 0 ]]"

# --- 2b. Files not owned by ubuntu are handed back before git runs -----------
# Pretend the checkout belongs to someone else: every entry is then "foreign".
FAKE_OWNER=nobody run_deploy; rc=$?
own_at=$(grep -n 'ownership:' "$W/out.log" | head -n 1 | cut -d: -f1)
fetch_at=$(grep -n 'fetching origin/main' "$W/out.log" | head -n 1 | cut -d: -f1)
check "foreign files trigger a sudo chown -R"       "grep -qx 'chown -R nobody:nobody $API' '$W/sudo.log'"
check "the chown happens before the fetch"          "[[ -n '$own_at' && -n '$fetch_at' && $own_at -lt $fetch_at ]]"

# --- 3. A failed sync rolls back and leaves the service alone ----------------
C3=$(commit 3)
sc_before=$(lines "$W/systemctl.log")
FAKE_UV_FAIL_AT="$C3" run_deploy; rc=$?
check "failed sync exits non-zero"                  "[[ $rc -ne 0 ]]"
check "failed sync resets the checkout back"        "[[ \$(head_of) == '$C2' ]]"
check "failed sync re-syncs the old commit's lock"  "[[ \$(tail -n 2 '$W/uv.log' | head -n 1) == 'sync $C2' ]]"
check "failed sync does not reload the service"     "[[ \$(lines '$W/systemctl.log') -eq $sc_before ]]"
check "failed sync keeps the marker"                "[[ \$(marker) == '$C2' ]]"
check "failed sync says it rolled back"             "grep -q 'rolled back' '$W/out.log'"

# --- 4. The retry after a failure really deploys -----------------------------
run_deploy; rc=$?
check "retry after a failed sync deploys"           "[[ $rc -eq 0 && \$(marker) == '$C3' && \$(head_of) == '$C3' ]]"

# --- 5. HEAD already at the target is not "nothing to do" --------------------
# The 2026-11-26 trap: the old script reset HEAD before its sync died, and the
# re-run compared HEAD with HEAD and skipped.
C4=$(commit 4)
git -C "$API" fetch -q origin main && git -C "$API" reset -q --hard "$C4"
run_deploy; rc=$?
check "HEAD at target but not deployed: syncs"      "grep -q '^sync $C4' '$W/uv.log'"
check "HEAD at target but not deployed: records"    "[[ $rc -eq 0 && \$(marker) == '$C4' ]]"

# --- 6. A sync that exits 0 but leaves a broken venv also rolls back ----------
C5=$(commit 5)
sc_before=$(lines "$W/systemctl.log")
FAKE_CHECK_FAIL_AT="$C5" run_deploy; rc=$?
check "failed pip check exits non-zero"             "[[ $rc -ne 0 ]]"
check "failed pip check resets to the last deploy"  "[[ \$(head_of) == '$C4' ]]"
check "failed pip check does not reload"            "[[ \$(lines '$W/systemctl.log') -eq $sc_before ]]"

# --- 7. Unhealthy after reload and restart: no marker ------------------------
C6=$(commit 6)
FAKE_HEALTH=000 run_deploy; rc=$?
check "unhealthy deploy exits non-zero"             "[[ $rc -ne 0 ]]"
check "unhealthy deploy escalates to a restart"     "grep -q '^restart oracle-api' '$W/systemctl.log'"
check "unhealthy deploy is not recorded"            "[[ \$(marker) == '$C4' ]]"
run_deploy; rc=$?
check "the retry of an unhealthy deploy re-runs it" "[[ $rc -eq 0 && \$(marker) == '$C6' ]]"

# --- 8. A rollback that fails too drops the marker ---------------------------
# Otherwise HEAD (back at the last good commit) equals the marker, and re-running
# that ref would print "no change" over a broken venv.
C7=$(commit 7)
FAKE_UV_FAIL_AT="$C7 $C6" run_deploy; rc=$?
check "failed rollback exits non-zero"              "[[ $rc -ne 0 ]]"
check "failed rollback says so"                     "grep -q 'ROLLBACK FAILED' '$W/out.log'"
check "failed rollback removes the marker"          "[[ \$(marker) == none && \$(head_of) == '$C6' ]]"
syncs_before=$(grep -c "^sync $C6" "$W/uv.log")
run_deploy "$C6"; rc=$?
check "re-running the last good ref syncs again"    "[[ $rc -eq 0 && \$(grep -c '^sync $C6' '$W/uv.log') -gt $syncs_before && \$(marker) == '$C6' ]]"

# --- 9. A dist-info dir without METADATA is removed before the sync -----------
SP="$API/api/.venv/lib/python3.12/site-packages"
mkdir -p "$SP/litellm-1.109.1.dist-info/licenses" "$SP/fastapi-0.165.0.dist-info"
touch "$SP/fastapi-0.165.0.dist-info/METADATA"
C8=$(commit 8)
run_deploy; rc=$?
check "the METADATA-less dist-info is removed"      "[[ $rc -eq 0 && ! -e '$SP/litellm-1.109.1.dist-info' ]]"
check "a valid dist-info is kept"                   "[[ -e '$SP/fastapi-0.165.0.dist-info/METADATA' ]]"
check "the removal is logged"                       "grep -q 'litellm-1.109.1.dist-info' '$W/out.log'"

# --- 10. The marker never gets committed --------------------------------------
check ".deployed_sha is git-ignored"                "grep -qx '/.deployed_sha' '$DIR/../.gitignore'"

echo "== $pass passed, $fail failed =="
[[ $fail -eq 0 ]]
