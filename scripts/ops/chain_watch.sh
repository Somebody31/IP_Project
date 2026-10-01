#!/usr/bin/env bash
# Summarise the follow-up chain once the *current* invocation stops.
# Detached (setsid): no session, so DBUS/XDG are set explicitly for notify-send.
# Lives in the repo (scripts/ops/): a /tmp-hosted copy of this watcher was
# wiped mid-run on this box, so /tmp is not durable enough for a 10 h watch.
cd "$(dirname "$0")/../.." || exit 1

UNIT=dehaze-followup-chain.service
# The unit may not be started yet (watcher and `systemctl start` can be issued
# in either order) — a stopped oneshot keeps the *previous* invocation id, so
# wait for fresh activation before capturing it.
for _ in $(seq 1 120); do
  state=$(systemctl --user show -p ActiveState --value "$UNIT")
  case "$state" in active|activating) break ;; *) sleep 5 ;; esac
done
case "$state" in
  active|activating) ;;
  *) echo "chain never started within 10 min (state=$state); exiting" >&2; exit 1 ;;
esac
# captured now, while the current run is live: journal is append-only across runs,
# so filtering by invocation id is what keeps earlier attempts' stage lines out
# of the summary (systemd sets _SYSTEMD_INVOCATION_ID on every entry).
INV=$(systemctl --user show -p InvocationID --value "$UNIT")
if [ -n "$INV" ]; then
  JOURNAL=(journalctl --user -u "$UNIT" _SYSTEMD_INVOCATION_ID="$INV" --no-pager -o cat)
else
  JOURNAL=(journalctl --user -u "$UNIT" --no-pager -o cat)
fi

while :; do
  state=$(systemctl --user show -p ActiveState --value "$UNIT")
  case "$state" in
    active|activating|reloading) sleep 60 ;;
    *) break ;;
  esac
done

{
  echo "=== chain stopped $(date '+%F %T') state=${state} invocation=${INV:-unknown}"
  echo
  echo "=== stage transitions (this invocation only)"
  "${JOURNAL[@]}" | grep -E 'stage |complete' | tail -20
  echo
  echo "=== stage logs (tail)"
  for f in outputs/logs/train-context-seed1729.log outputs/logs/*seed43*.log outputs/logs/*seed7*.log \
           outputs/logs/train-spatial*.log outputs/logs/*probe*gate*mask*.log outputs/logs/*spatial*.log; do
    [ -f "$f" ] && { echo "--- $f"; tail -5 "$f"; }
  done
  echo
  echo "=== errors/tracebacks in this invocation (if any)"
  "${JOURNAL[@]}" | grep -iE 'error|traceback|fail|abort' | tail -8 || true
  echo
  echo "=== results (newest first)"
  ls -lt results/machine_learning/*.json 2>/dev/null | head -15
} > outputs/logs/chain-summary.txt 2>&1

tail -3 outputs/logs/chain-summary.txt | tr '\n' ' ' > /tmp/chain-notify.txt
export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/bus"
if command -v notify-send >/dev/null 2>&1; then
  notify-send -u normal "IP_Project follow-up chain finished" "$(cat /tmp/chain-notify.txt)" || true
fi
