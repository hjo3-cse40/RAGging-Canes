#!/usr/bin/env bash
# Run from any directory; optionally pause the legacy Nemotron capture loop.
set -euo pipefail
cd "$(dirname "$0")/.."
paused_pid=''
dashboard_pid=''
cleanup() {
  if [[ -n "$dashboard_pid" ]]; then
    kill "$dashboard_pid" 2>/dev/null || true
    wait "$dashboard_pid" 2>/dev/null || true
  fi
  if [[ -n "$paused_pid" ]]; then kill -CONT "$paused_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "${1:-}" == '--pause-nemotron' ]]; then
  shift
  mapfile -t pids < <(pgrep -f '^/usr/bin/python3 -u -m gx10.service ' || true)
  if [[ "${#pids[@]}" != 1 ]]; then
    echo 'Expected one legacy Nemotron process; inspect running capture loops before continuing.' >&2
    exit 2
  fi
  paused_pid=${pids[0]}
  kill -STOP "$paused_pid"
fi
"${DASHBOARD_PYTHON:-.venv/bin/python}" -m software.dashboard "$@" &
dashboard_pid=$!
wait "$dashboard_pid"
