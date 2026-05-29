#!/bin/bash
# pixel_watch.sh — Auto-resubmit watcher for PIXEL-2026 dataset generation.
# Checks every 5 minutes. If no pixel_generate job is running, cleans up and
# resubmits. Logs to /tmp/pixel_watch.log.
#
# Usage:  nohup bash scripts/pixel_watch.sh &
# Stop:   kill $(cat /tmp/pixel_watch.pid)

PROJ=/Data1/ec_23104075/projects/pixel-2026
LOG=/tmp/pixel_watch.log
PID_FILE=/tmp/pixel_watch.pid

echo $$ > "$PID_FILE"
echo "[$(date)] pixel_watch started (PID=$$)" >> "$LOG"

while true; do
    # Check if a pixel_generate PBS job is running or queued
    if qstat -u ec_23104075 2>/dev/null | grep -q "pixel_gen"; then
        echo "[$(date)] Job running — skipping." >> "$LOG"
    else
        echo "[$(date)] No pixel_generate job found. Resubmitting…" >> "$LOG"

        cd "$PROJ" || { echo "[$(date)] ERROR: cd failed" >> "$LOG"; sleep 60; continue; }

        # Checkpoint before resubmit
        COUNT=$(python -c "import json; d=json.load(open('data/raw/pixel_dataset.checkpoint.json')); print(d['total_count'])" 2>/dev/null || echo "?")
        echo "[$(date)] Checkpoint count before resubmit: $COUNT" >> "$LOG"

        # Clear HDF5 consistency flags (set by SIGKILL'd jobs)
        h5clear -s data/raw/pixel_dataset.h5 2>/dev/null && echo "[$(date)] h5clear OK" >> "$LOG"
        rm -f data/raw/pixel_dataset.lock
        rm -rf data/tmp/pixel_sim_*

        NEW_JOB=$(qsub scripts/pbs_generate.pbs 2>&1)
        echo "[$(date)] Submitted: $NEW_JOB" >> "$LOG"
    fi

    sleep 300
done
