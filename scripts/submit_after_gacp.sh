#!/bin/bash
# ============================================================
# submit_after_gacp.sh
# Waits until no GACP GPU jobs are running for this user,
# then submits the pixel denoiser with --resume.
#
# Usage:
#   nohup bash scripts/submit_after_gacp.sh >> logs/watchdog.log 2>&1 &
#   echo $!  > /tmp/pixel_watchdog.pid
#
# Stop:  kill $(cat /tmp/pixel_watchdog.pid)
# ============================================================

PROJ=/Data1/ec_23104075/projects/pixel-2026
USER=ec_23104075
LOG=$PROJ/logs/watchdog.log
POLL_SEC=60   # check every 60 seconds

cd "$PROJ" || exit 1
mkdir -p logs

echo "[$(date)] pixel watchdog started. Waiting for no GACP GPU jobs…" | tee -a "$LOG"

while true; do
    # Check if any workq job (GPU) is running for this user
    RUNNING_GPU=$(qstat -u "$USER" 2>/dev/null | awk '$4 ~ /gacp/ && $10 == "R" {print $1}')

    if [ -z "$RUNNING_GPU" ]; then
        # Also ensure no pixel denoiser is already running
        RUNNING_PIXEL=$(qstat -u "$USER" 2>/dev/null | awk '$4 ~ /pixel_den/ && $10 == "R" {print $1}')
        if [ -z "$RUNNING_PIXEL" ]; then
            echo "[$(date)] No GACP GPU jobs found. Submitting pixel denoiser…" | tee -a "$LOG"
            JOB=$(qsub "$PROJ/scripts/pbs_train_denoiser.pbs" 2>&1)
            echo "[$(date)] Submitted: $JOB" | tee -a "$LOG"
            # Wait for it to start
            sleep 10
            qstat -u "$USER" 2>/dev/null | tee -a "$LOG"
            break
        else
            echo "[$(date)] Pixel denoiser already running ($RUNNING_PIXEL). Exiting." | tee -a "$LOG"
            break
        fi
    else
        echo "[$(date)] GACP running ($RUNNING_GPU). Sleeping ${POLL_SEC}s…" | tee -a "$LOG"
        sleep "$POLL_SEC"
    fi
done
