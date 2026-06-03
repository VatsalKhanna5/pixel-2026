#!/bin/bash
# ============================================================
# submit_after_gacp.sh  —  Persistent PIXEL training manager
#
# Runs FOREVER until pixel denoiser completes 300 epochs.
# Each time GACP kills the pixel job, this watchdog waits
# for GACP to go idle, then resubmits (--resume picks up
# from the last saved checkpoint = 1 epoch ago).
#
# Usage:
#   nohup bash scripts/submit_after_gacp.sh >> logs/watchdog.log 2>&1 &
#   echo $! > /tmp/pixel_watchdog.pid
#
# Stop:  kill $(cat /tmp/pixel_watchdog.pid)
# Status: tail -f logs/watchdog.log
# ============================================================

PROJ=/Data1/ec_23104075/projects/pixel-2026
USER=ec_23104075
TOTAL_EPOCHS=300
POLL_SEC=30          # check every 30 seconds
LATEST_CKPT="$PROJ/experiments/denoiser_v1/denoiser_latest.pt"

cd "$PROJ" || exit 1
mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/watchdog.log; }

log "============================================"
log "Persistent pixel watchdog started (PID=$$)"
log "Will run until $TOTAL_EPOCHS epochs complete"
log "============================================"

get_current_epoch() {
    python3 -c "
import torch, sys
try:
    c = torch.load('$LATEST_CKPT', map_location='cpu')
    print(c['epoch'])
except:
    print(0)
" 2>/dev/null
}

gacp_running() {
    qstat -u "$USER" 2>/dev/null | awk '$4 ~ /gacp/ && $10 == "R"' | grep -q .
}

pixel_running() {
    qstat -u "$USER" 2>/dev/null | awk '$4 ~ /pixel_den/ && $10 == "R"' | grep -q .
}

while true; do
    # Check if training is already complete
    DONE_EPOCH=$(get_current_epoch)
    if [ "$DONE_EPOCH" -ge "$TOTAL_EPOCHS" ] 2>/dev/null; then
        log "Training complete! Reached epoch $DONE_EPOCH/$TOTAL_EPOCHS. Exiting."
        exit 0
    fi

    # Check if pixel is already running
    if pixel_running; then
        log "Pixel running... epoch=$DONE_EPOCH/$TOTAL_EPOCHS. Waiting."
        sleep "$POLL_SEC"
        continue
    fi

    # Pixel is NOT running. Wait for GACP to be idle, then submit.
    if gacp_running; then
        log "GACP running, pixel not running (epoch=$DONE_EPOCH). Waiting for GACP to finish."
        sleep "$POLL_SEC"
        continue
    fi

    # Neither GACP nor pixel running. Submit pixel.
    log "No GACP. Submitting pixel (will resume from epoch=$DONE_EPOCH)."
    JOB=$(qsub "$PROJ/scripts/pbs_train_denoiser.pbs" 2>&1)
    log "Submitted: $JOB"
    sleep "$POLL_SEC"
done
