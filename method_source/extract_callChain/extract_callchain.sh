#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APK_DIR=${1:?Usage: extract_callchain.sh /home/lb/data/Dataset/apks_period/YEAR/benign}
cd "$SCRIPT_DIR"
LOG_FILE="callchain_automation.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ICCBot call-chain extraction..." | tee -a "$LOG_FILE"
bash "$SCRIPT_DIR/runICCBot.sh" "$APK_DIR" 2>&1 | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Call-chain extraction completed." | tee -a "$LOG_FILE"
