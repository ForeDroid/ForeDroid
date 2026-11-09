#!/bin/bash

# 自动化执行Foredroid分析工具的脚本
# 使用方法: bash extract_ui.sh /home/lb/data/Dataset/apks_period/2014/benign

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
if [[ -z "${JAVA:-}" && -x "$SCRIPT_DIR/tool/java8/bin/java" ]]; then
    JAVA="$SCRIPT_DIR/tool/java8/bin/java"
fi
JAVA=${JAVA:-java}
PYTHON=${PYTHON:-python3}
BUILD_DIR="$SCRIPT_DIR/ocr/build"
JAR="$SCRIPT_DIR/jars/Backstage-UIAnalysis.jar"
JAVA_COMPAT=()
JAVA_VERSION=$("$JAVA" -version 2>&1)
if [[ "$JAVA_VERSION" != *'"1.8.'* ]]; then
    JAVA_COMPAT=(--add-opens java.base/java.util=ALL-UNNAMED
                 --add-opens java.base/java.lang.reflect=ALL-UNNAMED
                 --add-opens java.base/java.text=ALL-UNNAMED
                 --add-opens java.desktop/java.awt.font=ALL-UNNAMED)
fi

# 检查参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /home/lb/data/Dataset/apks_period/YEAR/benign"
    exit 1
fi

APK_DIR=$1
LOG_FILE="gui_automation.log"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Check OCR dependencies before starting expensive static analysis.
[[ -d "$APK_DIR" ]] || { echo "APK directory not found: $APK_DIR" >&2; exit 1; }
[[ -f "$BUILD_DIR/OcrUIAnalysis.class" ]] || {
    echo 'Missing OCR bridge; see README_OCR.md for build instructions' >&2; exit 1;
}
"$PYTHON" - <<'CHECK_OCR'
import os, sys
sys.path.insert(0, 'ocr')
from ocr_assets import Recognizer
Recognizer(executable=os.environ.get('TESSERACT', 'tesseract'),
           language=os.environ.get('OCR_LANGUAGE', 'eng'))
CHECK_OCR

echo "[$TIMESTAMP] Starting Foredroid analysis pipeline..." | tee -a $LOG_FILE

###############################################################################
# Module 1: GUI Information Extraction with Backstage
###############################################################################
# echo -e "\n=== Module 1: GUI Information Extraction with Backstage ===" | tee -a $LOG_FILE
# cd extract_Gui


# Step 1(1): Run GUI Analysis
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Running Backstage GUI analysis..." | tee -a $LOG_FILE
bash run_backstage.sh "$APK_DIR" 2>&1 | tee -a $LOG_FILE

# Step 2: Tesseract enriches widget text before Backstage exports labels/context.
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Recognizing drawable text and exporting GUI labels..." | tee -a "$LOG_FILE"
"$JAVA" "${JAVA_COMPAT[@]}" -cp "$BUILD_DIR:$JAR" OcrUIAnalysis \
    "$SCRIPT_DIR/output" "$SCRIPT_DIR/outputBenignUi_train.txt" \
    "$SCRIPT_DIR/ocr/ocr_assets.py" "$PYTHON" \
    2>&1 | tee -a "$LOG_FILE"

# Clean only after OCR has consumed the decoded resources.
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Removing unused files..." | tee -a "$LOG_FILE"
bash removeOtherFolders.sh "$APK_DIR" ./output 2>&1 | tee -a "$LOG_FILE"

# # Step 3: API
echo "[$(date +"%Y-%m-%d %H:%M:%S")] API Things..." | tee -a $LOG_FILE
mkdir -p api-results
API_TMP=$(mktemp -d)
trap 'rm -rf -- "$API_TMP"' EXIT
"$JAVA" -jar jars/Backstage-APIResultsProcessor.jar -i results -o "$API_TMP" 2>&1 | tee -a "$LOG_FILE"
# Normalize the exporter's thread-numbered files to the training filename.
"$PYTHON" - "$API_TMP" "$SCRIPT_DIR/api-results/api_train.txt" <<'MERGE_API'
import csv
import os
from pathlib import Path
import sys
import tempfile

csv.field_size_limit(sys.maxsize)
source, target = map(Path, sys.argv[1:])
files = sorted(source.glob('api_*.txt'))
if not files:
    raise RuntimeError('API exporter produced no files; check its preceding log')
fd, pending = tempfile.mkstemp(prefix='.api-train-', dir=target.parent)
try:
    with os.fdopen(fd, 'w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output, delimiter=';', quoting=csv.QUOTE_ALL)
        header = None
        for path in files:
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.reader(stream, delimiter=';')
                current_header = next(reader)
                if header is None:
                    header = current_header
                    writer.writerow(header)
                elif current_header != header:
                    raise ValueError(f'API header mismatch: {path.name}')
                writer.writerows(reader)
    os.replace(pending, target)
finally:
    if os.path.exists(pending):
        os.unlink(pending)
MERGE_API

TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
# cd ..
echo -e "\n[$TIMESTAMP] Foredroid analysis pipeline completed successfully!" | tee -a $LOG_FILE
