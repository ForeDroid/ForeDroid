#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APK_DIR=$(realpath "${1:?Usage: runICCBot.sh /home/lb/data/Dataset/apks_period/YEAR/benign}")
[[ -d "$APK_DIR" ]] || { echo "APK directory not found: $APK_DIR" >&2; exit 1; }
cd "$SCRIPT_DIR"
JAVA=${JAVA:-java}
OUTPUT_DIR="$SCRIPT_DIR/output"
ANDROID_JAR=${ANDROID_JAR:-"$SCRIPT_DIR/lib/platforms"}
TIME=30
MAX_PATH_NUMBER=100
CLIENT="CTGClient"
TIMEOUT=1800
mkdir -p "$OUTPUT_DIR"

# Accept a year/label folder, a year folder, or the dataset root.
status=0
count=0
while IFS= read -r -d '' apk; do
    count=$((count + 1))
    apk_name=$(basename "$apk")
    report_file="$OUTPUT_DIR/${apk_name%.apk}/CallGraphInfo/rule_analysis_report.json"
    if [[ -f "$report_file" ]]; then
        echo "Report already exists for $apk_name, skipping."
        continue
    fi
    echo "Processing $apk_name ..."
    if timeout "$TIMEOUT" "$JAVA" -jar "$SCRIPT_DIR/ICCBot.jar" \
        -client "$CLIENT" \
        -maxPathNumber "$MAX_PATH_NUMBER" \
        -androidJar "$ANDROID_JAR" \
        -path "$(dirname "$apk")" \
        -outputDir "$OUTPUT_DIR" \
        -name "$apk_name" \
        -time "$TIME"; then
        :
    else
        result=$?
        echo "Analysis failed for $apk_name (exit $result)" >&2
        status=1
    fi
done < <(find "$APK_DIR" -type f -name '*.apk' -print0)
[[ "$count" -gt 0 ]] || { echo "No APK files found under $APK_DIR" >&2; exit 1; }
exit "$status"
