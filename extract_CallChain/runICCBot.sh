#!/bin/bash

# Set paths
APK_DIR="/media/hdisk/jiaming/apk/apk_downloads/2018/malware/"
OUTPUT_DIR="/media/hdisk/jiaming/output/2018/malware/"
ANDROID_JAR="lib/platforms"
TIME=30
MAX_PATH_NUMBER=100
CLIENT="CTGClient"
TIMEOUT=300  # Timeout in seconds

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each APK in the input directory
for apk in "$APK_DIR"/*.apk; do
    apk_name=$(basename "$apk")
    echo "Processing $apk_name ..."

    call_graph_info_dir="$OUTPUT_DIR/$(basename "$apk_name" .apk)/CallGraphInfo"
    report_file="$call_graph_info_dir/rule_analysis_report.json"

    if [ -f "$report_file" ]; then
        echo "Report already exists for $apk_name, skipping."
        continue
    fi

    timeout "$TIMEOUT" java -jar ICCBot.jar \
        -client $CLIENT \
        -maxapppathnumber $MAX_PATH_NUMBER \
        -platforms $ANDROID_JAR \
        -input "$apk" \
        -output "$OUTPUT_DIR"
done

