#!/bin/bash

# Set paths
FOLDER=/media/hdisk/jiaming/collect_apks/data1/Androzoo
ANDROID_JAR=./android-18/android.jar
CURRENT_DIR=/media/openharmony/hdisk/jiaming/backstage
JDK=/usr/lib/jvm/java-8-openjdk-amd64/jre/bin
OUTPUT_DIR=/media/hdisk/jiaming/backstage_output

run_analysis() {
    file=$1
    appName=$(basename "${file}" .apk)

    # Skip if already processed
    if [ -d "$OUTPUT_DIR/$appName" ]; then
        echo "Skipping $appName: Already exists."
        return
    fi

    # Decompile APK
    cd $OUTPUT_DIR
    $JDK/java -jar $CURRENT_DIR/tool/apktool_2.9.3.jar -s -f d $file -o $appName
    cd $CURRENT_DIR

    echo "Processing $appName..."

    # Run Backstage analysis
    $JDK/java -Xmx40g -Xss5m -jar ./Backstage.jar \
        -apk ${file} \
        -androidJar $ANDROID_JAR \
        -apkToolOutput $OUTPUT_DIR/$appName \
        -rAnalysis \
        -uiTimeoutValue 30 -uiTimeoutUnit SECONDS \
        -rTimeoutValue 30 -rTimeoutUnit SECONDS \
        -maxDepthMethodLevel 15 \
        -numThreads 24 \
        -rLimitByPackageName

    echo "Analysis finished for $appName."

    # Cleanup
    cd $OUTPUT_DIR
    bash $CURRENT_DIR/tool/removeAllFolders.sh $appName
    cd $CURRENT_DIR
}

export -f run_analysis

# Run in parallel (adjust `-j` as needed)
ls $FOLDER/*.apk | parallel -j 4 --timeout 1800 run_analysis

