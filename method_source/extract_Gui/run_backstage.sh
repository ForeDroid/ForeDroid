#!/bin/bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FOLDER=$(realpath "${1:?Usage: run_backstage.sh /home/lb/data/Dataset/apks_period/YEAR/benign}")
export SCRIPT_DIR
if [[ -z "${JAVA:-}" && -x "$SCRIPT_DIR/tool/java8/bin/java" ]]; then
    JAVA="$SCRIPT_DIR/tool/java8/bin/java"
fi
export JAVA=${JAVA:-java}


run_analysis() {
    set -euo pipefail

    ANDROID_JAR_DIR="$SCRIPT_DIR/android-18/android_jars"
    CURRENT_DIR="$SCRIPT_DIR"
    OUTPUT_DIR="$SCRIPT_DIR/output"

    APKTOOL_NEW=$CURRENT_DIR/tool/apktool_2.12.1.jar
    APKTOOL_OLD=$CURRENT_DIR/tool/apktool_2.9.3.jar

    file=$1
    appName=$(basename "${file}" .apk)
    APK_OUT_DIR="$OUTPUT_DIR/$appName"

    if [ -f "$APK_OUT_DIR/appSerialized.txt" ]; then
        echo "Skipping $appName: Already exists."
        return
    fi

    mkdir -p "$OUTPUT_DIR"
    cd "$OUTPUT_DIR"


    echo "Decompiling $appName with apktool 2.9.3..."
    "$JAVA" -jar "$APKTOOL_OLD" -s -f d "$file" -o "$appName" >/dev/null 2>&1
    # echo "Decompiling $appName with apktool 2.12.1..."
    # "$JAVA" -jar "$APKTOOL_NEW" d -f "$file" -o "$appName" >/dev/null 2>&1

    # # ===== 如果失败或没生成 apktool.yml → 回退旧版本 =====
    # if [ ! -f "$APK_OUT_DIR/apktool.yml" ]; then
    #     echo "⚠️  apktool 2.12.1 failed, retrying with 2.9.3..."
    #     rm -rf "$APK_OUT_DIR"
    #     "$JAVA" -jar "$APKTOOL_OLD" -s -f d "$file" -o "$appName" >/dev/null 2>&1

    #     if [ ! -f "$APK_OUT_DIR/apktool.yml" ]; then
    #         echo "❌ Both apktool versions failed for $appName, skipping."
    #         return
    #     fi
    # fi

    # ===== 读取 targetSdkVersion =====
    APKTOOL_YML="$APK_OUT_DIR/apktool.yml"
    TARGET_SDK=$(awk '/targetSdkVersion:/ {gsub(/[^0-9]/, "", $2); print $2; exit}' "$APKTOOL_YML")

    # 判断 TARGET_SDK 是否为空，如果为空则提取 minSdkVersion
    if [ -z "$TARGET_SDK" ]; then
        echo "未找到 targetSdkVersion，将使用 minSdkVersion 替代"
        MIN_SDK=$(awk '/minSdkVersion:/ {gsub(/[^0-9]/, "", $2); print $2; exit}' "$APKTOOL_YML")
        MIN_SDK=${MIN_SDK:-18}
        
        # # 额外的健壮性检查：如果 minSdkVersion 也不存在，给出提示并退出
        # if [ -z "$TARGET_SDK" ]; then
        #     echo "错误：既未找到 targetSdkVersion，也未找到 minSdkVersion"
        #     exit 1
        # fi
    fi

    if [ -z "$TARGET_SDK" ]; then
        TARGET_SDK=$MIN_SDK
        if [ "$MIN_SDK" -lt 18 ]; then  # minSdkVersion 小于 18 时，强制把TARGET_SDK设为 28
            TARGET_SDK=28
        fi

    fi

    # if [ -z "$TARGET_SDK" ]; then
    #     TARGET_SDK=18
    # fi

    if [ "$TARGET_SDK" -lt 18  ]; then
        TARGET_SDK=18
    fi

    
    # if [ "$TARGET_SDK" == 29 ] || [ "$TARGET_SDK" == 30 ]; then
    #     TARGET_SDK=28
    # fi

    echo "Detected targetSdkVersion: $TARGET_SDK"

    # ===== 自动选择 android.jar =====
    cd "$CURRENT_DIR"  # 切换到包含 android_jars 的目录

    SELECTED_JAR=""
    AVAILABLE_VERSIONS=$(ls $ANDROID_JAR_DIR | grep -oP '(?<=android-)\d+' | sort -n)

    echo "Available Android versions: $AVAILABLE_VERSIONS"

    # 寻找大于等于targetSdkVersion的最小版本
    BEST_MATCH=""
    for ver in $AVAILABLE_VERSIONS; do
        if [ "$ver" -ge "$TARGET_SDK" ]; then
            if [ -z "$BEST_MATCH" ] || [ "$ver" -lt "$BEST_MATCH" ]; then
                BEST_MATCH=$ver
            fi
        fi
    done

    # 如果找到了合适的版本
    if [ -n "$BEST_MATCH" ]; then
        SELECTED_JAR=$BEST_MATCH
        echo "Found best match: android-$SELECTED_JAR.jar"
    else
        # 如果没有大于等于的版本，回退到可用的最大版本
        MAX_VER=0
        for ver in $AVAILABLE_VERSIONS; do
            if [ "$ver" -gt "$MAX_VER" ]; then
                MAX_VER=$ver
            fi
        done
        if [ "$MAX_VER" -gt 0 ]; then
            SELECTED_JAR=$MAX_VER
            echo "Warning: No version >= $TARGET_SDK found. Using max available: android-$SELECTED_JAR.jar"
        else
            SELECTED_JAR=18  # 默认值
            echo "Error: No android.jar found. Using default: android-$SELECTED_JAR.jar"
        fi
    fi

    ANDROID_JAR="$ANDROID_JAR_DIR/android-$SELECTED_JAR.jar"

    echo "Using android.jar version: android-$SELECTED_JAR.jar"
    echo "$appName -> targetSdk=$TARGET_SDK -> android-$SELECTED_JAR.jar" >> "$CURRENT_DIR/sdk_selection.log"

    cd "$CURRENT_DIR"

    echo "Running Backstage analysis for $appName..."

    "$JAVA" -Xmx24g -Xss3m -jar ./Backstage.jar \
        -apk "$file" \
        -androidJar "$ANDROID_JAR" \
        -apkToolOutput "$APK_OUT_DIR" \
        -rAnalysis \
        -uiTimeoutValue 120 -uiTimeoutUnit SECONDS \
        -rTimeoutValue 120 -rTimeoutUnit SECONDS \
        -maxDepthMethodLevel 15 \
        -numThreads 12 \
        -rLimitByPackageName

    echo "Analysis finished for $appName."
}

export -f run_analysis
find "$FOLDER" -type f -name '*.apk' -print0 | parallel -0 -j 2 --timeout 1800 --halt soon,fail=1 run_analysis
