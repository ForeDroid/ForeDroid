ANDROID_JAR=./android-18/android.jar
CURRENT_DIR=/media/openharmony/hdisk/jiaming/backstage
JDK=/usr/lib/jvm/java-8-openjdk-amd64/jre/bin
file=$1

appName=`basename ${file} .apk`
export fileName=$(basename $file)

# 修改输出目录为 /media/ming/toshiba_ext/jiaming/backstage_output
OUTPUT_DIR=/media/openharmony/hdisk/jiaming/backstage_output

# 检查是否已经存在同名文件夹
if [ -d "$OUTPUT_DIR/$appName" ]; then
  echo "Skipping $fileName: Folder $OUTPUT_DIR/$appName already exists."
  exit 0
fi

# 进入输出目录并运行 apktool
cd $OUTPUT_DIR
$JDK/java -jar $CURRENT_DIR/output/apktool_2.9.3.jar -s -f d $file -o $appName
cd $CURRENT_DIR

echo $appName
echo $file

echo "Running analysis for $fileName"
$JDK/java -Xmx40g -Xss5m -jar ./Backstage.jar -apk ${file} -androidJar $ANDROID_JAR -apkToolOutput $OUTPUT_DIR/$appName -rAnalysis -uiTimeoutValue 30 -uiTimeoutUnit SECONDS -rTimeoutValue 30 -rTimeoutUnit SECONDS -maxDepthMethodLevel 15 -numThreads 24 -rLimitByPackageName

echo "Analysis for $fileName finished"

# 删除临时文件，更新路径为新输出目录
cd $OUTPUT_DIR
bash $CURRENT_DIR/output/removeAllFolders.sh $appName
cd $CURRENT_DIR
