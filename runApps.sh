#please install http://www.gnu.org/software/parallel/ before using it
# -apk /home/ming/work/backstage/apks/fraudapp.apk -androidJar /home/ming/Android/Sdk/platforms/android-18/android.jar -apkToolOutput output/fraudapp/ -rAnalysis -uiTimeoutValue 30 -uiTimeoutUnit SECONDS -rTimeoutValue 30 -rTimeoutUnit SECONDS -maxDepthMethodLevel 15 -numThreads 24 -rLimitByPackageName
#FOLDER=/home/ming/work/ICSE22ArtifactsZip/GPMalware_ICSE22_1029/apks
#FOLDER=/media/wuyang/toshiba_ext/jiaming/data/Benign/apks

FOLDER=/media/openharmony/hdisk/jiaming/collect_apks/data1/Androzoo

runApp() {
    bash runApp_output.sh $FOLDER/$1
}

export -f runApp

ls $FOLDER/*.apk | parallel -j $1 --timeout 1800 runApp
