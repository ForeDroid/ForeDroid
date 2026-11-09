# #!/bin/bash

# BASE_DIR="$1"

# if [ -z "$BASE_DIR" ] || [ ! -d "$BASE_DIR" ]; then
#     echo "用法: $0 <父目录路径>"
#     exit 1
# fi

# # 遍历 BASE_DIR 下的所有子文件夹
# for dir in "$BASE_DIR"/*/; do
#     [ -d "$dir" ] || continue
#     echo "正在清理目录: $dir"

#     # 遍历子文件夹里的所有文件/目录（包括隐藏文件）
#     for item in "$dir"* "$dir".*; do
#         name=$(basename "$item")

#         # 跳过 . 和 ..
#         if [[ "$name" == "." || "$name" == ".." ]]; then
#             continue
#         fi

#         # 白名单：保留这些文件/文件夹
#         if [[ "$name" == "appSerialized.txt" || \
#               "$name" == "images" || \
#               "$name" == "AndroidManifest.xml" || \
#               "$name" == "res" ]]; then
#             # echo "  保留: $name"
#             continue
#         fi

#         # echo "  删除: $name"
#         rm -rf "$item"
#     done

#     echo "目录 $dir 清理完成."

# done

# echo "清理完成 ✅"


#!/bin/bash

# 用法说明：该脚本会清理目标文件夹中与APK文件夹内APK文件同名的文件夹
# APK文件夹路径: $BASE_DIR/APKs/
# 目标文件夹路径: $BASE_DIR/

APK_DIR="$1"
TARGET_DIR="$2"

if [ -z "$APK_DIR" ] || [ ! -d "$APK_DIR" ]; then
    echo "用法: $0 <APK目录> <目标目录>"
    echo "注意: APK文件应放在 $APK_DIR 目录下"
    exit 1
fi

if [ -z "$TARGET_DIR" ] || [ ! -d "$TARGET_DIR" ]; then
    echo "用法: $0 <APK目录> <目标目录>"
    echo "注意: 目标目录应存在: $TARGET_DIR"
    exit 1
fi

# 检查APKs目录是否存在
if [ ! -d "$APK_DIR" ]; then
    echo "错误: APK目录不存在: $APK_DIR"
    echo "请在父目录下创建APKs文件夹，并将APK文件放入其中"
    exit 1
fi

echo "APK目录: $APK_DIR"
echo "目标目录: $TARGET_DIR"
echo "开始清理..."

# 获取APK文件名列表（不包括扩展名）
apk_files=()
while IFS= read -r apk_file; do
    # 提取文件名（不含扩展名）
    name=$(basename "$apk_file" .apk)
    # 如果没有.apk扩展名，则使用原文件名
    if [[ "$name" == "$apk_file" ]]; then
        name=$(basename "$apk_file")
    fi
    apk_files+=("$name")
done < <(find "$APK_DIR" -type f \( -name "*.apk" -o -name "*.APK" \) | sort)

if [ ${#apk_files[@]} -eq 0 ]; then
    echo "警告: 在 $APK_DIR 中没有找到APK文件"
    exit 0
fi

echo "找到 ${#apk_files[@]} 个APK文件: ${apk_files[*]}"

# 遍历APK文件名，清理对应的文件夹
cleaned_count=0
for apk_name in "${apk_files[@]}"; do
    target_folder="$TARGET_DIR/$apk_name"
    
    if [ -d "$target_folder" ]; then
        echo "正在清理目录: $target_folder"
        # 遍历目标文件夹里的所有文件/目录（包括隐藏文件）
        for item in "$target_folder"/* "$target_folder".*; do
            # 跳过不存在的文件（当文件夹为空时）
            [ -e "$item" ] || continue
            name=$(basename "$item")
            # 跳过 . 和 ..
            if [[ "$name" == "." || "$name" == ".." ]]; then
                continue
            fi
            # 白名单：保留这些文件/文件夹
            if [[ "$name" == "appSerialized.txt" || \
                  "$name" == "images" || \
                  "$name" == "AndroidManifest.xml" || \
                  "$name" == "res" || \
                  "$name" == "apktool.yml" 
                  ]]; then
                # echo "  保留: $name"
                continue
            fi
            # echo "  删除: $name"
            rm -rf "$item"
        done
        echo "目录 $apk_name 清理完成."
        ((cleaned_count++))
    else
        echo "跳过: 目录 $apk_name 不存在"
    fi
done

echo "清理完成 ✅"
echo "共清理了 $cleaned_count 个文件夹"