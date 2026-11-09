# 处理APK数据合并工具 - UI和API任一文件存在都可以合并
import os
import argparse
import csv
import sys
from typing import Dict, List, Tuple, Optional

# 解除CSV字段长度限制
csv.field_size_limit(sys.maxsize)

CONFIG = {
    "FINAL_TARGET_HEADER": [],   # 先占位，后面再赋值也行
}

CONFIG['FINAL_TARGET_HEADER'] = [
    "icon", "type", "label", "context", "declaringClass", 
    "hascallback", "uid", "container", "rawtext", "apk", 
    "id", "callback", "api", "sensitive", "apitype", 
    "category", "depth",
    "relation_strength"  # 新增列：关系强度
]


# ===================== 辅助函数 =====================
def read_csv_file(file_path: str, delimiter: str = ';') -> Tuple[List[str], List[List[str]]]:
    """读取CSV文件，返回标题和数据"""
    if not os.path.exists(file_path):
        return [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar='"')
            header = next(reader)
            data = [row for row in reader if row]
            return header, data
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return [], []

def build_col_index(header: List[str]) -> Dict[str, int]:
    """根据header生成 列名->索引 的映射（自动去引号）"""
    return {col.strip('"'): idx for idx, col in enumerate(header)}


def extract_apk_name_from_row(row: List[str], header: List[str]) -> str:
    """从行中提取APK名称（基于列名）"""
    col_map = build_col_index(header)

    apk_idx = col_map.get("apk", -1)
    if apk_idx != -1 and apk_idx < len(row):
        return row[apk_idx].strip('"').replace('.apk', '')
    return ''


# def create_field_mapping(ui_header: List[str], api_header: List[str], target_header: List[str]) -> Dict:
#     ui_col = build_col_index(ui_header)
#     api_col = build_col_index(api_header)

#     mapping = {}

#     for target_field in target_header:
#         mapping[target_field] = {'source': 'none', 'index': -1}

#         # 👇 declaringClass 特判：强制用 API
#         if target_field == "declaringClass" and "declaringClass" in api_col:
#             mapping[target_field] = {'source': 'api', 'index': api_col["declaringClass"]}
#             continue

#         # 其他字段：UI优先，其次API
#         if target_field in ui_col:
#             mapping[target_field] = {'source': 'ui', 'index': ui_col[target_field]}
#         elif target_field in api_col:
#             mapping[target_field] = {'source': 'api', 'index': api_col[target_field]}

#     return mapping
def create_field_mapping(ui_header: List[str], api_header: List[str], target_header: List[str]) -> Dict:
    ui_col = build_col_index(ui_header)
    api_col = build_col_index(api_header)

    mapping = {}
    for target_field in target_header:
        # 同时保存该字段在 UI 和 API 中的索引（如果不存在则为 -1）
        mapping[target_field] = {
            'ui_index': ui_col.get(target_field, -1),
            'api_index': api_col.get(target_field, -1)
        }
    return mapping



def process_ui_only_case(ui_header: List[str], ui_data: List[List[str]], mapping: Dict, writer) -> int:
    """处理只有UI文件的情况 - 特别注意填充关系强度字段"""
    print("处理UI-only数据...")
    final_records_count = 0

    for ui_row in ui_data:
        ui_row = [c.strip('"') for c in ui_row]
        final_row = []

        # for target_field in CONFIG['FINAL_TARGET_HEADER']:
        #     if target_field == "relation_strength":
        #         final_row.append("WEAK")
        #         continue

        #     field_info = mapping[target_field]
        #     if field_info['source'] == 'ui' and field_info['index'] < len(ui_row):
        #         final_row.append(ui_row[field_info['index']])
        #     else:
        #         final_row.append("")

        # writer.writerow(final_row)
        # final_records_count += 1
        for target_field in CONFIG['FINAL_TARGET_HEADER']:
            if target_field == "relation_strength":
                final_row.append("WEAK")
                continue

            field_info = mapping[target_field]
            # 👇 改为使用 ui_index 判断
            if field_info['ui_index'] != -1 and field_info['ui_index'] < len(ui_row):
                final_row.append(ui_row[field_info['ui_index']])
            else:
                final_row.append("")
        writer.writerow(final_row)
        final_records_count += 1

    return final_records_count



def process_api_only_case(api_header: List[str], api_data: List[List[str]], mapping: Dict, writer) -> int:
    """处理只有API文件的情况 - 特别注意填充关系强度字段和关键字段（apk、type、uid、declaringClass）"""
    print("处理API-only数据...")
    final_records_count = 0

    api_col = build_col_index(api_header)

    apk_idx = api_col.get("apk", -1)
    uid_idx = api_col.get("uid", -1)
    declaring_class_idx = api_col.get("declaringClass", -1)
    type_idx = api_col.get("type", -1)

    print(f"    API关键字段索引: apk={apk_idx}, type={type_idx}, uid={uid_idx}, declaringClass={declaring_class_idx}")

    for api_row in api_data:
        api_row = [c.strip('"') for c in api_row]

        apk_value = api_row[apk_idx].replace('.apk', '') if apk_idx != -1 and apk_idx < len(api_row) else ""
        type_value = api_row[type_idx] if type_idx != -1 and type_idx < len(api_row) else ""
        uid_value = api_row[uid_idx] if uid_idx != -1 and uid_idx < len(api_row) else ""
        declaring_class_value = api_row[declaring_class_idx] if declaring_class_idx != -1 and declaring_class_idx < len(api_row) else ""

        final_row = []

        for target_field in CONFIG['FINAL_TARGET_HEADER']:
            if target_field == "relation_strength":
                final_row.append("WEAK")
                continue

            field_info = mapping[target_field]

            if target_field == "apk":
                final_row.append(apk_value)
            elif target_field == "type":
                final_row.append(type_value)
            elif target_field == "uid":
                final_row.append(uid_value)
            elif target_field == "declaringClass":
                final_row.append(declaring_class_value)

            # elif field_info['source'] == 'api' and field_info['index'] < len(api_row):
            #     final_row.append(api_row[field_info['index']])
            # else:
            #     final_row.append("")

        # writer.writerow(final_row)
        # final_records_count += 1
            elif field_info['api_index'] != -1 and field_info['api_index'] < len(api_row):
                final_row.append(api_row[field_info['api_index']])
            else:
                final_row.append("")
        writer.writerow(final_row)
        final_records_count += 1

    print(f"    处理了 {final_records_count} 条API-only记录")
    return final_records_count


# ===================== 核心函数 =====================
def merge_apk_data(apk_name: str, api_header, api_data,
                   ui_header, ui_data, writer) -> Dict:

    result_info = {
        'apk': apk_name,
        'has_api': bool(api_data),
        'has_ui': bool(ui_data),
        'ui_only_count': 0,
        'api_only_count': 0,
        'matched_count': 0,
        'error': None
    }

    print(f"  APK {apk_name}: UI数据={bool(ui_data)}, API数据={bool(api_data)}")

    # ================== 只有UI ==================
    if ui_data and not api_data:
        mapping = create_field_mapping(ui_header, [], CONFIG['FINAL_TARGET_HEADER'])
        count = process_ui_only_case(ui_header, ui_data, mapping, writer)
        result_info['ui_only_count'] = count
        result_info['matched_count'] = count
        return result_info

    # ================== 只有API ==================
    if api_data and not ui_data:
        mapping = create_field_mapping([], api_header, CONFIG['FINAL_TARGET_HEADER'])
        count = process_api_only_case(api_header, api_data, mapping, writer)
        result_info['api_only_count'] = count
        result_info['matched_count'] = count
        return result_info

    # ================== 同时有UI和API ==================
    ui_col = build_col_index(ui_header)
    api_col = build_col_index(api_header)

    ui_uid_idx = ui_col.get("uid", -1)
    ui_container_idx = ui_col.get("container", -1)

    api_uid_idx = api_col.get("uid", -1)
    api_declaring_class_idx = api_col.get("declaringClass", -1)

    ui_by_uid, api_by_uid = {}, {}

    # 组织UI数据
    for row in ui_data:
        if ui_uid_idx != -1 and ui_uid_idx < len(row):
            uid = row[ui_uid_idx].strip('"')
            ui_by_uid.setdefault(uid, []).append({
                'row': row,
                'container': row[ui_container_idx].strip('"') if ui_container_idx != -1 and ui_container_idx < len(row) else "",
                'matched': False
            })

    # 组织API数据
    for row in api_data:
        if api_uid_idx != -1 and api_uid_idx < len(row):
            uid = row[api_uid_idx].strip('"')
            api_by_uid.setdefault(uid, []).append({
                'row': row,
                'matched': False
            })

    mapping = create_field_mapping(ui_header, api_header, CONFIG['FINAL_TARGET_HEADER'])

    matched_count = 0

    # ================== 匹配UI + API ==================
    for uid, ui_items in ui_by_uid.items():
        api_items = api_by_uid.get(uid, [])
        if not api_items:
            continue

        for ui_item in ui_items:
            for api_item in api_items:
                ui_row = [c.strip('"') for c in ui_item['row']]
                api_row = [c.strip('"') for c in api_item['row']]

                # ====== 保持你原来的匹配逻辑不变 ======
                ui_container = ui_item['container']
                api_declaring_class = api_row[api_declaring_class_idx] if api_declaring_class_idx < len(api_row) else ""

                if ui_container == api_declaring_class:
                    if matched_count < 5:  # 只打印前5条匹配的关系强度判断
                        print(f"  UI container: {ui_container}")
                        print(f"  API declaringClass: {api_declaring_class}")
                        print(f"  判断关系强度为 STRONG\n")
                    relation_strength = "STRONG"
                elif api_declaring_class == "default_value":
                    relation_strength = "MEDIUM"
                else:
                    relation_strength = "WEAK"
                # =====================================

                # final_row = []
                # for target_field in CONFIG['FINAL_TARGET_HEADER']:
                #     if target_field == "relation_strength":
                #         final_row.append(relation_strength)
                #         continue

                #     field_info = mapping[target_field]

                #     if field_info['source'] == 'ui' and field_info['index'] < len(ui_row):
                #         final_row.append(ui_row[field_info['index']])
                #     elif field_info['source'] == 'api' and field_info['index'] < len(api_row):
                #         final_row.append(api_row[field_info['index']])
                #     else:
                #         final_row.append("")

                # writer.writerow(final_row)
                # matched_count += 1
                # ui_item['matched'] = True
                # api_item['matched'] = True
                final_row =[]
                for target_field in CONFIG['FINAL_TARGET_HEADER']:
                    if target_field == "relation_strength":
                        final_row.append(relation_strength)
                        continue

                    field_info = mapping[target_field]
                    
                    # declaringClass 强制优先用 API
                    if target_field == "declaringClass" and field_info['api_index'] != -1 and field_info['api_index'] < len(api_row):
                        final_row.append(api_row[field_info['api_index']])
                        continue

                    val = ""
                    # 优先从 UI 取值
                    if field_info['ui_index'] != -1 and field_info['ui_index'] < len(ui_row):
                        val = ui_row[field_info['ui_index']]
                    
                    # 💡 核心修复：如果 UI 里这个字段是空的，或者压根没这个字段，立刻尝试从 API 补充！
                    if val == "" and field_info['api_index'] != -1 and field_info['api_index'] < len(api_row):
                        val = api_row[field_info['api_index']]
                    
                    # 💡 核心修复：清理 apk 字段的后缀
                    if target_field == "apk":
                        val = val.replace(".apk", "")
                        
                    final_row.append(val)

                writer.writerow(final_row)
                matched_count += 1
                ui_item['matched'] = True
                api_item['matched'] = True

    result_info['matched_count'] = matched_count
    print(f"    匹配成功 {matched_count} 条")

    # ================== 未匹配UI ==================
    # ui_only_count = 0
    # for ui_items in ui_by_uid.values():
    #     for ui_item in ui_items:
    #         if not ui_item['matched']:
    #             ui_row = [c.strip('"') for c in ui_item['row']]
    #             final_row = []

    #             for target_field in CONFIG['FINAL_TARGET_HEADER']:
    #                 if target_field == "relation_strength":
    #                     final_row.append("WEAK")
    #                     continue

    #                 field_info = mapping[target_field]
    #                 if field_info['source'] == 'ui' and field_info['index'] < len(ui_row):
    #                     final_row.append(ui_row[field_info['index']])
    #                 else:
    #                     final_row.append("")

    #             writer.writerow(final_row)
    #             ui_only_count += 1

    # result_info['ui_only_count'] = ui_only_count
    ui_only_count = 0
    for ui_items in ui_by_uid.values():
        for ui_item in ui_items:
            if not ui_item['matched']:
                ui_row =[c.strip('"') for c in ui_item['row']]
                final_row =[]
                for target_field in CONFIG['FINAL_TARGET_HEADER']:
                    if target_field == "relation_strength":
                        final_row.append("WEAK")
                        continue
                    field_info = mapping[target_field]
                    # 未匹配的UI行，直接从 ui_index 拿数据
                    if field_info['ui_index'] != -1 and field_info['ui_index'] < len(ui_row):
                        final_row.append(ui_row[field_info['ui_index']])
                    else:
                        final_row.append("")
                writer.writerow(final_row)
                ui_only_count += 1
    result_info['ui_only_count'] = ui_only_count

    # ================== 未匹配API ==================
    # api_only_count = 0
    # for api_items in api_by_uid.values():
    #     for api_item in api_items:
    #         if not api_item['matched']:
    #             api_row = [c.strip('"') for c in api_item['row']]
    #             final_row = []

    #             for target_field in CONFIG['FINAL_TARGET_HEADER']:
    #                 if target_field == "relation_strength":
    #                     final_row.append("WEAK")
    #                     continue

    #                 field_info = mapping[target_field]
    #                 if field_info['source'] == 'api' and field_info['index'] < len(api_row):
    #                     final_row.append(api_row[field_info['index']])
    #                 else:
    #                     final_row.append("")

    #             writer.writerow(final_row)
    #             api_only_count += 1
    api_only_count = 0
    for api_items in api_by_uid.values():
        for api_item in api_items:
            if not api_item['matched']:
                api_row =[c.strip('"') for c in api_item['row']]
                final_row =[]
                for target_field in CONFIG['FINAL_TARGET_HEADER']:
                    if target_field == "relation_strength":
                        final_row.append("WEAK")
                        continue
                    field_info = mapping[target_field]

                    val = ""
                    if field_info['api_index'] != -1 and field_info['api_index'] < len(api_row):
                        val = api_row[field_info['api_index']]
                    # 💡 核心修复：清理 apk 字段的后缀
                    if target_field == "apk":
                        val = val.replace(".apk", "")
                    final_row.append(val)
                    
                writer.writerow(final_row)
                api_only_count += 1

    result_info['api_only_count'] = api_only_count
    result_info['matched_count'] += ui_only_count + api_only_count

    print(f"    UI-only: {ui_only_count}, API-only: {api_only_count}")
    print(f"    总记录: {result_info['matched_count']}")

    return result_info


# ===================== 处理所有APK =====================
def process_and_merge_all_apks(api_file_path: str, ui_file_path: str, output_dir: str) -> List[Dict]:
    """处理所有APK，输出单个文件"""
    print("步骤1: 读取并组织数据\n" + "-"*40)
    os.makedirs(output_dir, exist_ok=True)

    api_header, api_data = read_csv_file(api_file_path)
    ui_header, ui_data = read_csv_file(ui_file_path)
    
    # 如果没有UI数据，检查是否有API数据
    if not ui_data and not api_data:
        print(f"错误: UI文件 {ui_file_path} 和 API文件 {api_file_path} 都为空或无法读取")
        return []
    
    # 分别按APK组织数据
    api_by_apk, ui_by_apk = {}, {}
    
    # 处理API数据
    for row in api_data:
        apk_name = extract_apk_name_from_row(row, api_header)
        if apk_name:
            api_by_apk.setdefault(apk_name, []).append(row)
    
    # 处理UI数据
    for row in ui_data:
        apk_name = extract_apk_name_from_row(row, ui_header)
        if apk_name:
            ui_by_apk.setdefault(apk_name, []).append(row)
    
    # 合并所有APK（包括只有API或只有UI的情况）
    all_apks = set(list(api_by_apk.keys()) + list(ui_by_apk.keys()))

    final_output_path = os.path.join(output_dir, "merge_ui_api_with_relation.txt")
    print(f"\n步骤2: 合并所有数据到单个文件\n输出文件: {final_output_path}\n" + "-"*40)
    results = []

    with open(final_output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_ALL)
        writer.writerow(CONFIG['FINAL_TARGET_HEADER'])
        
        for apk in sorted(all_apks):
            print(f"\n处理APK: {apk}")
            ui_apk_data = ui_by_apk.get(apk, [])
            api_apk_data = api_by_apk.get(apk, [])
            result = merge_apk_data(apk, api_header, api_apk_data, ui_header, ui_apk_data, writer)
            results.append(result)

    print(f"\n步骤3: 处理完成\n已将所有数据合并到: {final_output_path}")
    return results

# ===================== 生成汇总 =====================
def generate_summary(results: List[Dict], output_dir: str):
    """生成处理结果摘要"""
    print("\n" + "="*50 + "\n处理结果摘要\n" + "="*50)
    total_records = apk_with_api = apk_with_ui = apk_with_both = apk_ui_only = apk_api_only = error_count = 0
    for result in results:
        apk = result['apk']
        has_api = result.get('has_api', False)
        has_ui = result.get('has_ui', False)
        
        if result.get('error'):
            print(f"{apk}: 错误 - {result['error']}")
            error_count += 1
        else:
            count = result.get('matched_count', 0)
            ui_only = result.get('ui_only_count', 0)
            api_only = result.get('api_only_count', 0)
            matched = count - ui_only - api_only
            
            # 判断APK类型
            if has_ui and has_api:
                status = "(UI+API数据)"
                apk_with_both += 1
            elif has_ui and not has_api:
                status = "(仅UI数据)"
                apk_ui_only += 1
            elif not has_ui and has_api:
                status = "(仅API数据)"
                apk_api_only += 1
            else:
                status = "(无数据)"
            
            details = []
            if matched > 0:
                details.append(f"匹配: {matched}")
            if ui_only > 0:
                details.append(f"UI-only: {ui_only}")
            if api_only > 0:
                details.append(f"API-only: {api_only}")
            
            details_str = f" ({', '.join(details)})" if details else ""
            print(f"{apk}: {count} 条记录 {status}{details_str}")
            
            total_records += count
            if has_api: apk_with_api += 1
            if has_ui: apk_with_ui += 1

    print("-"*50)
    print(f"总共处理了 {len(results)} 个APK:")
    print(f"  - 同时有UI和API数据的APK: {apk_with_both} 个")
    print(f"  - 仅UI数据的APK: {apk_ui_only} 个")
    print(f"  - 仅API数据的APK: {apk_api_only} 个")
    print(f"  - 有API数据的APK总数: {apk_with_api} 个")
    print(f"  - 有UI数据的APK总数: {apk_with_ui} 个")
    if error_count > 0: 
        print(f"  - 处理出错的APK: {error_count} 个")
    print(f"生成了 {total_records} 条最终合并记录")

    # 保存摘要文件
    summary_file = os.path.join(output_dir, "merge_ui_api_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("APK合并处理汇总报告\n" + "="*50 + "\n")
        f.write(f"处理的APK数量: {len(results)}\n")
        f.write(f"APK类型分布:\n")
        f.write(f"  - 同时有UI和API数据的APK: {apk_with_both} 个\n")
        f.write(f"  - 仅UI数据的APK: {apk_ui_only} 个\n")
        f.write(f"  - 仅API数据的APK: {apk_api_only} 个\n")
        f.write(f"  - 有API数据的APK总数: {apk_with_api} 个\n")
        f.write(f"  - 有UI数据的APK总数: {apk_with_ui} 个\n")
        if error_count > 0: 
            f.write(f"  - 处理出错的APK: {error_count} 个\n")
        f.write(f"总最终合并记录数: {total_records}\n\n")
        f.write("各APK处理结果:\n" + "-"*50 + "\n")
        for result in results:
            apk = result['apk']
            has_api = result.get('has_api', False)
            has_ui = result.get('has_ui', False)
            
            if result.get('error'):
                f.write(f"{apk}: 错误 - {result['error']}\n")
            else:
                count = result.get('matched_count', 0)
                ui_only = result.get('ui_only_count', 0)
                api_only = result.get('api_only_count', 0)
                matched = count - ui_only - api_only
                
                if has_ui and has_api:
                    status = "(UI+API数据)"
                elif has_ui and not has_api:
                    status = "(仅UI数据)"
                elif not has_ui and has_api:
                    status = "(仅API数据)"
                else:
                    status = "(无数据)"
                
                details = []
                if matched > 0:
                    details.append(f"匹配记录: {matched}")
                if ui_only > 0:
                    details.append(f"UI-only: {ui_only}")
                if api_only > 0:
                    details.append(f"API-only: {api_only}")
                
                details_str = f" ({', '.join(details)})" if details else ""
                f.write(f"{apk}: {count} 条记录 {status}{details_str}\n")
    print(f"\n汇总报告已保存到: {summary_file}\n" + "="*50)

# ===================== 主程序 =====================
def main():
    print("="*60 + "\nAPK数据合并工具 - 支持UI-only和API-only\n" + "="*60)
    # api_file_path = "./api-results/api_10.txt"
    # ui_file_path = "outputBenignUi_10.txt"
    # api_file_path = "./api-results/api_3.txt"
    # ui_file_path = "outputBenignUi_3.txt"
    # api_file_path = "./api-results/api_GP.txt"
    # ui_file_path = "outputBenignUi_GP.txt"
    # api_file_path = "./api-results/api_benign.txt"
    # ui_file_path = "outputBenignUi_benign.txt"
    # api_file_path = "./api-results/api_malware.txt"
    # ui_file_path = "outputBenignUi_malware.txt"
    parser = argparse.ArgumentParser(description="Merge Backstage UI/OCR and API exports")
    parser.add_argument('--api-file', default='./api-results/api_train.txt')
    parser.add_argument('--ui-file', default='outputBenignUi_train.txt')
    parser.add_argument('--output-dir', default='./merge/train')
    args = parser.parse_args()
    api_file_path = args.api_file
    ui_file_path = args.ui_file


    # output_dir = "./merge/GP"
    # output_dir = "./merge/benign"
    # output_dir = "./merge/malware"
    output_dir = args.output_dir

    if not os.path.exists(ui_file_path) and not os.path.exists(api_file_path):
        print(f"错误: UI文件 {ui_file_path} 和 API文件 {api_file_path} 都不存在")
        return
    
    print(f"API文件: {api_file_path} {'(存在)' if os.path.exists(api_file_path) else '(不存在)'}")
    print(f"UI文件: {ui_file_path} {'(存在)' if os.path.exists(ui_file_path) else '(不存在)'}")
    print(f"输出目录: {output_dir}")

    results = process_and_merge_all_apks(api_file_path, ui_file_path, output_dir)
    if results:
        generate_summary(results, output_dir)
    else:
        print("\n没有找到可处理的APK，程序结束")

    print("\n" + "="*60 + "\n程序执行完成\n" + "="*60)

if __name__ == "__main__":
    main()