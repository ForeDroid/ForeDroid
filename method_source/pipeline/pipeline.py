import argparse
import fcntl
import concurrent.futures as futures
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from output_layout import replace_if_changed,raw_root
import subprocess
import sys
import tempfile
import time
import os
import json
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
from datetime import datetime

# ====================== 新增：仅记录最终统计的函数 ======================
def write_final_stats(message, stats_file="./final_stats.txt"):
    """
    仅写入每个函数的最终统计结果到TXT，最简实现
    """
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    # 追加模式，每行加时间戳
    with open(stats_file, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


# ====================== 1️⃣ 提取调用链 ======================
def extract_call_chains(input_root_folder, output_csv_path):
    """
    遍历 APK 文件夹，提取调用链 JSON 数据，生成 CSV。
    """
    if os.path.exists(output_csv_path):
        print(f"[SKIP] {output_csv_path} already exists, skip extract_call_chains.")
        return

    records = []
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    apk_folders = [f for f in os.listdir(input_root_folder) if os.path.isdir(os.path.join(input_root_folder, f))]
    total_record_count = 0

    for apk_folder in tqdm(apk_folders, desc="Processing APK folders"):
        json_file = os.path.join(input_root_folder, apk_folder, "CallGraphInfo", "rule_analysis_report.json")
        if not os.path.exists(json_file):
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        for item in data:
            findings = item.get('findings', [])
            if len(findings) < 2:
                continue
            sapcg_list = findings[0]
            call_chain_list = findings[1]
            if not sapcg_list:
                continue

            sapcg_info = sapcg_list[0]
            sensitiveAPI = sapcg_info.get('sensitiveAPI', '')

            for call_entry in call_chain_list:
                call_chain = call_entry.get('call_chain', [])
                call_chain_str = " -> ".join(call_chain)
                entry_method = call_chain[0] if call_chain else ''

                entry_point_total = call_entry.get('entry_point', {})
                entry_point = entry_point_total.get('entry_point', '')
                entry_actions = ", ".join(entry_point_total.get('actions', []))
                origin_entry_point = entry_point_total.get('origin_entry_point', '')
                origin_actions = ", ".join(entry_point_total.get('origin_actions', []))
                origin_call_chain = entry_point_total.get('origin_call_chain', [])
                origin_call_chain_str = " -> ".join(origin_call_chain)
                origin_entry_method = origin_call_chain[0] if origin_call_chain else ''

                record = {
                    'apk_name': apk_folder,
                    'entry_method': entry_method,
                    'sensitiveAPI': sensitiveAPI,
                    'call_chain': call_chain_str,
                    'entry_point': entry_point,
                    'entry_point_actions': entry_actions,
                    'origin_entry_point': origin_entry_point,
                    'origin_entry_point_actions': origin_actions,
                    'origin_call_chain': origin_call_chain_str,
                    'origin_entry_method': origin_entry_method
                }

                records.append(record)
                if len(records)>=10000:
                    pd.DataFrame(records).to_csv(output_csv_path,mode='a',header=total_record_count==0,index=False,encoding='utf-8')
                    total_record_count+=len(records)
                    records.clear()

    if records:
        pd.DataFrame(records).to_csv(output_csv_path,mode='a',header=total_record_count==0,index=False,encoding='utf-8')
        total_record_count+=len(records)
    if not total_record_count:
        pd.DataFrame(columns=['apk_name','entry_method','sensitiveAPI','call_chain','entry_point',
            'entry_point_actions','origin_entry_point','origin_entry_point_actions','origin_call_chain',
            'origin_entry_method']).to_csv(output_csv_path,index=False)
    print(f"[INFO] 调用链提取完成，CSV 保存为: {output_csv_path}")

    # 仅记录最终结果
    write_final_stats(f"[INFO] 调用链提取完成：生成 {total_record_count} 条记录，CSV保存为 {output_csv_path}")


# ====================== 1️⃣b 合并调用链列（保留 origin 信息） ======================
def merge_columns(input_csv_path, output_csv_path, chunksize=100000):
    if os.path.exists(output_csv_path):
        print(f"[SKIP] {output_csv_path} already exists, skip merge_columns.")
        return

    print("[INFO] 开始合并调用链列...")
    first_write = True
    current_id = 1

    for chunk in pd.read_csv(input_csv_path, encoding='utf-8', chunksize=chunksize):
        if 'sapcg_file' in chunk.columns:
            chunk = chunk.drop(columns=['sapcg_file'])
        
        chunk['entry_point'] = chunk['origin_entry_point'].fillna(chunk['entry_point'])
        chunk['call_chain'] = chunk['origin_call_chain'].fillna('') + " -> " + chunk['call_chain']
        chunk['entry_point_actions'] = chunk['origin_entry_point_actions'].fillna(chunk['entry_point_actions'])
        chunk['entry_method'] = chunk['origin_entry_method'].fillna(chunk['entry_method'])
        
        chunk = chunk.drop(columns=[
            'origin_entry_point',
            'origin_call_chain',
            'origin_entry_point_actions',
            'origin_entry_method'
        ])

        chunk = chunk[chunk['entry_point'].notnull() & (chunk['entry_point'] != '')]
        chunk['id'] = range(current_id, current_id + len(chunk))
        current_id += len(chunk)

        chunk.to_csv(output_csv_path, mode='a', header=first_write, index=False, encoding='utf-8')
        first_write = False

    print(f"[INFO] 调用链列合并完成，CSV 保存为: {output_csv_path}")

     # 仅记录最终结果
    write_final_stats(f"[INFO] 调用链列合并完成，CSV保存为 {output_csv_path}")



def merge_callchain_gui_info(input_csv_path_1, input_csv_path_2, output_csv_path):
    """
    合并调用链与GUI语义信息
    改进：弱绑定（页面环境）的聚合方式与强绑定完全一致，分字段聚合后再合并。
    """
    if os.path.exists(output_csv_path):
        print(f"[SKIP] {output_csv_path} already exists, skip merge_callchain_gui_info.")
        return

    print("[INFO] 开始合并调用链和GUI信息...")
    
    # 1. 读取调用链数据
    df_call = pd.read_csv(input_csv_path_1, encoding='utf-8')
    print(f"[INFO] 调用链记录数: {len(df_call)}")
    
    # 辅助列用于强匹配
    df_call['apk'] = df_call['apk_name']
    df_call['call_chain_split'] = df_call['call_chain'].fillna('').str.split(" -> ")
    exploded = df_call[['id', 'apk', 'call_chain_split']].explode('call_chain_split')
    exploded = exploded.rename(columns={'call_chain_split': 'callback'}).dropna()
    call_chain_methods = set(exploded['callback'])
    
    # 保留原有中间输出
    method_path = output_csv_path.replace("merged_gui_callchain.csv", "call_chain_methods.csv")
    pd.DataFrame({'唯一方法': list(call_chain_methods)}).to_csv(method_path, index=False, encoding='utf-8')
    print("[INFO] 唯一方法已保存到 call_chain_unique_methods.csv")
    
    # ====================== 2️⃣ 读取GUI数据 ======================
    df_gui = pd.read_csv(input_csv_path_2, sep=';', encoding='utf-8')
    df_gui = df_gui.drop(columns=['id'], errors='ignore')
    gui_keep_cols = [
        'apk', 
        'callback', 
        'label', 
        'context', 
        'rawtext',
        'declaringClass', 
        # 'uid', 
        'container',
        # 'type', 
        # 'sensitive'
    ]
    df_gui_all = df_gui[[c for c in gui_keep_cols if c in df_gui.columns]].copy()
    print(f"[INFO] GUI记录数: {len(df_gui_all)}")

    # ================= 🆕 新增：数据清洗与修复 =================
    # 修复 declaringClass 为 default_value 的情况，使用 container 替代
    invalid_mask = (df_gui_all['declaringClass'] == 'default_value')
    fix_count = invalid_mask.sum()
    if fix_count > 0:
        print(f"[INFO] 检测到 {fix_count} 条数据的 declaringClass 无效，正在使用 container 修复...")
        df_gui_all.loc[invalid_mask, 'declaringClass'] = df_gui_all.loc[invalid_mask, 'container']
    # ==========================================================


    # 2. 读取所有 GUI 数据
    # df_gui_all = pd.read_csv(input_csv_path_2, sep=';', encoding='utf-8', dtype={'uid': str})
    
    # # 仅保留相关的 APK 数据
    # df_gui_all = df_gui_all[df_gui_all['apk'].isin(df_call['apk'])].copy()
    # print(f"[INFO] 相关APK的GUI总记录数: {len(df_gui_all)}")

    # ================= 策略 A: 强绑定 (Strong Binding) =================
    # 逻辑：GUI.callback == CallChain.method
    df_gui_strong = df_gui_all[
        df_gui_all['apk'].isin(df_call['apk']) &
        df_gui_all['callback'].isin(call_chain_methods)
    ]
    print(f"[INFO] 强关联GUI匹配数: {len(df_gui_strong)}")
    
    # df_gui_strong = df_gui_strong.merge(exploded[['id', 'callback']], on='callback', how='left')
    df_gui_strong = df_gui_strong.merge(
        exploded[['id', 'apk', 'callback']],
        on=['apk', 'callback'],
        how='inner'
    )
    
    print(f"[INFO] 合并后共有 {len(df_gui_strong)} 条GUI记录")

    # 你的强绑定聚合逻辑 (保持不变)
    gui_agg_strong = df_gui_strong.groupby('id', as_index=False).agg({
        'label': lambda x: ' ||| '.join(set(str(item) for item in x if pd.notna(item) and str(item).strip() != '')),
        'context': lambda x: ' ||| '.join(set(str(item) for item in x if pd.notna(item) and str(item).strip() != '')),
        'rawtext': lambda x: ' ||| '.join(set(str(item) for item in x if pd.notna(item) and str(item).strip() != '')),
        'declaringClass': lambda x: ', '.join(set(str(item) for item in x if pd.notna(item))),
        'container': lambda x: ', '.join(set(str(item) for item in x if pd.notna(item))),
        # 'type': lambda x: ', '.join(set(str(item) for item in x if pd.notna(item))),
        # 'sensitive': lambda x: ', '.join(set(str(item) for item in x if pd.notna(item)))
    })
    # 保存强聚合文件
    gui_strong_path = output_csv_path.replace("merged_gui_callchain.csv", "gui_strong.csv")
    gui_agg_strong.to_csv(gui_strong_path, index=False, encoding='utf-8')

    # ================= 策略 B: 弱绑定 (Context Injection) =================
    # 逻辑：利用没有 callback 的行，按 Activity (apk + declaringClass) 聚合
    # 目标：生成属于该页面的 label, context, rawtext 等集合
    
    mask_no_callback = df_gui_all['callback'].isna() | (df_gui_all['callback'] == '')
    # 只要这几个字段里有任意一个有值，就说明是有效的环境信息
    mask_has_content = df_gui_all['rawtext'].notna() | df_gui_all['label'].notna() | df_gui_all['context'].notna()
    
    df_gui_passive = df_gui_all[mask_no_callback & mask_has_content].copy()
    
    print(f"[INFO] 提取到 {len(df_gui_passive)} 条页面环境数据，正在按Activity聚合...")

    # 弱绑定聚合逻辑：与强绑定完全一致，只是 GroupBy 的对象变了
    # 注意：这里去掉了 declaringClass，因为它是分组键, 把同一个 Activity 里的所有静态GUI语义合并成一个页面语义包
    # 这个页面整体在表达什么功能
    context_agg = df_gui_passive.groupby(['apk', 'declaringClass'], as_index=False).agg({
        'label': lambda x: ' ||| '.join(set(str(item) for item in x if pd.notna(item) and str(item).strip() != '')),
        'context': lambda x: ' ||| '.join(set(str(item) for item in x if pd.notna(item) and str(item).strip() != '')),
        'rawtext': lambda x: ' ||| '.join(set(str(item) for item in x if pd.notna(item) and str(item).strip() != '')),
        # declaringClass 不需要聚合，因为它是分组键
        'container': lambda x: ', '.join(set(str(item) for item in x if pd.notna(item))),
        # 'type': lambda x: ', '.join(set(str(item) for item in x if pd.notna(item))),
        # 'sensitive': lambda x: ', '.join(set(str(item) for item in x if pd.notna(item)))
    })
    # 保存弱聚合文件
    gui_weak_path = output_csv_path.replace("merged_gui_callchain.csv", "gui_weak.csv")
    context_agg.to_csv(gui_weak_path, index=False, encoding='utf-8')


    # 重命名列，防止 merge 时冲突 (加上 page_ 前缀)
    rename_dict = {
        'label': 'page_label',
        'context': 'page_context',
        'rawtext': 'page_rawtext',
        'container': 'page_container',
        # 'type': 'page_type',
        # 'sensitive': 'page_sensitive'
    }
    context_agg.rename(columns=rename_dict, inplace=True)

    # ================= 合并回主表 =================
    print("[INFO] 正在执行多阶段合并...")
    
    # 1. 合并强关联信息 (Button Level)
    df_merged = df_call.merge(gui_agg_strong, on='id', how='left')
    
    # 2. 合并页面环境信息 (Page Level)
    # 匹配条件: 同一个 APK，且 入口点类名 (entry_point) 等于 GUI 的 declaringClass
    df_merged = df_merged.merge(
        context_agg, 
        left_on=['apk', 'entry_point'], 
        right_on=['apk', 'declaringClass'], 
        how='left'
    )
    
    # ================= 字段融合 (Fusion) =================
    # 将 "按钮属性" 和 "页面属性" 合并到同一个字段中
    # 策略：如果两者都有，用 ' ||| ' 连接；如果只有一个，取那一个。
    
    def fuse_columns(row, base_col, page_col, sep=' ||| '):
        val_base = str(row[base_col]) if pd.notna(row[base_col]) and str(row[base_col]).strip() != '' else ""
        val_page = str(row[page_col]) if pd.notna(row[page_col]) and str(row[page_col]).strip() != '' else ""
        
        if val_base and val_page:
            return f"{val_base}{sep}{val_page}"
        elif val_base:
            return val_base
        elif val_page:
            return val_page
        else:
            return pd.NA

    print("[INFO] 正在融合 Button 和 Page 的语义字段...")
    
    # 对每个字段进行融合
    # 文本类字段用 ||| 分隔
    df_merged['label'] = df_merged.apply(lambda r: fuse_columns(r, 'label', 'page_label', ' ||| '), axis=1)
    df_merged['context'] = df_merged.apply(lambda r: fuse_columns(r, 'context', 'page_context', ' ||| '), axis=1)
    df_merged['rawtext'] = df_merged.apply(lambda r: fuse_columns(r, 'rawtext', 'page_rawtext', ' ||| '), axis=1)
    
    # 结构类字段用 , 分隔 (保持你原有的风格)
    df_merged['container'] = df_merged.apply(lambda r: fuse_columns(r, 'container', 'page_container', ', '), axis=1)
    # df_merged['type'] = df_merged.apply(lambda r: fuse_columns(r, 'type', 'page_type', ', '), axis=1)
    # df_merged['sensitive'] = df_merged.apply(lambda r: fuse_columns(r, 'sensitive', 'page_sensitive', ', '), axis=1)

    # 填补 declaringClass (如果强匹配没匹配到，就用页面匹配的填)
    if 'declaringClass_y' in df_merged.columns: # declaringClass_y 来自 context_agg
        df_merged['declaringClass_x'] = df_merged['declaringClass_x'].fillna(df_merged['declaringClass_y'])

    # ================= 清理与保存 =================
    # 删除临时列
    cols_to_drop = [
        'declaringClass_y', 
        'page_label', 'page_context', 'page_rawtext', 
        'page_container', 'page_type', 'page_sensitive',
        'container'
    ]
    df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns], inplace=True)
    
    # 修复列名
    if 'declaringClass_x' in df_merged.columns:
        df_merged.rename(columns={'declaringClass_x': 'declaringClass'}, inplace=True)

    # 这是新加的内容（重新生成id）
    print("[INFO] 重新生成连续id...")
    df_merged = df_merged.reset_index(drop=True)
    df_merged["id"] = df_merged.index + 1


    df_merged.to_csv(output_csv_path, index=False, encoding='utf-8')

    # 统计有多少数据因此变得丰富了
    enriched_count = df_merged[df_merged['rawtext'].str.contains(r'\|\|\|', na=False)].shape[0]
    
    print(f"[INFO] 数据融合完成。")
    print(f"[INFO] 有 {enriched_count} 条数据同时包含了 Button 和 Page 的语义信息。")
    print(f"[DONE] 结果保存到: {output_csv_path}")
    # 仅记录最终结果
    write_final_stats(f"[INFO] 调用链+GUI合并完成：总记录 {len(df_merged)} 条，{enriched_count} 条包含Button+Page语义，CSV保存为 {output_csv_path}")




# ====================== 3️⃣ 严格过滤 ======================
def deduplicate_scenarios(input_csv_path, output_csv_path) -> None:
    """
    对功能场景数据集进行去重，将“调用记录级数据”转换为“功能场景级样本”。
    去重标准（场景唯一性）：
        apk_name + entry_point + entry_method + call_chain + sensitiveAPI
    参数：
        input_csv_path: 原始完整数据CSV路径
        output_csv_path: 去重后CSV输出路径
        reset_id: 是否重新生成连续id（默认True）
    输出：
        在 output_csv_path 生成去重后的数据文件
    """

    print(f"[INFO] 读取数据集: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    original_count = len(df)
    print(f"[INFO] 原始样本数: {original_count}")

    # 功能场景唯一性判定字段
    dedup_columns = [
        "apk_name",
        "entry_point",
        "entry_method",
        "call_chain",
        "sensitiveAPI",
        "call_chain_split",
        "entry_point_actions",
        "label",
        "context",
        "rawtext",
        # "declaringClass",
        # # "uid",
        # "container"
    ]

    print("[INFO] 开始按功能场景粒度去重...")
    df_dedup = df.drop_duplicates(subset=dedup_columns, keep="first")

    dedup_count = len(df_dedup)
    removed = original_count - dedup_count
    print(f"[INFO] 去重后样本数: {dedup_count}")
    print(f"[INFO] 移除重复样本数: {removed}")
    # 重新编号（可选）
    # if reset_id:
    #     print("[INFO] 重新生成连续id...")
    #     df_dedup = df_dedup.reset_index(drop=True)
    #     df_dedup["id"] = df_dedup.index + 1

    # print("[INFO] 重新生成连续id...")
    # df_dedup = df_dedup.reset_index(drop=True)
    # df_dedup["id"] = df_dedup.index + 1

    df_dedup.to_csv(output_csv_path, index=False)
    print(f"[INFO] 保存去重后的数据集: {output_csv_path}")
    print("[DONE] 功能场景去重完成 ✅")

    # 仅记录最终结果
    write_final_stats(f"[INFO] 功能场景去重完成：原始 {original_count} 条，去重后 {dedup_count} 条，移除 {removed} 条重复样本，保存去重后的数据集: {output_csv_path}")



def filter_complete_records(input_csv_path, output_csv_path):
    print("[INFO] 开始过滤入口点样本...")
    df = pd.read_csv(input_csv_path, encoding='utf-8')
    original_count = len(df)

    # 标准化清理
    cols_to_fix = ['context', 'label', 'rawtext', 'entry_point_actions', 'sensitiveAPI']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace(['', '.', 'nan', 'NaN', 'NAN', 'null', 'NULL', 'None', 'NOAPI'], pd.NA)
    
    # 1. 定义 GUI 样本逻辑
    mask_gui = df['context'].notna() | df['label'].notna() | df['rawtext'].notna()
    # 2. 定义 System 样本逻辑 (带Action的)
    mask_system = df['entry_point_actions'].notna()

    # 3. 定义 Basic 样本逻辑（核心修改点）
    # 条件：非GUI 且 非System 且 是有效方法 且 【必须包含敏感API】
    mask_has_api = df['sensitiveAPI'].notna()
    mask_basic = (~mask_gui & ~mask_system & df.apply(is_semantic_method, axis=1) & mask_has_api)
    # mask_basic = (~mask_gui & ~mask_system & df.apply(is_semantic_method, axis=1))

    # 4. 初步筛选
    mask_keep = mask_gui | mask_system | mask_basic
    df_filtered = df[mask_keep].copy()

    print("[INFO] 过滤统计:")
    print(f"  原始总数: {len(df)}")
    print(f"  保留总数: {len(df_filtered)}")
    print(f"    - GUI类型: {mask_gui.sum()}")
    print(f"    - System类型: {mask_system.sum()}")
    print(f"    - Basic类型: {mask_basic.sum()}")
    
    # 5. 【新增】同类行为去重（Intra-App De-duplication）
    # 理由：如果一个APP在5个不同的Basic方法里都调用了同一个获取位置的API，
    # 我们只给大模型保留1个，防止数据偏置（Bias）。
    df_filtered['sample_type'] = 'Basic'
    df_filtered.loc[mask_gui, 'sample_type'] = 'GUI'
    df_filtered.loc[mask_system & ~mask_gui, 'sample_type'] = 'System'

    # 针对 Basic 和 System 类型进行基于 (APK, 敏感API) 的去重
    df_gui = df_filtered[df_filtered['sample_type'] == 'GUI']
    df_others = df_filtered[df_filtered['sample_type'] != 'GUI']
    print(f"[INFO] Basic/System 类型记录数: {len(df_others)}")
    
    # 后台行为去重逻辑：同一个APP，同样的敏感行为，只留一个入口点    
    # df_others_unique = df_others.drop_duplicates(subset=['apk_name', 'sensitiveAPI'])
    # 第一层：去掉完全重复的行为路径
    df_others_unique = df_others.drop_duplicates(['apk_name', 'call_chain'])
    print(f"[INFO] 去重call_chain后 Basic/System 类型记录数: {len(df_others_unique)}")
    # 第二层：限制同一API在同一APP里的最大样本数（防爆炸）
    df_others_unique['api_rank'] = df_others_unique.groupby(['apk_name', 'sensitiveAPI']).cumcount()
    # df_others_unique = df_others_unique[df_others_unique['api_rank'] < 10]   # 每个APK中的某个敏感API最多保留10种行为路径
    df_others_unique = df_others_unique[df_others_unique['api_rank'] < 50]   # 为了跑学姐实验改的
    print(f"[INFO] 去重敏感API后 Basic/System 类型记录数: {len(df_others_unique)}")
    # # 可选的另一种方法：先按频率排序，再保留前12条
    # df_others_unique = df_others.drop_duplicates(['apk_name', 'call_chain'])
    # # 新增：计算每个行为路径的出现次数（基于原始数据）
    # df_others_unique['count'] = df_others_unique.groupby(['apk_name', 'sensitiveAPI', 'call_chain'])['call_chain'].transform('count')
    # # 按频率降序排序（次数多的在前），再按组编号
    # df_others_unique = df_others_unique.sort_values(['apk_name', 'sensitiveAPI', 'count'], ascending=[True, True, False])
    # df_others_unique['api_rank'] = df_others_unique.groupby(['apk_name', 'sensitiveAPI']).cumcount()
    # df_others_unique = df_others_unique[df_others_unique['api_rank'] < 10]
    df_others_unique = df_others_unique.drop(columns=['api_rank'], errors='ignore')


    df_system = df_others_unique[df_others_unique['sample_type'] == 'System']
    df_basic = df_others_unique[df_others_unique['sample_type'] == 'Basic']
    df_final = pd.concat([df_gui, df_others_unique], ignore_index=True)
    final_count = len(df_final)

    print(f"[INFO] 对 Basic 和 System 类型去重前总记录数: {len(df_filtered)}")
    print(f"[INFO] 对 Basic 和 System 类型去重后总记录数: {len(df_final)}")

    # # 统计数量
    # df_gui = df_filtered[df_filtered['sample_type'] == 'GUI']
    # df_system = df_filtered[df_filtered['sample_type'] == 'System']
    # df_basic = df_filtered[df_filtered['sample_type'] == 'Basic']
    # df_final = pd.concat([df_gui, df_system, df_basic], ignore_index=True)

    # 保存结果与统计
    print(f"[INFO] 过滤统计:")
    print(f"  原始总数: {len(df)}")
    print(f"  最终保留总数: {len(df_final)}")
    print(f"  - GUI类型: {len(df_gui)}")
    print(f"  - System类型: {len(df_system)}")
    print(f"  - Basic类型: {len(df_basic)}")
    df_final.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"[INFO] 保存过滤结果到: {output_csv_path}")

    # 仅记录最终结果
    write_final_stats(f"[INFO] 严格过滤完成：原始 {original_count} 条，最终保留 {final_count} 条（GUI:{len(df_gui)} | System:{len(df_system)} | Basic:{len(df_basic)}）")


def is_semantic_method(row) -> bool:
    """
    判断一个入口点是否具有足够的语义含金量。
    改进点：增加对Android低语义生命周期函数的过滤。
    """
    method_full = str(row.get('entry_method', '')).lower()
    if pd.isna(method_full) or not method_full.strip():
        return False

    # 1. 提取方法名部分（从 <class: type name(...)> 中提取 name）
    # 简单的逻辑：取冒号后面，左括号前面的部分
    try:
        method_name = method_full.split(': ')[1].split('(')[0].split(' ')[1]
    except:
        method_name = method_full

    # 2. 【新增】排除高频、低语义的 Android 生命周期方法
    # 这些方法通常不承载具体的业务功能，只是系统的常规切换
    noise_lifecycle = {
        'onstop', 'onpause', 'ondestroy', 'onstart', 
        'onrestart', 'onsaveinstancestate', 'onrestoreinstancestate',
        'onattachedtowindow', 'ondetachedfromwindow', 'onwindowfocuschanged',
        'onconfigurationchanged', 'onlowmemory', 'ontrimmemory',
        'onanimationstart', 'onanimationend', 'onanimationrepeat'
    }
    if method_name in noise_lifecycle:
        return False

    # 3. 【保留】你原有的正则过滤（处理合成方法）
    noise_patterns = [
        r'access\$\d+', r'lambda\$\d+', r'\$[0-9]+', 
        r'^equals$', r'^hashcode$', r'^tostring$', r'^<init>$', r'^<clinit>$'
    ]
    for p in noise_patterns:
        if re.search(p, method_name):
            return False
    
    # 4. 长度过滤
    if len(method_name) <= 2:
        return False

    return True


def split_csv_by_rows(input_csv_path, output_prefix, rows_per_file=1500):
    """
    将一个大的CSV拆分成多个小CSV文件。
    参数：
        input_csv_path : 原始大CSV路径
        output_prefix  : 输出文件前缀（例如 ./output/final_training_data_part）
        rows_per_file  : 每个文件最大行数
    """
    print(f"[INFO] 开始拆分CSV文件: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        print(f"[ERROR] 文件不存在: {input_csv_path}")
        return
    
    chunk_iter = pd.read_csv(input_csv_path, chunksize=rows_per_file, encoding='utf-8')
    
    file_index = 1
    total_rows = 0
    for chunk in chunk_iter:
        output_path = f"{output_prefix}_{file_index}.csv"
        chunk.to_csv(output_path, index=False, encoding='utf-8')
        rows_written = len(chunk)
        total_rows += rows_written
        print(f"[INFO] 已写入 {rows_written} 行 -> {output_path}")
        file_index += 1
    
    print(f"[DONE] 拆分完成，共生成 {file_index - 1} 个文件，总行数 {total_rows}")


# ====================== 主函数 ======================
def run_single(callchain_dir, gui_file, output_dir):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_ROOT_FOLDER = str(Path(callchain_dir).resolve())
    GUI_CSV = str(Path(gui_file).resolve())
    OUTPUT_ROOT_FOLDER = str(Path(output_dir).resolve())
    if not Path(INPUT_ROOT_FOLDER).is_dir() or not Path(GUI_CSV).is_file():
        raise FileNotFoundError('Missing ICCBot output or GUI merge input')
    Path(OUTPUT_ROOT_FOLDER).mkdir(parents=True,exist_ok=True)
    os.chdir(OUTPUT_ROOT_FOLDER)

    # 中间文件路径
    RAW_CALLCHAIN_CSV = os.path.join(OUTPUT_ROOT_FOLDER, "raw_call_chains.csv")
    MERGED_CALLCHAIN_CSV = os.path.join(OUTPUT_ROOT_FOLDER, "merged_call_chains.csv")
    MERGED_GUI_CSV = os.path.join(OUTPUT_ROOT_FOLDER, "merged_gui_callchain.csv")

    DEDUP_GUI_CSV = os.path.join(OUTPUT_ROOT_FOLDER, "deduplicated_gui_callchain.csv")
    # DEDUP_GUI_CSV = "./output/111.csv"
    
    # 最终输出文件
    FINAL_OUTPUT_CSV = os.path.join(OUTPUT_ROOT_FOLDER, "final_training_data.csv")
    # FINAL_OUTPUT_CSV = "./output/222.csv"
    
    # 步骤1：提取调用链
    if not os.path.exists(RAW_CALLCHAIN_CSV):
        run_stage(extract_call_chains, [INPUT_ROOT_FOLDER], RAW_CALLCHAIN_CSV)
    else:
        print(f"[SKIP] {RAW_CALLCHAIN_CSV} 已存在")
    
    # 步骤1b：合并调用链列
    if not os.path.exists(MERGED_CALLCHAIN_CSV):
        run_stage(merge_columns, [RAW_CALLCHAIN_CSV], MERGED_CALLCHAIN_CSV)
    else:
        print(f"[SKIP] {MERGED_CALLCHAIN_CSV} 已存在")
    
    # 步骤2：合并GUI
    if not os.path.exists(MERGED_GUI_CSV):
        run_stage(merge_callchain_gui_info, [MERGED_CALLCHAIN_CSV, GUI_CSV], MERGED_GUI_CSV)
    else:
        print(f"[SKIP] {MERGED_GUI_CSV} 已存在")
    
    # # 步骤2：合并GUI
    # if not os.path.exists(FINAL_OUTPUT_CSV):
    #     merge_callchain_gui_info(MERGED_CALLCHAIN_CSV, GUI_CSV, FINAL_OUTPUT_CSV)
    # else:
    #     print(f"[SKIP] {FINAL_OUTPUT_CSV} 已存在")


    # 步骤3：功能场景入口点多余数据去重
    if not os.path.exists(FINAL_OUTPUT_CSV):
        run_stage(deduplicate_scenarios, [MERGED_GUI_CSV], FINAL_OUTPUT_CSV)
    else:
        print(f"[SKIP] {FINAL_OUTPUT_CSV} 已存在")


    
        # ====================== 步骤5：拆分大CSV为多个小文件 ======================
    # SPLIT_PREFIX = "./output/final_training_data_part"
    # split_csv_by_rows(FINAL_OUTPUT_CSV, SPLIT_PREFIX, rows_per_file=1500)

    # 验证最终输出
    if os.path.exists(FINAL_OUTPUT_CSV):
        df_final = pd.read_csv(FINAL_OUTPUT_CSV, encoding='utf-8')
        print("\n" + "="*50)
        print("最终数据集统计信息:")
        print("="*50)
        print(f"总样本数: {len(df_final)}")
        print(f"有GUI label的样本: {df_final['label'].notna().sum()}")
        print(f"有GUI context的样本: {df_final['context'].notna().sum()}")
        print(f"有系统回调的样本: {df_final['entry_point_actions'].notna().sum()}")
        print(f"列名: {list(df_final.columns)}")
        print("="*50)

def run_stage(function, inputs, output):
    target=Path(output)
    with tempfile.TemporaryDirectory(prefix='.fusion-',dir=target.parent) as tmp:
        staged=Path(tmp)/target.name
        function(*inputs,str(staged))
        if not staged.is_file():
            raise RuntimeError(f'No output from {function.__name__}')
        for path in Path(tmp).iterdir():
            if path!=staged:
                os.replace(path,target.parent/path.name)
        os.replace(staged,target)


def merge_splits(root, batches):
    csv.field_size_limit(sys.maxsize)
    seen={}
    for split, selected in [('train',[b for b in batches if b[0]<2018]),
                            ('test',[b for b in batches if b[0]>=2018])]:
        if not selected:continue
        target=root/split/'final_training_data.csv'
        target.parent.mkdir(parents=True,exist_ok=True)
        count=0; apks=set();header=None;stats=[]
        pending=target.with_suffix('.csv.tmp')
        try:
            with pending.open('w',newline='',encoding='utf-8') as output:
                writer=csv.writer(output)
                for year,label in selected:
                    source=raw_root(root)/str(year)/label/'pipeline/final_training_data.csv'
                    batch_rows=0
                    with source.open(newline='',encoding='utf-8') as stream:
                        reader=csv.reader(stream);current=next(reader)
                        if header is None:
                            header=current;writer.writerow(header+['dataset_year','dataset_class'])
                        if current!=header:raise ValueError(f'Inconsistent columns: {source}')
                        id_idx=header.index('id');apk_idx=header.index('apk_name')
                        for row in reader:
                            if len(row)!=len(header):raise ValueError(f'Invalid CSV row: {source}')
                            count+=1;batch_rows+=1;apks.add(row[apk_idx]);row[id_idx]=str(count)
                            writer.writerow(row+[str(year),label])
                    stats.append((year,label,batch_rows))
            replace_if_changed(pending,target)
        finally:pending.unlink(missing_ok=True)
        seen[split]=apks
        print(f'SPLIT {split}: rows={count}, APKs={len(apks)}, batches={stats}, output={target}',flush=True)
    print(f'Train/test APK SHA overlap: {len(seen.get("train",set()) & seen.get("test",set()))}',flush=True)


def main():
    parser=argparse.ArgumentParser(description='Fuse existing ICCBot and merged GUI results; no LLM calls')
    parser.add_argument('--output-root',type=Path,default=Path('/home/lb/data/Dataset/ForeDroid_output/2014-2017'))
    parser.add_argument('--years',type=int,nargs='+',help='Explicit years; pre-2018 benign is train, 2018+ both classes are test')
    parser.add_argument('--workers',type=int,default=6,help='Parallel year/class batches, 1..6')
    parser.add_argument('--callchain-dir',type=Path)
    parser.add_argument('--gui-file',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    single=[args.callchain_dir,args.gui_file,args.output_dir]
    if any(x is not None for x in single):
        if not all(x is not None for x in single):parser.error('Provide all three single-batch paths')
        run_single(*single)
        return
    if not 1<=args.workers<=6:parser.error('workers must be 1..6')
    root=args.output_root.resolve()
    if root.name=='2014-2017' and args.years and any(y not in range(2014,2018) for y in args.years):parser.error('Training root accepts only 2014–2017')
    lock=(root/'run.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.years and any(y<2000 or y>2100 for y in args.years):parser.error('Invalid year')
    batches=[(year,label) for year in sorted(set(args.years)) for label in (['benign'] if year<2018 else ['benign','malware'])] if args.years else [(year,'benign') for year in range(2014,2018)]
    for year,label in batches:
        batch=raw_root(root)/str(year)/label
        if not (batch/'callchain/output').is_dir() or not (batch/'gui/merge/train/merge_ui_api_with_relation.txt').is_file():
            raise FileNotFoundError(f'Missing fusion input: {batch}')
    def launch(batch):
        year,label=batch;folder=raw_root(root)/str(year)/label;out=folder/'pipeline';out.mkdir(exist_ok=True)
        started=time.monotonic()
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with (out/'pipeline.log').open('a') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),
                '--callchain-dir',str(folder/'callchain/output'),
                '--gui-file',str(folder/'gui/merge/train/merge_ui_api_with_relation.txt'),
                '--output-dir',str(out)],stdout=log,stderr=subprocess.STDOUT,env=env)
        if result.returncode:raise RuntimeError(f'{year}/{label} fusion failed; inspect {out}/pipeline.log')
        print(f'BATCH {year}/{label} complete in {time.monotonic()-started:.1f}s',flush=True)
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(launch,b) for b in batches]
        for future in futures.as_completed(pending):future.result()
    merge_splits(root,batches)
    print(f'Fusion finished: batches={batches}. Pre-2018 benign -> train; 2018+ -> test. No LLM or model training run.',flush=True)


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from output_layout import dispatch,TRAIN
    dispatch(__file__,'--output-root',TRAIN,year_flag='--years',training=False)
    main()
