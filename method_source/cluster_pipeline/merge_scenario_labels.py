# === merge_scenario_labels.py ===
# 将 GUI 管道和 Background 管道的场景标签合并为统一的 cluster 列。
#
# 变动说明（相较 merge_scenario_labels_1.py）：
#   由于 pipeline.py 已在数据入口处按有无 GUI 文本分流，两条管道处理不重叠的数据，
#   因此本脚本不再需要 "GUI 覆盖 Background" 的逻辑，只做简单的 offset 合并。
#
# 变动说明（相较之前的版本）：
#   update_cluster_names 现在同时接收 gui_label_json 和 bg_label_json，
#   将 GUI 和 Background 的场景标签都写入 cluster_names.csv。
#
# 输入:
#   --bg_clustered_csv: Background 管道产出的 all_apks_with_clusters.csv
#                       (仅含无 GUI 文本的记录，含 cluster, id, apk_name 等)
#   --gui_assigned_csv: gui_scenario_assign.py 产出的 CSV
#                       (仅含有 GUI 文本的记录，含 cluster, apk, id)
#   --output_dir: 统一输出目录
#   --gui_label_json: gui_scenario_labels.json（GUI 场景标签映射）
#   --bg_label_json: labels.json（Background 场景标签映射）
#
# 逻辑:
#   1. Background cluster 保持不变 (0, 1, 2, ...)
#   2. GUI cluster = cluster + offset, offset = max(bg_cluster) + 1
#   3. 简单拼接，不再需要覆盖

import os
import csv
import json
import argparse
import pandas as pd
import numpy as np

from group_by_cluster import save_clusters_separately, save_cluster_id_mapping


def merge_scenario_labels(bg_clustered_csv, gui_assigned_csv, output_dir,
                          gui_label_json=None, bg_label_json=None):
    """
    合并 GUI 和 Background 场景标签。
    背景和 GUI 数据不重叠，直接 offset 拼接。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读入两个 CSV
    print(f"读入 Background 聚类结果: {bg_clustered_csv}")
    bg_df = pd.read_csv(bg_clustered_csv, sep=';', dtype={'id': str, 'cluster': str})
    bg_df['cluster'] = bg_df['cluster'].astype(str)

    print(f"读入 GUI 场景分配结果: {gui_assigned_csv}")
    gui_df = pd.read_csv(gui_assigned_csv, sep=';', dtype={'id': str})

    # 检查所需列
    gui_required = {'cluster'}
    gui_missing = gui_required - set(gui_df.columns)
    if gui_missing:
        raise ValueError(f"GUI CSV 缺少列: {gui_missing}")

    # 过滤有效的 GUI 标签
    gui_valid_mask = pd.to_numeric(gui_df['cluster'],errors='coerce').ge(0)
    gui_valid = gui_df[gui_valid_mask].copy()
    gui_invalid = gui_df[~gui_valid_mask].copy()

    n_gui_total = len(gui_df)
    n_gui_valid = len(gui_valid)
    print(f"GUI 记录总数: {n_gui_total}, 有效标签: {n_gui_valid}")

    # 2. 计算 offset
    bg_clusters = pd.to_numeric(bg_df['cluster'], errors='coerce')
    bg_max_valid = bg_clusters[bg_clusters >= 0]
    if len(bg_max_valid) > 0:
        bg_max = int(bg_max_valid.max())
    else:
        bg_max = -1
    offset = bg_max + 1 if bg_max >= 0 else 10000
    print(f"Background 最大 cluster: {bg_max}, GUI offset: {offset}")

    # 3. GUI: cluster = cluster + offset
    gui_valid['cluster'] = gui_valid['cluster'].astype(str)
    gui_numeric = pd.to_numeric(gui_valid['cluster'], errors='coerce')
    gui_valid['cluster'] = (gui_numeric + offset).astype(int).astype(str)

    # 无效 GUI 标签标为 -1 (噪声)
    if len(gui_invalid) > 0:
        gui_invalid['cluster'] = '-1'
        gui_valid = pd.concat([gui_valid, gui_invalid], ignore_index=True)

    n_gui_overridden = len(gui_valid)
    print(f"GUI cluster 映射: {n_gui_valid} 条 → offset + cluster")
    print(f"Background 保留: {len(bg_df)} 条")

    # 4. 拼接
    unified_df = pd.concat([bg_df, gui_valid], ignore_index=True)
    print(f"合并后总记录数: {len(unified_df)}")

    # 5. 保存
    unified_csv = os.path.join(output_dir, "all_apks_with_clusters.csv")
    unified_df.to_csv(unified_csv, index=False, sep=';', quoting=csv.QUOTE_ALL)
    print(f"统一聚类结果已保存: {unified_csv}")

    # 6. 分组导出
    cluster_label_dir = os.path.join(output_dir, "clusters_by_label")
    save_clusters_separately(unified_df, output_dir=cluster_label_dir, cluster_column='cluster')
    print(f"按 cluster 分组已保存: {cluster_label_dir}")

    # 7. cluster_id_mapping
    bg_id_col = 'id'
    save_cluster_id_mapping(unified_df,
                            output_file=os.path.join(output_dir, "cluster_id_mapping.csv"),
                            cluster_column='cluster',
                            id_column=bg_id_col)

    # 8. 更新 cluster_names（同时包含 GUI 和 Background 标签）
    update_cluster_names(unified_df, gui_label_json, bg_label_json, offset, output_dir)

    print("\n=== 场景标签合并完成 ===")
    print(f"总记录数: {len(unified_df)}")
    print(f"统一 cluster 数: {unified_df['cluster'].nunique()}")

    return unified_df


def update_cluster_names(merged_df, gui_label_json, bg_label_json, offset, output_dir):
    """
    更新 cluster_names.csv，同时写入 GUI 和 Background 的场景标签。

    参数:
        merged_df: 合并后的统一 DataFrame
        gui_label_json: gui_scenario_labels.json（key 为 GUI 原始 cluster ID，需加 offset）
        bg_label_json: labels.json（key 为 Background cluster ID，无需 offset）
        offset: GUI cluster 的偏移量
        output_dir: 输出目录
    """
    existing_names = {}
    existing_path = os.path.join(output_dir, "cluster_names.csv")
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path, sep=';', dtype={'cluster': str})
        for _, row in existing.iterrows():
            existing_names[str(row['cluster'])] = row['name']

    cluster_counts = merged_df['cluster'].value_counts().to_dict()

    # GUI 标签映射：key 需要加 offset（因为 GUI cluster = 原始 cluster + offset）
    gui_label_map = {}
    if gui_label_json and os.path.exists(gui_label_json):
        with open(gui_label_json, 'r') as f:
            raw_map = json.load(f)
        gui_label_map = {str(int(k) + offset): v for k, v in raw_map.items()}
        print(f"加载 GUI 场景标签: {len(gui_label_map)} 个 → offset 后 cluster")

    # Background 标签映射：key 就是 cluster ID，不需要 offset
    bg_label_map = {}
    if bg_label_json and os.path.exists(bg_label_json):
        with open(bg_label_json, 'r') as f:
            bg_label_map = json.load(f)
        print(f"加载 Background 场景标签: {len(bg_label_map)} 个")

    # 为每个 cluster 分配名称：优先 GUI 标签 → BG 标签 → existing（非默认） → fallback
    rows = []
    for cid in sorted(cluster_counts.keys(),
                       key=lambda x: (int(x) if x.lstrip('-').isdigit() else -1)):
        name = None
        # 1. 优先使用 GUI 场景标签
        if cid in gui_label_map:
            name = f"GUI: {gui_label_map[cid]}"
        # 2. 其次使用 Background 场景标签
        if name is None and cid in bg_label_map:
            name = f"BG: {bg_label_map[cid]}"
        # 3. 保留 existing 中的非默认名称（说明可能是手动命名过的）
        if name is None:
            existing = existing_names.get(cid, None)
            if existing is not None and not existing.startswith('cluster_'):
                name = existing
        # 4. fallback 到默认 cluster_{cid}
        if name is None:
            name = f"cluster_{cid}"
        rows.append({'cluster': cid, 'name': name, 'count': cluster_counts[cid]})

    names_df = pd.DataFrame(rows)
    names_df.sort_values('count', ascending=False, inplace=True)
    names_path = os.path.join(output_dir, "cluster_names.csv")
    names_df.to_csv(names_path, index=False, sep=';')
    print(f"cluster_names 已更新: {names_path}")
    print(f"  其中 GUI 标签: {sum(1 for r in rows if r['name'].startswith('GUI: '))} 个")
    print(f"  其中 Background 标签: {sum(1 for r in rows if r['name'].startswith('BG: '))} 个")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 GUI 和 Background 场景标签")
    parser.add_argument('--bg_clustered_csv', required=False, default="./train-more/bg_centroids/bg_clustered_data.csv",
                        help='Background 管道产出的 all_apks_with_clusters.csv')
    parser.add_argument('--gui_assigned_csv', required=False, default="./train-more/gui_centroids/gui_clustered_data.csv",
                        help='GUI管道产出的 CSV')
    parser.add_argument('--output_dir', required=False, default="./train-more/output",
                        help='统一输出目录')
    parser.add_argument('--gui_label_json', default="./train-more/gui_centroids/gui_scenario_labels.json",
                        help='gui_scenario_labels.json 路径（用于更新 cluster_names）')
    parser.add_argument('--bg_label_json', default="./train-more/bg_centroids/bg_scenario_labels.json",
                        help='bg_scenario_labels.json 路径（用于更新 cluster_names）')
    args = parser.parse_args()

    merge_scenario_labels(
        bg_clustered_csv=args.bg_clustered_csv,
        gui_assigned_csv=args.gui_assigned_csv,
        output_dir=args.output_dir,
        gui_label_json=args.gui_label_json,
        bg_label_json=args.bg_label_json,
    )