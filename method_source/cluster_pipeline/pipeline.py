# === pipeline_2.py ===
# ForeDroid 功能场景建模 — 统一推理管道 (基于预训练质心)
# ============================================================
# 对应论文 Section 3.2 Functional Scenario Modeling。
#
# 相对于 pipeline.py 的核心改动：
#   1. Background 管道不再重新聚类，而是加载 bg_scenario_cluster.py 产出的预训练质心，
#      通过余弦相似度分配标签（与 GUI 管道对称）。
#   2. 整体流程：分流 → 两套质心分配 → offset 合并 → 分组导出。
#
# 使用方式（推理阶段）：
#   python3 pipeline_2.py \
#     --input_csv <info_fusion产出的final_training_data_with_desc.csv> \
#     --output_dir <输出目录> \
#     --gui_centroids_dir <gui_scenario_cluster.py产出的质心目录> \
#     --bg_centroids_dir <bg_scenario_cluster.py产出的质心目录> \
#     --min_cluster_size 15 \
#     --model_path ../models/GoogleNews-vectors-negative300.bin \
#     --gui_model_name all-MiniLM-L6-v2 \
#     --gui_cache_folder ../models/model-all-MiniLM-L6-v2

import argparse
import os
import sys
import csv
import json
import numpy as np
import pandas as pd

# ── 复用现有模块 ────────────────────────────────────────────
from group_by_cluster import save_clusters_separately, save_cluster_id_mapping
from api_stats import generate_cluster_api_stats

# GUI 管道依赖
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Background 管道依赖
from entry_point_filter import EntryPointFilter
from scenario_common import valid_gui_text, gui_raw_text, gui_doc_text, ROOT, MODELS
from sklearn.preprocessing import normalize


# ══════════════════════════════════════════════════════════════
#  分流函数
# ══════════════════════════════════════════════════════════════

def is_gui_entry(row) -> bool:
    """
    判断一条记录是否为 GUI 入口点。
    条件：label / context / rawtext 中至少有一个非空有效值。
    """
    return any(valid_gui_text(row.get(col)) for col in ['label','context','rawtext'])


def split_by_gui(df: pd.DataFrame) -> tuple:
    """按 GUI 文本分流，返回 (gui_df, bg_df)。"""
    mask_gui = df.apply(is_gui_entry, axis=1)
    gui_df = df[mask_gui].copy()
    bg_df = df[~mask_gui].copy()
    print(f"分流结果: GUI 入口点 {len(gui_df)} 条, Background 入口点 {len(bg_df)} 条")
    return gui_df, bg_df


# ══════════════════════════════════════════════════════════════
#  GUI 文本预处理
# ══════════════════════════════════════════════════════════════

def preprocess_gui_text(label, context, rawtext, nlp):
    """
    拼接 label + context + rawtext，做词形还原 + 保留介词助动词。
    与 gui_scenario_cluster.py / gui_scenario_assign.py 一致。
    """
    text=gui_raw_text(label,context,rawtext)
    return gui_doc_text(nlp(text)) if text else ''


# ══════════════════════════════════════════════════════════════
#  GUI 管道 (基于预训练质心，对应论文 3.2.1 Step 3)
# ══════════════════════════════════════════════════════════════

def run_gui_pipeline(gui_df, centroids_dir,
                     model_name="all-MiniLM-L6-v2",
                     cache_folder="../models/model-all-MiniLM-L6-v2"):
    """
    对 GUI 入口点执行 BERT 嵌入 + 预训练质心余弦相似度分配。
    返回带 cluster 和 gui_scenario_label 列的 DataFrame。
    """
    if len(gui_df) == 0:
        print("⚠ GUI 子集为空，跳过 GUI 管道。")
        gui_df['cluster'] = ""
        # gui_df['gui_scenario_label'] = ""
        return gui_df

    # 1. 加载预训练质心和标签
    centroids_path = os.path.join(centroids_dir, "gui_centroids.npy")
    cluster_ids_path = os.path.join(centroids_dir, "gui_cluster_ids.json")
    labels_path = os.path.join(centroids_dir, "gui_scenario_labels.json")

    for p, name in [(centroids_path, "质心矩阵"), (cluster_ids_path, "簇ID列表"), (labels_path, "场景标签")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到{name}: {p}，请先运行 gui_scenario_cluster.py")

    print(f"加载 GUI 质心: {centroids_path}")
    centroid_matrix = np.load(centroids_path)

    with open(cluster_ids_path, 'r') as f:
        cluster_ids = json.load(f)
    with open(labels_path, 'r') as f:
        scenario_labels = json.load(f)

    print(f"已加载 {len(cluster_ids)} 个 GUI 场景簇, 质心维度: {centroid_matrix.shape}")

    # 2. 加载模型
    print("加载 spaCy ...")
    nlp = spacy.load("en_core_web_sm")
    print(f"加载 BERT: {model_name}")
    model = SentenceTransformer(model_name, cache_folder=cache_folder, local_files_only=True)

    # 3. 预处理
    tqdm.pandas(desc="预处理 GUI 文本")
    gui_df['semantic_text'] = gui_df.progress_apply(
        lambda row: preprocess_gui_text(
            row.get('label', ''),
            row.get('context', ''),
            row.get('rawtext', ''),
            nlp,
        ),
        axis=1,
    )

    mask_empty = gui_df['semantic_text'].str.strip() == ""
    n_empty = mask_empty.sum()
    if n_empty > 0:
        print(f"⚠ {n_empty} 条 GUI 记录预处理后语义为空, 不分配标签")

    # 4. BERT 嵌入
    non_empty_texts = gui_df[~mask_empty]['semantic_text'].tolist()
    if non_empty_texts:
        print(f"生成 BERT 嵌入，共 {len(non_empty_texts)} 条...")
        non_empty_embeddings = model.encode(non_empty_texts, show_progress_bar=True)
    else:
        non_empty_embeddings = np.array([])

    # 5. 余弦相似度分配 (论文 Eq.2 & Eq.3)
    # assigned_labels = []
    assigned_clusters = []
    if len(non_empty_embeddings) > 0:
        similarities = cosine_similarity(non_empty_embeddings, centroid_matrix)
        best_indices = np.argmax(similarities, axis=1)
        for idx in best_indices:
            cid = cluster_ids[idx]
            assigned_clusters.append(cid)
            # assigned_labels.append(scenario_labels.get(cid, f"Cluster_{cid}"))

    # 6. 合并分配结果
    gui_df['cluster'] = ""
    # gui_df['gui_scenario_label'] = ""
    non_empty_indices = gui_df[~mask_empty].index
    for i, orig_idx in enumerate(non_empty_indices):
        if i < len(assigned_clusters):
            gui_df.at[orig_idx, 'cluster'] = assigned_clusters[i]
            # gui_df.at[orig_idx, 'gui_scenario_label'] = assigned_labels[i]

    n_assigned = (gui_df['cluster'] != "").sum()
    print(f"GUI 管道完成: {n_assigned}/{len(gui_df)} 条已分配场景标签")

    # 清理中间列
    gui_df.drop(columns=['semantic_text'], inplace=True, errors='ignore')
    # gui_csv = os.path.join("./test/output", "gui_clusters.csv")
    # gui_df.to_csv(gui_csv, index=False, sep=';', quoting=csv.QUOTE_ALL)

    return gui_df


# ══════════════════════════════════════════════════════════════
#  Background 管道 (基于预训练质心，对应论文 3.2.2 Step 3)
# ══════════════════════════════════════════════════════════════

def run_bg_pipeline(bg_df, centroids_dir,
                    model_path='../models/GoogleNews-vectors-negative300.bin'):
    """
    对 Background 入口点：EntryPointFilter 预处理 + Word2Vec 向量化
    → PCA 降维 → 与预训练质心余弦相似度分配。
    返回带 cluster 和 bg_scenario_label 列的 DataFrame。
    """
    if len(bg_df) == 0:
        print("⚠ Background 子集为空，跳过 Background 管道。")
        bg_df['cluster'] = ""
        return bg_df

    # 1. 加载预训练质心和标签
    centroids_path = os.path.join(centroids_dir, "bg_centroids.npy")
    cluster_ids_path = os.path.join(centroids_dir, "bg_cluster_ids.json")
    labels_path = os.path.join(centroids_dir, "bg_scenario_labels.json")
    pca_model_path = os.path.join(centroids_dir, "bg_pca_model.pkl")

    for p, name in [(centroids_path, "质心矩阵"), (cluster_ids_path, "簇ID列表"),
                     (labels_path, "场景标签"), (pca_model_path, "PCA模型")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到{name}: {p}，请先运行 bg_scenario_cluster.py")

    print(f"加载 BG 质心: {centroids_path}")
    centroid_matrix = np.load(centroids_path)

    with open(cluster_ids_path, 'r') as f:
        cluster_ids = json.load(f)
    with open(labels_path, 'r') as f:
        scenario_labels = json.load(f)

    import joblib
    pca = joblib.load(pca_model_path)

    print(f"已加载 {len(cluster_ids)} 个 Background 场景簇, 质心维度: {centroid_matrix.shape}")

    # 2. EntryPointFilter 预处理 + Word2Vec 向量化
    print(f"加载 Word2Vec 模型: {model_path}")
    ep_filter = EntryPointFilter(model_path=model_path)

    tmp_dir = os.path.dirname(centroids_dir) or '.'
    tmp_input = os.path.join(tmp_dir, "_tmp_bg_pipeline_input.csv")
    bg_df.to_csv(tmp_input, index=False)

    tmp_words = os.path.join(tmp_dir, "_tmp_bg_pipeline_words.csv")
    tmp_vec = os.path.join(tmp_dir, "_tmp_bg_pipeline_vec.csv")

    ep_filter.process_csv(tmp_input, tmp_words, tmp_vec)

    vec_df = pd.read_csv(tmp_vec, sep=';', index_col=0)
    X = vec_df.values

    for f in [tmp_input, tmp_words, tmp_vec]:
        if os.path.exists(f):
            os.remove(f)

    print(f"生成 Word2Vec 向量: {X.shape}")

    # 3. PCA 降维（与训练时相同变换）
    X_normalized = normalize(X, norm='l2')
    X_reduced = pca.transform(X_normalized)
    print(f"PCA 变换后向量维度: {X_reduced.shape}")

    # 4. 余弦相似度分配 (论文 Eq.2 & Eq.3 的逻辑)
    similarities = cosine_similarity(X_reduced, centroid_matrix)
    best_indices = np.argmax(similarities, axis=1)

    # assigned_labels = []
    assigned_clusters = []
    for idx in best_indices:
        cid = cluster_ids[idx]
        assigned_clusters.append(cid)
        # assigned_labels.append(scenario_labels.get(cid, f"Cluster_{cid}"))

    bg_df['cluster'] = assigned_clusters
    # bg_df['bg_scenario_label'] = assigned_labels

    n_assigned = (bg_df['cluster'] != "").sum()
    print(f"Background 管道完成: {n_assigned}/{len(bg_df)} 条已分配场景标签")

    return bg_df


# ══════════════════════════════════════════════════════════════
#  合并函数：将两条管道结果统一
# ══════════════════════════════════════════════════════════════

def merge_scenario_pipelines(bg_df, gui_df, output_dir,
                             gui_centroids_dir=None, bg_centroids_dir=None):
    """
    合并 Background 和 GUI 聚类结果。
    Background cluster (0, 1, 2, ...) + GUI cluster + offset → 统一 cluster 列。
    """
    # 1. 给 Background 记录创建统一的 cluster 列（直接使用 cluster）
    if 'cluster' not in bg_df.columns:
        bg_df['cluster'] = bg_df['cluster'].astype(str)

    # # 1a. 计算 offset（基于 Background 的 cluster 列）
    bg_max = -1
    if bg_centroids_dir:
        bg_cluster_ids_path = os.path.join(bg_centroids_dir, "bg_cluster_ids.json")
        if os.path.exists(bg_cluster_ids_path):
            with open(bg_cluster_ids_path, 'r') as f:
                bg_cluster_ids = json.load(f)
            valid_ids = [int(c) for c in bg_cluster_ids if str(c).lstrip('-').isdigit()]
            if valid_ids:
                bg_max = max(valid_ids)

    offset = bg_max + 1 if bg_max >= 0 else 10000
    print(f"训练集 Background 最大 cluster: {bg_max}, GUI offset: {offset}")


    # 2. GUI: cluster = cluster + offset
    #    注意：不能覆盖 gui_df['cluster']！（run_gui_pipeline 已分配好 cluster）
    #    只对有效 cluster 做 offset 偏移
    mask_gui_valid = (gui_df['cluster'].notna() &
                      (gui_df['cluster'] != "") &
                      (gui_df['cluster'] != '-1'))
    if mask_gui_valid.any():
        gui_numeric = pd.to_numeric(gui_df.loc[mask_gui_valid, 'cluster'], errors='coerce')
        gui_df.loc[mask_gui_valid, 'cluster'] = (gui_numeric + offset).astype(int).astype(str)
        print(f"GUI 有效 cluster: {mask_gui_valid.sum()} 条 → offset + cluster")

    # 未分配的 GUI 记录给 -1 (噪声)
    gui_df.loc[~mask_gui_valid, 'cluster'] = '-1'
    print(f"GUI 记录: 有效 {mask_gui_valid.sum()} 条, 噪声 -1: {(~mask_gui_valid).sum()} 条")

    # 3. 拼接
    unified_df = pd.concat([bg_df, gui_df], ignore_index=True)
    print(f"合并后总记录数: {len(unified_df)}")

    # 4. 保存
    unified_csv = os.path.join(output_dir, "all_apks_with_clusters.csv")
    unified_df.to_csv(unified_csv, index=False, sep=';', quoting=csv.QUOTE_ALL)
    print(f"统一聚类结果已保存: {unified_csv}")

    # # 5. 分组导出
    # cluster_label_dir = os.path.join(output_dir, "clusters_by_label")
    # save_clusters_separately(unified_df, output_dir=cluster_label_dir, cluster_column='cluster')

    # # 6. cluster_id_mapping
    # save_cluster_id_mapping(unified_df,
    #                         output_file=os.path.join(output_dir, "cluster_id_mapping.csv"),
    #                         cluster_column='cluster',
    #                         id_column='id')

    # # 7. API stats
    # generate_cluster_api_stats(unified_csv, output_file=os.path.join(output_dir, "cluster_api_stats.json"))

    # 8. 生成 cluster_names.csv（同时包含 GUI 和 Background 场景标签）
    save_cluster_names(unified_df, gui_centroids_dir, bg_centroids_dir, offset, output_dir)

    return unified_df


def save_cluster_names(merged_df, gui_centroids_dir, bg_centroids_dir, offset, output_dir):
    """
    生成 cluster_names.csv，同时写入 GUI 和 Background 的场景标签。

    参数:
        merged_df: 合并后的统一 DataFrame
        gui_centroids_dir: GUI 预训练质心目录（含 gui_scenario_labels.json）
        bg_centroids_dir: Background 预训练质心目录（含 bg_scenario_labels.json）
        offset: GUI cluster 的偏移量
        output_dir: 输出目录
    """
    import json

    existing_names = {}
    existing_path = os.path.join(output_dir, "cluster_names.csv")
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path, sep=';', dtype={'cluster': str})
        for _, row in existing.iterrows():
            existing_names[str(row['cluster'])] = row['name']

    cluster_counts = merged_df['cluster'].value_counts().to_dict()

    # GUI 标签映射：key 需要加 offset（因为 GUI cluster = 原始 cluster + offset）
    gui_label_map = {}
    if gui_centroids_dir:
        gui_label_path = os.path.join(gui_centroids_dir, "gui_scenario_labels.json")
        if os.path.exists(gui_label_path):
            with open(gui_label_path, 'r') as f:
                raw_map = json.load(f)
            gui_label_map = {str(int(k) + offset): v for k, v in raw_map.items()}
            print(f"加载 GUI 场景标签: {len(gui_label_map)} 个 → offset 后 cluster")

    # Background 标签映射：key 就是 cluster ID，不需要 offset
    bg_label_map = {}
    if bg_centroids_dir:
        bg_label_path = os.path.join(bg_centroids_dir, "bg_scenario_labels.json")
        if os.path.exists(bg_label_path):
            with open(bg_label_path, 'r') as f:
                bg_label_map = json.load(f)
            print(f"加载 Background 场景标签: {len(bg_label_map)} 个")

    # 为每个 cluster 分配名称：优先 existing → GUI 标签 → BG 标签 → fallback
    rows = []
    for cid in sorted(cluster_counts.keys(),
                       key=lambda x: (int(x) if x.lstrip('-').isdigit() else -1)):
        # name = existing_names.get(cid, None)
        # if name is None and cid in gui_label_map:
        #     name = f"GUI: {gui_label_map[cid]}"
        # if name is None and cid in bg_label_map:
        #     name = f"BG: {bg_label_map[cid]}"
        # if name is None:
        #     name = f"cluster_{cid}"
        # rows.append({'cluster': cid, 'name': name, 'count': cluster_counts[cid]})
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


# ══════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════

def run_pipeline(input_csv, output_dir,
                 min_cluster_size=15,
                 model_path='../models/GoogleNews-vectors-negative300.bin',
                 gui_centroids_dir=None,
                 bg_centroids_dir=None,
                 gui_model_name="all-MiniLM-L6-v2",
                 gui_cache_folder="../models/model-all-MiniLM-L6-v2"):
    """
    论文 Section 3.2 统一推理管道（基于预训练质心，不再重新聚类）。

    参数:
        input_csv: info_fusion 产出的 final_training_data_with_desc.csv
                   必须包含 id, apk_name, entry_point, entry_method, entry_point_actions,
                   call_chain, sensitiveAPI, label, context, rawtext 等列
        output_dir: 输出目录
        min_cluster_size: 保留参数（当前管道中不使用，仅用于兼容）
        model_path: Word2Vec 模型路径（Background 管道用）
        gui_centroids_dir: GUI 预训练质心目录 (gui_scenario_cluster.py 产出)
        bg_centroids_dir: Background 预训练质心目录 (bg_scenario_cluster.py 产出)
        gui_model_name: BERT 模型名
        gui_cache_folder: BERT 模型缓存目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 默认质心目录
    if gui_centroids_dir is None:
        gui_centroids_dir = os.path.join(os.path.dirname(output_dir) or '.', "gui_centroids")
    if bg_centroids_dir is None:
        bg_centroids_dir = os.path.join(os.path.dirname(output_dir) or '.', "bg_centroids")

    # 0. 读取原始数据
    print(f"读取输入数据: {input_csv}")
    df = pd.read_csv(input_csv, dtype={'id': str})
    print(f"总记录数: {len(df)}")

    df['id'] = df['id'].astype(str)

    # 1. 分流
    gui_df, bg_df = split_by_gui(df)

    # 2. GUI 管道（论文 3.2.1 Step 3，预训练质心分配）
    gui_df = run_gui_pipeline(gui_df, gui_centroids_dir,
                              model_name=gui_model_name,
                              cache_folder=gui_cache_folder)

    # 3. Background 管道（论文 3.2.2 Step 3，预训练质心分配）
    bg_df = run_bg_pipeline(bg_df, bg_centroids_dir, model_path=model_path)

    # 4. 合并（同时传入质心目录以生成含 GUI + BG 场景标签的 cluster_names.csv）
    unified_df = merge_scenario_pipelines(bg_df, gui_df, output_dir,
                                          gui_centroids_dir=gui_centroids_dir,
                                          bg_centroids_dir=bg_centroids_dir)

    print("\n=== 功能场景建模完成 ===")
    print(f"总记录数: {len(unified_df)}")
    print(f"总 cluster 数: {unified_df['cluster'].nunique()}")
    print(f"输出目录: {output_dir}")

    return unified_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ForeDroid 场景建模管道（推理版，基于预训练质心）')
    parser.add_argument('--input_csv', required=False,default=str(ROOT/'test/final_training_data_with_desc.csv'),
                        help='info_fusion 产出的 final_training_data_with_desc.csv')
    parser.add_argument('--output_dir', required=False,default=str(ROOT/'test/cluster_pipeline/output'),
                        help='输出目录路径')
    parser.add_argument('--min_cluster_size', type=int, default=15,
                        help='HDBSCAN 最小簇大小 (保留参数，推理时未使用)')
    parser.add_argument('--model_path', default=str(MODELS/'word2vec-google-news-300/GoogleNews-vectors-negative300.bin'),
                        help='Word2Vec 模型路径 (Background 管道)')
    parser.add_argument('--gui_centroids_dir', default=str(ROOT/'train/cluster_pipeline/gui_centroids'),
                        help='GUI 预训练质心目录 (含 gui_centroids.npy 等)')
    parser.add_argument('--bg_centroids_dir', default=str(ROOT/'train/cluster_pipeline/bg_centroids'),
                        help='Background 预训练质心目录 (含 bg_centroids.npy 等)')
    parser.add_argument('--gui_model_name', default=str(MODELS/'all-MiniLM-L6-v2'),
                        help='BERT 模型名')
    parser.add_argument('--gui_cache_folder', default=str(MODELS),
                        help='BERT 模型缓存目录')

    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        model_path=args.model_path,
        gui_centroids_dir=args.gui_centroids_dir,
        bg_centroids_dir=args.bg_centroids_dir,
        gui_model_name=args.gui_model_name,
        gui_cache_folder=args.gui_cache_folder,
    )