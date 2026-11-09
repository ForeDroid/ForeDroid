# === gui_scenario_cluster.py ===
# 对应论文 3.2.1 GUI-Associated Scenario Modeling — Step 2: Scenario Label Generation
# 输入: extract_Gui 产出的 CSV（含 label, context, callback, rawtext, apk 等字段）
# 输出: GUI 场景簇质心 centroids.npy + 场景标签映射 scenario_labels.json

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 环境变量要放在其他库加载之前，否则有可能不生效
import csv
import json
import argparse
import numpy as np
import pandas as pd
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scenario_common import gui_mask, gui_raw_text, gui_doc_text, ROOT, MODELS

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'      # 一开始下载模型时会访问Hugging Face的服务器。设置环境变量可以让Python通过代理服务器访问外部网络，从而成功加载模型。
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'     # 所以要开vpn，并且确保代理服务器地址和端口正确。否则，下载模型时可能会遇到网络连接错误。

def preprocess_gui_text(label, context, rawtext, nlp):
    """
    轻量 NLP 预处理，对应论文 3.2.1 Step2 (1) Textual Preprocessing：
    - 词形还原
    - 去标点、数字
    - 保留介词和助动词（"log in" ≠ "log out"）
    - rawtext 不为 NO_RAWTEXT 时拼入 rawtext 文字
    """
    text=gui_raw_text(label,context,rawtext)
    return gui_doc_text(nlp(text)) if text else ''


def load_and_preprocess(input_csv, nlp):
    """
    读取 extract_Gui 输出的 CSV，过滤无回调的 UI 元素，预处理拼接语义文本。
    返回 DataFrame，包含 id, semantic_text, apk, label 等字段。
    """
    print(f"读取 GUI 数据: {input_csv}")
    df = pd.read_csv(input_csv, sep=',', dtype=str)

    required_cols = {"label", "context", "rawtext", "id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入 CSV 缺少必需列: {missing}")

    # 过滤：只保留有 GUI 文本的记录（label/context/rawtext 至少一个非空）
    df = df[gui_mask(df)].copy()
    print(f"有效 GUI 入口点数量: {len(df)}")

    if len(df) == 0:
        raise ValueError("没有找到任何带有回调方法的 UI 元素，请检查输入数据")

    # 预处理生成语义文本
    tqdm.pandas(desc="预处理 GUI 文本")
    df['semantic_text'] = df.progress_apply(
        lambda row: preprocess_gui_text(row['label'], row['context'], row['rawtext'], nlp),
        axis=1
    )

    # 过滤预处理后为空的
    before_filter = len(df)
    df = df[df['semantic_text'].str.strip() != ""].copy()
    print(f"预处理后有效语义文本数量: {len(df)} (过滤掉 {before_filter - len(df)} 条)")

    return df


def cluster_gui_scenarios(input_csv, output_dir, min_cluster_size=10,
                          model_name="all-MiniLM-L6-v2", cache_folder="../models/model-all-MiniLM-L6-v2"):
    """
    主流程：BERT 嵌入 → HDBSCAN 聚类 → 保存质心和标签。
    对应论文 3.2.1 Step2 (2) UI Semantic Embedding 和 (3) Unsupervised Clustering。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载 spaCy（用于预处理）
    print("加载 spaCy 模型...")
    nlp = spacy.load("en_core_web_sm")

    # 1. 加载并预处理
    df = load_and_preprocess(input_csv, nlp)

    # 2. BERT 嵌入（论文 Equation 1: v_e = BERT(text ∥ context)）
    print(f"加载 BERT 模型: {model_name}")
    model = SentenceTransformer(model_name, cache_folder=cache_folder, local_files_only=True)

    semantic_texts = df['semantic_text'].tolist()
    print(f"生成 BERT 嵌入，共 {len(semantic_texts)} 条...")
    embeddings = model.encode(semantic_texts, show_progress_bar=True)
    print(f"嵌入维度: {embeddings.shape[1]}")

    # 3. HDBSCAN 聚类
    print(f"执行 HDBSCAN 聚类 (min_cluster_size={min_cluster_size})...")
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', n_jobs=-1)
    labels = hdb.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"HDBSCAN 完成: {n_clusters} 个簇, {n_noise} 个噪声点 (共 {len(labels)} 条)")

    df['cluster'] = labels

    # 过滤噪声点
    valid_df = df[df['cluster'] != -1].copy()
    valid_embeddings = embeddings[df['cluster'] != -1]

    # 4. 计算每个簇的质心并选取最接近质心的原始 label 作为场景标签
    #    论文描述：选取离聚类中心最近的 UI 元素作为人类可读的场景标签
    cluster_ids = sorted(valid_df['cluster'].unique())
    centroids = {}
    scenario_labels = {}

    print("计算簇质心并生成场景标签...")
    for cid in tqdm(cluster_ids, desc="Processing clusters"):
        mask = valid_df['cluster'] == cid
        cluster_vecs = valid_embeddings[mask.values]
        cluster_df = valid_df[mask]

        # 质心向量
        centroid = np.mean(cluster_vecs, axis=0)
        centroids[str(cid)] = centroid

        # 选取余弦相似度最接近质心的 UI 元素的原始 label
        sims = cosine_similarity([centroid], cluster_vecs)[0]
        best_idx = np.argmax(sims)
        best_label = cluster_df.iloc[best_idx]['label']
        # 确保 label 是字符串
        if isinstance(best_label, float) or best_label is None:
            best_label = str(best_label)
        scenario_labels[str(cid)] = str(best_label).strip()

    # 5. 保存结果
    # 质心矩阵 (shape: n_clusters x embedding_dim)
    centroid_matrix = np.array([centroids[str(c)] for c in cluster_ids])
    centroids_path = os.path.join(output_dir, "gui_centroids.npy")
    np.save(centroids_path, centroid_matrix)
    print(f"质心矩阵已保存: {centroids_path} (shape: {centroid_matrix.shape})")

    # 簇ID列表
    cluster_ids_path = os.path.join(output_dir, "gui_cluster_ids.json")
    with open(cluster_ids_path, 'w') as f:
        json.dump([str(c) for c in cluster_ids], f)

    # 场景标签映射
    labels_path = os.path.join(output_dir, "gui_scenario_labels.json")
    with open(labels_path, 'w') as f:
        json.dump(scenario_labels, f, indent=2, ensure_ascii=False)
    print(f"场景标签已保存: {labels_path}")

    # 聚类统计摘要
    cluster_counts = valid_df['cluster'].value_counts().to_dict()
    summary_path = os.path.join(output_dir, "gui_cluster_summary.csv")
    summary_rows = []
    for cid in cluster_ids:
        summary_rows.append({
            "cluster_id": cid,
            "scenario_label": scenario_labels[str(cid)],
            "count": cluster_counts.get(cid, 0)
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values("count", ascending=False, inplace=True)
    summary_df.to_csv(summary_path, index=False, sep=';')
    print(f"聚类摘要已保存: {summary_path}")

    # 带簇标签的完整数据
    full_output_path = os.path.join(output_dir, "gui_clustered_data.csv")
    df.to_csv(full_output_path, index=False, sep=';', quoting=csv.QUOTE_ALL)
    print(f"完整聚类数据已保存: {full_output_path}")

    print(f"\n=== GUI 场景聚类完成 ===")
    print(f"总簇数: {len(cluster_ids)}")
    print(f"噪声点: {n_noise}")
    print(f"前 10 大场景:")
    for _, row in summary_df.head(10).iterrows():
        print(f"  Cluster {row['cluster_id']}: \"{row['scenario_label']}\" ({row['count']} 条)")

    return centroids, scenario_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI 场景聚类 (论文 3.2.1 Step2)")
    parser.add_argument('--input_csv', required=False, default=str(ROOT/'train/final_training_data_with_desc.csv'),
                        help='extract_Gui 输出的 CSV 文件路径 (分隔符为 ,)')
    parser.add_argument('--output_dir', required=False, default=str(ROOT/'train/cluster_pipeline/gui_centroids'),
                        help='输出目录路径')
    parser.add_argument('--min_cluster_size', type=int, default=25,
                        help='HDBSCAN 最小簇大小 (默认: 15，调整以控制簇数约 ~800)')
    parser.add_argument('--model_name', default=str(MODELS/'all-MiniLM-L6-v2'),
                        help='SentenceTransformer BERT 模型名')
    parser.add_argument('--cache_folder', default=str(MODELS),
                        help='BERT 模型缓存目录')
    args = parser.parse_args()

    cluster_gui_scenarios(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        model_name=args.model_name,
        cache_folder=args.cache_folder
    )
# model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="../models/model-all-MiniLM-L6-v2", local_files_only=True)  