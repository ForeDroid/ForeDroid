# === bg_scenario_cluster.py ===
# 对应论文 3.2.2 Background Callback-Associated Scenario Modeling — Step 2: Scenario Label Generation
# 输入: info_fusion 产出的 CSV（仅无 GUI 文本的 Background 记录）
#       必须包含: id, entry_point, entry_method, entry_point_actions
# 输出: Background 场景簇质心 bg_centroids.npy + 场景标签映射 bg_scenario_labels.json
#
# 与 gui_scenario_cluster.py 对称，只是 embedding 方式不同：
#   GUI 用 BERT (SentenceTransformer)
#   Background 用 Word2Vec 平均向量

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from entry_point_filter import EntryPointFilter
from scenario_common import gui_mask, ROOT, MODELS
from cluster_entry import do_cluster_vectors
from group_by_cluster import merge_original_with_clusters



def load_and_preprocess_bg(input_csv, model_path, sep=','):
    """
    调用 EntryPointFilter 对 Background 入口点做预处理和 Word2Vec 向量化。
    返回向量 DataFrame (index=id, 列为向量维度) 和原始清洗结果 DataFrame。
    """
    print(f"读取 Background 数据: {input_csv}")
    df = pd.read_csv(input_csv, sep=sep, dtype={'id': str})
    # 过滤：只保留非 GUI 的记录（label/context/rawtext 全空或无效）
    df = df[~gui_mask(df)].copy()

    print(f"总 Background 记录数: {len(df)}")

    # 检查必要列
    required_cols = {'id', 'entry_point', 'entry_method', 'entry_point_actions'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入 CSV 缺少必需列: {missing}")

    # 写临时 CSV 供 EntryPointFilter 消费（process_csv 读文件）
    tmp_input = os.path.join(os.path.dirname(input_csv) or '.', "_tmp_bg_cluster_input.csv")
    df.to_csv(tmp_input, index=False, sep=sep)

    # EntryPointFilter 做预处理 + Word2Vec 向量化
    print(f"加载 Word2Vec 模型: {model_path}")
    ep_filter = EntryPointFilter(model_path=model_path)

    tmp_words = os.path.join(os.path.dirname(tmp_input), "_tmp_bg_words.csv")
    tmp_vec = os.path.join(os.path.dirname(tmp_input), "_tmp_bg_vec.csv")

    ep_filter.process_csv(tmp_input, tmp_words, tmp_vec)

    # 读取向量
    vec_df = pd.read_csv(tmp_vec, sep=';', index_col=0)
    words_df = pd.read_csv(tmp_words, sep=';', dtype={'id': str})

    # 清理临时文件
    for f in [tmp_input, tmp_words, tmp_vec]:
        if os.path.exists(f):
            os.remove(f)

    print(f"向量形状: {vec_df.shape}, 清洗文本: {len(words_df)} 条")
    return vec_df, words_df


def cluster_bg_scenarios(input_csv, output_dir, min_cluster_size=10,
                         n_components=100,
                         model_path='../models/GoogleNews-vectors-negative300.bin',
                         sep=','):
    """
    主流程：Word2Vec 向量化 → PCA 降维 → HDBSCAN 聚类 → 保存质心和标签。
    对应论文 3.2.2 Step 2。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载并向量化
    vec_df, words_df = load_and_preprocess_bg(input_csv, model_path, sep=sep)

    X = vec_df.values
    ids = vec_df.index.tolist()
    print(f"向量维度: {X.shape}")

    # 2. PCA + HDBSCAN（调用纯聚类函数）
    labels, X_reduced, pca, hdb = do_cluster_vectors(
        X, n_components=n_components, min_cluster_size=min_cluster_size
    )

    # 保存 PCA 模型（供 bg_scenario_assign.py 推理时做相同变换）
    import joblib
    pca_model_path = os.path.join(output_dir, "bg_pca_model.pkl")
    joblib.dump(pca, pca_model_path)
    print(f"PCA 模型已保存: {pca_model_path}")

    # 3. 合并 labels 到 words_df
    words_df['cluster'] = labels
    valid_df = words_df[words_df['cluster'] != -1].copy()

    # 只取有效簇的向量（PCA 降维后）
    valid_mask = labels != -1
    valid_embeddings = X_reduced[valid_mask]
    valid_ids = np.array(ids)[valid_mask]

    if len(valid_df) == 0:
        raise ValueError("聚类结果中没有有效簇，请调整 min_cluster_size 参数")

    # 4. 计算每个簇的质心并选取最接近质心的原始 entry_point_cleaned 作为场景标签
    cluster_ids = sorted(valid_df['cluster'].unique())
    centroids = {}
    scenario_labels = {}

    print("计算簇质心并生成场景标签...")
    for cid in tqdm(cluster_ids, desc="Processing clusters"):
        # 该簇的 PCA 降维后向量
        mask = labels[valid_mask] == cid
        cluster_vecs = valid_embeddings[mask]
        cluster_df = valid_df[valid_df['cluster'] == cid]

        # 质心向量（基于 PCA 降维后的空间）
        centroid = np.mean(cluster_vecs, axis=0)
        centroids[str(cid)] = centroid

        # 选取余弦相似度最接近质心的记录的 entry_point_cleaned 作为场景标签
        sims = cosine_similarity([centroid], cluster_vecs)[0]
        best_idx = np.argmax(sims)
        best_text = cluster_df.iloc[best_idx]['entry_point_cleaned']
        if isinstance(best_text, float) or best_text is None:
            best_text = str(best_text)
        scenario_labels[str(cid)] = str(best_text).strip()

    # 5. 保存结果
    # 质心矩阵 (shape: n_clusters x n_components)
    centroid_matrix = np.array([centroids[str(c)] for c in cluster_ids])
    centroids_path = os.path.join(output_dir, "bg_centroids.npy")
    np.save(centroids_path, centroid_matrix)
    print(f"质心矩阵已保存: {centroids_path} (shape: {centroid_matrix.shape})")

    # 簇ID列表
    cluster_ids_path = os.path.join(output_dir, "bg_cluster_ids.json")
    with open(cluster_ids_path, 'w') as f:
        json.dump([str(c) for c in cluster_ids], f)

    # 场景标签映射
    labels_path = os.path.join(output_dir, "bg_scenario_labels.json")
    with open(labels_path, 'w') as f:
        json.dump(scenario_labels, f, indent=2, ensure_ascii=False)
    print(f"场景标签已保存: {labels_path}")

    # 聚类统计摘要
    cluster_counts = valid_df['cluster'].value_counts().to_dict()
    summary_path = os.path.join(output_dir, "bg_cluster_summary.csv")
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
    full_output_path = os.path.join(output_dir, "bg_clustered_data.csv")
    # words_df.to_csv(full_output_path, index=False, sep=';', quoting=csv.QUOTE_ALL)
    cluster_labels_df = words_df[['id', 'cluster']].copy()
    merge_original_with_clusters(input_csv, cluster_labels_df,
                             os.path.join(output_dir, "bg_clustered_data.csv"))
    
    print(f"完整聚类数据已保存: {full_output_path}")

    n_clusters = len(cluster_ids)
    n_noise = list(labels).count(-1)
    print(f"\n=== Background 场景聚类完成 ===")
    print(f"总簇数: {n_clusters}")
    print(f"噪声点: {n_noise}")
    print(f"前 10 大场景:")
    for _, row in summary_df.head(10).iterrows():
        print(f"  Cluster {row['cluster_id']}: \"{row['scenario_label']}\" ({row['count']} 条)")

    return centroids, scenario_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Background 场景聚类 (论文 3.2.2 Step2)")
    parser.add_argument('--input_csv', required=False, default=str(ROOT/'train/final_training_data_with_desc.csv'),
                        help='info_fusion 产出的 CSV 文件路径（仅无 GUI 文本的 Background 记录，分隔符默认为 ,）')
    parser.add_argument('--output_dir', required=False, default=str(ROOT/'train/cluster_pipeline/bg_centroids'),
                        help='输出目录路径')
    parser.add_argument('--min_cluster_size', type=int, default=50,
                        help='HDBSCAN 最小簇大小 (默认: 15，调整以控制簇数约 ~2000)')
    parser.add_argument('--n_components', type=int, default=100,
                        help='PCA 降维目标维度 (默认: 100)')
    parser.add_argument('--model_path', default=str(MODELS/'word2vec-google-news-300/GoogleNews-vectors-negative300.bin'),
                        help='Word2Vec 模型路径')
    parser.add_argument('--sep', default=',',
                        help='CSV 分隔符 (默认: ,)')
    args = parser.parse_args()

    cluster_bg_scenarios(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        n_components=args.n_components,
        model_path=args.model_path,
        sep=args.sep,
    )