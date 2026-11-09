# === File: cluster_entry_1.py ===
# 相较 cluster_entry.py，拆出 do_cluster_vectors() 纯聚类函数，
# 供 bg_scenario_cluster.py 调用（不触发 t-SNE/wordcloud 可视化）。
# 原有 cluster_vectors() 保留不变，内部调用 do_cluster_vectors()。
import csv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
from collections import Counter
import joblib

from sklearn.cluster import HDBSCAN  # 要求 scikit-learn >= 1.3.0


# ══════════════════════════════════════════════════════════════
#  纯聚类函数（新增）：PCA + HDBSCAN，返回 labels, X_reduced, pca, hdb
#  供 bg_scenario_cluster.py 使用，不带任何可视化
# ══════════════════════════════════════════════════════════════

def do_cluster_vectors(X, n_components=100, min_cluster_size=15, min_samples=None, random_state=42):
    """
    纯聚类逻辑：L2归一化 → PCA降维 → HDBSCAN。
    
    参数:
        X: numpy array, shape (n_samples, n_features), 原始向量
        n_components: PCA 降维目标维度
        min_cluster_size: HDBSCAN 最小簇大小
        min_samples: HDBSCAN min_samples, None 则由算法自动决定
        random_state: 随机种子
    
    返回:
        labels: numpy array, 聚类标签 (-1 表示噪声)
        X_reduced: PCA 降维后的向量
        pca: 已训练的 PCA 模型
        hdb: 已训练的 HDBSCAN 模型
    """
    print("进行 L2 归一化...")
    X_normalized = normalize(X, norm='l2')

    print(f"进行 PCA 降维至 {n_components} 维...")
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_normalized)
    print(f"PCA 降维完成，形状为 {X_reduced.shape}")
    print(f"PCA 保留方差比例: {pca.explained_variance_ratio_.sum():.4f}")

    print(f"初始化 HDBSCAN 模型 (min_cluster_size={min_cluster_size})...")
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, n_jobs=-1)
    labels = hdb.fit_predict(X_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"HDBSCAN 训练完成。自动发现了 {n_clusters} 个簇，噪声点: {n_noise}/{len(labels)}")

    return labels, X_reduced, pca, hdb


# ══════════════════════════════════════════════════════════════
#  以下为原有代码，cluster_vectors() 内部调用 do_cluster_vectors()
# ══════════════════════════════════════════════════════════════

def cluster_tsne_top20(labels, X_reduced, min_cluster_size, output_file):
    import pandas as pd
    
    # 1. 修复 sklearn 警告，将 n_iter 替换为 max_iter
    tsne = TSNE(n_components=2, random_state=42, max_iter=300)
    X_tsne = tsne.fit_transform(X_reduced)
    # 2. 使用 pandas 的 value_counts 替代 np.bincount，它完美支持负数(-1)
    label_counts = pd.Series(labels).value_counts()
    # 将 -1 (噪声) 排除在 "Top 20 正常簇" 之外
    valid_counts = label_counts[label_counts.index != -1]
    N = 20
    # 获取数量最多的前 20 个正常簇
    top_labels = valid_counts.head(N).index.tolist()
    # 3. 设置颜色映射
    cmap_base = plt.cm.get_cmap('tab20', N)
    color_map_dict = {lbl: cmap_base(i) for i, lbl in enumerate(top_labels)}
    noise_color = "black"      # 噪声点(-1)用显眼的黑色表示
    others_color = "lightgray" # 第20名以后的正常小簇用浅灰色表示

    plot_colors =[]
    for lbl in labels:
        if lbl in top_labels:
            plot_colors.append(color_map_dict[lbl])
        # elif lbl == -1:                          # 显示噪声点
        #     plot_colors.append(noise_color)
        else:
            plot_colors.append(others_color)

    # 4. 开始画图
    plt.figure(figsize=(12, 8))
    # alpha=0.6 让点稍微透明，重叠密集的地方颜色更深
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=plot_colors, s=10, alpha=0.6)
    plt.title(f"t-SNE Visualization of Top 20 Clusters + Noise (min_cluster_size={min_cluster_size})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # 5. 制作图例
    legend_elements =[mpatches.Patch(color=color_map_dict[lbl], label=f"Cluster {lbl}") for lbl in top_labels]
    # if -1 in labels:         # 只有当存在噪声点时才添加图例
    #     legend_elements.append(mpatches.Patch(color=noise_color, label="Noise (-1)"))
    legend_elements.append(mpatches.Patch(color=others_color, label="Other Clusters"))
    # 把图例放在图的右侧外边，防止挡住数据点
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout() # 自动调整边界，防止图例被切掉
    
    plt.savefig(output_file, dpi=300)
    plt.show()


def assign_cluster_names_and_sort(merged_df, top_n_words=3, output_file='sorted_cluster_names.csv'):
    if not {'cluster', 'entry_point_cleaned'}.issubset(merged_df.columns):
        raise ValueError("merged_df must contain 'cluster' and 'entry_point_cleaned' columns.")

    cluster_counts = merged_df['cluster'].value_counts().to_dict()
    cluster_name_mapping = {}

    for cluster in merged_df['cluster'].unique():
        texts = merged_df[merged_df['cluster'] == cluster]['entry_point_cleaned']
        combined = ' '.join(texts.dropna().tolist())
        word_counts = Counter(combined.split())
        top_words = [word for word, _ in word_counts.most_common(top_n_words)]
        cluster_name_mapping[cluster] = '_'.join(top_words) if top_words else f"Cluster_{cluster}"

    df = pd.DataFrame({
        'cluster': list(cluster_name_mapping.keys()),
        'name': list(cluster_name_mapping.values()),
        'count': [cluster_counts[k] for k in cluster_name_mapping.keys()]
    })
    df.sort_values(by='count', ascending=False).reset_index(drop=True).to_csv(output_file, index=False)
    print(f"Cluster names saved to {output_file}")
    return df


# 原有的 cluster_vectors，内部调用 do_cluster_vectors + 可视化
def cluster_vectors(vec_file, model_dir, output_vec_file, entry_point_words_file, merged_id_words_cluster_file, tsne_file, wordcloud_dir, min_cluster_size=15, min_samples=None, n_components=100):
    os.makedirs(model_dir, exist_ok=True)

    print("读取向量文件...")
    vec_df = pd.read_csv(vec_file, sep=';', index_col=0)
    X = vec_df.values
    ids = vec_df.index.tolist()
    print(f"读取完成，形状为 {X.shape}")

    # === 调用纯聚类函数 ===
    labels, X_reduced, pca, hdb = do_cluster_vectors(
        X, n_components=n_components, min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    # 保存模型
    pca_model_path = os.path.join(model_dir, f'pca_model_{min_cluster_size}.pkl')
    joblib.dump(pca, pca_model_path)
    print(f"PCA 模型已保存到 '{pca_model_path}'")

    hdbscan_model_path = os.path.join(model_dir, f'hdbscan_model_{min_cluster_size}.pkl')
    joblib.dump(hdb, hdbscan_model_path)
    print(f"HDBSCAN 模型已保存到 '{hdbscan_model_path}'")

    print("获取聚类标签...")
    vec_df['cluster'] = labels
    print("聚类标签已添加到 DataFrame。")

    print(f"保存聚类结果到 {output_vec_file}...")
    vec_df.to_csv(output_vec_file, sep=';')
    print("聚类结果保存完成。")

    print(f"读取 {entry_point_words_file}...")
    ep_df = pd.read_csv(entry_point_words_file, sep=';', dtype={'id': str})
    cluster_info_df = pd.DataFrame({'id': ids, 'cluster': labels})
    cluster_info_df['id'] = cluster_info_df['id'].astype(str)

    merged_df = pd.merge(ep_df, cluster_info_df, on='id')
    merged_df.to_csv(merged_id_words_cluster_file, index=False, sep=';', quoting=csv.QUOTE_ALL)
    assign_cluster_names_and_sort(merged_df, 3, os.path.join(os.path.dirname(merged_id_words_cluster_file), 'cluster_names.csv'))

    # 词云
    top_clusters = merged_df[merged_df['cluster'] != -1]['cluster'].value_counts().index[:20].tolist()
    os.makedirs(f"{wordcloud_dir}_{min_cluster_size}", exist_ok=True)

    for count, c in enumerate(top_clusters, 1):
        all_words = " ".join(merged_df[merged_df['cluster'] == c]['entry_point_cleaned'].dropna())
        if not all_words.strip(): continue
        wc = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(all_words)
        wc.to_file(f"{wordcloud_dir}_{min_cluster_size}/cluster_{c}_top{count}.png")
        print(f"Wordcloud saved for cluster {c}")
    
    # t-SNE 可视化
    print("生成 t-SNE 可视化图...")
    cluster_tsne_top20(
        labels=labels,
        X_reduced=X_reduced,
        min_cluster_size=min_cluster_size,
        output_file=tsne_file
    )
    print(f"t-SNE 可视化图已保存到 {tsne_file}")

    print("All processing done.")