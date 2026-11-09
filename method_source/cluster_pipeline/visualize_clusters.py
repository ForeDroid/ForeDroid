# === visualize_clusters.py ===
# 聚类训练完成后，基于已产出的聚类结果 CSV 生成可视化：
#   1. t-SNE 散点图（Top N 簇 + 噪声点）
#   2. 每个 Top N 簇的词云（查看簇内语义连贯性）
#
# 支持两种类型：
#   --type gui : 读取 gui_clustered_data.csv，用 semantic_text + BERT 重新嵌入
#   --type bg  : 读取 bg_clustered_data.csv，用 entry_point/method/actions + Word2Vec + PCA 重新嵌入
#
# 用法示例：
#   python3 visualize_clusters.py --type gui \
#       --clustered_csv ./train/gui_centroids/gui_clustered_data.csv \
#       --output_dir ./train/gui_centroids/visualizations
#
#   python3 visualize_clusters.py --type bg \
#       --clustered_csv ./train/bg_centroids/bg_clustered_data.csv \
#       --output_dir ./train/bg_centroids/visualizations \
#       --bg_pca_model ./train/bg_centroids/bg_pca_model.pkl

import os
import re
import csv
import json
import argparse
import numpy as np
import pandas as pd

# matplotlib 使用 Agg 后端（不弹出 GUI 窗口，直接保存文件）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud

from sklearn.manifold import TSNE
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
#  公共辅助函数
# ══════════════════════════════════════════════════════════════

def extract_method_name(full_signature):
    """从完整方法签名中提取方法名，如 '<com.foo.Bar: void onCreate(android.os.Bundle)>' -> 'onCreate'"""
    if not isinstance(full_signature, str):
        return ''
    match = re.match(r'<[^:]+:\s*[^ ]+\s+([^(]+)\(', full_signature.strip())
    return match.group(1).strip() if match else full_signature


def keep_after_last_dot(s):
    """保留字符串中最后一个点之后的部分，如 'com.foo.Bar' -> 'Bar'"""
    return s.rsplit('.', 1)[-1].strip() if isinstance(s, str) else ''


def camel_split(s):
    """驼峰命名拆分，如 'onCreate' -> 'on Create'"""
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return re.sub(r'([a-z\d])([A-Z])', r'\1 \2', s)


def load_clustered_data(clustered_csv):
    """
    读取聚类输出 CSV（自动检测分隔符），过滤噪声点 (-1)。
    返回 DataFrame，cluster 列转为 int 类型。
    """
    # 自动检测分隔符（支持分号 ; 和逗号 ,）
    with open(clustered_csv, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    sep = ';' if ';' in first_line else ','
    print(f"检测到 CSV 分隔符: '{sep}'")
    df = pd.read_csv(clustered_csv, sep=sep, dtype={'id': str})
    # 过滤 cluster 为空或 -1 (噪声) 的记录
    valid_mask = (df['cluster'].notna() &
                  (df['cluster'].astype(str).str.strip() != '') &
                  (df['cluster'].astype(str).str.strip() != '-1'))
    df = df[valid_mask].copy()
    df['cluster'] = df['cluster'].astype(int)
    print(f"读取聚类数据: {len(df)} 条有效记录 (已剔除噪声点)")
    return df


# ══════════════════════════════════════════════════════════════
#  嵌入生成（GUI: BERT；BG: Word2Vec + PCA）
# ══════════════════════════════════════════════════════════════

def generate_gui_embeddings(df, model_name="all-MiniLM-L6-v2",
                            cache_folder="../models/model-all-MiniLM-L6-v2"):
    """
    GUI 管道：用 semantic_text 重新生成 BERT 嵌入。
    对应论文 Equation 1: v_e = BERT(text label ∥ context label)
    """
    from sentence_transformers import SentenceTransformer

    print(f"加载 BERT 模型: {model_name}")
    model = SentenceTransformer(model_name, cache_folder=cache_folder, local_files_only=True)

    texts = df['semantic_text'].fillna('').tolist()
    print(f"生成 BERT 嵌入，共 {len(texts)} 条...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"嵌入维度: {embeddings.shape[1]}")
    return embeddings


def generate_bg_embeddings(df, model_path='../models/GoogleNews-vectors-negative300.bin',
                           pca_model_path=None):
    """
    Background 管道：用 entry_point + entry_method + entry_point_actions
    通过 EntryPointFilter 生成 Word2Vec 平均向量，再用保存的 PCA 模型降维。
    对应论文 Equation 4。
    """
    import tempfile
    import joblib
    from sklearn.preprocessing import normalize
    from entry_point_filter import EntryPointFilter

    # 1. 检查必要列
    required_cols = {'id', 'entry_point', 'entry_method', 'entry_point_actions'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"BG 数据缺少必需列: {required_cols - set(df.columns)}")

    # 2. 构造临时 CSV（EntryPointFilter 的 process_csv 读逗号分隔）
    tmp_dir = tempfile.mkdtemp(prefix='_bg_vis_')
    tmp_input = os.path.join(tmp_dir, "input.csv")
    tmp_words = os.path.join(tmp_dir, "words.csv")
    tmp_vec = os.path.join(tmp_dir, "vec.csv")

    df[['id', 'entry_point', 'entry_method', 'entry_point_actions']].to_csv(tmp_input, index=False)

    # 3. EntryPointFilter 预处理 + Word2Vec 向量化
    print(f"加载 Word2Vec 模型: {model_path}")
    ep_filter = EntryPointFilter(model_path=model_path)
    print("执行预处理和向量化...")
    ep_filter.process_csv(tmp_input, tmp_words, tmp_vec)

    vec_df = pd.read_csv(tmp_vec, sep=';', index_col=0)
    X = vec_df.values
    print(f"Word2Vec 向量维度: {X.shape}")

    # 4. PCA 降维（使用训练时保存的模型）
    if pca_model_path and os.path.exists(pca_model_path):
        print(f"加载 PCA 模型: {pca_model_path}")
        pca = joblib.load(pca_model_path)
        X_normalized = normalize(X, norm='l2')
        X_reduced = pca.transform(X_normalized)
        print(f"PCA 降维后维度: {X_reduced.shape}")
    else:
        print("警告: 未提供 PCA 模型路径，跳过降维")
        X_reduced = X

    # 注意：vec_df 的行顺序与 df 的 id 顺序一致（process_csv 按输入行顺序处理）
    # 但需要确保 id 顺序匹配
    vec_ids = vec_df.index.astype(str).tolist()
    df_ids = df['id'].astype(str).tolist()
    if vec_ids != df_ids:
        print("警告: 向量行顺序与原始数据不一致，正在对齐...")
        # 按 id 对齐
        id_to_vec = dict(zip(vec_ids, range(len(vec_ids))))
        aligned_indices = [id_to_vec[i] for i in df_ids if i in id_to_vec]
        if len(aligned_indices) != len(df_ids):
            raise ValueError("部分 id 在向量文件中缺失，请检查输入数据")
        X_reduced = X_reduced[aligned_indices]

    # 清理临时文件
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return X_reduced


# ══════════════════════════════════════════════════════════════
#  t-SNE 散点图
# ══════════════════════════════════════════════════════════════

def plot_tsne_top_n(embeddings, labels, output_file, top_n=20, title_suffix=""):
    """
    将嵌入向量降维到 2D，绘制 Top N 个最大簇的散点图。
    - 前 Top N 簇：每种颜色标注
    - 噪声（-1，如果有）：黑色
    - 其他小簇：浅灰色
    """
    print(f"执行 t-SNE 降维 (n_components=2)...")
    tsne = TSNE(n_components=2, random_state=42, max_iter=300)
    X_tsne = tsne.fit_transform(embeddings)

    # 统计每个簇的样本数，找出 Top N
    label_counts = pd.Series(labels).value_counts()
    top_labels = label_counts.head(top_n).index.tolist()
    print(f"Top {top_n} 个簇: {top_labels}")

    # 颜色映射
    cmap_base = plt.cm.get_cmap('tab20', max(top_n, 20))
    color_map_dict = {lbl: cmap_base(i) for i, lbl in enumerate(top_labels)}
    noise_color = "black"
    others_color = "lightgray"

    plot_colors = []
    for lbl in labels:
        if lbl in top_labels:
            plot_colors.append(color_map_dict[lbl])
        elif lbl == -1:
            plot_colors.append(noise_color)
        else:
            plot_colors.append(others_color)

    # 绘图
    plt.figure(figsize=(14, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=plot_colors, s=10, alpha=0.6)
    plt.title(f"t-SNE Visualization of Top {top_n} Clusters{title_suffix}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # 图例
    legend_elements = [mpatches.Patch(color=color_map_dict[lbl], label=f"Cluster {lbl} ({label_counts[lbl]})")
                       for lbl in top_labels]
    if -1 in labels:
        legend_elements.append(mpatches.Patch(color=noise_color, label="Noise (-1)"))
    if len(top_labels) < len(label_counts):
        legend_elements.append(mpatches.Patch(color=others_color, label="Other Clusters"))
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE 散点图已保存: {output_file}")


# ══════════════════════════════════════════════════════════════
#  词云生成
# ══════════════════════════════════════════════════════════════

def generate_wordclouds(df, text_column, output_dir, top_n=20, max_words=200):
    """
    为 Top N 个最大簇生成词云图。
    text_column: 用于生成词云的文本列（GUI 用 semantic_text；BG 用拼接过的方法名/类名）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 统计每个簇的样本数
    label_counts = df['cluster'].value_counts()
    top_labels = label_counts.head(top_n).index.tolist()

    # 读取场景标签映射（向上查找）
    scenario_labels = {}
    base_dir = os.path.dirname(output_dir)
    found = False
    for depth in range(3):  # 向上追溯最多 3 层
        candidates = [
            os.path.join(base_dir, "scenario_labels.json"),
            os.path.join(base_dir, "gui_scenario_labels.json"),
            os.path.join(base_dir, "bg_scenario_labels.json"),
        ]
        for p in candidates:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    scenario_labels = json.load(f)
                print(f"加载场景标签映射: {p}")
                found = True
                break
        if found:
            break
        base_dir = os.path.dirname(base_dir)
    if not found:
        print("警告: 未找到场景标签映射文件，词云文件名将不包含标签")

    print(f"生成 {len(top_labels)} 个簇的词云...")
    for count, cid in enumerate(tqdm(top_labels, desc="Generating wordclouds")):
        cluster_texts = df[df['cluster'] == cid][text_column].dropna().tolist()
        all_words = " ".join(cluster_texts)
        if not all_words.strip():
            print(f"  ⚠ Cluster {cid} 无文本，跳过词云")
            continue

        wc = WordCloud(width=800, height=400, background_color='white',
                       max_words=max_words, collocations=False).generate(all_words)

        # 文件名包含场景标签
        label = scenario_labels.get(str(cid), "")
        if label:
            safe_label = re.sub(r'[^\w\-]+', '_', label)[:50]
            filename = f"cluster_{cid}_top{count+1}_{safe_label}.png"
        else:
            filename = f"cluster_{cid}_top{count+1}.png"

        out_path = os.path.join(output_dir, filename)
        wc.to_file(out_path)
        print(f"  ✅ Cluster {cid}: {label or '(无标签)'} -> {filename}")

    print(f"词云已保存到: {output_dir}")


# ══════════════════════════════════════════════════════════════
#  BG 词云文本拼接
# ══════════════════════════════════════════════════════════════

def build_bg_text_column(df):
    """
    为 BG 数据构建词云用文本列：
    从 entry_method 提取方法名 + entry_point 取类名 + entry_point_actions，做驼峰拆分。
    如 'com.foo.SplashActivity' + 'onCreate' + 'android.intent.action.MAIN'
      -> 'Splash Activity on Create MAIN'
    """
    def build_text(row):
        parts = []
        # 类名（取最后一段）
        ep = keep_after_last_dot(row.get('entry_point', ''))
        if ep:
            parts.append(camel_split(ep))
        # 方法名
        method = extract_method_name(row.get('entry_method', ''))
        if method:
            parts.append(camel_split(method))
        # intent 动作（去掉 android.intent.action. 前缀）
        action = row.get('entry_point_actions', '')
        if isinstance(action, str) and action:
            action_clean = re.sub(r'^android\.intent\.action\.', '', action)
            parts.append(camel_split(action_clean))
        return " ".join([p for p in parts if p])

    return df.apply(build_text, axis=1)


# ══════════════════════════════════════════════════════════════
#  主流程（GUI）
# ══════════════════════════════════════════════════════════════

def visualize_gui_clusters(clustered_csv, output_dir, top_n=20, max_words=200,
                           model_name="all-MiniLM-L6-v2",
                           cache_folder="../models/model-all-MiniLM-L6-v2"):
    """GUI 聚类可视化：BERT 嵌入 → t-SNE 散点图 → 词云"""
    print("=" * 60)
    print("GUI 聚类可视化")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    df = load_clustered_data(clustered_csv)

    # 2. 确认 semantic_text 列存在
    if 'semantic_text' not in df.columns:
        raise ValueError("GUI 聚类数据缺少 semantic_text 列，请先运行 gui_scenario_cluster.py")

    # 3. 生成 BERT 嵌入
    embeddings = generate_gui_embeddings(df, model_name=model_name, cache_folder=cache_folder)

    # 4. t-SNE 散点图
    tsne_path = os.path.join(output_dir, "gui_tsne_top20.png")
    plot_tsne_top_n(embeddings, df['cluster'].values, tsne_path,
                    top_n=top_n, title_suffix=" (GUI)")

    # 5. 词云
    wordcloud_dir = os.path.join(output_dir, "wordclouds")
    generate_wordclouds(df, text_column='semantic_text', output_dir=wordcloud_dir,
                        top_n=top_n, max_words=max_words)

    print(f"\n=== GUI 聚类可视化完成 ===")
    print(f"t-SNE 散点图: {tsne_path}")
    print(f"词云目录: {wordcloud_dir}")


# ══════════════════════════════════════════════════════════════
#  主流程（BG）
# ══════════════════════════════════════════════════════════════

def visualize_bg_clusters(clustered_csv, output_dir, top_n=20, max_words=200,
                          model_path='../models/GoogleNews-vectors-negative300.bin',
                          pca_model_path=None):
    """Background 聚类可视化：Word2Vec + PCA → t-SNE 散点图 → 词云"""
    print("=" * 60)
    print("Background 聚类可视化")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    df = load_clustered_data(clustered_csv)

    # 2. 生成嵌入向量
    embeddings = generate_bg_embeddings(df, model_path=model_path, pca_model_path=pca_model_path)

    # 3. t-SNE 散点图
    tsne_path = os.path.join(output_dir, "bg_tsne_top20.png")
    plot_tsne_top_n(embeddings, df['cluster'].values, tsne_path,
                    top_n=top_n, title_suffix=" (Background)")

    # 4. 构建词云文本列并生成词云
    df['_vis_text'] = build_bg_text_column(df)
    wordcloud_dir = os.path.join(output_dir, "wordclouds")
    generate_wordclouds(df, text_column='_vis_text', output_dir=wordcloud_dir,
                        top_n=top_n, max_words=max_words)

    print(f"\n=== Background 聚类可视化完成 ===")
    print(f"t-SNE 散点图: {tsne_path}")
    print(f"词云目录: {wordcloud_dir}")


# ══════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="聚类结果可视化：t-SNE 散点图 + 词云 (支持 GUI / Background 两种管道)"
    )
    parser.add_argument('--type', choices=['gui', 'bg'], required=True,
                        help='可视化类型: gui = GUI场景聚类, bg = Background场景聚类')
    parser.add_argument('--clustered_csv', required=True,
                        help='聚类输出 CSV 路径 (如 gui_clustered_data.csv 或 bg_clustered_data.csv)')
    parser.add_argument('--output_dir', required=True,
                        help='可视化输出目录')
    parser.add_argument('--top_n', type=int, default=20,
                        help='t-SNE 和词云显示的最大簇数 (默认: 20)')
    parser.add_argument('--max_words', type=int, default=200,
                        help='词云最大词数 (默认: 200)')

    # GUI 参数
    parser.add_argument('--gui_model_name', default='all-MiniLM-L6-v2',
                        help='GUI 管道 BERT 模型名')
    parser.add_argument('--gui_cache_folder', default='../models/model-all-MiniLM-L6-v2',
                        help='GUI 管道 BERT 模型缓存目录')

    # BG 参数
    parser.add_argument('--bg_model_path', default='../models/GoogleNews-vectors-negative300.bin',
                        help='Background 管道 Word2Vec 模型路径')
    parser.add_argument('--bg_pca_model', default=None,
                        help='Background 管道 PCA 模型路径 (bg_pca_model.pkl)')

    args = parser.parse_args()

    if args.type == 'gui':
        visualize_gui_clusters(
            clustered_csv=args.clustered_csv,
            output_dir=args.output_dir,
            top_n=args.top_n,
            max_words=args.max_words,
            model_name=args.gui_model_name,
            cache_folder=args.gui_cache_folder,
        )
    else:
        visualize_bg_clusters(
            clustered_csv=args.clustered_csv,
            output_dir=args.output_dir,
            top_n=args.top_n,
            max_words=args.max_words,
            model_path=args.bg_model_path,
            pca_model_path=args.bg_pca_model,
        )