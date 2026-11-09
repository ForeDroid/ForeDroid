# === visualize_integrated_clusters.py ===
# 整合可视化：将 GUI + Background 场景聚类结果放在一起查看
#
# 输入：train/output/ 目录（merge_scenario_labels.py 产出的整合结果）
#   - all_apks_with_clusters.csv : 所有记录（semantic_text 只有 GUI 记录有值）
#   - cluster_names.csv          : cluster -> 场景标签映射（GUI: xxx / BG: xxx）
#
# 功能：
#   1. 一张 t-SNE 散点图同时展示 GUI（圆点）和 BG（叉号）簇
#   2. 对 Top N 个最大簇分别生成词云（文件名含场景标签）
#
# 用法示例：
#   python3 visualize_integrated_clusters.py \
#       --clustered_csv ./train/output/all_apks_with_clusters.csv \
#       --cluster_names ./train/output/cluster_names.csv \
#       --output_dir ./train/output/visualizations

import os
import re
import json
import argparse
import numpy as np
import pandas as pd

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
    """从完整方法签名中提取方法名"""
    if not isinstance(full_signature, str):
        return ''
    match = re.match(r'<[^:]+:\s*[^ ]+\s+([^(]+)\(', full_signature.strip())
    return match.group(1).strip() if match else full_signature


def keep_after_last_dot(s):
    """保留字符串中最后一个点之后的部分"""
    return s.rsplit('.', 1)[-1].strip() if isinstance(s, str) else ''


def camel_split(s):
    """驼峰命名拆分"""
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return re.sub(r'([a-z\d])([A-Z])', r'\1 \2', s)


def detect_sep(filepath):
    """检测 CSV 分隔符"""
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    return ';' if ';' in first_line else ','


def load_cluster_names(names_csv):
    """加载 cluster -> 场景标签映射"""
    names = pd.read_csv(names_csv, sep=detect_sep(names_csv), dtype={'cluster': str})
    mapping = dict(zip(names['cluster'], names['name']))
    print(f"加载场景标签映射: {len(mapping)} 个簇")
    return mapping


def split_gui_bg(df):
    """
    通过 semantic_text 区分 GUI/BG 记录。
    semantic_text 只有 GUI 数据才有值。
    """
    mask_gui = (df['semantic_text'].notna() &
                (df['semantic_text'].astype(str).str.strip() != ''))
    gui_df = df[mask_gui].copy()
    bg_df = df[~mask_gui].copy()
    print(f"记录总数: {len(df)} -> GUI: {len(gui_df)}, Background: {len(bg_df)}")
    return gui_df, bg_df


# ══════════════════════════════════════════════════════════════
#  嵌入生成
# ══════════════════════════════════════════════════════════════

def generate_gui_embeddings(df, model_name="all-MiniLM-L6-v2",
                            cache_folder="../models/model-all-MiniLM-L6-v2"):
    """GUI: BERT 嵌入 semantic_text"""
    from sentence_transformers import SentenceTransformer

    if len(df) == 0:
        return np.array([])

    print(f"加载 BERT 模型: {model_name}")
    model = SentenceTransformer(model_name, cache_folder=cache_folder, local_files_only=True)

    texts = df['semantic_text'].fillna('').tolist()
    print(f"生成 GUI BERT 嵌入，共 {len(texts)} 条...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"GUI 嵌入维度: {embeddings.shape[1]}")
    return embeddings


def generate_bg_embeddings(df, model_path='../models/GoogleNews-vectors-negative300.bin',
                           pca_model_path=None):
    """BG: Word2Vec 平均向量 + PCA 降维"""
    import tempfile
    import joblib
    from sklearn.preprocessing import normalize
    from entry_point_filter import EntryPointFilter

    if len(df) == 0:
        return np.array([])

    tmp_dir = tempfile.mkdtemp(prefix='_bg_integ_')
    tmp_input = os.path.join(tmp_dir, "input.csv")
    tmp_words = os.path.join(tmp_dir, "words.csv")
    tmp_vec = os.path.join(tmp_dir, "vec.csv")

    df[['id', 'entry_point', 'entry_method', 'entry_point_actions']].to_csv(tmp_input, index=False)

    print(f"加载 Word2Vec 模型: {model_path}")
    ep_filter = EntryPointFilter(model_path=model_path)
    print("执行 Background 预处理和向量化...")
    ep_filter.process_csv(tmp_input, tmp_words, tmp_vec)

    vec_df = pd.read_csv(tmp_vec, sep=';', index_col=0)
    X = vec_df.values
    print(f"BG Word2Vec 向量维度: {X.shape}")

    # 对齐 id 顺序
    vec_ids = vec_df.index.astype(str).tolist()
    df_ids = df['id'].astype(str).tolist()
    if vec_ids != df_ids:
        print("警告: 向量行顺序与原始数据不一致，正在对齐...")
        id_to_vec = dict(zip(vec_ids, range(len(vec_ids))))
        aligned_indices = [id_to_vec[i] for i in df_ids if i in id_to_vec]
        X = X[aligned_indices]

    # PCA 降维（使用训练时保存的模型）
    if pca_model_path and os.path.exists(pca_model_path):
        print(f"加载 BG PCA 模型: {pca_model_path}")
        pca = joblib.load(pca_model_path)
        X_normalized = normalize(X, norm='l2')
        X_reduced = pca.transform(X_normalized)
        print(f"BG PCA 降维后维度: {X_reduced.shape}")
    else:
        print("警告: 未提供 BG PCA 模型路径，跳过降维")
        X_reduced = X

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return X_reduced


# ══════════════════════════════════════════════════════════════
#  整合 t-SNE 散点图（GUI + BG 同时展示）
# ══════════════════════════════════════════════════════════════

def plot_integrated_tsne(gui_emb, bg_emb, gui_df, bg_df, output_file, top_n=20,
                         cluster_names=None, n_components=100):
    """
    在一张 t-SNE 图中同时展示 GUI 和 BG 聚类结果。
    - GUI 用圆形（o）标记，BG 用叉号（x）标记
    - Top N 簇用不同颜色，其他簇用灰色
    - 图例标注 Top N 簇的名称（来自 cluster_names.csv）
    
    注意：GUI 使用 BERT 嵌入（384维），BG 使用 PCA 降维后（100维），
    维度不同。拼接前先对 GUI 嵌入做 PCA 降维到 n_components 维，
    使两者维度一致。
    """
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA

    print("\n生成整合 t-SNE 散点图...")

    n_gui = len(gui_emb)
    n_bg = len(bg_emb)

    # 统一维度：GUI (BERT, 384维) -> PCA 降到与 BG 相同维度
    if n_gui > 0 and n_bg > 0 and gui_emb.shape[1] != bg_emb.shape[1]:
        print(f"GUI 嵌入维度 ({gui_emb.shape[1]}) 与 BG 嵌入维度 ({bg_emb.shape[1]}) 不一致")
        print(f"对 GUI 嵌入执行 PCA 降维到 {n_components} 维...")
        pca = PCA(n_components=min(n_components, gui_emb.shape[1], len(gui_emb)), random_state=42)
        gui_emb = pca.fit_transform(gui_emb)
        print(f"GUI 嵌入 PCA 降维后维度: {gui_emb.shape[1]}")
    elif n_gui > 0 and n_bg > 0 and gui_emb.shape[1] == bg_emb.shape[1]:
        print(f"GUI 和 BG 嵌入维度一致: {gui_emb.shape[1]}")

    # 拼接收敛到同一个 t-SNE 空间
    all_emb = np.vstack([gui_emb, bg_emb]) if n_gui > 0 and n_bg > 0 else (
        gui_emb if n_gui > 0 else bg_emb
    )
    all_labels = np.concatenate([gui_df['cluster'].astype(int).values,
                                 bg_df['cluster'].astype(int).values]) if n_gui > 0 and n_bg > 0 else (
        gui_df['cluster'].astype(int).values if n_gui > 0 else bg_df['cluster'].astype(int).values
    )
    is_gui = np.array([True] * n_gui + [False] * n_bg)

    print(f"t-SNE 输入: {all_emb.shape}, GUI: {n_gui}, BG: {n_bg}")
    tsne = TSNE(n_components=2, random_state=42, max_iter=300)
    X_tsne = tsne.fit_transform(all_emb)
    print("t-SNE 降维完成")

    # Top N 簇
    label_counts = pd.Series(all_labels).value_counts()
    top_labels = label_counts.head(top_n).index.tolist()

    # 颜色映射
    cmap_base = plt.cm.get_cmap('tab20', max(top_n, 20))
    color_map_dict = {lbl: cmap_base(i) for i, lbl in enumerate(top_labels)}
    others_color = "lightgray"

    # 绘图
    plt.figure(figsize=(16, 12))

    for i, lbl in enumerate(top_labels):
        mask = (all_labels == lbl) & is_gui
        if mask.any():
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        c=[color_map_dict[lbl]], s=12, alpha=0.6, marker='o',
                        edgecolors='none')
        mask = (all_labels == lbl) & ~is_gui
        if mask.any():
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        c=[color_map_dict[lbl]], s=8, alpha=0.6, marker='x')

    # 其他（非 Top）簇
    others_mask = ~np.isin(all_labels, top_labels)
    if others_mask.any():
        gui_others = others_mask & is_gui
        bg_others = others_mask & ~is_gui
        if gui_others.any():
            plt.scatter(X_tsne[gui_others, 0], X_tsne[gui_others, 1],
                        c=others_color, s=8, alpha=0.3, marker='o', edgecolors='none')
        if bg_others.any():
            plt.scatter(X_tsne[bg_others, 0], X_tsne[bg_others, 1],
                        c=others_color, s=6, alpha=0.3, marker='x')

    plt.title(f"Integrated t-SNE Visualization: GUI + Background Clusters (Top {top_n})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # 图例
    legend_elements = []
    for lbl in top_labels:
        name = cluster_names.get(str(lbl), f"Cluster {lbl}") if cluster_names else f"Cluster {lbl}"
        if len(name) > 40:
            name = name[:37] + "..."
        legend_elements.append(mpatches.Patch(color=color_map_dict[lbl],
                                              label=f"{name} (n={label_counts[lbl]})"))
    legend_elements.append(mpatches.Patch(color=others_color, label="Other Clusters"))
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
               fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"整合 t-SNE 散点图已保存: {output_file}")


# ══════════════════════════════════════════════════════════════
#  整合词云生成
# ══════════════════════════════════════════════════════════════

def build_bg_text(row):
    """为 BG 记录构建词云文本"""
    parts = []
    ep = keep_after_last_dot(row.get('entry_point', ''))
    if ep:
        parts.append(camel_split(ep))
    method = extract_method_name(row.get('entry_method', ''))
    if method:
        parts.append(camel_split(method))
    action = row.get('entry_point_actions', '')
    if isinstance(action, str) and action:
        action_clean = re.sub(r'^android\.intent\.action\.', '', action)
        parts.append(camel_split(action_clean))
    return " ".join([p for p in parts if p])


def generate_integrated_wordclouds(gui_df, bg_df, output_dir, top_n=20, max_words=200,
                                   cluster_names=None):
    """
    对整合后的 Top N 个最大簇生成词云。
    - GUI 记录使用 semantic_text
    - BG 记录使用 entry_point + entry_method + entry_point_actions 拼接
    """
    os.makedirs(output_dir, exist_ok=True)

    # 合并两部分的 cluster + 文本
    gui_parts = gui_df[['cluster', 'semantic_text']].copy()
    gui_parts['_text'] = gui_parts['semantic_text'].fillna('')

    bg_parts = bg_df[['cluster']].copy()
    bg_parts['_text'] = bg_df.apply(build_bg_text, axis=1)

    merged = pd.concat([gui_parts, bg_parts], ignore_index=True)
    merged['cluster'] = merged['cluster'].astype(int)

    # Top N 簇
    label_counts = merged['cluster'].value_counts()
    top_labels = label_counts.head(top_n).index.tolist()
    print(f"\n生成 {len(top_labels)} 个簇的词云...")

    for count, cid in enumerate(tqdm(top_labels, desc="Generating wordclouds")):
        cluster_texts = merged[merged['cluster'] == cid]['_text'].dropna().tolist()
        all_words = " ".join(cluster_texts)
        if not all_words.strip():
            print(f"  ⚠ Cluster {cid} 无文本，跳过词云")
            continue

        wc = WordCloud(width=800, height=400, background_color='white',
                       max_words=max_words, collocations=False).generate(all_words)

        # 文件名包含场景标签
        label = cluster_names.get(str(cid), "") if cluster_names else ""
        if label:
            # 清理标签中的非法字符（去掉 GUI: / BG: 前缀）
            clean_label = re.sub(r'^(GUI|BG):\s*', '', label)
            safe_label = re.sub(r'[^\w\-]+', '_', clean_label)[:50]
            filename = f"cluster_{cid}_{safe_label}.png"
        else:
            filename = f"cluster_{cid}.png"

        out_path = os.path.join(output_dir, filename)
        wc.to_file(out_path)
        print(f"  ✅ Cluster {cid}: {label or '(无标签)'} -> {filename}")

    print(f"词云已保存到: {output_dir}")


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════

def visualize_integrated_clusters(clustered_csv, cluster_names_csv, output_dir,
                                  top_n=20, max_words=200,
                                  gui_model_name="all-MiniLM-L6-v2",
                                  gui_cache_folder="../models/model-all-MiniLM-L6-v2",
                                  bg_model_path='../models/GoogleNews-vectors-negative300.bin',
                                  bg_pca_model=None):
    """GUI + Background 整合聚类可视化"""
    print("=" * 60)
    print("整合聚类可视化 (GUI + Background)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    print(f"读取整合数据: {clustered_csv}")
    sep = detect_sep(clustered_csv)
    df = pd.read_csv(clustered_csv, sep=sep, dtype={'id': str})

    # 过滤噪声点
    valid_mask = (df['cluster'].notna() &
                  (df['cluster'].astype(str).str.strip() != '') &
                  (df['cluster'].astype(str).str.strip() != '-1'))
    df = df[valid_mask].copy()
    df['cluster'] = df['cluster'].astype(int)
    print(f"有效记录: {len(df)} 条 (已剔除噪声点)")

    # 2. 加载场景标签
    cluster_names = None
    if cluster_names_csv and os.path.exists(cluster_names_csv):
        cluster_names = load_cluster_names(cluster_names_csv)

    # 3. 区分 GUI / BG
    gui_df, bg_df = split_gui_bg(df)

    if len(gui_df) == 0 and len(bg_df) == 0:
        raise ValueError("没有找到有效的 GUI 或 Background 记录")

    # 4. 生成嵌入
    gui_emb = generate_gui_embeddings(gui_df, model_name=gui_model_name,
                                      cache_folder=gui_cache_folder) if len(gui_df) > 0 else np.array([])
    bg_emb = generate_bg_embeddings(bg_df, model_path=bg_model_path,
                                    pca_model_path=bg_pca_model) if len(bg_df) > 0 else np.array([])

    # 5. 整合 t-SNE 散点图
    tsne_path = os.path.join(output_dir, "integrated_tsne_top20.png")
    plot_integrated_tsne(gui_emb, bg_emb, gui_df, bg_df, tsne_path,
                         top_n=top_n, cluster_names=cluster_names)

    # 6. 整合词云
    wordcloud_dir = os.path.join(output_dir, "wordclouds")
    generate_integrated_wordclouds(gui_df, bg_df, wordcloud_dir,
                                   top_n=top_n, max_words=max_words,
                                   cluster_names=cluster_names)

    print(f"\n=== 整合聚类可视化完成 ===")
    print(f"t-SNE 散点图: {tsne_path}")
    print(f"词云目录: {wordcloud_dir}")


# ══════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="整合可视化：GUI + Background 场景聚类一起查看 (t-SNE 散点图 + 词云)"
    )
    parser.add_argument('--clustered_csv', required=False,
                        default='./train/output/all_apks_with_clusters.csv',
                        help='整合聚类输出 CSV 路径')
    parser.add_argument('--cluster_names', required=False,
                        default='./train/output/cluster_names.csv',
                        help='cluster_names.csv 路径 (含 GUI: xxx / BG: xxx 标签)')
    parser.add_argument('--output_dir', required=False,
                        default='./train/output/visualizations',
                        help='可视化输出目录')
    parser.add_argument('--top_n', type=int, default=20,
                        help='显示的最大簇数 (默认: 20)')
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
    parser.add_argument('--bg_pca_model', default='./train/bg_centroids/bg_pca_model.pkl',
                        help='Background 管道 PCA 模型路径')

    args = parser.parse_args()

    visualize_integrated_clusters(
        clustered_csv=args.clustered_csv,
        cluster_names_csv=args.cluster_names,
        output_dir=args.output_dir,
        top_n=args.top_n,
        max_words=args.max_words,
        gui_model_name=args.gui_model_name,
        gui_cache_folder=args.gui_cache_folder,
        bg_model_path=args.bg_model_path,
        bg_pca_model=args.bg_pca_model,
    )