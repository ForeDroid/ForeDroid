# === File: cluster_entry_vec_2.py ===
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

def cluster_tsne_top20(labels, X_reduced, num_clusters, output_file):
    tsne = TSNE(n_components=2, random_state=42, n_iter=300)
    X_tsne = tsne.fit_transform(X_reduced)

    counts = np.bincount(labels)
    sorted_indices = np.argsort(counts)[::-1]
    N = 20
    top_labels = sorted_indices[:N]
    top_labels_set = set(top_labels)

    cmap_base = plt.cm.get_cmap('tab20', N)
    color_map_dict = {lbl: cmap_base(i) for i, lbl in enumerate(top_labels)}
    others_color = "gray"
    plot_colors = [color_map_dict.get(lbl, others_color) for lbl in labels]

    plt.figure(figsize=(12, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=plot_colors, s=10)
    plt.title(f"t-SNE Visualization of Top 20 Clusters + Others (clusters={num_clusters})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    legend_elements = [mpatches.Patch(color=color_map_dict[lbl], label=f"Cluster {lbl}") for lbl in top_labels]
    legend_elements.append(mpatches.Patch(color=others_color, label="Others"))
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

def cluster_vectors(vec_file, model_dir, output_vec_file, entry_point_words_file, merged_id_words_cluster_file, tsne_file, wordcloud_dir, num_clusters=100, batch_size=1000, n_components=100):
    os.makedirs(model_dir, exist_ok=True)

    vec_df = pd.read_csv(vec_file, sep=';', index_col=0)
    X = vec_df.values
    ids = vec_df.index.tolist()

    X_normalized = normalize(X, norm='l2')

    if n_components:
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X_normalized)
        joblib.dump(pca, os.path.join(model_dir, f'pca_model_{num_clusters}.pkl'))
    else:
        X_reduced = X_normalized

    mbk = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, random_state=42, max_iter=100, init='k-means++', n_init=10, verbose=1)
    mbk.fit(X_reduced)
    labels = mbk.labels_
    vec_df['cluster'] = labels
    vec_df.to_csv(output_vec_file, sep=';')
    joblib.dump(mbk, os.path.join(model_dir, f'kmeans_model_{num_clusters}.pkl'))

    ep_df = pd.read_csv(entry_point_words_file, sep=';', dtype={'id': str})
    merged_df = pd.merge(ep_df, pd.DataFrame({'id': ids, 'cluster': labels}), on='id')
    merged_df.to_csv(merged_id_words_cluster_file, index=False, sep=';', quoting=csv.QUOTE_ALL)
    assign_cluster_names_and_sort(merged_df, 3, os.path.join(os.path.dirname(merged_id_words_cluster_file), 'cluster_names.csv'))

    top_clusters = merged_df['cluster'].value_counts().index[:20].tolist()
    os.makedirs(f"{wordcloud_dir}_{num_clusters}", exist_ok=True)

    for count, c in enumerate(top_clusters, 1):
        all_words = " ".join(merged_df[merged_df['cluster'] == c]['entry_point_cleaned'].dropna())
        if not all_words.strip(): continue
        wc = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(all_words)
        wc.to_file(f"{wordcloud_dir}_{num_clusters}/cluster_{c}_top{count}.png")
        print(f"Wordcloud saved for cluster {c}")

    print("All processing done.")

