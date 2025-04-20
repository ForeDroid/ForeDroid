import os
import csv
import pandas as pd

def save_clusters_separately(merged_df, output_dir='clusters_by_label', cluster_column='cluster'):
    os.makedirs(output_dir, exist_ok=True)
    for cluster in merged_df[cluster_column].unique():
        cluster_df = merged_df[merged_df[cluster_column] == cluster]
        cluster_df.to_csv(os.path.join(output_dir, f'cluster_{cluster}.csv'), index=False, sep=';', quoting=csv.QUOTE_ALL)

def save_cluster_id_mapping(merged_df, output_file='cluster_id_mapping.csv', cluster_column='cluster', id_column='id'):
    cluster_id_df = merged_df.groupby(cluster_column)[id_column].apply(lambda x: ','.join(x)).reset_index()
    cluster_id_df.to_csv(output_file, index=False, sep=';', quoting=csv.QUOTE_ALL)

def merge_original_with_clusters(original_csv, cluster_labels_df, output_csv='all_apks_with_clusters.csv'):
    original_df = pd.read_csv(original_csv, sep=',', dtype={'id': str})
    cluster_labels_df['id'] = cluster_labels_df['id'].astype(str)
    merged_df = pd.merge(original_df, cluster_labels_df, on='id', how='inner')
    merged_df.to_csv(output_csv, index=False, sep=';', quoting=csv.QUOTE_ALL)
    return merged_df