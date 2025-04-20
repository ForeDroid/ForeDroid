import argparse
import os
import pandas as pd
from .entry_point_filter import EntryPointFilter
from .cluster_entry import cluster_vectors
from .group_by_cluster import merge_original_with_clusters, save_clusters_separately, save_cluster_id_mapping
from .api_stats import generate_cluster_api_stats

def run_pipeline(input_csv, output_dir, vector_output, cluster_label_dir, label_map_output, api_stat_output, num_clusters, model_path):
    ep_filter = EntryPointFilter(model_path=model_path)
    cleaned_words_file = os.path.join(output_dir, "entry_point_words.csv")
    vector_file = os.path.join(output_dir, "entry_point_vec.csv")
    ep_filter.process_csv(input_csv, cleaned_words_file, vector_file)

    merged_clusters_file = os.path.join(output_dir, "merged_id_words_clusters.csv")
    cluster_vectors(vector_file, os.path.join(output_dir, "models"), vector_output,
                    cleaned_words_file, merged_clusters_file, 
                    os.path.join(output_dir, "tsne.png"),
                    os.path.join(output_dir, "wordclouds"),
                    num_clusters=num_clusters)

    merged_df = merge_original_with_clusters(input_csv, pd.read_csv(merged_clusters_file, sep=';'), vector_output)
    save_clusters_separately(merged_df, output_dir=os.path.join(output_dir, "clusters_by_label"))
    save_cluster_id_mapping(merged_df, output_file=label_map_output)
    generate_cluster_api_stats(vector_output, output_file=api_stat_output)