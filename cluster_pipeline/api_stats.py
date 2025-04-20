import pandas as pd
import json
import os

def generate_cluster_api_stats(merged_vec_file, output_file='cluster_api_stats.json'):
    if not os.path.exists(merged_vec_file):
        raise FileNotFoundError(f"{merged_vec_file} not found")
    df = pd.read_csv(merged_vec_file, sep=';', dtype={'id': str})
    if 'cluster' not in df.columns or 'sensitiveAPI' not in df.columns:
        raise ValueError("Missing 'cluster' or 'sensitiveAPI' columns in input")
    stats = {str(k): v['sensitiveAPI'].value_counts().to_dict()
             for k, v in df.groupby('cluster')}
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)