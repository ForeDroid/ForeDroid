import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set proxy if needed
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder='/home/ming/model')

def calculate_similarity(test_chain, test_apk, cluster_id, vector_dir):
    """
    Compute max similarity of a generated description against a cluster (excluding same APK).
    """
    vector_path = os.path.join(vector_dir, f"cluster_{cluster_id}.npy")
    apk_path = os.path.join(vector_dir, f"cluster_{cluster_id}_apk.json")

    if not os.path.exists(vector_path) or not os.path.exists(apk_path):
        print(f"Missing {vector_path} or {apk_path}, returning similarity 0")
        return 0

    group_vecs = np.load(vector_path)
    with open(apk_path, 'r') as f:
        group_apks = json.load(f)

    filtered_vecs = [group_vecs[i] for i, apk in enumerate(group_apks) if apk != test_apk]
    if not filtered_vecs:
        print(f"Cluster {cluster_id} only contains test APK {test_apk}, returning similarity 0")
        return 0

    filtered_vecs = np.array(filtered_vecs)
    test_vec = model.encode([test_chain])
    similarities = cosine_similarity(test_vec, filtered_vecs)
    return float(np.max(similarities))

def detect_anomalies_in_all_apks(new_apk_df, vector_dir, threshold=0.7):
    """
    Run anomaly detection on generated descriptions for unknown APKs.
    """
    anomaly_scores = []
    max_similarities = []

    for _, row in tqdm(new_apk_df.iterrows(), total=len(new_apk_df), desc="Detecting anomalies"):
        test_chain = row['generated_description']
        test_apk = row['apk_name']
        cluster_label = str(row['cluster'])

        max_similarity = calculate_similarity(test_chain, test_apk, cluster_label, vector_dir)
        anomaly_scores.append(1 if max_similarity < threshold else 0)
        max_similarities.append(max_similarity)

    new_apk_df['anomaly_score'] = anomaly_scores
    new_apk_df['max_similarity'] = max_similarities
    print("Anomaly detection complete.")
    return new_apk_df

def aggregate_anomaly_scores(df, threshold=1.0, agg_method="sum"):
    """
    Aggregate anomaly scores per APK and classify as anomalous.
    """
    if agg_method == "sum":
        aggregation = df.groupby('apk_name')['anomaly_score'].sum().reset_index()
    elif agg_method == "mean":
        aggregation = df.groupby('apk_name')['anomaly_score'].mean().reset_index()
    else:
        raise ValueError("agg_method must be 'sum' or 'mean'")

    aggregation.rename(columns={'anomaly_score': 'total_anomaly_score'}, inplace=True)
    aggregation['is_anomalous'] = aggregation['total_anomaly_score'] > threshold

    print("APK-level anomaly classification complete.")
    print("Summary:")
    print(aggregation['is_anomalous'].value_counts())
    return aggregation

def main():
    num_clusters = 2000
    similarity_threshold = 0.5
    label = 'benign'
    data_dir = '/media/ming/ExtremeSSD/jiaming/processIccBot/data2/output/similarity_llm'
    vector_dir = f'{data_dir}/precomputed_vectors'

    input_file = f'/media/ming/ExtremeSSD/jiaming/processIccBot/data2/output/llm_unknown_apks_vectors_benign_test_api/unknown_apks_all_apks_with_clusters.csv'
    new_apk_df = pd.read_csv(input_file, sep=';', dtype={'id': str})

    df_with_anomalies = detect_anomalies_in_all_apks(new_apk_df, vector_dir, similarity_threshold)
    anomaly_outfile = f'{data_dir}/{label}_with_sim_new_apk_with_anomalies.csv'
    df_with_anomalies.to_csv(anomaly_outfile, index=False, sep=';')
    print(f"Saved anomaly results to {anomaly_outfile}")

    aggregation = aggregate_anomaly_scores(df_with_anomalies)
    agg_outfile = f'{data_dir}/{similarity_threshold}_{label}_with_sim_aggregation_anomalies_scores.csv'
    aggregation.to_csv(agg_outfile, index=False, sep=';')
    print(f"Saved aggregated scores to {agg_outfile}")

if __name__ == '__main__':
    main()

