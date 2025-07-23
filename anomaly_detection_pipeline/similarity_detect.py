import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder='/home/ming/model')

def calculate_topk_similarity(test_chain, test_apk, cluster_id, vector_dir, k=50):
    
    vector_path = os.path.join(vector_dir, f"cluster_{cluster_id}.npy")
    apk_path = os.path.join(vector_dir, f"cluster_{cluster_id}_apk.json")

    if not os.path.exists(vector_path) or not os.path.exists(apk_path):
        print(f"Missing {vector_path} or {apk_path}, returning similarity 0")
        return 0.0

    group_vecs = np.load(vector_path)
    with open(apk_path, 'r') as f:
        group_apks = json.load(f)

    filtered_vecs = [group_vecs[i] for i, apk in enumerate(group_apks) if apk != test_apk]
    if not filtered_vecs:
        print(f"Cluster {cluster_id} only contains test APK {test_apk}, returning similarity 0")
        return 0.0

    filtered_vecs = np.array(filtered_vecs)
    test_vec = model.encode([test_chain])
    similarities = cosine_similarity(test_vec, filtered_vecs)[0]  # shape: (N,)
    k = min(k, len(similarities))
    topk_avg = float(np.mean(np.partition(similarities, -k)[-k:])) 
    return topk_avg

def detect_anomalies_in_all_apks(new_apk_df, vector_dir, k=50):
    anomaly_scores = []
    avg_similarities = []

    for _, row in tqdm(new_apk_df.iterrows(), total=len(new_apk_df), desc="Detecting anomalies"):
        test_chain = row['generated_description']
        test_apk = row['apk_name']
        cluster_label = str(row['cluster'])

        avg_topk_similarity = calculate_topk_similarity(test_chain, test_apk, cluster_label, vector_dir, k)
        anomaly_score = 1 - avg_topk_similarity 

        anomaly_scores.append(anomaly_score)
        avg_similarities.append(avg_topk_similarity)

    new_apk_df['avg_topk_similarity'] = avg_similarities
    new_apk_df['anomaly_score'] = anomaly_scores
    print("Anomaly detection complete.")
    return new_apk_df

def aggregate_anomaly_scores(df, threshold=0.5, agg_method="mean"):
    """
    APK aggerate anomaly_score
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
    k_neighbors = 50  
    threshold = 0.5   
    label = 'benign'
    data_dir = '/media/ming/ExtremeSSD/jiaming/processIccBot/data2/output/similarity_llm'
    vector_dir = f'{data_dir}/precomputed_vectors'

    input_file = f'/media/ming/ExtremeSSD/jiaming/processIccBot/data2/output/llm_unknown_apks_vectors_benign_test_api/unknown_apks_all_apks_with_clusters.csv'
    new_apk_df = pd.read_csv(input_file, sep=';', dtype={'id': str})

    
    df_with_anomalies = detect_anomalies_in_all_apks(new_apk_df, vector_dir, k=k_neighbors)
    anomaly_outfile = f'{data_dir}/{label}_top{k_neighbors}_anomaly_scores.csv'
    df_with_anomalies.to_csv(anomaly_outfile, index=False, sep=';')
    print(f"Saved anomaly results to {anomaly_outfile}")

   
    aggregation = aggregate_anomaly_scores(df_with_anomalies, threshold=threshold, agg_method="mean")
    agg_outfile = f'{data_dir}/{label}_top{k_neighbors}_agg_anomaly_scores.csv'
    aggregation.to_csv(agg_outfile, index=False, sep=';')
    print(f"Saved aggregated scores to {agg_outfile}")

if __name__ == '__main__':
    main()
