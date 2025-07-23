import pandas as pd
import os
import argparse
from sentence_transformers import SentenceTransformer, util
import torch

def compute_topk_similarity_vectors(merged_file, cluster_dir, output_file, k=50):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    df = pd.read_csv(merged_file, sep=';', dtype={'id': str})
    id2desc = dict(zip(df['id'], df['behavior_description']))

    results = []
    for fname in os.listdir(cluster_dir):
        if not fname.endswith('.csv'):
            continue
        cluster_df = pd.read_csv(os.path.join(cluster_dir, fname), sep=';', dtype={'id': str})
        ids = cluster_df['id'].tolist()
        descs = [id2desc.get(i, "") for i in ids]
        embeddings = model.encode(descs, convert_to_tensor=True)

        for i, eid in enumerate(ids):
            if len(embeddings) == 1:
                avg_sim = 0.0
            else:
               
                sim_scores = util.cos_sim(embeddings[i], embeddings)[0]  
                sim_scores[i] = -1.0 
                topk = torch.topk(sim_scores, min(k, len(sim_scores)-1)).values
                avg_sim = float(topk.mean())

            anomaly_score = 1 - avg_sim
            results.append({'id': eid, 'anomaly_score': anomaly_score})

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, sep=';', index=False)
    print(f"[Done] Saved anomaly scores to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_file", type=str, required=True)
    parser.add_argument("--cluster_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    compute_topk_similarity_vectors(args.merged_file, args.cluster_dir, args.output_file, args.k)
