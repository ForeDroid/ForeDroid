import pandas as pd
import os
import argparse
from sentence_transformers import SentenceTransformer, util

def compute_similarity_vectors(merged_file, cluster_dir, output_file):
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
            others = embeddings[:i] + embeddings[i+1:] if len(embeddings) > 1 else embeddings
            if len(others) > 0:
                sim_scores = util.cos_sim(embeddings[i], others)
                max_sim = float(sim_scores.max())
            else:
                max_sim = 1.0
            results.append({'id': eid, 'max_similarity': max_sim})

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, sep=';', index=False)
    print(f"[Done] Saved similarity vectors to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_file", type=str, required=True)
    parser.add_argument("--cluster_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    compute_similarity_vectors(args.merged_file, args.cluster_dir, args.output_file)