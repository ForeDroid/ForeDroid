"""Explicit model loading and aligned behavior vectors; no work on import."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def precompute_cluster_vectors(input_dir,output_dir,model_path='/home/lb/models/foredroid/all-MiniLM-L6-v2',batch_size=128):
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(str(model_path),local_files_only=True,device='cpu')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    for file in sorted(Path(input_dir).glob('cluster_*.csv')):
        cid=file.stem.removeprefix('cluster_')
        if cid=='-1':continue
        frame=pd.read_csv(file,sep=';',dtype=str,keep_default_na=False)
        if not {'generated_description','apk_name'}<=set(frame):raise ValueError(f'Missing columns: {file}')
        if frame['generated_description'].str.strip().eq('').any() or frame['apk_name'].str.strip().eq('').any():raise ValueError(f'Missing descriptions/APKs: {file}')
        embeddings=model.encode(frame['generated_description'].tolist(),batch_size=batch_size,normalize_embeddings=True,show_progress_bar=True)
        target=output/f'cluster_{cid}.npy';temp=target.with_suffix('.npy.tmp')
        with temp.open('wb') as f:np.save(f,embeddings)
        temp.replace(target)
        target=output/f'cluster_{cid}_apk.json';temp=target.with_suffix('.json.tmp');temp.write_text(json.dumps(frame['apk_name'].tolist()));temp.replace(target)


if __name__=='__main__':
    from run_pipeline import main
    main(default_only='references')
