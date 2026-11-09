#!/usr/bin/env python3
"""Run the train/test scenario DAG with bounded CPU parallelism and resume."""
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

BASE=Path(__file__).resolve().parent
ROOT=Path('/home/lb/data/Dataset/ForeDroid_output/2014-2017')
MODELS=Path('/home/lb/models/foredroid')


def publish(path,data):
    pending=path.with_name(path.name+'.tmp');pending.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');pending.replace(path)


def branch_folder(root,split,branch):
    return root/split/'cluster_pipeline'/(f'{branch}_centroids' if split=='train' else branch)


def merge_results(root,split,training_root=None):
    import pandas as pd
    import numpy as np
    from merge_scenario_labels import update_cluster_names
    from group_by_cluster import save_clusters_separately,save_cluster_id_mapping
    gui=branch_folder(root,split,'gui')/(f'gui_clustered_data.csv' if split=='train' else 'gui_assigned_data.csv')
    bg=branch_folder(root,split,'bg')/(f'bg_clustered_data.csv' if split=='train' else 'bg_assigned_data.csv')
    models=(training_root or root)/'train/cluster_pipeline';out=root/split/'cluster_pipeline/output';out.mkdir(exist_ok=True)
    bgids=[int(x) for x in json.loads((models/'bg_centroids/bg_cluster_ids.json').read_text())]
    offset=max(bgids)+1 if bgids else 10000
    signature=hashlib.sha256(json.dumps([(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in [gui,bg,models/'bg_centroids/bg_cluster_ids.json']]+[offset]).encode()).hexdigest()
    checkpoint=out/'manifest.json'
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved.get('signature')==signature and all((out/p).exists() for p in saved['artifacts']):
            print(f'RESUME {split} merge: completed output retained',flush=True);return saved['summary']
    gui_df=pd.read_csv(gui,sep=';',dtype=str,keep_default_na=False);bg_df=pd.read_csv(bg,sep=';',dtype=str,keep_default_na=False)
    if set(gui_df['id']) & set(bg_df['id']):raise ValueError('GUI and BG IDs overlap')
    for frame in [gui_df,bg_df]:frame['cluster']=pd.to_numeric(frame['cluster'],errors='raise').astype(int)
    valid=gui_df['cluster'].ge(0);gui_df.loc[valid,'cluster']+=offset
    merged=pd.concat([bg_df,gui_df],ignore_index=True);merged['cluster']=merged['cluster'].astype(str)
    original=pd.read_csv(root/split/'final_training_data_with_desc.csv',usecols=['id'],dtype=str)
    if len(merged)!=len(original) or not merged['id'].is_unique or set(merged['id'])!=set(original['id']):raise ValueError('Merged scenario output dropped or duplicated input IDs')
    order={value:i for i,value in enumerate(original['id'])};merged['_order']=merged['id'].map(order);merged.sort_values('_order',inplace=True);merged.drop(columns='_order',inplace=True)
    with tempfile.TemporaryDirectory(prefix='.merge-',dir=out) as tmp:
        tmp=Path(tmp)
        merged.to_csv(tmp/'all_apks_with_clusters.csv',sep=';',index=False)
        update_cluster_names(merged,str(models/'gui_centroids/gui_scenario_labels.json'),str(models/'bg_centroids/bg_scenario_labels.json'),offset,str(tmp))
        save_cluster_id_mapping(merged,str(tmp/'cluster_id_mapping.csv'))
        # Downstream normal-behavior vectors consume the original training cluster files.
        if split=='train':
            groups=out/'clusters_by_label';groups.mkdir(exist_ok=True)
            for cid,frame in merged.groupby('cluster',sort=False):
                pending=groups/f'cluster_{cid}.csv.tmp';frame.to_csv(pending,sep=';',index=False);pending.replace(groups/f'cluster_{cid}.csv')
            expected={f'cluster_{cid}.csv' for cid in merged['cluster'].unique()}
            for old in groups.glob('cluster_*.csv'):
                if old.name not in expected:old.unlink()
        artifacts=[p.name for p in tmp.iterdir()]
        for p in tmp.iterdir():os.replace(p,out/p.name)
    if split=='train':artifacts+=['clusters_by_label']
    summary=dict(rows=len(merged),gui_rows=len(gui_df),bg_rows=len(bg_df),noise=int(merged['cluster'].eq('-1').sum()),clusters=int(merged.loc[merged['cluster'].ne('-1'),'cluster'].nunique()),offset=offset)
    publish(checkpoint,dict(signature=signature,artifacts=artifacts,summary=summary))
    print(f'MERGED {split}: {summary}',flush=True);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--models',type=Path,default=MODELS)
    parser.add_argument('--training-root',type=Path)
    parser.add_argument('--test-years',type=int,nargs='+',default=[2018])
    parser.add_argument('--jobs',type=int,default=32);parser.add_argument('--nlp-workers',type=int,default=4);parser.add_argument('--encode-batch',type=int,default=128)
    parser.add_argument('--gui-min',type=int,default=25);parser.add_argument('--bg-min',type=int,default=50);parser.add_argument('--components',type=int,default=100)
    parser.add_argument('--retries',type=int,default=3)
    args=parser.parse_args()
    if min(args.jobs,args.nlp_workers,args.encode_batch,args.components)<1 or min(args.gui_min,args.bg_min)<2:parser.error('Invalid resource or clustering parameters')
    root=args.root.resolve();args.root=root
    control=root/'cluster_pipeline';control.mkdir(exist_ok=True)
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    splits=['test'] if args.training_root else (['train','test'] if (root/'test/final_training_data_with_desc.csv').exists() else ['train'])
    for split in splits:
        if not (root/split/'final_training_data_with_desc.csv').is_file():raise FileNotFoundError(f'Missing {split} descriptions')
    for file in ['all-MiniLM-L6-v2/model.safetensors','word2vec-google-news-300/GoogleNews-vectors-negative300.bin']:
        if not (args.models/file).is_file():raise FileNotFoundError(f'Missing local model: {file}')
    started=time.time();state=dict(state='running',pid=os.getpid(),started_at=started,jobs_per_branch=args.jobs,nlp_workers=args.nlp_workers,branches={},merges={},updated_at=time.time())
    status=control/'status.json'
    def snapshot():
        state['updated_at']=time.time();publish(status,state)
    def launch(split,branch):
        folder=branch_folder(root,split,branch);folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(BASE/'scenario_worker.py'),'--root',str(root),'--models',str(args.models),'--split',split,'--branch',branch,
            '--jobs',str(args.jobs),'--nlp-workers',str(args.nlp_workers),'--encode-batch',str(args.encode_batch),'--gui-min',str(args.gui_min),'--bg-min',str(args.bg_min),'--components',str(args.components)]
        if args.training_root:command+=['--training-root',str(args.training_root),'--test-years',*map(str,args.test_years)]
        env=dict(os.environ,OMP_NUM_THREADS=str(args.jobs),OPENBLAS_NUM_THREADS=str(args.jobs),MKL_NUM_THREADS=str(args.jobs),NUMEXPR_NUM_THREADS=str(args.jobs),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        for attempt in range(1,args.retries+1):
            with (folder/'run.log').open('ab') as log:
                log.write(f'\nATTEMPT {attempt}\n'.encode());log.flush()
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            if result.returncode==0:return
            if result.returncode==2:raise RuntimeError(f'{split}/{branch}: input/configuration error; inspect {folder}/run.log')
            if attempt<args.retries:time.sleep(min(30,5*attempt))
        raise RuntimeError(f'{split}/{branch}: failed after {args.retries} attempts; checkpoints preserved')
    futures_map={};submitted=set()
    try:
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            for branch in ['gui','bg']:
                first=splits[0]
                futures_map[pool.submit(launch,first,branch)]=(first,branch);submitted.add((first,branch))
            while futures_map:
                ready,_=futures.wait(futures_map,timeout=3,return_when=futures.FIRST_COMPLETED)
                for future in ready:
                    split,branch=futures_map.pop(future);future.result();state['branches'][f'{split}/{branch}']='finished'
                    if split=='train' and 'test' in splits:
                        futures_map[pool.submit(launch,'test',branch)]=('test',branch);submitted.add(('test',branch))
                snapshot()
            for split in splits:
                state['stage']=f'merge_{split}';snapshot();state['merges'][split]=merge_results(root,split,training_root=args.training_root);snapshot()
        state['state']='finished';state['stage']='finished';snapshot()
        print('SCENARIOS FINISHED: training models fitted on 2014-2017 benign only; 2018 assigned with frozen PCA and centroids.',flush=True)
    except Exception as error:
        state.update(state='failed',error=str(error));snapshot();raise


if __name__=='__main__':
    sys.path.insert(0,str(BASE.parent))
    from output_layout import dispatch,TRAIN
    dispatch(__file__,'--root',TRAIN,year_flag='--test-years',training=True)
    main()
