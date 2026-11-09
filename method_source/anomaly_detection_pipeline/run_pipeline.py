#!/usr/bin/env python3
"""Offline, resumable behavior embedding -> reference scores -> OCSVM evaluation."""
import argparse
import concurrent.futures as futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
ROOT=Path('/home/lb/data/Dataset/ForeDroid_output/2014-2017')
MODEL=Path('/home/lb/models/foredroid/all-MiniLM-L6-v2')
DEPENDENCIES={'embedding_train':set(),'embedding_test':set(),'references':{'embedding_train'},
    'scores_train':{'embedding_train','references'},'scores_test':{'embedding_test','references'},
    'train':{'scores_train'},'evaluate':{'train','scores_test'}}


def publish(path,data):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def task_set(only):
    targets={'all':['evaluate'],'embedding':['embedding_train','embedding_test'],'references':['references'],'scores':['scores_train','scores_test'],'train':['train'],'evaluate':['evaluate']}[only]
    needed=set(targets)
    while True:
        more=set().union(*(DEPENDENCIES[name] for name in needed))-needed
        if not more:return needed
        needed.update(more)


def main(default_only='all'):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--only',choices=['all','embedding','references','scores','train','evaluate'],default=default_only)
    parser.add_argument('--jobs',type=int,default=32);parser.add_argument('--batch',type=int,default=128);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--score-threads',type=int,default=4);parser.add_argument('--score-batch',type=int,default=256);parser.add_argument('--grid-workers',type=int,default=4)
    parser.add_argument('--neighbors',type=int,default=50);parser.add_argument('--selection',choices=['all','top10','top80percent'],default='top80percent');parser.add_argument('--top-percent',type=float,default=.8);parser.add_argument('--retries',type=int,default=3)
    args=parser.parse_args()
    if min(args.jobs,args.batch,args.workers,args.score_threads,args.score_batch,args.grid_workers,args.neighbors,args.retries)<1 or not 0<args.top_percent<=1:parser.error('Invalid resource or feature parameters')
    root=args.root.resolve();args.root=root;base=root/'anomaly_detection_pipeline';base.mkdir(exist_ok=True)
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    training_only=not (root/'test').exists()
    for split in (['train'] if training_only else ['train','test']):
        if not (root/split/'cluster_pipeline/output/all_apks_with_clusters.csv').is_file():raise FileNotFoundError(f'Missing clustered {split} data')
    if not (args.model/'model.safetensors').is_file():raise FileNotFoundError('Missing local behavior embedding model')
    needed=task_set(args.only)
    if training_only:
        if args.only=='evaluate':parser.error('Use pipeline/run_inference.py for the separate testing root')
        needed-={'embedding_test','scores_test','evaluate'}
    completed=set();submitted=set();pending={}
    state=dict(state='running',pid=os.getpid(),started_at=time.time(),updated_at=time.time(),tasks={name:'pending' for name in needed},neighbors=args.neighbors,jobs=args.jobs,score_workers=args.workers,score_threads=args.score_threads,grid_workers=args.grid_workers,selection=args.selection)
    target=base/'status.json'
    def snapshot():state['updated_at']=time.time();publish(target,state)
    def launch(name):
        if name.startswith('embedding_') or name.startswith('scores_'):stage,split=name.split('_',1)
        else:stage,split=name,'train'
        command=[sys.executable,'-u',str(BASE/'worker.py'),'--root',str(root),'--model',str(args.model),'--stage',stage,'--split',split,
            '--jobs',str(args.jobs),'--batch',str(args.batch),'--workers',str(args.workers),'--score-threads',str(args.score_threads),'--score-batch',str(args.score_batch),'--grid-workers',str(args.grid_workers),'--neighbors',str(args.neighbors),'--selection',args.selection,'--top-percent',str(args.top_percent)]
        env=dict(os.environ,OMP_NUM_THREADS=str(args.jobs),OPENBLAS_NUM_THREADS=str(args.jobs),MKL_NUM_THREADS=str(args.jobs),NUMEXPR_NUM_THREADS=str(args.jobs),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
        for attempt in range(1,args.retries+1):
            with (base/f'{name}.log').open('ab') as log:
                log.write(f'\nATTEMPT {attempt}\n'.encode());log.flush();result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            if result.returncode==0:return
            if result.returncode==2:raise RuntimeError(f'{name}: data/configuration error; inspect {base}/{name}.log')
            if attempt<args.retries:time.sleep(min(30,5*attempt))
        raise RuntimeError(f'{name} failed after {args.retries} attempts; checkpoints retained')
    try:
        snapshot()
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            while completed!=needed:
                for name in sorted(needed-submitted):
                    if DEPENDENCIES[name]<=completed:
                        pending[pool.submit(launch,name)]=name;submitted.add(name);state['tasks'][name]='running'
                snapshot()
                if not pending:raise RuntimeError('Unresolved pipeline dependency')
                ready,_=futures.wait(pending,timeout=3,return_when=futures.FIRST_COMPLETED)
                for future in ready:
                    name=pending.pop(future);future.result();completed.add(name);state['tasks'][name]='finished';print('FINISHED',name,flush=True)
                snapshot()
        state.update(state='finished');snapshot()
        print('ANOMALY PIPELINE FINISHED; models/scalers/reference library fitted on training benign data only.',flush=True)
    except Exception as error:
        state.update(state='failed',error=str(error));snapshot();raise


if __name__=='__main__':
    sys.path.insert(0,str(BASE.parent))
    from output_layout import TEST
    argv=sys.argv[1:]
    if '--root' in argv and Path(argv[argv.index('--root')+1]).name=='2018-2022':
        raise SystemExit(subprocess.call([sys.executable,str(BASE.parent/'pipeline/run_inference.py'),'--root',argv[argv.index('--root')+1]]))
    main()
