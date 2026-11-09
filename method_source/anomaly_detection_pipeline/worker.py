#!/usr/bin/env python3
"""Resumable behavior embeddings, training references, scores and OCSVM stages."""
import argparse
from contextlib import ExitStack
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from similarity_detect import topk_cosine,aggregate_anomaly_scores
from modeling import train_detector,evaluate_detector

ROOT=Path('/home/lb/data/Dataset/ForeDroid_output/2014-2017')
MODEL=Path('/home/lb/models/foredroid/all-MiniLM-L6-v2')


def publish(path,data):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def save_npy(path,values):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.save(f,values)
    temp.replace(path)


def save_csv(path,frame):
    temp=path.with_name(path.name+'.tmp');frame.to_csv(temp,sep=';',index=False);temp.replace(path)


class Worker:
    def __init__(self,args):
        self.a=args;self.root=args.root;self.base=args.root/'anomaly_detection_pipeline';self.base.mkdir(exist_ok=True)
        self.name=args.stage+('_'+args.split if args.stage in ('embedding','scores') else '')
        self.cache=self.base/'checkpoints'/self.name;self.cache.mkdir(parents=True,exist_ok=True)
        self.manifest=self.cache/'manifest.json';self.status=self.base/f'{self.name}_status.json'
        sources=[(args.training_root or args.root)/'train/cluster_pipeline/output/all_apks_with_clusters.csv',args.root/'test/cluster_pipeline/output/all_apks_with_clusters.csv']
        sources=[p for p in sources if p.is_file()]
        model=args.model/'model.safetensors'
        profile=dict(inputs=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sources],model=(str(model),model.stat().st_size,model.stat().st_mtime_ns),neighbors=args.neighbors,selection=args.selection,top_percent=args.top_percent,revision=1,code=hashlib.sha256(b''.join((Path(__file__).parent/name).read_bytes() for name in ['worker.py','similarity_detect.py','modeling.py'])).hexdigest())
        self.signature=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()
        self.saved=json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
        if self.saved and self.saved.get('signature')!=self.signature:raise ValueError('Input/configuration changed; use a fresh anomaly output or explicitly clear its checkpoints')
        if not self.saved:publish(self.manifest,dict(signature=self.signature,complete=False))
        self.started=time.time();self.lock=threading.Lock();self.last=None;self.stop=threading.Event()

    def update(self,done=0,total=0,detail='',state='running',**extra):
        record=dict(stage=self.name,state=state,pid=os.getpid(),done=done,total=total,detail=detail,started_at=self.started,updated_at=time.time(),signature=self.signature)
        record.update(extra)
        with self.lock:self.last=record;publish(self.status,record)

    def heartbeat(self):
        while not self.stop.wait(5):
            with self.lock:
                if self.last and self.last['state']=='running':self.last['updated_at']=time.time();publish(self.status,self.last)

    def source(self,split):return ((self.a.training_root or self.root) if split=='train' else self.root)/split/'cluster_pipeline/output/all_apks_with_clusters.csv'

    def metadata(self,split):
        path=self.source(split)
        frame=pd.read_csv(path,sep=';',usecols=['id','apk_name','cluster','dataset_year','dataset_class','generated_description'],dtype=str,keep_default_na=False)
        if not frame['id'].is_unique:raise ValueError('Duplicate input record IDs')
        if not frame['generated_description'].str.strip().ne('').all():raise ValueError('Empty descriptions')
        if split=='train':
            if not frame['dataset_year'].isin(['2014','2015','2016','2017']).all() or not frame['dataset_class'].eq('benign').all():raise ValueError('Training provenance violation')
        elif not frame['dataset_year'].isin([str(y) for y in self.a.test_years]).all() or not frame['dataset_class'].isin(['benign','malware']).all():raise ValueError('Test provenance violation')
        frame['cluster']=pd.to_numeric(frame['cluster'],errors='raise').astype(int).astype(str)
        return frame

    def embedding(self):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(self.a.jobs)
        self.update(detail='读取场景数据及行为描述')
        meta_file=self.cache/'metadata.pkl'
        if meta_file.exists():frame=pd.read_pickle(meta_file)
        else:
            frame=self.metadata(self.a.split);temp=meta_file.with_suffix('.pkl.tmp');frame.to_pickle(temp);temp.replace(meta_file)
        texts=frame['generated_description'].tolist();eligible=frame['cluster'].ne('-1').to_numpy()
        unique=list(dict.fromkeys(text for text,valid in zip(texts,eligible) if valid));lookup={text:i for i,text in enumerate(unique)}
        model=SentenceTransformer(str(self.a.model),local_files_only=True,device='cpu');dimension=model.get_sentence_embedding_dimension()
        if not unique:
            save_npy(self.cache/'embeddings.npy',np.zeros((len(frame),dimension),dtype='float32'))
            publish(self.cache/'summary.json',dict(rows=len(frame),eligible_rows=0,noise_rows=len(frame),unique_descriptions=0));return
        partial=self.cache/'unique.partial.npy';marker=self.cache/'embedding_progress.json';start=0
        if partial.exists() and marker.exists():
            vectors=np.lib.format.open_memmap(partial,mode='r+');start=json.loads(marker.read_text())['done']
            if vectors.shape!=(len(unique),dimension):raise ValueError('Embedding checkpoint mismatch')
        else:vectors=np.lib.format.open_memmap(partial,mode='w+',dtype='float32',shape=(len(unique),dimension))
        for i in range(start,len(unique),self.a.batch):
            batch=unique[i:i+self.a.batch]
            encoded=model.encode(batch,batch_size=self.a.batch,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            if not np.isfinite(encoded).all() or (np.linalg.norm(encoded,axis=1)==0).any():raise ValueError('Invalid or zero behavior embedding')
            vectors[i:i+len(batch)]=encoded;vectors.flush();publish(marker,dict(done=i+len(batch)))
            self.update(i+len(batch),len(unique),'MiniLM 批量行为嵌入；重复描述复用，噪声不进入参考库')
        output=np.zeros((len(frame),dimension),dtype='float32')
        for index,(text,valid) in enumerate(zip(texts,eligible)):
            if valid:output[index]=vectors[lookup[text]]
        save_npy(self.cache/'embeddings.npy',output);del vectors;partial.unlink(missing_ok=True);marker.unlink(missing_ok=True)
        summary=dict(rows=len(frame),eligible_rows=int(eligible.sum()),noise_rows=int((~eligible).sum()),unique_descriptions=len(unique))
        publish(self.cache/'summary.json',summary)

    def references(self):
        trained=self.base/'checkpoints/embedding_train';frame=pd.read_pickle(trained/'metadata.pkl');vectors=np.load(trained/'embeddings.npy',mmap_mode='r')
        folder=self.base/'precomputed_vectors';folder.mkdir(exist_ok=True)
        groups={cid:idx for cid,idx in frame.groupby('cluster',sort=False).indices.items() if cid!='-1'}
        progress=self.cache/'clusters.json';done=set(json.loads(progress.read_text())) if progress.exists() else set()
        for cid,idx in groups.items():
            target=folder/f'cluster_{cid}.npy';apks=folder/f'cluster_{cid}_apk.json'
            if cid in done and target.exists() and apks.exists():continue
            save_npy(target,np.asarray(vectors[idx]));publish(apks,frame.iloc[idx]['apk_name'].tolist())
            done.add(cid);publish(progress,sorted(done,key=int));self.update(len(done),len(groups),'保存仅训练良性样本的场景参考库，排除 cluster=-1')
        publish(folder/'manifest.json',dict(signature=self.signature,clusters=sorted(groups,key=int),cluster_sizes={cid:len(idx) for cid,idx in groups.items()},rows=sum(map(len,groups.values()))))

    def scores(self):
        branch=self.base/'checkpoints'/f'embedding_{self.a.split}';meta=pd.read_pickle(branch/'metadata.pkl');vectors=np.load(branch/'embeddings.npy',mmap_mode='r')
        groups={cid:idx for cid,idx in meta.groupby('cluster',sort=False).indices.items() if cid!='-1'}
        partial=self.cache/'scores.partial.npy';marker=self.cache/'score_progress.json'
        if partial.exists() and marker.exists():
            values=np.lib.format.open_memmap(partial,mode='r+');done=set(json.loads(marker.read_text()))
            if values.shape!=(len(meta),4):raise ValueError('Score checkpoint shape mismatch')
        else:
            values=np.lib.format.open_memmap(partial,mode='w+',dtype='float64',shape=(len(meta),4));values[:]=np.nan;values[:,2:]=0;values.flush();done=set()
        references=((self.a.training_root/'anomaly_detection_pipeline') if self.a.training_root else self.base)/'precomputed_vectors'
        sizes=json.loads((references/'manifest.json').read_text())['cluster_sizes']
        work={cid:len(idx)*sizes[cid] for cid,idx in groups.items()}
        def score_cluster(cid,idx):
            ref=np.load(references/f'cluster_{cid}.npy',mmap_mode='r');apks=json.loads((references/f'cluster_{cid}_apk.json').read_text())
            result=np.empty((len(idx),4),dtype='float64')
            lookup={apk:i for i,apk in enumerate(dict.fromkeys(apks))}
            ref_codes=np.asarray([lookup[apk] for apk in apks],dtype='int32')
            query_apks=np.asarray([lookup.get(apk,-1) for apk in meta.iloc[idx]['apk_name']],dtype='int32')
            if not np.isfinite(ref).all() or (np.linalg.norm(ref,axis=1)==0).any():raise ValueError('Invalid reference embedding')
            # Bound every dense similarity block to about 32 MiB per task.
            block=max(1,min(self.a.score_batch,8_000_000//max(1,len(ref))))
            for i in range(0,len(idx),block):
                rows=idx[i:i+block];means,counts,used=topk_cosine(vectors[rows],ref,query_apks[i:i+block],ref_codes,self.a.neighbors,normalized=True)
                result[i:i+len(rows)]=np.column_stack([means,1-means,counts,used])
            return result
        self.update(len(done),len(groups),'按场景并行 Top-K，训练与测试均排除同 APK 的所有参考记录',work_done=sum(work[c] for c in done),work_total=sum(work.values()))
        with threadpool_limits(limits=self.a.score_threads),futures.ThreadPoolExecutor(max_workers=self.a.workers) as pool:
            pending={pool.submit(score_cluster,cid,idx):(cid,idx) for cid,idx in groups.items() if cid not in done}
            for future in futures.as_completed(pending):
                cid,idx=pending[future];values[idx]=future.result();values.flush();done.add(cid);publish(marker,sorted(done,key=int))
                self.update(len(done),len(groups),'已提交场景分数断点；无跨 APK 参考的记录标为不可评分',work_done=sum(work[c] for c in done),work_total=sum(work.values()))
        full=pd.read_csv(self.source(self.a.split),sep=';',dtype=str,keep_default_na=False)
        if full['id'].tolist()!=meta['id'].tolist():raise ValueError('Score output record order changed')
        full['avg_topk_similarity']=values[:,0];full['anomaly_score']=values[:,1];full['reference_count']=values[:,2];full['neighbors_used']=values[:,3]
        full['score_status']=np.where(full['cluster'].eq('-1'),'unassigned_scene',np.where(np.isfinite(values[:,1]),'scored','no_cross_apk_reference'))
        out=self.root/self.a.split/'anomaly_detection_pipeline';out.mkdir(exist_ok=True)
        save_csv(out/f'{self.a.split}_top{self.a.neighbors}_anomaly_scores.csv',full)
        save_csv(out/f'{self.a.split}_top{self.a.neighbors}_agg_anomaly_scores.csv',aggregate_anomaly_scores(full))
        summary=dict(rows=len(full),scored_rows=int(full['score_status'].eq('scored').sum()),unassigned_rows=int(full['score_status'].eq('unassigned_scene').sum()),no_reference_rows=int(full['score_status'].eq('no_cross_apk_reference').sum()))
        publish(out/'score_summary.json',summary)
        del values;partial.unlink(missing_ok=True);marker.unlink(missing_ok=True)

    def train(self):
        path=self.root/'train/anomaly_detection_pipeline'/f'train_top{self.a.neighbors}_anomaly_scores.csv'
        frame=pd.read_csv(path,sep=';',usecols=['apk_name','cluster','anomaly_score','dataset_class'],dtype={'apk_name':str,'cluster':str,'dataset_class':str})
        folder=self.base/'models';folder.mkdir(exist_ok=True)
        summary=train_detector(frame,folder,self.update,workers=self.a.grid_workers,selection=self.a.selection,top_percent=self.a.top_percent,neighbors=self.a.neighbors)
        print('TRAIN_SUMMARY',json.dumps(summary),flush=True)

    def evaluate(self):
        path=self.root/'test/anomaly_detection_pipeline'/f'test_top{self.a.neighbors}_anomaly_scores.csv'
        frame=pd.read_csv(path,sep=';',usecols=['apk_name','cluster','anomaly_score','dataset_class'],dtype={'apk_name':str,'cluster':str,'dataset_class':str})
        self.update(detail='使用固定场景维度、训练 scaler 和 OCSVM 预测；测试标签只用于评估')
        summary=evaluate_detector(frame,self.base/'models/ocsvm_model.pkl',self.base/'results')
        print('TEST_METRICS',json.dumps(summary),flush=True)

    def artifacts(self):
        split=self.a.split
        return dict(embedding=[self.cache/'metadata.pkl',self.cache/'embeddings.npy',self.cache/'summary.json'],
            references=[self.base/'precomputed_vectors/manifest.json'],scores=[self.root/split/'anomaly_detection_pipeline'/f'{split}_top{self.a.neighbors}_anomaly_scores.csv',self.root/split/'anomaly_detection_pipeline/score_summary.json',self.root/split/'anomaly_detection_pipeline'/f'{split}_top{self.a.neighbors}_agg_anomaly_scores.csv'],
            train=[self.base/'models/ocsvm_model.pkl',self.base/'models/training_summary.json',self.base/'models/training_apk_status.csv'],evaluate=[self.base/'results/predictions.csv',self.base/'results/metrics.json'])[self.a.stage]

    def run(self):
        threading.Thread(target=self.heartbeat,daemon=True).start();self.update(detail='检查断点')
        if self.saved.get('complete') and all(p.exists() for p in self.artifacts()):
            self.update(1,1,'复用完整阶段断点',state='finished');self.stop.set();return
        getattr(self,self.a.stage)()
        if not all(p.exists() for p in self.artifacts()):raise RuntimeError('Stage outputs incomplete')
        publish(self.manifest,dict(signature=self.signature,complete=True,finished_at=time.time(),seconds=time.time()-self.started))
        self.update(1,1,'阶段完成',state='finished');self.stop.set()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--training-root',type=Path,help='Read-only frozen training references for test inference')
    parser.add_argument('--test-years',type=int,nargs='+',default=[2018])
    parser.add_argument('--stage',choices=['embedding','references','scores','train','evaluate'],required=True);parser.add_argument('--split',choices=['train','test'],default='train')
    parser.add_argument('--neighbors',type=int,default=50);parser.add_argument('--jobs',type=int,default=32);parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--score-threads',type=int,default=4);parser.add_argument('--score-batch',type=int,default=256);parser.add_argument('--grid-workers',type=int,default=4)
    parser.add_argument('--selection',choices=['all','top10','top80percent'],default='top80percent');parser.add_argument('--top-percent',type=float,default=.8)
    args=parser.parse_args();args.root=args.root.resolve();args.training_root=args.training_root.resolve() if args.training_root else None;worker=None
    if args.training_root and (args.stage not in ('embedding','scores') or args.split!='test'):parser.error('External training root permits test embedding/scores only')
    try:
        worker=Worker(args);worker.run()
    except Exception as error:
        if worker:worker.update(detail=str(error),state='failed')
        traceback.print_exc();sys.exit(2 if isinstance(error,(ValueError,FileNotFoundError,AssertionError)) else 1)


if __name__=='__main__':main()
