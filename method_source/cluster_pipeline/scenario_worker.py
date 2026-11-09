#!/usr/bin/env python3
"""Checkpointed scenario features, training-only fitting and centroid assignment."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import threading
import traceback
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scenario_common import ROOT, MODELS, gui_mask, gui_raw_text, gui_doc_text


def atomic_json(path,data):
    pending=path.with_name(path.name+'.tmp');pending.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');pending.replace(path)


def atomic_npy(path,array):
    pending=path.with_name(path.name+'.tmp')
    with pending.open('wb') as f:np.save(f,array)
    pending.replace(path)


def atomic_pickle(path,frame):
    pending=path.with_name(path.name+'.tmp');frame.to_pickle(pending);pending.replace(path)


def atomic_csv(path,frame,sep=';'):
    pending=path.with_name(path.name+'.tmp');frame.to_csv(pending,sep=sep,index=False);pending.replace(path)


class Worker:
    def __init__(self,args):
        self.a=args;self.train=args.split=='train';self.branch=args.branch
        self.folder=args.root/args.split/'cluster_pipeline'/('gui_centroids' if self.branch=='gui' else 'bg_centroids') if self.train else args.root/'test/cluster_pipeline'/self.branch
        self.folder.mkdir(parents=True,exist_ok=True);self.cache=self.folder/'checkpoints';self.cache.mkdir(exist_ok=True)
        self.source=args.root/args.split/'final_training_data_with_desc.csv'
        stat=self.source.stat()
        model=args.models/('all-MiniLM-L6-v2/model.safetensors' if self.branch=='gui' else 'word2vec-google-news-300/GoogleNews-vectors-negative300.bin')
        ms=model.stat()
        signature=dict(input=str(self.source),size=stat.st_size,mtime=stat.st_mtime_ns,model=str(model),model_size=ms.st_size,model_mtime=ms.st_mtime_ns,
                       gui_min=args.gui_min,bg_min=args.bg_min,components=args.components,revision=1)
        if not self.train:
            trained=(args.training_root or args.root)/'train/cluster_pipeline'/('gui_centroids' if self.branch=='gui' else 'bg_centroids')
            signature['training']={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in trained.glob('*') if p.is_file() and p.suffix in ('.npy','.pkl','.json') and p.name!='status.json'}
        if not self.train and args.test_years!=[2018]:signature['test_years']=sorted(set(args.test_years))
        self.signature=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        self.manifest=self.cache/'manifest.json'
        old=json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
        if old and old.get('signature')!=self.signature:
            raise ValueError(f'Checkpoint inputs changed: {self.folder}. Use a new output root or explicitly clear this branch checkpoint.')
        self.saved=old or dict(signature=self.signature,stages={})
        self.started=time.time();self.stage_started=self.started;self.stage='starting';self.previous=None
        self.status_lock=threading.Lock();self.last_status=None;self.halt=threading.Event()

    def update(self,done=0,total=0,detail='',state='running',**extra):
        data=dict(state=state,pid=os.getpid(),split=self.a.split,branch=self.branch,stage=self.stage,done=done,total=total,
                  started_at=self.started,stage_started=self.stage_started,updated_at=time.time(),detail=detail,signature=self.signature)
        data.update(extra)
        with self.status_lock:
            self.last_status=data
            atomic_json(self.folder/'status.json',data)

    def heartbeat(self):
        while not self.halt.wait(5):
            with self.status_lock:
                if self.last_status and self.last_status['state']=='running':
                    self.last_status['updated_at']=time.time()
                    atomic_json(self.folder/'status.json',self.last_status)

    def stage_run(self,name,artifacts,function):
        self.stage=name;self.stage_started=time.time();self.update(detail='检查断点')
        if self.saved['stages'].get(name) and all((self.folder/p).is_file() for p in artifacts):
            print(f'RESUME {self.a.split}/{self.branch}: {name}',flush=True);return
        print(f'STAGE {self.a.split}/{self.branch}: {name}',flush=True)
        function()
        for path in artifacts:
            if not (self.folder/path).is_file():raise RuntimeError(f'Stage did not publish {path}')
        self.saved['stages'][name]=dict(seconds=time.time()-self.stage_started,finished_at=time.time())
        atomic_json(self.manifest,self.saved)
        self.update(1,1,'阶段完成')

    def prepare(self):
        frame=pd.read_csv(self.source,dtype=str,keep_default_na=False)
        required={'id','apk_name','generated_description','label','context','rawtext','entry_point','entry_method','entry_point_actions','dataset_year','dataset_class'}
        if required-set(frame):raise ValueError(f'Missing input columns: {required-set(frame)}')
        if not frame['id'].is_unique:raise ValueError('Duplicate IDs in input')
        if frame['generated_description'].str.strip().eq('').any():raise ValueError('Empty behavior descriptions in input')
        if self.train:
            if not frame['dataset_year'].isin(['2014','2015','2016','2017']).all() or not frame['dataset_class'].eq('benign').all():raise ValueError('Training split contains test or malware data')
        elif not frame['dataset_year'].isin([str(y) for y in self.a.test_years]).all() or not frame['dataset_class'].isin(['benign','malware']).all():raise ValueError('Test split contains unexpected years or classes')
        mask=gui_mask(frame);frame=frame[mask if self.branch=='gui' else ~mask].copy().reset_index(drop=True)
        atomic_pickle(self.cache/'input.pkl',frame)
        self.update(len(frame),len(frame),'GUI/BG 分流完成，保留完整记录与描述')

    def preprocess(self):
        frame=pd.read_pickle(self.cache/'input.pkl')
        db=sqlite3.connect(self.cache/'text_cache.sqlite',timeout=60);db.execute('PRAGMA cache_size=-65536')
        db.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY,text TEXT,vector BLOB)');db.commit()
        existing={k:(t,v) for k,t,v in db.execute('SELECT key,text,vector FROM features')}
        if self.branch=='gui':
            raw=[gui_raw_text(*values) for values in frame[['label','context','rawtext']].itertuples(index=False,name=None)]
        else:
            raw=[json.dumps(values,ensure_ascii=False) for values in frame[['entry_point','entry_method','entry_point_actions']].itertuples(index=False,name=None)]
        unique=list(dict.fromkeys(raw));keys={text:hashlib.sha256(text.encode()).hexdigest() for text in unique}
        missing=[text for text in unique if keys[text] not in existing];done=len(unique)-len(missing)
        self.update(done,len(unique),'加载本地 NLP 模型')
        if self.branch=='gui':
            import spacy
            nlp=spacy.load('en_core_web_sm',disable=['parser','ner'])
            nlp.max_length=max(nlp.max_length,max(map(len,missing),default=0)+1)
            for start in range(0,len(missing),512):
                batch=missing[start:start+512]
                documents=nlp.pipe(batch,batch_size=64,n_process=self.a.nlp_workers)
                records=[]
                for text,doc in zip(batch,documents):records.append((keys[text],gui_doc_text(doc),None))
                db.executemany('INSERT OR REPLACE INTO features VALUES (?,?,?)',records);db.commit()
                existing.update((k,(t,v)) for k,t,v in records);done+=len(records);self.update(done,len(unique),'多进程词形还原；重复文本复用')
        else:
            from entry_point_filter import EntryPointFilter
            ep=EntryPointFilter(model_path=str(self.a.models/'word2vec-google-news-300/GoogleNews-vectors-negative300.bin'))
            ep.clean_entry_point=lru_cache(maxsize=200000)(ep.clean_entry_point)
            ep.clean_and_tokenize_action=lru_cache(maxsize=100000)(ep.clean_and_tokenize_action)
            ep.make_feature_vec=lru_cache(maxsize=100000)(ep.make_feature_vec)
            for start in range(0,len(missing),512):
                records=[]
                for text in missing[start:start+512]:
                    point,method,actions=json.loads(text)
                    cleaned=' '.join(filter(None,[ep.clean_entry_point(ep.keep_after_last_dot(point)),
                        ep.clean_entry_point(ep.extract_method_name(method)),ep.clean_and_tokenize_action(actions).strip()]))
                    records.append((keys[text],cleaned,ep.make_feature_vec(cleaned).astype('float32').tobytes()))
                db.executemany('INSERT OR REPLACE INTO features VALUES (?,?,?)',records);db.commit()
                existing.update((k,(t,v)) for k,t,v in records);done+=len(records);self.update(done,len(unique),'入口点清洗及 Word2Vec；重复入口复用')
        cleaned=[existing[keys[text]][0] for text in raw]
        frame['semantic_text' if self.branch=='gui' else 'entry_point_cleaned']=cleaned
        atomic_pickle(self.cache/'prepared.pkl',frame)
        if self.branch=='bg':
            vectors=np.asarray([np.frombuffer(existing[keys[text]][1],dtype='float32') for text in raw],dtype='float32').reshape(len(raw),300)
            atomic_npy(self.cache/'embeddings.npy',vectors)
        db.close()

    def embed(self):
        if self.branch=='bg':return
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(self.a.jobs)
        frame=pd.read_pickle(self.cache/'prepared.pkl');texts=frame['semantic_text'].tolist()
        unique=list(dict.fromkeys(text for text in texts if text));lookup={text:i for i,text in enumerate(unique)}
        model=SentenceTransformer(str(self.a.models/'all-MiniLM-L6-v2'),local_files_only=True,device='cpu')
        dimension=model.get_sentence_embedding_dimension()
        if not unique:
            atomic_npy(self.cache/'embeddings.npy',np.zeros((len(texts),dimension),dtype='float32'));return
        pending=self.cache/'unique_embeddings.partial.npy';progress=self.cache/'embedding_progress.json'
        start=0
        if pending.exists() and progress.exists():
            info=json.loads(progress.read_text());start=info['done'];matrix=np.lib.format.open_memmap(pending,mode='r+')
            if matrix.shape!=(len(unique),dimension):raise ValueError('Embedding checkpoint shape mismatch')
        else:matrix=np.lib.format.open_memmap(pending,mode='w+',dtype='float32',shape=(len(unique),dimension))
        for index in range(start,len(unique),self.a.encode_batch):
            batch=unique[index:index+self.a.encode_batch]
            matrix[index:index+len(batch)]=model.encode(batch,batch_size=self.a.encode_batch,show_progress_bar=False,convert_to_numpy=True)
            matrix.flush();atomic_json(progress,dict(done=index+len(batch)))
            self.update(index+len(batch),len(unique),'本地 MiniLM 批量嵌入；已保存向量断点')
        complete=np.zeros((len(texts),dimension),dtype='float32')
        for i,text in enumerate(texts):
            if text:complete[i]=matrix[lookup[text]]
        atomic_npy(self.cache/'embeddings.npy',complete)
        del matrix;pending.unlink(missing_ok=True);progress.unlink(missing_ok=True)

    def fit_or_assign(self):
        frame=pd.read_pickle(self.cache/'prepared.pkl');X=np.load(self.cache/'embeddings.npy')
        if len(X)!=len(frame):raise ValueError('Feature/ID alignment mismatch')
        if self.train:
            if self.branch=='bg':
                self.update(detail='仅训练集拟合 L2 → PCA')
                components=min(self.a.components,X.shape[0],X.shape[1])
                pca=PCA(n_components=components,random_state=42)
                X=pca.fit_transform(normalize(X,norm='l2')).astype('float32')
                pending=self.folder/'bg_pca_model.pkl.tmp';joblib.dump(pca,pending);pending.replace(self.folder/'bg_pca_model.pkl')
                valid=np.ones(len(X),dtype=bool)
            else:valid=frame['semantic_text'].str.strip().ne('').to_numpy()
            labels=np.full(len(X),-1,dtype=int);minimum=self.a.gui_min if self.branch=='gui' else self.a.bg_min
            self.update(0,int(valid.sum()),f'HDBSCAN 全量训练，{self.a.jobs} 个 CPU 工作线程；此阶段无逐行进度')
            if valid.sum()>=minimum:
                labels[valid]=HDBSCAN(min_cluster_size=minimum,metric='euclidean',n_jobs=self.a.jobs).fit_predict(X[valid])
            ids=sorted(int(x) for x in np.unique(labels) if x>=0)
            centroids=[];names={};summary=[]
            for index,cid in enumerate(ids):
                positions=np.flatnonzero(labels==cid);vectors=X[positions];centroid=vectors.mean(axis=0)
                best=positions[int(np.argmax(cosine_similarity([centroid],vectors)[0]))]
                field='label' if self.branch=='gui' else 'entry_point_cleaned';name=frame.iloc[best][field]
                if not name and self.branch=='gui':name=frame.iloc[best]['semantic_text']
                centroids.append(centroid);names[str(cid)]=name;summary.append(dict(cluster_id=cid,scenario_label=name,count=len(positions)))
                self.update(index+1,len(ids),'计算训练簇质心与场景标签')
            centroid_matrix=np.asarray(centroids,dtype='float32').reshape(len(ids),X.shape[1])
            atomic_npy(self.folder/f'{self.branch}_centroids.npy',centroid_matrix)
            atomic_json(self.folder/f'{self.branch}_cluster_ids.json',[str(cid) for cid in ids])
            atomic_json(self.folder/f'{self.branch}_scenario_labels.json',names)
            summary_frame=pd.DataFrame(summary,columns=['cluster_id','scenario_label','count']).sort_values('count',ascending=False)
            atomic_csv(self.folder/f'{self.branch}_cluster_summary.csv',summary_frame)
            print(f'FIT {self.branch}: rows={len(frame)}, clusters={len(ids)}, noise={int((labels<0).sum())}',flush=True)
        else:
            trained=(self.a.training_root or self.a.root)/'train/cluster_pipeline'/f'{self.branch}_centroids'
            centroids=np.load(trained/f'{self.branch}_centroids.npy')
            ids=np.asarray(json.loads((trained/f'{self.branch}_cluster_ids.json').read_text()),dtype=int)
            if self.branch=='bg':
                pca=joblib.load(trained/'bg_pca_model.pkl')
                X=pca.transform(normalize(X,norm='l2')).astype('float32')
                valid=np.ones(len(X),dtype=bool)
            else:valid=frame['semantic_text'].str.strip().ne('').to_numpy()
            labels=np.full(len(X),-1,dtype=int)
            positions=np.flatnonzero(valid)
            if len(ids):
                for start in range(0,len(positions),4096):
                    selected=positions[start:start+4096]
                    labels[selected]=ids[np.argmax(cosine_similarity(X[selected],centroids),axis=1)]
                    self.update(start+len(selected),len(positions),'仅用训练质心分配测试场景，不拟合任何模型')
        atomic_npy(self.cache/'labels.npy',labels)

    def export(self):
        frame=pd.read_pickle(self.cache/'prepared.pkl');frame['cluster']=np.load(self.cache/'labels.npy')
        filename=f'{self.branch}_clustered_data.csv' if self.train else f'{self.branch}_assigned_data.csv'
        atomic_csv(self.folder/filename,frame)
        self.update(len(frame),len(frame),'分支结果已发布')

    def run(self):
        threading.Thread(target=self.heartbeat,daemon=True).start()
        self.stage_run('prepare',['checkpoints/input.pkl'],self.prepare)
        artifacts=['checkpoints/prepared.pkl']+(['checkpoints/embeddings.npy'] if self.branch=='bg' else [])
        self.stage_run('preprocess',artifacts,self.preprocess)
        self.stage_run('embedding',['checkpoints/embeddings.npy'],self.embed)
        artifacts=['checkpoints/labels.npy']
        if self.train:
            artifacts += [f'{self.branch}_centroids.npy',f'{self.branch}_cluster_ids.json',f'{self.branch}_scenario_labels.json',f'{self.branch}_cluster_summary.csv']
            if self.branch=='bg':artifacts+=['bg_pca_model.pkl']
        self.stage_run('fit' if self.train else 'assign',artifacts,self.fit_or_assign)
        self.stage_run('export',[f'{self.branch}_clustered_data.csv' if self.train else f'{self.branch}_assigned_data.csv'],self.export)
        self.stage='finished';self.update(1,1,'分支完成',state='finished');self.halt.set()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--models',type=Path,default=MODELS)
    parser.add_argument('--test-years',type=int,nargs='+',default=[2018])
    parser.add_argument('--training-root',type=Path,help='Read-only frozen training artifacts for test assignment')
    parser.add_argument('--split',choices=['train','test'],required=True);parser.add_argument('--branch',choices=['gui','bg'],required=True)
    parser.add_argument('--jobs',type=int,default=32);parser.add_argument('--nlp-workers',type=int,default=4);parser.add_argument('--encode-batch',type=int,default=128)
    parser.add_argument('--gui-min',type=int,default=25);parser.add_argument('--bg-min',type=int,default=50);parser.add_argument('--components',type=int,default=100)
    args=parser.parse_args();args.root=args.root.resolve();args.training_root=args.training_root.resolve() if args.training_root else None
    if args.training_root and args.split!='test':parser.error('--training-root is test-inference only')
    worker=Worker(args)
    try:worker.run()
    except Exception as error:
        worker.update(detail=str(error),state='failed',retryable=not isinstance(error,(ValueError,FileNotFoundError,AssertionError)))
        traceback.print_exc();sys.exit(2 if isinstance(error,(ValueError,FileNotFoundError,AssertionError)) else 1)


if __name__=='__main__':main()
