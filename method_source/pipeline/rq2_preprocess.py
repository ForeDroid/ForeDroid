#!/usr/bin/env python3
"""Resumable preprocessing of the non-temporal GPMalware RQ2 corpus.

Year 0 is an internal compatibility sentinel, NOT a calendar year. Every fused
row also carries dataset_name=GPMalware. No RQ1 data/model is trained or modified.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
ROOT=Path('/home/lb/data/Dataset/ForeDroid_output/RQ2/GPMalware')
SOURCE=Path('/home/lb/data/Dataset/GPMalware')
TRAIN=Path('/home/lb/data/Dataset/ForeDroid_output/2014-2017')
STAGES=['extract','gui_merge','fusion','test_input','descriptions','gui_scenarios','bg_scenarios','merge_scenarios','embedding','scores','coverage_audit']

def publish(p,data):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');q.replace(p)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--dataset',type=Path,default=SOURCE);parser.add_argument('--training-root',type=Path,default=TRAIN)
 parser.add_argument('--phase',choices=['all','extract'],default='all')
 parser.add_argument('--gui-workers',type=int,default=64);parser.add_argument('--chain-workers',type=int,default=64);parser.add_argument('--ocr-workers',type=int,default=16);parser.add_argument('--llm-workers',type=int,default=32)
 args=parser.parse_args();root=args.root.resolve();source=args.dataset.resolve();train=args.training_root.resolve()
 if not (1<=args.gui_workers<=64 and 1<=args.chain_workers<=64 and 1<=args.ocr_workers<=64 and 1<=args.llm_workers<=64):parser.error('Invalid concurrency')
 root.mkdir(parents=True,exist_ok=True);extract=root/'extraction';extract.mkdir(exist_ok=True)
 if args.phase=='extract':
  from batch_extract import Runner
  csv.field_size_limit(2**31-1)
  runner_args=SimpleNamespace(output=str(extract),dataset=str(root/'input_view'),batches=[(0,'malware')],gui_workers=args.gui_workers,chain_workers=args.chain_workers,export_workers=args.ocr_workers,min_free_memory_gb=96,gui_timeout=1800,chain_timeout=1800,python=sys.executable,retry_failed=False,pilot=None)
  Runner(runner_args).run();return
 # The controller lock must differ from the child description generator's run.lock.
 lock=(root/'rq2_controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 apks=sorted(source.glob('*.apk'))
 if not apks:raise ValueError('No APKs in GPMalware source')
 inventory=[dict(apk_name=p.stem,path=str(p),size=p.stat().st_size,mtime=p.stat().st_mtime_ns) for p in apks]
 config=dict(dataset='GPMalware',calendar_year=None,internal_year_sentinel=0,inventory=inventory,training_root=str(train))
 config_path=root/'rq2_input_config.json'
 if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('RQ2 inputs changed; inspect checkpoints')
 publish(config_path,config)
 dataset_link=root/'input_view/0/malware';dataset_link.parent.mkdir(parents=True,exist_ok=True)
 if not dataset_link.exists():dataset_link.symlink_to(source,target_is_directory=True)
 if dataset_link.resolve()!=source:raise ValueError('Unexpected input symlink')
 for p in [train/'train/cluster_pipeline/gui_centroids/gui_centroids.npy',train/'train/cluster_pipeline/bg_centroids/bg_pca_model.pkl',train/'anomaly_detection_pipeline/precomputed_vectors/manifest.json']:
  if not p.is_file():raise FileNotFoundError(p)
 model=PROJECT/'anomaly_detection_pipeline/models/mean_all_model.pkl';model_sha=hashlib.sha256(model.read_bytes()).hexdigest()
 saved=json.loads((root/'rq2_checkpoints.json').read_text()) if (root/'rq2_checkpoints.json').exists() else {}
 started=time.time()
 def status(stage,state='running',**kw):
  publish(root/'rq2_status.json',dict(dataset='GPMalware',state=state,stage=stage,pid=os.getpid(),started_at=started,updated_at=time.time(),total_apks=len(apks),stages=STAGES,completed_stages=list(saved),**kw))
  print(stage,state,kw,flush=True)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='32',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='32',TOKENIZERS_PARALLELISM='false')
 def launch(cmd,name):
  logs=root/'logs';logs.mkdir(exist_ok=True)
  with (logs/(name+'.log')).open('ab') as out:
   out.write(('\n'+time.strftime('%Y-%m-%d %H:%M:%S')+'\n').encode());out.flush()
   p=subprocess.run(cmd,env=env,cwd=PROJECT,stdout=out,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError(f'{name} failed ({p.returncode}); see {logs/(name+".log")}')
 def stage(name,artifacts,fn):
  status(name)
  if saved.get(name) and all(p.exists() for p in artifacts):print('RESUME',name,flush=True);return
  fn()
  if not all(p.exists() for p in artifacts):raise RuntimeError('Missing stage output: '+name)
  saved[name]=dict(finished_at=time.time());publish(root/'rq2_checkpoints.json',saved)
 py=sys.executable;batch=extract/'0/malware';gui=batch/'gui';fused=batch/'pipeline/final_training_data.csv';test=root/'test'
 common=[str(PROJECT/'pipeline/rq2_preprocess.py'),'--root',str(root),'--dataset',str(source),'--training-root',str(train)]
 try:
  stage('extract',[extract/'status.json',extract/'state.sqlite'],lambda:launch([py,'-u',*common,'--phase','extract','--gui-workers',str(args.gui_workers),'--chain-workers',str(args.chain_workers),'--ocr-workers',str(args.ocr_workers)],'extract'))
  stage('gui_merge',[gui/'merge/train/merge_ui_api_with_relation.txt'],lambda:launch([py,str(PROJECT/'extract_Gui/merge_ui_api.py'),'--api-file',str(gui/'api-results/api_train.txt'),'--ui-file',str(gui/'outputBenignUi_train.txt'),'--output-dir',str(gui/'merge/train')],'gui_merge'))
  stage('fusion',[fused],lambda:launch([py,'-u',str(PROJECT/'pipeline/pipeline.py'),'--callchain-dir',str(batch/'callchain/output'),'--gui-file',str(gui/'merge/train/merge_ui_api_with_relation.txt'),'--output-dir',str(batch/'pipeline')],'fusion'))
  def test_input():
   csv.field_size_limit(sys.maxsize);test.mkdir(exist_ok=True)
   with fused.open(newline='') as inp,(test/'final_training_data.csv.tmp').open('w',newline='') as out:
    r=csv.DictReader(inp);w=csv.DictWriter(out,fieldnames=r.fieldnames+['dataset_year','dataset_class','dataset_name']);w.writeheader();rows=0
    for rows,row in enumerate(r,1):
     row.update(id=str(rows),dataset_year='0',dataset_class='malware',dataset_name='GPMalware');w.writerow(row)
   if not rows:raise ValueError('No fused behavior rows; cannot continue')
   (test/'final_training_data.csv.tmp').replace(test/'final_training_data.csv')
   publish(root/'fusion_summary.json',dict(rows=rows,raw_apks=len(apks),dataset='GPMalware',calendar_year=None))
  stage('test_input',[test/'final_training_data.csv'],test_input)
  stage('descriptions',[test/'final_training_data_with_desc.csv'],lambda:launch([py,'-u',str(PROJECT/'pipeline/generate_description.py'),'--output-root',str(root),'--splits','test','--model','MiniMax-M3','--workers',str(args.llm_workers),'--reuse-cache',str(train/'descriptions.sqlite')],'descriptions'))
  for branch in ('gui','bg'):
   stage(branch+'_scenarios',[test/f'cluster_pipeline/{branch}/{branch}_assigned_data.csv'],lambda b=branch:launch([py,'-u',str(PROJECT/'cluster_pipeline/scenario_worker.py'),'--root',str(root),'--training-root',str(train),'--test-years','0','--split','test','--branch',b,'--jobs','32','--nlp-workers','4'],b+'_scenarios'))
  def merge():
   sys.path.insert(0,str(PROJECT/'cluster_pipeline'))
   from run_scenarios import merge_results
   merge_results(root,'test',training_root=train)
  stage('merge_scenarios',[test/'cluster_pipeline/output/all_apks_with_clusters.csv'],merge)
  for action,artifact in [('embedding',root/'anomaly_detection_pipeline/checkpoints/embedding_test/embeddings.npy'),('scores',test/'anomaly_detection_pipeline/test_top50_anomaly_scores.csv')]:
   stage(action,[artifact],lambda a=action:launch([py,'-u',str(PROJECT/'anomaly_detection_pipeline/worker.py'),'--root',str(root),'--training-root',str(train),'--test-years','0','--split','test','--stage',a,'--selection','all','--jobs','32','--workers','8','--score-threads','4'],action))
  def audit():
   import pandas as pd
   f=pd.read_csv(test/'anomaly_detection_pipeline/test_top50_anomaly_scores.csv',sep=';',usecols=['apk_name','anomaly_score','cluster'])
   rows=[]
   for p in apks:
    q=f[f.apk_name.eq(p.stem)];rows.append(dict(apk_name=p.stem,behavior_rows=len(q),scored_rows=int(q.anomaly_score.notna().sum()),status='no_fused_behavior' if q.empty else ('scored' if q.anomaly_score.notna().any() else 'no_scorable_behavior')))
   pd.DataFrame(rows).to_csv(root/'preprocessing_coverage.csv',index=False)
  stage('coverage_audit',[root/'preprocessing_coverage.csv'],audit)
  assert hashlib.sha256(model.read_bytes()).hexdigest()==model_sha
  status('finished',state='finished',scores=str(test/'anomaly_detection_pipeline/test_top50_anomaly_scores.csv'))
 except Exception as error:
  status('failed',state='failed',error=str(error));raise
if __name__=='__main__':main()
