#!/usr/bin/env python3
"""Frozen future-year inference: wait for descriptions, assign, score, evaluate."""
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
TRAIN=Path('/home/lb/data/Dataset/ForeDroid_output/2014-2017')
ROOT=Path('/home/lb/data/Dataset/ForeDroid_output/2018-2022')
MODEL=PROJECT/'anomaly_detection_pipeline/models/mean_all_model.pkl'

def publish(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

sys.path.insert(0,str(PROJECT/'anomaly_detection_pipeline'))
from mean_ocsvm import predict_scores,validate_package

def evaluate(root,model_path,years,dataset,*,inventory_path=None,output=None):
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
    package=joblib.load(model_path)
    validate_package(package)
    cols=['apk_name','cluster','anomaly_score','dataset_year','dataset_class']
    frame=pd.read_csv(root/'test/anomaly_detection_pipeline/test_top50_anomaly_scores.csv',sep=';',usecols=cols,dtype={c:str for c in cols if c!='anomaly_score'})
    if not frame.dataset_year.isin([str(y) for y in years]).all():raise ValueError('Unexpected inference year')
    pred=predict_scores(frame,package)
    if inventory_path is not None:
        columns=['dataset_year','apk_name','dataset_class']
        raw=pd.read_csv(inventory_path,sep=';',usecols=columns,dtype=str)
        if raw.empty or raw.isna().any().any() or not raw.dataset_class.isin(['benign','malware']).all():
            raise ValueError('Invalid test inventory')
        if set(raw.dataset_year)!=set(map(str,years)):
            raise ValueError('Test inventory years must match --years')
    else:
        inventory=[]
        for y in years:
            for label in ('benign','malware'):
                folder=dataset/str(y)/label
                if not folder.is_dir():raise FileNotFoundError(folder)
                inventory.extend(dict(dataset_year=str(y),apk_name=p.stem,dataset_class=label) for p in sorted(folder.rglob('*.apk')))
        raw=pd.DataFrame(inventory)
    if raw.groupby(['dataset_year','apk_name']).dataset_class.nunique().gt(1).any():raise ValueError('Raw APK class conflict')
    raw=raw.drop_duplicates(['dataset_year','apk_name'])
    merged=raw.merge(pred,on=['dataset_year','apk_name'],how='left',suffixes=('','_pred'),validate='one_to_one',indicator=True)
    if len(pred)!=int(merged['_merge'].eq('both').sum()):raise ValueError('Prediction APK missing from raw inventory')
    present=merged['_merge'].eq('both')
    if not merged.loc[present,'dataset_class'].equals(merged.loc[present,'dataset_class_pred']):raise ValueError('Prediction class provenance mismatch')
    merged['prediction_status']=np.where(~present,'no_fused_behavior',np.where(merged.predicted_label.notna(),'predicted','no_scorable_scene'))
    merged.drop(columns=['_merge','dataset_class_pred'],inplace=True)
    def measure(g):
        q=g[g.predicted_label.notna()];y=q.dataset_class.eq('malware').astype(int);p=q.predicted_label.astype(int)
        result=dict(total_apks=len(g),scorable_apks=len(q),unscorable_apks=len(g)-len(q),coverage=len(q)/len(g) if len(g) else 0,status_counts=g.prediction_status.value_counts().to_dict())
        if not len(q):return result
        tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
        result.update(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p,zero_division=0)),f1=float(f1_score(y,p,zero_division=0)),auc=float(roc_auc_score(y,q.anomaly_score)) if y.nunique()==2 else None,benign_fpr=float(fp/(tn+fp)) if tn+fp else None,malware_only_f1=float(2*tp/(2*tp+fn)) if 2*tp+fn else None)
        return result
    result=dict(model=str(model_path),model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),model_refit=False,threshold=0,representation=package['representation'],rms_scale=package['rms_scale'],method='Scene means + training-mean imputation + coverage normalization + training RMS calibration + frozen OCSVM',overall=measure(merged),years={str(y):measure(merged[merged.dataset_year.eq(str(y))]) for y in years},scope='Exploratory evaluation; representation examined on GPMalware and 2018–2022, models/scalers fit only on 2014–2017 benign data')
    out=output if output is not None else root/'anomaly_detection_pipeline/results';out.mkdir(parents=True,exist_ok=True)
    temp=out/'predictions.csv.tmp';merged.to_csv(temp,sep=';',index=False);temp.replace(out/'predictions.csv')
    publish(out/'metrics.json',result)
    for y in years:
        folder=out/str(y);folder.mkdir(exist_ok=True)
        merged[merged.dataset_year.eq(str(y))].to_csv(folder/'predictions.csv',sep=';',index=False);publish(folder/'metrics.json',result['years'][str(y)])
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--training-root',type=Path,default=TRAIN)
    parser.add_argument('--ocsvm-model',type=Path,default=MODEL);parser.add_argument('--years',type=int,nargs='+',default=[2018,2019,2020,2021,2022])
    parser.add_argument('--dataset',type=Path,default=Path('/home/lb/data/Dataset/apks_period'))
    parser.add_argument('--inventory',type=Path,help='Recompute evaluation from saved scores and a semicolon-delimited APK inventory (or published predictions.csv)')
    parser.add_argument('--replay-output',type=Path,help='Output directory for evaluation using --inventory')
    args=parser.parse_args();root=args.root.resolve();trained=args.training_root.resolve();started=time.time()
    if args.inventory or args.replay_output:
        if not (args.inventory and args.replay_output):parser.error('--inventory and --replay-output must be used together')
        result=evaluate(root,args.ocsvm_model,args.years,args.dataset,inventory_path=args.inventory,output=args.replay_output)
        print(json.dumps(result['overall'],ensure_ascii=False,indent=2));return
    if root==trained:parser.error('Inference and frozen training roots must differ')
    if min(args.years)<2018:parser.error('Testing requires 2018 or later')
    def status(stage,state='running',**extra):
        data=dict(state=state,stage=stage,pid=os.getpid(),started_at=started,updated_at=time.time(),years=args.years,**extra)
        publish(root/'inference_status.json',data);print(json.dumps(data,ensure_ascii=False),flush=True)
    model_hash=hashlib.sha256(args.ocsvm_model.read_bytes()).hexdigest()
    import joblib
    validate_package(joblib.load(args.ocsvm_model))
    try:
        while True:
            p=root/'description_status.json'
            data=json.loads(p.read_text()) if p.exists() else {}
            if data.get('state')=='finished' and (root/'test/final_training_data_with_desc.csv').is_file():break
            if data.get('state')=='blocked':raise RuntimeError('Description generator blocked: '+data.get('error','unknown error'))
            time.sleep(15)
        lock=(root/'run.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX)
        config=dict(training_root=str(trained),model=str(args.ocsvm_model),model_sha256=model_hash,years=args.years,description_input=[(root/'test/final_training_data_with_desc.csv').stat().st_size,(root/'test/final_training_data_with_desc.csv').stat().st_mtime_ns])
        cp=root/'inference_config.json'
        previous=json.loads(cp.read_text()) if cp.exists() else None
        model_changed=previous is not None and previous!=config
        if model_changed:
            unchanged=lambda c:{k:v for k,v in c.items() if k not in ('model','model_sha256')}
            if unchanged(previous)!=unchanged(config):raise ValueError('Inference inputs/config changed; review checkpoints before resume')
            prior_status=root/'inference_status.json'
            if prior_status.exists() and json.loads(prior_status.read_text()).get('state')=='finished' and (root/'test/anomaly_detection_pipeline/test_top50_anomaly_scores.csv').is_file():
                status('evaluate_updated_ocsvm')
                result=evaluate(root,args.ocsvm_model,args.years,args.dataset)
                assert model_hash==hashlib.sha256(args.ocsvm_model.read_bytes()).hexdigest()
                publish(cp,config)
                status('finished',state='finished',metrics=result['overall'])
                print('Updated model evaluated using existing behavior scores.',flush=True);return
        publish(cp,config)
        finished=root/'inference_status.json'
        if not model_changed and finished.exists() and json.loads(finished.read_text()).get('state')=='finished' and (root/'anomaly_detection_pipeline/results/predictions.csv').exists():
            print('RESUME: completed frozen inference retained.',flush=True);return
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='32',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='32')
        def launch(cmd,log):
            log.parent.mkdir(parents=True,exist_ok=True)
            for attempt in range(1,4):
                with log.open('ab') as f:
                    f.write(f'\nATTEMPT {attempt}\n'.encode());f.flush()
                    r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
                if r.returncode==0:return
                if r.returncode==2 or attempt==3:raise RuntimeError(f'Stage failed: {log}; exit={r.returncode}')
                time.sleep(5*attempt)
        status('assign_scenarios')
        def assign(branch):
            launch([sys.executable,'-u',str(PROJECT/'cluster_pipeline/scenario_worker.py'),'--root',str(root),'--training-root',str(trained),'--split','test','--test-years',*map(str,args.years),'--branch',branch,'--jobs','32','--nlp-workers','4'],root/'test/cluster_pipeline'/branch/'run.log')
        with futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(assign,['gui','bg']))
        sys.path.insert(0,str(PROJECT/'cluster_pipeline'))
        from run_scenarios import merge_results
        merge_results(root,'test',training_root=trained)
        for stage in ('embedding','scores'):
            status(stage)
            launch([sys.executable,'-u',str(PROJECT/'anomaly_detection_pipeline/worker.py'),'--root',str(root),'--training-root',str(trained),'--stage',stage,'--split','test','--test-years',*map(str,args.years),'--selection','all','--jobs','32','--workers','8','--score-threads','4'],root/'anomaly_detection_pipeline'/f'{stage}_test.log')
        status('evaluate_frozen_mean_ocsvm')
        result=evaluate(root,args.ocsvm_model,args.years,args.dataset)
        assert model_hash==hashlib.sha256(args.ocsvm_model.read_bytes()).hexdigest()
        status('finished',state='finished',metrics=result['overall'])
        print('Finished frozen inference; training artifacts not refitted.',flush=True)
    except Exception as e:
        status('failed',state='failed',error=str(e));raise

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from output_layout import dispatch,TEST
    dispatch(__file__,'--root',TEST,year_flag='--years',training=True)
    main()
