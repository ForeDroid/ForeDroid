#!/usr/bin/env python3
"""All-F-Droid training reference experiment; 2018–2022 testing with checkpoints."""
from pathlib import Path
import argparse,concurrent.futures as futures,csv,fcntl,hashlib,json,os,signal,subprocess,sys,threading,time
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
DATA=Path('/home/lb/data/Dataset');ROOT=DATA/'ForeDroid_output/F-Droid'

def publish(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def training_plan(records,index_sha256):
    return dict(dataset='F-Droid',index_sha256=index_sha256,policy='all downloaded F-Droid applications are training inputs; testing is 2018–2022 only',
                internal_year_sentinel=0,calendar_year=None,testing_dataset='2018-2022',
                training=[dict(package_name=r['package_name'],apk_name=r['sha256']) for r in sorted(records,key=lambda r:r['package_name'])])


def wait_for_extraction(root,pid,retire):
    """Adopt the active unchanged extraction while its old controller is stopped."""
    guard=(root/'adoption.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def command(pid):
        try:return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except OSError:return ''
    started=time.time()
    while command(pid):
        if 'fdroid_pipeline/worker.py extract' not in command(pid) or str(root/'extraction') not in command(pid):raise ValueError('Unexpected extraction process')
        publish(root/'status.json',dict(state='running',stage='extract',pid=os.getpid(),extraction_pid=pid,started_at=started,updated_at=time.time(),total_apks=4933,training_apks=4933,stage_count=18,completed_stages=[],detail='Adopting unchanged full-dataset extraction; all F-Droid APKs are training inputs'))
        time.sleep(10)
    old=command(retire)
    if old:
        if 'fdroid_pipeline/run.py' not in old:raise ValueError('Unexpected retired controller')
        state=(Path('/proc')/str(retire)/'status').read_text()
        if '\nState:\tT' not in state:raise ValueError('Retired controller must be stopped before replacement')
        os.kill(retire,signal.SIGKILL)
    (root/'adoption.json').unlink(missing_ok=True)
    subprocess.run(['tmux','kill-window','-t','foredroid_fdroid:extract_owner'],capture_output=True)
    status=root/'extraction/status.json'
    return status.exists() and json.loads(status.read_text()).get('state')=='finished'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--gui-workers',type=int,default=32);p.add_argument('--chain-workers',type=int,default=32);p.add_argument('--ocr-workers',type=int,default=16);p.add_argument('--llm-workers',type=int,default=32)
    p.add_argument('--wait-extraction-pid',type=int,help=argparse.SUPPRESS);p.add_argument('--retire-controller-pid',type=int,help=argparse.SUPPRESS)
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'CLEANUP_IN_PROGRESS').exists():p.error('Old F-Droid output cleanup is still running; wait for it to finish')
    if not all(1<=x<=64 for x in [args.gui_workers,args.chain_workers,args.ocr_workers,args.llm_workers]):p.error('Concurrency must be 1–64')
    adoption=root/'adoption.json'
    if args.wait_extraction_pid:
        publish(adoption,dict(extraction_pid=args.wait_extraction_pid,controller_pid=args.retire_controller_pid))
    elif adoption.exists():
        previous=json.loads(adoption.read_text());args.wait_extraction_pid=previous['extraction_pid'];args.retire_controller_pid=previous['controller_pid']
    adopted=False
    if args.wait_extraction_pid:
        if not args.retire_controller_pid:p.error('Missing retired controller PID')
        adopted=wait_for_extraction(root,args.wait_extraction_pid,args.retire_controller_pid)
    lock=(root/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX if args.wait_extraction_pid else fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=DATA/'apks_fdroid';manifest=json.loads((source/'.download/manifest.json').read_text());download=json.loads((source/'.download/status.json').read_text())
    if download['state']!='finished' or download['failed']:raise ValueError('Finish verified F-Droid downloads first')
    if not manifest.get('selection_complete') or manifest.get('pilot_limit') is not None:raise ValueError('Finish the full legacy-version selection, not a pilot')
    if manifest.get('compatibility_policy')!='all-dex-035-and-legacy-typespec-v1':raise ValueError('Expected the newly selected legacy-compatible F-Droid corpus')
    records=manifest['records'];n=len(records)
    if len({r['package_name'] for r in records})!=n or len({r['sha256'] for r in records})!=n:raise ValueError('Duplicate package/hash in downloaded manifest')
    for r in records:
        path=source/(r['sha256']+'.apk')
        if not path.is_file() or path.stat().st_size!=r['size']:raise ValueError('Missing or changed APK: '+r['package_name'])
    train_ids={r['sha256'] for r in records}
    plan=training_plan(records,manifest['index_sha256']);planfile=root/'training_manifest.json'
    if planfile.exists() and json.loads(planfile.read_text())!=plan:raise ValueError('F-Droid training input changed; use a new root')
    if not planfile.exists():publish(planfile,plan)
    training=root/'training';historical=root/'testing/2018-2022'
    for folder in [training,historical]:(folder/('train' if folder==training else 'test')).mkdir(parents=True,exist_ok=True)
    def prov(folder,name,splitname,inventory):
        publish(folder/'provenance.json',dict(dataset=name,split=splitname,identities=[[str(x['dataset_year']),x['apk_name'],x['dataset_class']] for x in inventory],training_manifest=str(planfile)))
        publish(folder/'inventory.json',inventory)
    inventory=[dict(dataset_year='0',apk_name=x,dataset_class='benign') for x in sorted(train_ids)]
    prov(training,'F-Droid','train',inventory)
    checkpoints=root/'checkpoints.json';done=json.loads(checkpoints.read_text()) if checkpoints.exists() else {}
    started=time.time();halt=threading.Event();current={};status_lock=threading.Lock()
    def state(stage,status='running',**extra):
        nonlocal current
        with status_lock:
            current=dict(state=status,stage=stage,pid=os.getpid(),started_at=started,updated_at=time.time(),total_apks=n,training_apks=len(train_ids),stage_count=18,completed_stages=list(done),**extra);publish(root/'status.json',current)
        print(stage,status,extra,flush=True)
    def heartbeat():
        while not halt.wait(10):
            with status_lock:
                if current and current['state']=='running':current['updated_at']=time.time();publish(root/'status.json',current)
    threading.Thread(target=heartbeat,daemon=True).start()
    scratch=Path('/tmp/foredroid-fdroid-extract');scratch.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(scratch),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='32',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='32')
    py=sys.executable;worker=PROJECT/'fdroid_pipeline/worker.py'
    def launch(cmd,name):
        logs=root/'logs';logs.mkdir(exist_ok=True)
        with (logs/(name+'.log')).open('ab') as f:
            f.write(('\n'+time.strftime('%Y-%m-%d %H:%M:%S')+'\n').encode());f.flush()
            result=subprocess.run([str(x) for x in cmd],cwd=PROJECT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed ({result.returncode}); see {logs/(name+".log")}')
    def stage(name,artifacts,function):
        state(name)
        if name in done and all(p.exists() for p in artifacts):print('RESUME',name,flush=True);return
        function()
        if not all(p.exists() for p in artifacts):raise RuntimeError('Missing stage output: '+name)
        done[name]=dict(finished_at=time.time());publish(checkpoints,done)
    def link(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_symlink():
            if target.resolve()!=source.resolve():raise ValueError('Unexpected input link: '+str(target))
        elif target.exists():raise ValueError('Refusing to replace input: '+str(target))
        else:target.symlink_to(source)
    def branches(folder,splitname,trained=None,years=(0,)):
        def run(branch):
            cmd=[py,worker,'scenario','--root',folder,'--split',splitname,'--branch',branch,'--jobs','32','--nlp-workers','4','--test-years',*map(str,years)]
            if trained:cmd+=['--training-root',trained]
            launch(cmd,folder.name+'_'+splitname+'_'+branch)
        with futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,['gui','bg']))
    def merge(folder,splitname,trained=None):
        sys.path.insert(0,str(PROJECT/'cluster_pipeline'));from run_scenarios import merge_results
        merge_results(folder,splitname,training_root=trained)
    def anomaly(folder,action,splitname,trained=None,years=(0,)):
        cmd=[py,worker,'anomaly','--root',folder,'--stage',action,'--split',splitname,'--selection','all','--jobs','32','--workers','8','--score-threads','4','--test-years',*map(str,years)]
        if trained:cmd+=['--training-root',trained]
        launch(cmd,folder.name+'_'+action+'_'+splitname)
    def scorefile(folder,splitname):return folder/splitname/'anomaly_detection_pipeline'/f'{splitname}_top50_anomaly_scores.csv'
    model=training/'anomaly_detection_pipeline/models/mean_all_model.pkl'
    formal_model=PROJECT/'anomaly_detection_pipeline/models/mean_all_model.pkl';formal_sha=hashlib.sha256(formal_model.read_bytes()).hexdigest()
    try:
        view=root/'input_view/0/benign';link(source,view)
        extraction=root/'extraction';batch=extraction/'0/benign';gui=batch/'gui'
        stage('extract',[extraction/'status.json',extraction/'state.sqlite'],lambda:None if adopted else launch([py,worker,'extract','--root',extraction,'--view',root/'input_view','--gui-workers',args.gui_workers,'--chain-workers',args.chain_workers,'--ocr-workers',args.ocr_workers],'extract'))
        stage('gui_merge',[gui/'merge/train/merge_ui_api_with_relation.txt'],lambda:launch([py,PROJECT/'extract_Gui/merge_ui_api.py','--api-file',gui/'api-results/api_train.txt','--ui-file',gui/'outputBenignUi_train.txt','--output-dir',gui/'merge/train'],'gui_merge'))
        fused=batch/'pipeline/final_training_data.csv'
        stage('fusion',[fused],lambda:launch([py,PROJECT/'pipeline/pipeline.py','--callchain-dir',batch/'callchain/output','--gui-file',gui/'merge/train/merge_ui_api_with_relation.txt','--output-dir',batch/'pipeline'],'fusion'))
        def training_rows():
            csv.field_size_limit(sys.maxsize)
            target=training/'train/final_training_data.csv';rows=0
            with fused.open(newline='') as inp,target.with_suffix('.csv.tmp').open('w',newline='') as out:
                reader=csv.DictReader(inp);fields=[c for c in reader.fieldnames if c not in ('dataset_year','dataset_class','dataset_name')]+['dataset_year','dataset_class','dataset_name']
                writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
                for row in reader:
                    if row['apk_name'] not in train_ids:raise ValueError('Unknown fused F-Droid APK')
                    rows+=1;row.update(id=str(rows),dataset_year='0',dataset_class='benign',dataset_name='F-Droid');writer.writerow(row)
            if not rows:raise ValueError('No fused training behaviors')
            target.with_suffix('.csv.tmp').replace(target)
            publish(root/'fusion_summary.json',dict(training_apks=n,training_rows=rows))
        stage('prepare_training',[training/'train/final_training_data.csv'],training_rows)
        stage('train_descriptions',[training/'train/final_training_data_with_desc.csv'],lambda:launch([py,PROJECT/'pipeline/generate_description.py','--output-root',training,'--splits','train','--model','MiniMax-M3','--workers',args.llm_workers,'--reuse-cache',DATA/'ForeDroid_output/2014-2017/descriptions.sqlite'],'train_descriptions'))
        stage('train_scenarios',[training/'train/cluster_pipeline/gui_centroids/gui_clustered_data.csv',training/'train/cluster_pipeline/bg_centroids/bg_clustered_data.csv'],lambda:branches(training,'train'))
        stage('train_merge_scenarios',[training/'train/cluster_pipeline/output/all_apks_with_clusters.csv'],lambda:merge(training,'train'))
        stage('train_embedding',[training/'anomaly_detection_pipeline/checkpoints/embedding_train/embeddings.npy'],lambda:anomaly(training,'embedding','train'))
        stage('train_references',[training/'anomaly_detection_pipeline/precomputed_vectors/manifest.json'],lambda:anomaly(training,'references','train'))
        stage('train_scores',[scorefile(training,'train')],lambda:anomaly(training,'scores','train'))
        def fit():
            import pandas as pd,joblib
            from anomaly_detection_pipeline.mean_ocsvm import train
            cols=['apk_name','dataset_year','dataset_class','dataset_name','cluster','anomaly_score']
            f=pd.read_csv(scorefile(training,'train'),sep=';',usecols=cols,dtype={c:str for c in cols if c!='anomaly_score'})
            provenance_ids=set(f.apk_name)
            if not provenance_ids<=train_ids:raise ValueError('Unexpected training APK')
            clusters=sorted(f.loc[f.anomaly_score.notna()&f.cluster.ne('-1'),'cluster'].unique(),key=int)
            package=train(f,clusters,training_dataset='F-Droid');package.update(training_manifest_sha256=hashlib.sha256(planfile.read_bytes()).hexdigest(),reference_root=str(training))
            model.parent.mkdir(exist_ok=True);temp=model.with_suffix('.pkl.tmp');joblib.dump(package,temp);temp.replace(model)
        stage('fit_ocsvm',[model],fit)
        def training_coverage():
            import pandas as pd,joblib
            f=pd.read_csv(scorefile(training,'train'),sep=';',usecols=['apk_name','cluster','anomaly_score'],dtype={'apk_name':str,'cluster':str})
            p=joblib.load(model);vocabulary=set(map(str,p['clusters']))
            f['_reference']=f.cluster.ne('-1').astype(int);f['_scored']=f.anomaly_score.notna().astype(int)
            f['_eligible']=f.anomaly_score.notna()&f.cluster.isin(vocabulary)
            grouped=f.groupby('apk_name').agg(behavior_rows=('apk_name','size'),reference_rows=('_reference','sum'),scored_rows=('_scored','sum'),model_training_eligible=('_eligible','any'))
            grouped=grouped.reindex(sorted(train_ids),fill_value=0);grouped['model_training_eligible']=grouped.model_training_eligible.astype(bool)
            rows=grouped.reset_index().to_dict('records')
            out=training/'results';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'coverage.csv',index=False)
            publish(out/'coverage.json',dict(training_input_apks=n,fused_apks=sum(r['behavior_rows']>0 for r in rows),reference_apks=sum(r['reference_rows']>0 for r in rows),model_training_apks=p['training_apks']))
        stage('training_coverage',[training/'results/coverage.csv',training/'results/coverage.json'],training_coverage)
        def prepare_external(folder,source_root,name):
            import pandas as pd
            inv=pd.read_csv(source_root/'anomaly_detection_pipeline/results/predictions.csv',sep=';',dtype={'dataset_year':str,'apk_name':str})[['dataset_year','apk_name','dataset_class']].to_dict('records')
            excluded=[x for x in inv if x['apk_name'] in train_ids];inv=[x for x in inv if x['apk_name'] not in train_ids]
            publish(folder/'excluded_train_hash_overlap.json',excluded);prov(folder,name,'test',inv)
            src=source_root/'test/final_training_data_with_desc.csv';target=folder/'test/final_training_data_with_desc.csv'
            if not excluded:link(src,target)
            else:
                allowed={(str(x['dataset_year']),x['apk_name'],x['dataset_class']) for x in inv};csv.field_size_limit(sys.maxsize)
                with src.open(newline='') as inp,target.with_suffix('.csv.tmp').open('w',newline='') as out:
                    reader=csv.DictReader(inp);writer=csv.DictWriter(out,fieldnames=reader.fieldnames);writer.writeheader()
                    for row in reader:
                        if (row['dataset_year'],row['apk_name'],row['dataset_class']) in allowed:writer.writerow(row)
                target.with_suffix('.csv.tmp').replace(target)
        stage('prepare_historical',[historical/'test/final_training_data_with_desc.csv',historical/'inventory.json'],lambda:prepare_external(historical,DATA/'ForeDroid_output/2018-2022/runs','2018-2022'))
        for folder,years in [(historical,list(range(2018,2023)))]:
            tag=folder.name
            stage(tag+'_scenarios',[folder/'test/cluster_pipeline/gui/gui_assigned_data.csv',folder/'test/cluster_pipeline/bg/bg_assigned_data.csv'],lambda f=folder,y=years:branches(f,'test',training,y))
            stage(tag+'_merge_scenarios',[folder/'test/cluster_pipeline/output/all_apks_with_clusters.csv'],lambda f=folder:merge(f,'test',training))
            stage(tag+'_embedding',[folder/'anomaly_detection_pipeline/checkpoints/embedding_test/embeddings.npy'],lambda f=folder,y=years:anomaly(f,'embedding','test',training,y))
            stage(tag+'_scores',[scorefile(folder,'test')],lambda f=folder,y=years:anomaly(f,'scores','test',training,y))
            stage(tag+'_evaluate',[folder/'results/metrics.json',folder/'results/predictions.csv'],lambda f=folder:evaluate(f,model))
        assert hashlib.sha256(formal_model.read_bytes()).hexdigest()==formal_sha
        summary={f.name:json.loads((f/'results/metrics.json').read_text()) for f in [historical]};publish(root/'results.json',summary)
        state('finished','finished');halt.set()
    except Exception as error:state('failed','failed',error=str(error));halt.set();raise


def evaluate(root,model):
    import pandas as pd,numpy as np,joblib
    from anomaly_detection_pipeline.mean_ocsvm import predict_scores
    cols=['apk_name','dataset_year','dataset_class','cluster','anomaly_score']
    f=pd.read_csv(root/'test/anomaly_detection_pipeline/test_top50_anomaly_scores.csv',sep=';',usecols=cols,dtype={c:str for c in cols if c!='anomaly_score'})
    pred=predict_scores(f,joblib.load(model));raw=pd.DataFrame(json.loads((root/'inventory.json').read_text()));raw['dataset_year']=raw.dataset_year.astype(str)
    if raw.duplicated(['dataset_year','apk_name']).any():raise ValueError('Duplicate test identity')
    if not set(zip(pred.dataset_year,pred.apk_name))<=set(zip(raw.dataset_year,raw.apk_name)):raise ValueError('Test scores outside inventory')
    result=raw.merge(pred.drop(columns=['dataset_class']),on=['dataset_year','apk_name'],how='left',validate='one_to_one')
    result['prediction_status']=np.where(result.predicted_label.notna(),'predicted','unscorable')
    def metric(g):
        q=g[g.predicted_label.notna()];y=q.dataset_class.eq('malware');p=q.predicted_label.eq(1)
        tp,fp,fn,tn=[int(x.sum()) for x in [y&p,~y&p,y&~p,~y&~p]]
        both=bool(y.any() and (~y).any())
        return dict(total_apks=len(g),scorable_apks=len(q),unscorable_apks=len(g)-len(q),TP=tp,FP=fp,FN=fn,TN=tn,
                    precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                    f1=2*tp/(2*tp+fp+fn) if both and 2*tp+fp+fn else None,
                    malware_only_f1=2*tp/(2*tp+fn) if tp+fn else None,benign_fpr=fp/(tn+fp) if tn+fp else None)
    data=dict(model=str(model),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),training_dataset='F-Droid',overall=metric(result),years={year:metric(g) for year,g in result.groupby('dataset_year')},scope='All downloaded F-Droid apps used as training inputs; 2018–2022 test, exact training SHA overlap excluded; cross-corpus package/version novelty not established')
    out=root/'results';out.mkdir(exist_ok=True);temp=out/'predictions.csv.tmp';result.to_csv(temp,sep=';',index=False);temp.replace(out/'predictions.csv');publish(out/'metrics.json',data)
    print(root.name,data['overall'],flush=True)

if __name__=='__main__':main()
