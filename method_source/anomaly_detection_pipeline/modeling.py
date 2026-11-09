"""Train-only feature construction, benign validation and frozen evaluation."""
from pathlib import Path
import json
import math
import time
import multiprocessing as mp
import warnings
from sklearn.exceptions import ConvergenceWarning
import concurrent.futures as futures
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,roc_curve


def build_features(df,clusters=None,selection='top80percent',top_percent=.8,top_n=10):
    if df.groupby('apk_name')['dataset_class'].nunique().gt(1).any():raise ValueError('Conflicting labels for the same APK')
    metadata=df.groupby('apk_name',sort=False)['dataset_class'].agg(lambda x:x.iloc[0]).reset_index()
    valid=df[df['anomaly_score'].notna() & df['cluster'].astype(str).ne('-1')].copy()
    if clusters is None:clusters=sorted(valid['cluster'].astype(str).unique(),key=int)
    valid=valid[valid['cluster'].astype(str).isin(clusters)]
    selected=[]
    for _,group in valid.groupby('apk_name',sort=False):
        if selection=='top80percent':group=group.nlargest(max(1,math.ceil(len(group)*top_percent)),'anomaly_score')
        elif selection=='top10':group=group.nlargest(min(top_n,len(group)),'anomaly_score')
        elif selection!='all':raise ValueError(f'Unknown selection: {selection}')
        selected.append(group)
    chosen=pd.concat(selected,ignore_index=True) if selected else valid
    apks=metadata['apk_name'].to_numpy(dtype=str);apk_index={name:i for i,name in enumerate(apks)};cluster_index={str(c):i for i,c in enumerate(clusters)}
    X=np.zeros((len(apks),len(clusters)),dtype='float32');scorable=np.zeros(len(apks),dtype=bool)
    for (apk,cluster),score in chosen.groupby(['apk_name','cluster'])['anomaly_score'].max().items():
        i=apk_index[apk];X[i,cluster_index[str(cluster)]]=score;scorable[i]=True
    metadata['scorable']=scorable
    return X,metadata,clusters


MAX_ITER = 100000
CANDIDATE_TIMEOUT = 300


def _fit_candidate(send, params, fit, validation):
    try:
        start = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            model = OneClassSVM(**params, cache_size=256, max_iter=MAX_ITER).fit(fit)
        if model.fit_status_ != 0:
            result = dict(params=params, status='not_converged', iterations=int(model.n_iter_))
        else:
            retention = float(np.mean(model.predict(validation) == 1))
            result = dict(params=params, status='converged', benign_validation_retention=retention,
                          objective=retention-params['nu']*.1)
        result['seconds'] = time.monotonic()-start
        send.send(result)
    except Exception as error:
        send.send(dict(params=params, status='failed', error=str(error)))
    finally:
        send.close()


def bounded_candidate(params, fit, validation, timeout=CANDIDATE_TIMEOUT):
    context = mp.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_fit_candidate, args=(send, params, fit, validation))
    process.start();send.close()
    try:
        if receive.poll(timeout):
            try:return receive.recv()
            except EOFError:return dict(params=params, status='failed', error='Candidate process exited without a result')
        return dict(params=params, status='timeout', seconds=timeout)
    finally:
        if process.is_alive():process.terminate()
        process.join(5)
        if process.is_alive():process.kill();process.join()
        receive.close()


def train_detector(scores,folder,update,workers=4,selection='top80percent',top_percent=.8,neighbors=50):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if not scores['dataset_class'].eq('benign').all():raise ValueError('Non-benign records in detector training')
    X,metadata,clusters=build_features(scores,selection=selection,top_percent=top_percent)
    eligible=metadata['scorable'].to_numpy();normal=X[eligible]
    if len(normal)<10 or not len(clusters):raise ValueError('Too few scorable benign APKs or scenarios')
    fit_indices,validation_indices=train_test_split(np.arange(len(normal)),test_size=.2,random_state=42)
    scaler=StandardScaler().fit(normal[fit_indices]);fit=scaler.transform(normal[fit_indices]);validation=scaler.transform(normal[validation_indices])
    parameters=[dict(kernel=kernel,nu=nu,gamma=gamma) for kernel in ['rbf','linear'] for nu in [.001,.01,.05,.1] for gamma in (['scale','auto'] if kernel=='rbf' else ['scale'])]
    checkpoint=folder/'grid_search.json';previous=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    def key(params):return json.dumps(params,sort_keys=True)
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(bounded_candidate,p,fit,validation):p for p in parameters if key(p) not in previous}
        for future in futures.as_completed(pending):
            params=pending[future];previous[key(params)]=future.result()
            staged=checkpoint.with_suffix('.json.tmp');staged.write_text(json.dumps(previous,indent=2));staged.replace(checkpoint)
            update(len(previous),len(parameters),'仅用训练集内部良性验证选择超参数')
    candidates=[previous[key(p)] for p in parameters if previous[key(p)].get('status','converged')=='converged' and 'objective' in previous[key(p)]]
    if not candidates:raise ValueError('No converged OCSVM candidate; inspect grid_search.json')
    best=max(candidates,key=lambda row:row['objective'])
    update(len(parameters),len(parameters),'使用全部可评分良性 APK 重新拟合最终模型')
    scaler=StandardScaler().fit(normal);transformed=scaler.transform(normal)
    model=OneClassSVM(**best['params'],cache_size=512,max_iter=MAX_ITER);model.fit(transformed)
    if model.fit_status_ != 0:raise ValueError('Final OCSVM did not converge within iteration limit; model not published')
    package=dict(model=model,scaler=scaler,all_clusters=clusters,best_params=best['params'],standardize=True,
                 sample_selection=selection,top_percent=top_percent,top_n=10,train_samples=len(normal),
                 num_scenarios=len(clusters),k_neighbors=neighbors,training_years=[2014,2015,2016,2017],embedding_model='all-MiniLM-L6-v2',train_acc=float(np.mean(model.predict(transformed)==1)),
                 validation=best,reference_policy='training benign only; exclude every reference from query APK',
                 noise_policy='cluster -1 excluded; insufficient reference evidence is unscorable')
    target=folder/'ocsvm_model.pkl';temp=target.with_suffix('.pkl.tmp');joblib.dump(package,temp);temp.replace(target)
    # Explicitly retain training APKs that have no evaluable scenario.
    metadata['predicted_anomalous']=pd.Series(pd.NA,index=metadata.index,dtype='Int64')
    metadata.loc[eligible,'predicted_anomalous']=(model.predict(transformed)==-1).astype(int)
    metadata.to_csv(folder/'training_apk_status.csv',sep=';',index=False)
    summary=dict(scorable_apks=int(eligible.sum()),unscorable_apks=int((~eligible).sum()),scenarios=len(clusters),best_params=best['params'],validation=best,training_benign_retention=package['train_acc'])
    (folder/'training_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def evaluate_detector(scores,model_file,folder):
    package=joblib.load(model_file)
    X,metadata,_=build_features(scores,clusters=package['all_clusters'],selection=package['sample_selection'],top_percent=package['top_percent'],top_n=package['top_n'])
    eligible=metadata['scorable'].to_numpy();y=(metadata['dataset_class']=='malware').astype(int).to_numpy()
    metadata['true_label']=y;metadata['predicted_label']=pd.Series(pd.NA,index=metadata.index,dtype='Int64');metadata['anomaly_score']=np.nan
    if eligible.any():
        transformed=package['scaler'].transform(X[eligible]);pred=(package['model'].predict(transformed)==-1).astype(int);decision=-package['model'].decision_function(transformed)
        metadata.loc[eligible,'predicted_label']=pred;metadata.loc[eligible,'anomaly_score']=decision
    else:raise ValueError('No scorable test APKs')
    truth=y[eligible];matrix=confusion_matrix(truth,pred,labels=[0,1]);tn,fp,fn,tp=matrix.ravel()
    metrics=dict(total_apks=len(metadata),scorable_apks=int(eligible.sum()),unscorable_apks=int((~eligible).sum()),coverage=float(eligible.mean()),
        coverage_by_class={label:dict(total=int((metadata['dataset_class']==label).sum()),scorable=int(((metadata['dataset_class']==label)&metadata['scorable']).sum())) for label in ['benign','malware']},
        accuracy=float(accuracy_score(truth,pred)),precision=float(precision_score(truth,pred,zero_division=0)),recall=float(recall_score(truth,pred,zero_division=0)),f1=float(f1_score(truth,pred,zero_division=0)),
        auc=float(roc_auc_score(truth,decision)) if len(np.unique(truth))==2 else None,confusion_matrix=matrix.tolist(),false_positive_rate=float(fp/(fp+tn)) if fp+tn else None,
        false_negative_rate=float(fn/(fn+tp)) if fn+tp else None,evaluation_scope='Scorable APKs only; unscorable APKs are not labeled benign',threshold='OCSVM decision boundary 0, not tuned on test labels',best_params=package['best_params'])
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    metadata['status']=np.where(eligible,'scored','insufficient_scenario_evidence')
    metadata.to_csv(folder/'predictions.csv',sep=';',index=False);(folder/'metrics.json').write_text(json.dumps(metrics,indent=2))
    if len(np.unique(truth))==2:
        fpr,tpr,thresholds=roc_curve(truth,decision);pd.DataFrame(dict(false_positive_rate=fpr,true_positive_rate=tpr,threshold=thresholds)).to_csv(folder/'roc_curve.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5,4));image=ax.imshow(matrix,cmap='Blues');ax.set_xticks([0,1],['Benign','Malware']);ax.set_yticks([0,1],['Benign','Malware']);ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(folder/'confusion_matrix.png',dpi=160);plt.close(fig)
    if len(np.unique(truth))==2:
        fig,ax=plt.subplots(figsize=(5,4));ax.plot(fpr,tpr,label=f'AUC = {metrics["auc"]:.3f}');ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='OCSVM ROC');ax.legend();fig.tight_layout();fig.savefig(folder/'roc_curve.png',dpi=160);plt.close(fig)
    return metrics
