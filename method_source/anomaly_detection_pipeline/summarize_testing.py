"""Publish combined testing results without changing behavior IDs or embeddings."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from output_layout import TEST,groups

def summarize(root):
    import pandas as pd
    from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
    frames=[]
    for folder,years in groups(root):
        p=folder/'anomaly_detection_pipeline/results/predictions.csv'
        if not p.exists():raise FileNotFoundError(p)
        frame=pd.read_csv(p,sep=';',dtype={'dataset_year':str,'apk_name':str})
        if not frame.dataset_year.isin(list(map(str,years))).all():raise ValueError('Unexpected partition year')
        frames.append(frame)
    result=pd.concat(frames,ignore_index=True)
    if result.duplicated(['dataset_year','apk_name']).any():raise ValueError('Duplicate year/APK in combined results')
    def metric(frame):
        q=frame[frame.predicted_label.notna()];y=q.dataset_class.eq('malware').astype(int);p=q.predicted_label.astype(int)
        d=dict(total_apks=len(frame),scorable_apks=len(q),unscorable_apks=len(frame)-len(q),coverage=len(q)/len(frame),status_counts=frame.prediction_status.value_counts().to_dict())
        if len(q):
            tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
            d.update(TN=tn,FP=fp,FN=fn,TP=tp,precision=precision_score(y,p,zero_division=0),recall=recall_score(y,p,zero_division=0),f1=f1_score(y,p,zero_division=0),auc=roc_auc_score(y,q.anomaly_score) if y.nunique()==2 else None,benign_fpr=fp/(tn+fp) if tn+fp else None,malware_only_f1=2*tp/(2*tp+fn) if 2*tp+fn else None)
        return d
    out=root/'anomaly_detection_pipeline/results';out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'predictions.csv.tmp',sep=';',index=False);(out/'predictions.csv.tmp').replace(out/'predictions.csv')
    sources=[json.loads((folder/'anomaly_detection_pipeline/results/metrics.json').read_text()) for folder,_ in groups(root)]
    signatures={(s.get('model_sha256'),s.get('representation')) for s in sources}
    if len(signatures)!=1:raise ValueError('Mixed OCSVM models in testing results')
    model_meta={k:sources[0][k] for k in ('model','model_sha256','representation','rms_scale','scope') if k in sources[0]}
    data=dict(overall=metric(result),years={year:metric(g) for year,g in result.groupby('dataset_year')},method='Coverage-normalized mean OCSVM with training RMS calibration; unified cached behavior artifacts',training_years=[2014,2015,2016,2017],testing_years=list(range(2018,2023)),**model_meta)
    (out/'metrics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    for year,g in result.groupby('dataset_year'):
        d=out/year;d.mkdir(exist_ok=True);g.to_csv(d/'predictions.csv',sep=';',index=False);(d/'metrics.json').write_text(json.dumps(data['years'][year],indent=2)+'\n')
    print(json.dumps(data['overall'],ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=TEST);summarize(p.parse_args().root)
