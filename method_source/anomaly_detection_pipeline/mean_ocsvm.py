"""Mean OCSVM with scene-coverage normalization and training-only RMS calibration."""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

REPRESENTATION = 'coverage_normalized'


def validate_package(package):
    if (package.get('selection') != 'all' or package.get('aggregation') != 'mean'
            or package.get('representation') != REPRESENTATION):
        raise ValueError('Expected the coverage-normalized mean OCSVM package; retrain with mean_ocsvm.py')
    scale = package.get('rms_scale', 0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Missing or invalid training RMS scale')


def _coverage_vectors(X, observed, mean, scaler):
    """Use only observed scenes; zero is also a valid observed behavior score."""
    counts = observed.sum(axis=1)
    if (counts == 0).any():
        raise ValueError('Cannot normalize an APK without any observed scenes')
    Z = scaler.transform(np.where(observed, X.astype(np.float64), mean))
    Z[~observed] = 0
    return Z * np.sqrt(X.shape[1] / counts)[:, None]


def predict_scores(frame,package):
    """Mean all observed behavior scores per year/APK/scene; frozen fill/scaler."""
    import numpy as np
    import pandas as pd
    validate_package(package)
    frame=frame.copy();frame['dataset_year']=frame.dataset_year.astype(str);frame['cluster']=frame.cluster.astype(str)
    keys=['dataset_year','apk_name']
    if frame.groupby(keys).dataset_class.nunique().gt(1).any():raise ValueError('Conflicting classes for the same year/APK')
    meta=frame.groupby(keys,sort=False).dataset_class.first().reset_index()
    columns=[str(c) for c in package['clusters']];ci={c:i for i,c in enumerate(columns)}
    ai={(r.dataset_year,r.apk_name):i for i,r in meta.iterrows()}
    X=np.zeros((len(meta),len(columns)),dtype='float32');observed=np.zeros(X.shape,dtype=bool)
    valid=frame[frame.anomaly_score.notna() & frame.cluster.isin(ci)]
    for (year,apk,cid),score in valid.groupby(keys+['cluster']).anomaly_score.mean().items():
        i=ai[year,apk];j=ci[cid];X[i,j]=score;observed[i,j]=True
    eligible=observed.any(axis=1);meta['scorable']=eligible;meta['observed_scenes']=observed.sum(axis=1)
    meta['predicted_label']=pd.Series(pd.NA,index=meta.index,dtype='Int64');meta['anomaly_score']=np.nan
    if eligible.any():
        Z=_coverage_vectors(X[eligible],observed[eligible],package['missing_mean'],package['scaler'])/package['rms_scale']
        meta.loc[eligible,'predicted_label']=(package['model'].predict(Z)==-1).astype(int)
        meta.loc[eligible,'anomaly_score']=-package['model'].decision_function(Z)
    return meta


def train(scores, clusters, *, training_dataset=None):
    """Fit the explicitly validated benign training cohort; no test-label selection."""
    if not scores.dataset_class.eq('benign').all():
        raise ValueError('Training requires benign samples only')
    if training_dataset is None:
        if not scores.dataset_year.astype(int).isin([2014,2015,2016,2017]).all():
            raise ValueError('Training requires years 2014–2017')
    elif training_dataset == 'F-Droid':
        if ('dataset_name' not in scores or not scores.dataset_name.eq('F-Droid').all()
                or not scores.dataset_year.astype(str).eq('0').all()):
            raise ValueError('F-Droid training requires explicit dataset provenance and non-calendar sentinel 0')
    else:
        raise ValueError('Unsupported training dataset')
    scores=scores.copy();scores['cluster']=scores.cluster.astype(str)
    columns=[str(c) for c in clusters]
    # Match the verified model: an identical training APK in multiple years
    # contributes one training vector. Test identities remain year + APK.
    meta=scores.groupby('apk_name',sort=True).dataset_class.first().reset_index()
    ai={r.apk_name:i for i,r in meta.iterrows()}
    ci={c:i for i,c in enumerate(columns)}
    X=np.zeros((len(meta),len(columns)),dtype='float64');M=np.zeros(X.shape,dtype=bool)
    valid=scores[scores.anomaly_score.notna() & scores.cluster.astype(str).isin(ci)].copy()
    valid['cluster']=valid.cluster.astype(str)
    for (apk,c),value in valid.groupby(['apk_name','cluster']).anomaly_score.mean().items():
        i=ai[apk];j=ci[c];X[i,j]=np.float32(value);M[i,j]=True
    keep=M.any(axis=1);X=X[keep];M=M[keep]
    counts=M.sum(axis=0)
    if len(X)<10 or (counts==0).any():raise ValueError('Insufficient training evidence for selected scenes')
    mean=(X.astype(float)*M).sum(axis=0)/counts
    scaler=StandardScaler().fit(np.where(M,X,mean))
    Z=_coverage_vectors(X,M,mean,scaler)
    rms_scale=float(np.sqrt(np.mean(Z**2))) or 1.0
    model=OneClassSVM(kernel='rbf',nu=.001,gamma='auto',cache_size=1024,max_iter=100000).fit(Z/rms_scale)
    if model.fit_status_!=0:raise ValueError('OCSVM did not converge')
    return dict(model=model,scaler=scaler,missing_mean=mean,selection='all',aggregation='mean',clusters=columns,threshold=0,
                representation=REPRESENTATION,rms_scale=rms_scale,training_grouping='apk_name',
                training_years=[2014,2015,2016,2017] if training_dataset is None else None,
                training_dataset=training_dataset or '2014–2017 benign',training_apks=len(X))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-scores',type=Path,required=True)
    parser.add_argument('--scene-model',type=Path,required=True,help='Existing OCSVM package supplying the frozen scene vocabulary')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists; choose a new path')
    columns=['apk_name','dataset_year','dataset_class','cluster','anomaly_score']
    scores=pd.read_csv(args.train_scores,sep=';',usecols=columns,dtype={c:str for c in columns if c!='anomaly_score'})
    original=joblib.load(args.scene_model)
    package=train(scores,original.get('clusters',original.get('all_clusters')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+'.tmp');joblib.dump(package,temporary);temporary.replace(args.output)
    print(f'Saved {args.output}; scenes={len(package["clusters"])}; threshold=0')

if __name__=='__main__':main()
