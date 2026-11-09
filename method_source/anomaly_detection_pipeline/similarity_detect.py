"""Batched, same-APK-excluding behavioral similarity; no model loaded on import."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def topk_cosine(query_vectors,reference_vectors,query_apks,reference_apks,k=50,normalized=False):
    """Return mean top-k cosine, eligible reference count and used neighbor count.

    NaN is deliberate when no cross-APK reference exists. It is never replaced
    with zero similarity (which would invent a maximal anomaly score).
    """
    if k<1:raise ValueError('k must be positive')
    query=np.asarray(query_vectors,dtype='float32');reference=np.asarray(reference_vectors,dtype='float32')
    if query.ndim!=2 or reference.ndim!=2 or query.shape[1]!=reference.shape[1]:raise ValueError('Incompatible vector shapes')
    if len(query)!=len(query_apks) or len(reference)!=len(reference_apks):raise ValueError('APK/vector alignment mismatch')
    if not normalized and (not np.isfinite(query).all() or not np.isfinite(reference).all()):raise ValueError('Non-finite embeddings')
    if not len(reference):return np.full(len(query),np.nan),np.zeros(len(query),dtype=int),np.zeros(len(query),dtype=int)
    if normalized:
        qnorm=np.ones((len(query),1));rnorm=np.ones((len(reference),1))
    else:
        qnorm=np.linalg.norm(query,axis=1,keepdims=True);rnorm=np.linalg.norm(reference,axis=1,keepdims=True)
        query=np.divide(query,qnorm,out=np.zeros_like(query),where=qnorm>0)
        reference=np.divide(reference,rnorm,out=np.zeros_like(reference),where=rnorm>0)
    similarities=np.clip(query@reference.T,-1,1)
    eligible=(np.asarray(query_apks)[:,None]!=np.asarray(reference_apks)[None,:]) & (rnorm.ravel()>0)[None,:]
    eligible &= (qnorm.ravel()>0)[:,None]
    counts=eligible.sum(axis=1);similarities[~eligible]=-np.inf
    take=min(k,len(reference));top=np.partition(similarities,len(reference)-take,axis=1)[:,-take:]
    finite=np.isfinite(top);used=finite.sum(axis=1)
    average=np.divide(np.where(finite,top,0).sum(axis=1),used,out=np.full(len(query),np.nan,dtype='float32'),where=used>0)
    return average,counts,used


def calculate_topk_similarity(test_chain,test_apk,cluster_id,vector_dir,k=50,model=None):
    if str(cluster_id)=='-1':return float('nan')
    if model is None:raise ValueError('Pass the preloaded behavior embedding model')
    root=Path(vector_dir);vectors=np.load(root/f'cluster_{cluster_id}.npy');apks=json.loads((root/f'cluster_{cluster_id}_apk.json').read_text())
    query=model.encode([test_chain],normalize_embeddings=True,show_progress_bar=False)
    means,_,_=topk_cosine(query,vectors,[test_apk],apks,k)
    return float(means[0])


def aggregate_anomaly_scores(df,threshold=0.5,agg_method='mean'):
    if agg_method not in ('mean','sum'):raise ValueError('agg_method must be mean or sum')
    eligible=df[df['anomaly_score'].notna()]
    agg=eligible.groupby('apk_name')['anomaly_score'].agg(agg_method).rename('total_anomaly_score')
    result=pd.DataFrame({'apk_name':df['apk_name'].drop_duplicates()}).merge(agg,on='apk_name',how='left')
    result['scorable']=result['total_anomaly_score'].notna()
    result['is_anomalous']=pd.Series(pd.NA,index=result.index,dtype='boolean')
    result.loc[result['scorable'],'is_anomalous']=result.loc[result['scorable'],'total_anomaly_score']>threshold
    return result


if __name__=='__main__':
    from run_pipeline import main
    main(default_only='scores')
