"""Reconstruct the equal-buyer training reference without fitting recommenders."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

def run(path, out):
    U='Survey ResponseID'; I='ASIN/ISBN (Product Code)'; D='Order Date'
    Q='Quantity'; C='Category'; V='Purchase Price Per Unit'
    p=pd.read_csv(path,low_memory=False)
    p[D]=pd.to_datetime(p[D],errors='coerce')
    for col in [Q,V]:
        p[col]=pd.to_numeric(p[col].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False).str.strip(),errors='coerce')
    p=p.dropna(subset=[D,U,I,C])
    p=p[(p[Q]>0)&(p[V]>0)&~p[C].astype('string').str.strip().isin({'GIFT_CARD','ABIS_GIFT_CARD','Ecard Gift Certificate'})]
    tr=p[p[D]<pd.Timestamp('2022-04-19')].copy();va=p[p[D]>=pd.Timestamp('2022-04-19')]
    assert len(tr)==1381836 and len(va)==345141
    pairs=tr[[U,I]].drop_duplicates()
    while True:
        old=len(pairs)
        uc=pairs.groupby(U)[I].nunique();pairs=pairs[pairs[U].isin(uc[uc>=5].index)]
        ic=pairs.groupby(I)[U].nunique();pairs=pairs[pairs[I].isin(ic[ic>=5].index)]
        if len(pairs)==old:break
    users=set(pairs[U]);items=set(pairs[I]);q=tr.groupby(I)[Q].sum().reindex(sorted(items))
    b1=q.mean();b2=q[q>b1].mean()
    seg=pd.Series(np.where(q<=b1,'tail',np.where(q<=b2,'mid','head')),index=q.index)
    seen=tr.groupby(U)[I].agg(set).to_dict();future=va.groupby(U)[I].agg(set).to_dict()
    evaluated=sorted(u for u in users if (future.get(u,set())-seen[u])&items)
    assert len(users)==4507 and len(items)==30360 and len(evaluated)==4012
    ranked=q.rename('quantity').reset_index().sort_values(['quantity',I],ascending=[False,True])
    ranktail=set(ranked.iloc[int(np.ceil(len(ranked)*.10)):][I])
    rows=tr[tr[U].isin(evaluated)&tr[I].isin(items)].copy();rows['segment']=rows[I].map(seg)
    shares=pd.crosstab(rows[U],rows.segment,normalize='index').reindex(columns=['head','mid','tail'])
    ref=shares.mean();rank_ref=rows.assign(rank_tail=rows[I].isin(ranktail)).groupby(U).rank_tail.mean().mean()
    assert abs(ref['tail']*100-47.90)<.01
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'head_pct':100*ref['head'],'mid_pct':100*ref['mid'],'tail_pct':100*ref['tail'],'rank_tail_pct':100*rank_ref,'buyers':len(evaluated)}]).to_csv(out/'training_mix_reference.csv',index=False)
    (out/'training_mix_reference.json').write_text(json.dumps({'training_rows':len(tr),'validation_rows':len(va),'core_buyers':len(users),'catalog_products':len(items),'evaluated_buyers':len(evaluated),'breaks':[b1,b2],'reference':'Mean within-buyer shares of classifiable training purchase rows; same evaluated buyers as Top-20 lists.'},indent=2))
    print(ref*100,'rank tail',100*rank_ref,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--purchases',required=True);args=a.parse_args()
    run(args.purchases,Path(__file__).resolve().parents[1]/'Results/TrainingReference')
