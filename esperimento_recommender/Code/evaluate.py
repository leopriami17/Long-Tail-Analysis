"""Evaluate one fixed fit and archive complete per-buyer Top-30 lists."""
from pathlib import Path
import argparse,pickle,json,time,os,gzip
import numpy as np,pandas as pd
from discovery_recommender import TopPopularQuantity,CategoryConditionedQuantity,ItemKNNFast
from latent_numpy import Network
P=Path(__file__).resolve().parents[1];O=P/'Results/Chronological8020';U='Survey ResponseID';I='ASIN/ISBN (Product Code)'
a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--fit');a.add_argument('--rank-cache');args=a.parse_args()
with (P/'prepared.pkl').open('rb') as f:ctx=pickle.load(f)
users=ctx['eligible'];items=ctx['core_items'];nitems=len(items);iset=set(items);ismap={i:j for j,i in enumerate(items)};umap={u:j for j,u in enumerate(ctx['core_users'])};categories=ctx['categories'];segments=ctx['segment_map'];q=ctx['train_quantity_precore'];catnames=sorted(set(categories.values()));cmap={c:j for j,c in enumerate(catnames)};ci=np.array([cmap[categories[i]] for i in items]);ranked=q.rename('q').reset_index().sort_values(['q',I],ascending=[False,True]);ranktail=set(ranked.iloc[int(np.ceil(nitems*.1)):][I])
H=ctx['category_history'].reindex(columns=catnames,fill_value=0);H=H.div(H.sum(axis=1),axis=0)
availability=np.bincount(ci,weights=ctx['availability_product_share'],minlength=len(catnames))
start=time.time()
if args.model=='Global quantity':model=TopPopularQuantity(q)
elif args.model=='Category quantity':model=CategoryConditionedQuantity(ctx)
elif args.model=='ItemKNN':model=ItemKNNFast(ctx['core_pairs'],q,n_neighbors=50)
else:
 path=Path(args.fit);config=json.loads((path/'training.json').read_text());model=Network(len(ctx['core_users']),nitems,args.model,dtype=config['dtype'],layout=config['layout']);w=np.load(path/'weights.npz');model.p={k:w[k] for k in w.files}
cache=np.load(args.rank_cache)['ranks'] if args.rank_cache else None
rows=[];slots=[];counts={k:np.zeros(nitems,dtype=np.int64) for k in [10,20,30]}
for j,u in enumerate(users):
 if cache is not None:rec=[items[t] for t in cache[j]]
 elif args.model in ['BPR-MF','NCF']:
  sc=np.concatenate([model.score(umap[u],st,min(st+8192,nitems)) for st in range(0,nitems,8192)])
  sc[ctx['seen_indices'][u]]=-np.inf;ix=np.lexsort((np.arange(nitems),-sc))[:30];rec=[items[t] for t in ix]
 else:rec=model.recommend(u,ctx['seen'][u],k=30)
 assert len(rec)==30 and len(set(rec))==30 and not(set(rec)&ctx['seen'][u]) and set(rec)<=iset
 target=ctx['targets'][u];novel=ctx['novel'][u];nc=nitems-len(ctx['seen_indices'][u]);hist=H.loc[u].to_numpy()
 for rank,i in enumerate(rec,1):slots.append((args.model,u,rank,i,segments[i],categories[i],i in target))
 for K in [10,20,30]:
  sub=rec[:K];ii=np.array([ismap[i] for i in sub]);counts[K][ii]+=1;hit=np.array([i in target for i in sub]);h=int(hit.sum());ranks=np.flatnonzero(hit)+1
  ideal=np.sum(1/np.log2(np.arange(1,min(K,len(target))+1)+1));cg=np.bincount(ci[ii],minlength=len(catnames));mix=cg/K;nonzero=mix[mix>0]
  rows.append({'model':args.model,U:u,'K':K,'hits':h,'rankable_targets':len(target),'novel_targets':len(novel),'candidates':nc,'precision':h/K,'recall':h/len(target),'end_to_end_recall':h/len(novel),'ndcg':float(np.sum(1/np.log2(ranks+1))/ideal),'mrr':float(1/ranks[0]) if h else 0.,'hit_rate':int(h>0),'accuracy':(2*h+nc-K-len(target))/nc,'head_share':np.mean([segments[i]=='head' for i in sub]),'mid_share':np.mean([segments[i]=='mid' for i in sub]),'tail_share':np.mean([segments[i]=='tail' for i in sub]),'rank_tail_share':np.mean([i in ranktail for i in sub]),'category_count':int((cg>0).sum()),'category_entropy':float(-np.sum(nonzero*np.log(nonzero))/np.log(K)),'dominant_category_share':float(mix.max()),'category_ild':float(1-np.sum(cg*(cg-1))/(K*(K-1))),'category_tv':float(.5*np.abs(mix-hist).sum()),'new_category_share':float(mix[hist==0].sum())})
 if (j+1)%500==0:print(args.model,j+1,'buyers',round(time.time()-start,1),'s',flush=True)
slug=args.model.replace(' ','_');dest=O/'by_model'/slug;dest.mkdir(parents=True,exist_ok=True)
def atomic_csv(frame,path):
 tmp=path.with_name(path.name+'.tmp');frame.to_csv(tmp,index=False,compression='gzip' if str(path).endswith('.gz') else None)
 with tmp.open('rb') as f:os.fsync(f.fileno())
 if str(path).endswith('.gz'):gzip.decompress(tmp.read_bytes())
 os.replace(tmp,path)
pu=pd.DataFrame(rows);sl=pd.DataFrame(slots,columns=['model',U,'rank',I,'htb_segment','primary_category','relevant']);atomic_csv(pu,dest/'per_buyer_metrics.csv.gz');atomic_csv(sl,dest/'recommendation_slots.csv.gz')
metriccols=[c for c in pu if c not in ['model',U,'K']];summ=pu.groupby(['model','K'])[metriccols].mean().reset_index();summ['coverage']=[np.mean(counts[k]>0) for k in summ.K];summ['hits_total']=[int(pu[pu.K==k].hits.sum()) for k in summ.K];summ.to_csv(dest/'summary.csv',index=False)
ex=[]
for K in [10,20,30]:
 obs=np.bincount(ci,weights=counts[K]/(K*len(users)),minlength=len(catnames));hist=H.mean(axis=0).to_numpy()
 for j,c in enumerate(catnames):ex.append({'model':args.model,'K':K,'category':c,'recommendation_share':obs[j],'candidate_share':availability[j],'historical_share':hist[j],'gap_pp':100*(obs[j]-availability[j])})
pd.DataFrame(ex).to_csv(dest/'category_exposure.csv',index=False)
groups=pu.merge(ctx['profiles'][['tail_group','interest_group']],left_on=U,right_index=True,validate='many_to_one');groups.groupby(['model','K','tail_group'],observed=True)[metriccols].mean().reset_index().to_csv(dest/'tail_terciles.csv',index=False);groups.groupby(['model','K','interest_group'],observed=True)[metriccols].mean().reset_index().to_csv(dest/'fairness_groups.csv',index=False)
(dest/'verification.json').write_text(json.dumps({'model':args.model,'evaluated_buyers':len(users),'unique_30_item_lists':True,'seen_excluded':True,'shared_training_catalog':True,'seconds':time.time()-start},indent=2));print(summ.to_string(index=False),flush=True)
