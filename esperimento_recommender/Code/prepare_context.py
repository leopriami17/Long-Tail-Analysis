"""Build and verify the chronological population, without fitting models."""
from pathlib import Path
import argparse,hashlib,json,pickle
import numpy as np,pandas as pd
from discovery_recommender import iterative_kcore_pairs,htb_three_class
U='Survey ResponseID';I='ASIN/ISBN (Product Code)';D='Order Date';Q='Quantity';C='Category';V='Purchase Price Per Unit'
P=Path(__file__).resolve().parents[1];O=P/'Results/Chronological8020';O.mkdir(parents=True,exist_ok=True)
def main(path):
 p=pd.read_csv(path,low_memory=False);p[D]=pd.to_datetime(p[D],errors='coerce')
 for col in [Q,V]:p[col]=pd.to_numeric(p[col].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False).str.strip(),errors='coerce')
 p=p.dropna(subset=[D,U,I,C]);p=p[(p[Q]>0)&(p[V]>0)&~p[C].astype('string').str.strip().isin({'GIFT_CARD','ABIS_GIFT_CARD','Ecard Gift Certificate'})]
 daily=p.groupby(D).size().sort_index();boundary=daily.cumsum().sub(.8*len(p)).abs().idxmin();assert str(boundary.date())=='2022-04-18'
 tr=p[p[D]<=boundary].copy();va=p[p[D]>boundary].copy()
 pairs=iterative_kcore_pairs(tr[[U,I]],5);users=sorted(pairs[U].unique());items=sorted(pairs[I].unique());iset=set(items)
 q=tr.groupby(I)[Q].sum().reindex(items);seg,breaks,_=htb_three_class(q)
 seen=tr.groupby(U)[I].agg(set).to_dict();future=va.groupby(U)[I].agg(set).to_dict();targets={u:(future.get(u,set())-seen[u])&iset for u in users};novel={u:future.get(u,set())-seen[u] for u in users};eligible=sorted(u for u in users if targets[u])
 assert (len(p),len(tr),len(va),len(users),len(items),len(pairs),len(eligible))==(1726977,1381836,345141,4507,30360,308854,4012)
 assert np.allclose(breaks,[15.197529644268775,35.58833397857788])
 cc=tr[tr[I].isin(iset)].groupby([I,C]).size().rename('n').reset_index().sort_values([I,'n',C],ascending=[True,False,True]);cats=cc.drop_duplicates(I).set_index(I)[C].reindex(items)
 tt=tr[tr[I].isin(iset)].groupby([I,'Title']).size().rename('n').reset_index().sort_values([I,'n','Title'],ascending=[True,False,True]);titles=tt.drop_duplicates(I).set_index(I).Title.reindex(items)
 cat=pd.DataFrame({I:items,'quantity':q.values,'segment':seg.reindex(items).values,'primary_category':cats.values,'title':titles.values});cat.assign(embedding_item_index=np.arange(len(items))).to_csv(O/'training_catalog.csv.gz',index=False)
 pd.DataFrame({'embedding_user_index':np.arange(len(users)),U:users}).to_csv(O/'training_users.csv',index=False)
 s=tr[tr[I].isin(iset)].copy();s['segment']=s[I].map(seg);s['primary']=s[I].map(cats)
 shares=pd.crosstab(s[U],s.segment,normalize='index').reindex(users).fillna(0)
 categories=pd.crosstab(s[U],s.primary).reindex(users).fillna(0);category_shares=categories.div(categories.sum(axis=1),axis=0);entropy=-(category_shares*np.log(category_shares.where(category_shares>0))).sum(axis=1)
 profile=shares.copy();profile.columns=[c+'_share' for c in profile.columns];profile['category_entropy']=entropy;profile['training_catalog_rows']=categories.sum(axis=1);profile.index.name=U
 # Equal-sized groups on evaluated users; code order resolves exact ties.
 ep=profile.loc[eligible].copy();ep['tail_group']=pd.qcut(ep.tail_share.rank(method='first'),3,labels=['Low','Middle','High']);ep['interest_group']=pd.qcut(ep.category_entropy.rank(method='first'),3,labels=['Narrow','Middle','Broad']);ep.to_csv(O/'buyer_history.csv')
 audit=[]
 for u in users:audit.append({U:u,'validation_rows':int((va[U]==u).sum()) if False else 0,'target_products':len(future.get(u,set())),'repeat_products':len(future.get(u,set())&seen[u]),'novel_targets':len(novel[u]),'rankable_targets':len(targets[u])})
 audit=pd.DataFrame(audit);audit['validation_rows']=audit[U].map(va[U].value_counts()).fillna(0).astype(int);audit.to_csv(O/'target_audit.csv',index=False)
 usmap={u:j for j,u in enumerate(users)};ismap={i:j for j,i in enumerate(items)}
 def arr(a):return np.column_stack([a[U].map(usmap),a[I].map(ismap)]).astype(np.int64)
 orders={'raw':arr(pairs)}
 provisional=tr[[U,I,Q,C]].copy();provisional['quantity']=provisional[Q];provisional['category']=provisional[C]
 ctx={'provisional_train':provisional,'core_pairs':pairs,'core_users':users,'core_items':items,'train_quantity_precore':q,'segment_map':seg.to_dict(),'popularity_breaks':breaks,'seen':seen,'train_end':'2022-04-19','eligible':eligible,'targets':targets,'novel':novel,'future':future,'categories':cats.to_dict(),'catalog_table':cat,'profiles':ep,'category_history':categories.loc[eligible],'orders':orders}
 # Availability reference for categories; every buyer is weighted equally.
 base=np.zeros(len(items));seen_ix={}
 for u in eligible:
  ix=np.array([ismap[i] for i in seen[u]&iset],dtype=np.int64);seen_ix[u]=ix;w=1/(len(items)-len(ix));base+=w;base[ix]-=w
 ctx['seen_indices']=seen_ix;ctx['availability_product_share']=base/len(eligible)
 with (P/'prepared.pkl').open('wb') as f:pickle.dump(ctx,f,protocol=5)
 manifest={'clean_rows':len(p),'training_rows':len(tr),'validation_rows':len(va),'training_fraction':len(tr)/len(p),'train_max':'2022-04-18','validation_min':'2022-04-19','validation_max':str(va[D].max().date()),'core_buyers':len(users),'catalog_products':len(items),'core_pairs':len(pairs),'evaluated_buyers':len(eligible),'breaks':breaks,'catalog_sha256':hashlib.sha256('\n'.join(items).encode()).hexdigest(),'buyer_sha256':hashlib.sha256('\n'.join(eligible).encode()).hexdigest(),'validation_used_for_tuning':False,'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--purchases',required=True);main(a.parse_args().purchases)
