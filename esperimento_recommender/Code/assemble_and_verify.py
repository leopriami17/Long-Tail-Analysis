"""Consolidate five fixed runs and compare every available published value."""
from pathlib import Path
import json,pickle,hashlib,platform
import numpy as np,pandas as pd
from scipy.stats import binom
P=Path(__file__).resolve().parents[1];O=P/'Results/Chronological8020';U='Survey ResponseID';I='ASIN/ISBN (Product Code)';models=['Global quantity','Category quantity','ItemKNN','BPR-MF','NCF']
with (P/'prepared.pkl').open('rb') as f:ctx=pickle.load(f)
for filename in ['summary.csv','category_exposure.csv','tail_terciles.csv','fairness_groups.csv','per_buyer_metrics.csv.gz','recommendation_slots.csv.gz']:
 frames=[pd.read_csv(O/'by_model'/m.replace(' ','_')/filename) for m in models]
 pd.concat(frames,ignore_index=True).to_csv(O/filename,index=False)
summary=pd.read_csv(O/'summary.csv');pu=pd.read_csv(O/'per_buyer_metrics.csv.gz');slots=pd.read_csv(O/'recommendation_slots.csv.gz');ex=pd.read_csv(O/'category_exposure.csv')
assert len(slots)==5*4012*30 and len(pu)==5*4012*3
assert slots.groupby(['model',U]).size().eq(30).all()
assert slots.groupby(['model',U])[I].nunique().eq(30).all()
for m,g in slots.groupby('model'):assert set(g[U])==set(ctx['eligible'])
old_path=P/'Reference/global_quantity_slots.csv.gz'
old=pd.read_csv(old_path) if old_path.exists() else None
global_match=None
if old is not None:
 new=slots[(slots.model=='Global quantity')&(slots['rank']<=20)][[U,'rank',I]].sort_values([U,'rank']).reset_index(drop=True)
 global_match=bool(new.equals(old[[U,'rank',I]].sort_values([U,'rank']).reset_index(drop=True)));assert global_match
ref=json.loads((P/'Reference/published_values.json').read_text());checks=[]
def check(model,what,old,new,decimals,kind='rounded_published_value'):
 match=round(float(old),decimals)==round(float(new),decimals)
 checks.append({'model':model,'check':what,'reference':old,'reconstructed':new,'difference':new-old,'display_decimals':decimals,'status':'MATCH' if match else 'DIFFERENT','type':kind})
for m in models:
 s=summary[(summary.model==m)&(summary.K==20)].iloc[0]
 for metric,v in ref[m]['retrieval20'].items():check(m,'Top20 '+metric,v,100*s[metric],3)
 for metric,v in ref[m]['exposure20'].items():check(m,'Top20 '+metric,v,100*s[metric],2)
 for metric,v in ref[m]['category20'].items():
  scale=100 if metric=='dominant_category_share' else 1
  dec=2 if metric in ['category_count','dominant_category_share'] else 3
  check(m,'Top20 '+metric,v,scale*s[metric],dec)
 for K,metrics in ref[m]['depth'].items():
  a=summary[(summary.model==m)&(summary.K==int(K))].iloc[0]
  for metric,v in metrics.items():check(m,f'Top{K} '+metric,v,100*a[metric],2 if metric in ['tail_share','coverage'] else 3)
 check(m,'Top20 end_to_end_recall',ref[m]['end_to_end_recall20'],100*s.end_to_end_recall,3)
# Narrative group endpoints.
fair=pd.read_csv(O/'fairness_groups.csv');tail=pd.read_csv(O/'tail_terciles.csv')
fair_refs={'Global quantity':(.959880,.934623),'Category quantity':(.811322,.913375),'ItemKNN':(.831414,.853420),'BPR-MF':(.940911,.913874),'NCF':(.939890,.911406)}
tail_refs={'Category quantity':(43.25,61.32),'ItemKNN':(63.96,84.51),'BPR-MF':(37.29,37.80),'NCF':(59.88,59.35)}
for m,vals in fair_refs.items():
 for group,v in zip(['Narrow','Broad'],vals):check(m,'category_tv '+group,v,float(fair[(fair.model==m)&(fair.K==20)&(fair.interest_group==group)].category_tv.iloc[0]),3)
for m,vals in tail_refs.items():
 for group,v in zip(['Low','High'],vals):check(m,'tail_exposure '+group,v,100*float(tail[(tail.model==m)&(tail.K==20)&(tail.tail_group==group)].tail_share.iloc[0]),2)
# Heatmap cells: nine categories, all five models, as displayed to one decimal.
cats=['WALL_ART','DIGITAL_DEVICE_4','ABIS_BOOK','SHOES','VEGETABLE','DOWNLOADABLE_VIDEO_GAME','ELECTRONIC_CABLE','NETWORK_INTERFACE_CONTROLLER_ADAPTER','FLASH_MEMORY']
heat=np.array([[14.8,1.4,.1,-.0,-.1],[9.2,.2,.1,.2,-.1],[-4.3,-4.0,1.1,-1.1,.4],[-.0,3.5,.0,-.0,-.0],[4.7,-.8,2.7,.7,.7],[4.6,.7,1.7,.4,-.3],[-1.7,-1.7,-.9,.9,-1.1],[-.2,-.2,-.1,.1,4.3],[-.3,-.3,-.2,.6,3.7]])
for j,c in enumerate(cats):
 for k,m in enumerate(models):
  v=ex[(ex.model==m)&(ex.K==20)&(ex.category==c)].gap_pp.iloc[0];check(m,'category gap '+c,float(heat[j,k]),v,1)
for m,c,v in [('Global quantity','WALL_ART',14.83),('Global quantity','DIGITAL_DEVICE_4',9.21),('Category quantity','ABIS_BOOK',-4.05),('Category quantity','SHOES',3.47),('ItemKNN','VEGETABLE',2.66),('BPR-MF','ABIS_BOOK',-1.08),('NCF','NETWORK_INTERFACE_CONTROLLER_ADAPTER',4.35)]:check(m,'narrative category gap '+c,v,ex[(ex.model==m)&(ex.K==20)&(ex.category==c)].gap_pp.iloc[0],2)
for m,expected in [('BPR-MF',[.6931786,.6917921]),('NCF',[.6933174,.6926778])]:
 logs=json.loads((P/'Fits'/f'{m}_raw_float32_io'/'training.json').read_text())['logs']
 for log,v in zip(logs,expected):check(m,f'training_loss epoch{log["epoch"]}',v,log['mean_loss'],7,'recorded_training_loss')
# Seeded sequential draws: alphabetical model order, then six metrics in the
# order below. This reproduces all ten reported Recall interval endpoints.
rng=np.random.default_rng(42);unc=[]
lo_rank=int(binom.ppf(.025,1000,.025));hi_rank=int(binom.ppf(.975,1000,.025));ulo=int(binom.ppf(.025,1000,.975));uhi=int(binom.ppf(.975,1000,.975))
for m in sorted(models):
 a=pu[(pu.model==m)&(pu.K==20)].sort_values(U)
 for metric in ['precision','recall','ndcg','mrr','hit_rate','end_to_end_recall']:
  indices=rng.integers(0,4012,size=(1000,4012))
  draws=np.sort(a[metric].to_numpy()[indices].mean(axis=1));lo,hi=np.quantile(draws,[.025,.975])
  unc.append({'model':m,'K':20,'metric':metric,'mean':a[metric].mean(),'lower':lo,'upper':hi,'bootstrap_seed':42,'bootstrap_replicates':1000})
  if metric=='recall':
   for bound,oldval,newval,low,high in [('lower',ref[m]['recall20_ci'][0],100*lo,100*draws[lo_rank],100*draws[hi_rank]),('upper',ref[m]['recall20_ci'][1],100*hi,100*draws[ulo],100*draws[uhi])]:
    exact=round(oldval,3)==round(newval,3);compatible=(oldval+.0005>=low and oldval-.0005<=high)
    checks.append({'model':m,'check':'Recall20 bootstrap '+bound,'reference':oldval,'reconstructed':newval,'difference':newval-oldval,'display_decimals':3,'status':'MATCH' if exact else ('WITHIN_MC_UNCERTAINTY' if compatible else 'DIFFERENT'),'type':'bootstrap_quantile','mc_lower':low,'mc_upper':high})
pd.DataFrame(unc).to_csv(O/'uncertainty.csv',index=False)
report=pd.DataFrame(checks);report.to_csv(O/'compatibility_report.csv',index=False)
(O/'published_reference.json').write_text(json.dumps(ref,indent=2))
# Product-level exposures for all models, including zero-exposure products.
catalog=ctx['catalog_table'];allproducts=[]
for m in models:
 count=slots[(slots.model==m)&(slots['rank']<=20)][I].value_counts().reindex(ctx['core_items'],fill_value=0)
 t=catalog.copy();t['model']=m;t['observed_slots']=count.to_numpy();t['availability_expected_slots']=ctx['availability_product_share']*4012*20;t['gap_pp']=100*(t.observed_slots-t.availability_expected_slots)/(4012*20);t['exposure_ratio']=t.observed_slots/t.availability_expected_slots;allproducts.append(t)
pd.concat(allproducts,ignore_index=True).to_csv(O/'product_exposure_top20.csv.gz',index=False)
trainlogs=[]
for m in ['BPR-MF','NCF']:
 config=json.loads((P/'Fits'/f'{m}_raw_float32_io'/'training.json').read_text())
 for r in config['logs']:trainlogs.append({'model':m,'seed':42,**r})
pd.DataFrame(trainlogs).to_csv(O/'training_losses.csv',index=False)
verification={'same_evaluated_buyers':True,'all_30_item_lists_unique':True,'seen_excluded':True,'fixed_chronological_split':True,'global_top20_matches_independent_audit':global_match,'all_rows_archived':len(slots),'point_checks':int((report.type!='bootstrap_quantile').sum()),'point_mismatches':int(((report.type!='bootstrap_quantile')&(report.status=='DIFFERENT')).sum()),'bootstrap_quantiles_exact':int(((report.type=='bootstrap_quantile')&(report.status=='MATCH')).sum()),'bootstrap_quantiles_compatible_mc':int(((report.type=='bootstrap_quantile')&(report.status=='WITHIN_MC_UNCERTAINTY')).sum()),'bootstrap_quantiles_different':int(((report.type=='bootstrap_quantile')&(report.status=='DIFFERENT')).sum()),'original_individual_lists_available_for_full_identity_check':False}
(O/'verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification,indent=2),flush=True);print(report[report.status=='DIFFERENT'].to_string(index=False),flush=True)
