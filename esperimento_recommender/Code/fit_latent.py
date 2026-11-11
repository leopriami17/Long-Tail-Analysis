from pathlib import Path
import argparse,pickle,json,time,os,hashlib,platform
import numpy as np
from latent_numpy import Network,fit,gradient_checks
P=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--kind',required=True);a.add_argument('--order',default='raw');a.add_argument('--dtype',default='float32');a.add_argument('--layout',default='io');a.add_argument('--sigmoid',default='expit');a.add_argument('--square-first',action='store_true');a.add_argument('--tag',default='');a.add_argument('--label64',action='store_true');a.add_argument('--combine-user',action='store_true');a.add_argument('--state64',action='store_true');args=a.parse_args()
with (P/'prepared.pkl').open('rb') as f:ctx=pickle.load(f)
checks=gradient_checks();name=f'{args.kind}_{args.order}_{args.dtype}_{args.layout}'+('' if args.sigmoid=='expit' and not args.square_first else f'_{args.sigmoid}_sq{int(args.square_first)}')+('_label64' if args.label64 else '')+('_combined' if args.combine_user else '')+('_state64' if args.state64 else '')+args.tag;out=P/'Fits'/name;out.mkdir(parents=True,exist_ok=True)
m=Network(len(ctx['core_users']),len(ctx['core_items']),args.kind,dtype=args.dtype,layout=args.layout,sigmoid=args.sigmoid,label64=args.label64,combine_user=args.combine_user)
start=time.time()
def cb(row):print(name,row,flush=True)
logs=fit(m,ctx['orders'][args.order],len(ctx['core_items']),callback=cb,square_first=args.square_first,state64=args.state64)
with (out/'weights.tmp').open('wb') as f:
    np.savez_compressed(f,**m.p);f.flush();os.fsync(f.fileno())
os.replace(out/'weights.tmp',out/'weights.npz')
from threadpoolctl import threadpool_info
metadata={'kind':args.kind,'order':args.order,'dtype':args.dtype,'layout':args.layout,'sigmoid':args.sigmoid,'square_first':args.square_first,'label64':args.label64,'combine_user':args.combine_user,'state64':args.state64,'gradient_checks':checks,'logs':logs,'seconds':time.time()-start,'numpy_version':np.__version__,'python_version':platform.python_version(),'threadpools':threadpool_info(),'seed':42,'epochs':2,'learning_rate':.001,'batch_size':4096,'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('latent_numpy.py')]},'parameter_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in m.p.items()}}
(out/'training.tmp').write_text(json.dumps(metadata,indent=2));os.replace(out/'training.tmp',out/'training.json')
print('SAVED',out,flush=True)
