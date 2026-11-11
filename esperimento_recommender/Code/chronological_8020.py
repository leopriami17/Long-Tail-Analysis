"""Run the documented fixed experiment; no hyperparameter search or optimization."""
from pathlib import Path
import argparse,subprocess,sys,os,json
P=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--purchases',required=True);a.add_argument('--require-exact-reference',action='store_true');args=a.parse_args()
env=os.environ.copy();env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONHASHSEED':'42'})
def run(name,*options):subprocess.run([sys.executable,str(P/'Code'/name),*options],cwd=P,env=env,check=True)
run('prepare_context.py','--purchases',str(Path(args.purchases).resolve()))
for model in ['Global quantity','Category quantity','ItemKNN','BPR-MF','NCF']:
    options=['--model',model]
    if model in ['BPR-MF','NCF']:
        run('fit_latent.py','--kind',model)
        options+=['--fit',str(P/'Fits'/f'{model}_raw_float32_io')]
    run('evaluate.py',*options)
run('assemble_and_verify.py')
verification=json.loads((P/'Results/Chronological8020/verification.json').read_text())
print(json.dumps(verification,indent=2))
if args.require_exact_reference and (verification['point_mismatches'] or verification['bootstrap_quantiles_different']):
    sys.exit('The reconstructed results are not identical to every reported reference value. See compatibility_report.csv.')
