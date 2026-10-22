"""Record which pinned model revisions supply safetensors or PyTorch weights."""
import argparse,concurrent.futures,json,os
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='0'
from huggingface_hub import HfApi
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def one(path):
 e=json.loads(path.read_text());info=HfApi().model_info(e['id'],revision=e['revision']);files=[s.rfilename for s in info.siblings]
 r=dict(id=e['id'],revision=e['revision'],safetensors=[f for f in files if f.endswith('.safetensors') and '/' not in f],bin=[f for f in files if f.endswith('.bin') and '/' not in f])
 if not r['safetensors'] and not r['bin']:raise ValueError('No supported weights: '+e['id'])
 return r
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,(a.output/'metadata').glob('*.json')))
(a.output/'weight_availability.json').write_text(json.dumps(rows,indent=2));print('WEIGHTS_AUDITED',len(rows))
