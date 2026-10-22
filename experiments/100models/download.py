"""Prefetch pinned safetensors while shared-text preparation runs."""
import argparse,concurrent.futures,json,os,time,traceback
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='0'
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def one(path):
 e=json.loads(path.read_text());s=Path(e['snapshot']);index=s/'model.safetensors.index.json'
 if (s/'pytorch_model.bin').is_file():return
 bi=s/'pytorch_model.bin.index.json'
 if bi.exists() and all((s/v).is_file() for v in set(json.loads(bi.read_text())['weight_map'].values())):return
 if (s/'model.safetensors').is_file():return
 if index.is_file() and all((s/v).is_file() for v in set(json.loads(index.read_text())['weight_map'].values())):return
 available={r['id']:r for r in json.loads((a.output/'weight_availability.json').read_text())}
 weights=['*.safetensors'] if available[e['id']]['safetensors'] else ['*.bin']
 print('DOWNLOAD',e['id'],flush=True)
 try:
  snapshot_download(e['id'],revision=e['revision'],cache_dir=str(s.parents[2]),allow_patterns=weights+['*.json','*.model','*.txt','*.tiktoken','*.jinja'],max_workers=2)
  print('DOWNLOADED',e['id'],flush=True)
 except Exception:print('DOWNLOAD_FAILED',e['id'],traceback.format_exc(),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,sorted((a.output/'metadata').glob('*.json'))))
print('PREFETCH_FINISHED',flush=True)
