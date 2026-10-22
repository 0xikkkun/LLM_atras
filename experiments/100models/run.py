"""Resume-safe four-GPU queue; no partial cohort silently enters the analysis."""
import argparse, concurrent.futures, json, os, queue, subprocess, sys, time, traceback
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='0'
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--cache',required=True);p.add_argument('--stage',choices=['smoke','main'],required=True);a=p.parse_args()
models=json.loads((a.output/'models.json').read_text());cfg=json.loads((a.output/'config.json').read_text());indices=cfg['smoke_models'] if a.stage=='smoke' else list(range(len(models)))
from importlib.metadata import version
(a.output/'execution_environment.json').write_text(json.dumps({p:version(p) for p in ['torch','transformers','numpy','scipy','scikit-learn','huggingface-hub']},indent=2))
logs=a.output/a.stage/'logs';logs.mkdir(parents=True,exist_ok=True)
def complete(e):return a.output/a.stage/'features'/e['id'].replace('/','__')/'complete.json'
def download(e):
 s=Path(e['snapshot']);ix=s/'model.safetensors.index.json'
 if (s/'pytorch_model.bin').is_file():return
 bi=s/'pytorch_model.bin.index.json'
 if bi.exists() and all((s/v).is_file() for v in set(json.loads(bi.read_text())['weight_map'].values())):return
 if (s/'model.safetensors').is_file():return
 if ix.exists() and all((s/v).is_file() for v in set(json.loads(ix.read_text())['weight_map'].values())):return
 available={r['id']:r for r in json.loads((a.output/'weight_availability.json').read_text())}
 weights=['*.safetensors'] if available[e['id']]['safetensors'] else ['*.bin']
 for attempt in range(3):
  try:
   dest=snapshot_download(e['id'],revision=e['revision'],cache_dir=a.cache,
    allow_patterns=weights+['*.json','*.model','*.txt','*.tiktoken','*.jinja'],max_workers=2)
   if Path(dest)!=s:raise RuntimeError(('Snapshot changed',dest,str(s)))
   return
  except Exception:
   if attempt==2:raise
   time.sleep(5)
q=queue.Queue()
for i in indices:
 if not complete(models[i]).exists():q.put(i)
def worker(gpu):
 while True:
  try:i=q.get_nowait()
  except queue.Empty:return
  e=models[i];status=a.output/a.stage/f'worker_gpu{gpu}.json'
  try:
   status.write_text(json.dumps(dict(index=i,id=e['id'],phase='download',time=time.time())))
   print('START',a.stage,gpu,i,e['id'],flush=True);download(e)
   status.write_text(json.dumps(dict(index=i,id=e['id'],phase='extract',time=time.time())))
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
   with (logs/f'extract_{i:03d}.log').open('a') as f:
    subprocess.run([sys.executable,'-m','llm_atras','extract','--output',str(a.output),'--stage',a.stage,'--index',str(i)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   print('DONE',a.stage,gpu,i,e['id'],flush=True)
   status.write_text(json.dumps(dict(index=i,id=e['id'],phase='complete',time=time.time())))
  except Exception:
   text=traceback.format_exc();(logs/f'error_{i:03d}.log').write_text(text);print('FAILED',i,e['id'],text,flush=True)
  finally:q.task_done()
with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus.split(','))) as pool:
 for f in [pool.submit(worker,g) for g in a.gpus.split(',')]:f.result()
missing=[models[i]['id'] for i in indices if not complete(models[i]).exists()]
if missing:
 (a.output/a.stage/'incomplete.json').write_text(json.dumps(missing,indent=2));raise RuntimeError(f'{len(missing)} models incomplete; see incomplete.json and logs')
print('ALL_FEATURES_COMPLETE',a.stage,len(indices),flush=True)
env=dict(os.environ,HF_HUB_OFFLINE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
# Primary matched-cohort comparisons finish first; full 300-node GW is a separate costly phase.
subprocess.run([sys.executable,'-m','llm_atras','analyze','--output',str(a.output),'--stage',a.stage,'--quick'],env=env,check=True)
if a.stage=='smoke':
 validation=Path(__file__).with_name('validate_100.py')
 if not validation.exists():validation=Path(__file__).with_name('validate.py')
 subprocess.run([sys.executable,str(validation),'--output',str(a.output)],env=dict(env,CUDA_VISIBLE_DEVICES='0'),check=True)
print('PRIMARY_ANALYSIS_COMPLETE',a.stage,flush=True)

if a.stage=='main':
 report=Path(__file__).with_name('report_100.py')
 if not report.exists():report=Path(__file__).with_name('report.py')
 subprocess.run([sys.executable,str(report),'--output',str(a.output)],env=env,check=True)
 (a.output/'primary_complete.json').write_text(json.dumps(dict(models=len(indices),texts=300,time=time.time())))
 gw=Path(__file__).with_name('gw_100.py')
 if not gw.exists():gw=Path(__file__).with_name('gw.py')
 with (a.output/'gw_run.log').open('a') as log:
  subprocess.run([sys.executable,'-u',str(gw),'--output',str(a.output),'--workers','4'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([sys.executable,'-m','llm_atras','analyze','--output',str(a.output),'--stage','main'],env=env,check=True)
 subprocess.run([sys.executable,str(report),'--output',str(a.output)],env=env,check=True)
 (a.output/'full_complete.json').write_text(json.dumps(dict(models=len(indices),texts=300,time=time.time())))
 print('FULL_ANALYSIS_WITH_GW_COMPLETE',flush=True)
