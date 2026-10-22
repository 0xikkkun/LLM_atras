"""Checkpointed full-300-node approximate GW, independent CPU processes."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,concurrent.futures,itertools,json,time
from pathlib import Path
import numpy as np
from llm_atras.core import normalize

def initialize(root):
 global matrices
 root=Path(root);entries=json.loads((root/'models.json').read_text());base=root/'main/analysis/graphs'
 matrices={w:[normalize(np.load(base/(e['id'].replace('/','__')+'_'+w+'.npz'))['S']) for e in entries] for w in ['euclidean','fisher']}

def solve(item):
 import ot
 w,i,j=item;ss=matrices[w];p=np.ones(len(ss[0]))/len(ss[0]);values=[]
 for u,v in [(i,j),(j,i)]:
  cost=ot.gromov.gromov_wasserstein2(ss[u],ss[v],p,p,loss_fun='square_loss',symmetric=True,max_iter=100,tol_rel=1e-7,tol_abs=1e-8)
  values.append(float(cost))
 return w,dict(i=i,j=j,forward=values[0],reverse=values[1])

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=a.output/'main/analysis';entries=json.loads((a.output/'models.json').read_text());tasks=[]
 cache=out/'gw_checkpoints';cache.mkdir(exist_ok=True)
 for w in ['euclidean','fisher']:
  for i,j in itertools.combinations(range(len(entries)),2):
   if not (cache/f'{w}_{i:03d}_{j:03d}.json').exists():tasks.append((w,i,j))
 start=time.time();print('GW_PENDING_PAIRS',len(tasks),flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(a.output),)) as pool:
  futures=[pool.submit(solve,item) for item in tasks]
  for n,f in enumerate(concurrent.futures.as_completed(futures),1):
   w,row=f.result();path=cache/f'{w}_{row["i"]:03d}_{row["j"]:03d}.json';path.write_text(json.dumps(row,allow_nan=False))
   if n%20==0:print('GW_PROGRESS',n,len(tasks),'seconds',round(time.time()-start),flush=True)
 for w in ['euclidean','fisher']:
  rows=[json.loads((cache/f'{w}_{i:03d}_{j:03d}.json').read_text()) for i,j in itertools.combinations(range(len(entries)),2)]
  (out/f'gw_solver_{w}.json').write_text(json.dumps(rows,allow_nan=False))
 print('GW_COMPLETE',flush=True)
if __name__=='__main__':main()
