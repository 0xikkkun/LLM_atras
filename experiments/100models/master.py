"""Wait for preparation, require smoke success, then launch the 100-model run."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--runner',type=Path,required=True);p.add_argument('--cache',required=True);a=p.parse_args()
for _ in range(720):
 if (a.output/'config.json').exists():break
 time.sleep(5)
else:raise RuntimeError('Preparation did not finish; see prepare.log')
for stage in ['smoke','main']:
 with (a.output/(stage+'_run.log')).open('a') as log:
  subprocess.run([sys.executable,'-u',str(a.runner),'--output',str(a.output),'--cache',a.cache,'--stage',stage,'--gpus','0,1,2' if stage=='smoke' else '0,1,2,3'],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('STAGE_COMPLETE',stage,flush=True)
(a.output/'primary_complete.json').write_text(json.dumps(dict(models=100,texts=300,time=time.time())))
