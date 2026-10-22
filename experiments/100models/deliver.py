"""Copy completed outputs over SSH without copying raw extraction checkpoints."""
import argparse,json,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--remote',required=True);p.add_argument('--local',required=True,type=Path);a=p.parse_args();a.local.mkdir(parents=True,exist_ok=True)
last=None
for _ in range(1440):
 markers=[]
 for name in ['primary_complete.json','full_complete.json']:
  r=subprocess.run(['ssh','-o','BatchMode=yes',a.host,'test -f '+shlex.quote(a.remote+'/'+name)],capture_output=True)
  if r.returncode==0:markers.append(name)
 state=tuple(markers)
 if state and state!=last:
  subprocess.run(['rsync','-az','--exclude=*/features/','--exclude=metadata/','--exclude=*.log','--exclude=gw_checkpoints/',a.host+':'+a.remote+'/',str(a.local)+'/'],check=True)
  (a.local/'delivery_status.json').write_text(json.dumps(dict(markers=markers,updated=time.time()),indent=2));print('DELIVERED',markers,flush=True);last=state
 if 'full_complete.json' in markers:break
 time.sleep(60)
else:raise RuntimeError('Delivery timeout; experiment may still be running remotely')
