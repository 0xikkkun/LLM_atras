"""Print a compact machine-readable experiment status."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();root=a.output
meta=[json.loads(p.read_text()) for p in (root/'metadata').glob('*.json')]
def ready(e):
 s=Path(e['snapshot'])
 if any((s/n).is_file() for n in ['model.safetensors','pytorch_model.bin']):return True
 for n in ['model.safetensors.index.json','pytorch_model.bin.index.json']:
  if (s/n).is_file() and all((s/v).is_file() for v in set(json.loads((s/n).read_text())['weight_map'].values())):return True
 return False
r=dict(metadata_models=len(meta),weights_ready=sum(ready(e) for e in meta),prepared=(root/'config.json').exists(),smoke_features=len(list((root/'smoke/features').glob('*/complete.json'))),main_features=len(list((root/'main/features').glob('*/complete.json'))),reference_validated=(root/'smoke/reference_validation.json').exists(),primary_complete=(root/'primary_complete.json').exists(),full_complete=(root/'full_complete.json').exists(),gw_pairs_complete=len(list((root/'main/analysis/gw_checkpoints').glob('*.json'))),updated_unix=time.time())
print(json.dumps(r,indent=2))
