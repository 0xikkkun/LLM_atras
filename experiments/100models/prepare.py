"""Freeze metadata/tokenizers for 100 models, then prepare shared 300 texts."""
import argparse, concurrent.futures, hashlib, json, os, time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='0'
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--inventory',type=Path);p.add_argument('--source',type=Path,required=True);p.add_argument('--cache',required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
models=json.loads(a.manifest.read_text())['models']; inventory={m['id']:m for m in json.loads(a.inventory.read_text())} if a.inventory else {}
def one(item):
 name=item['id'];path=a.output/'metadata'/(name.replace('/','__')+'.json');path.parent.mkdir(exist_ok=True)
 if path.exists():
  cached=dict(json.loads(path.read_text()), **item)
  t=AutoTokenizer.from_pretrained(cached['snapshot'],local_files_only=True)
  if t.vocab_size>1000 and len(t.encode('Hello world',add_special_tokens=False))>0:return cached
 old=inventory.get(name)
 if old and item.get('revision') and Path(old['snapshot']).name!=item['revision']:old=None
 if old:
  snapshot=old['snapshot'];revision=Path(snapshot).name
  snapshot=snapshot_download(name,revision=revision,cache_dir=str(Path(snapshot).parents[2]),allow_patterns=['*.json','*.model','*.txt','*.tiktoken','*.jinja'],max_workers=2)
 else:
  revision=item.get('revision') or HfApi().model_info(name).sha
  snapshot=snapshot_download(name,revision=revision,cache_dir=a.cache,allow_patterns=['*.json','*.model','*.txt','*.tiktoken','*.jinja'],max_workers=2)
 config=json.loads((Path(snapshot)/'config.json').read_text())
 if config['model_type'] not in ['llama','mistral','qwen2','gpt2','gpt_neox']:raise ValueError((name,config['model_type']))
 tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
 assert tok.vocab_size>1000 and len(tok.encode('Hello world',add_special_tokens=False))>0, name
 assert tok.bos_token_id is not None or tok.eos_token_id is not None
 result=dict(**item,snapshot=snapshot,revision=revision,model_type=config['model_type'])
 path.write_text(json.dumps(result,indent=2));print('METADATA_OK',name,flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: entries=list(pool.map(one,models))
for i,e in enumerate(entries):e.update(index=i,label=e['id'].split('/')[-1])
toks=[AutoTokenizer.from_pretrained(e['snapshot'],local_files_only=True) for e in entries]
rows=json.loads(a.source.read_text());prepared=[]
for row in rows:
 full=row['text'];lo,hi=0,len(full)
 if all(max(len(t.encode(full,add_special_tokens=False))+1,len(t.encode(full,add_special_tokens=True)))<=512 for t in toks):lo=hi
 while lo<hi:
  mid=(lo+hi+1)//2
  if all(max(len(t.encode(full[:mid],add_special_tokens=False))+1,len(t.encode(full[:mid],add_special_tokens=True)))<=512 for t in toks):lo=mid
  else:hi=mid-1
 text=full[:lo];r=dict(row,text=text,sha256=hashlib.sha256(text.encode()).hexdigest(),bytes=len(text.encode()),token_lengths=[len(t.encode(text,add_special_tokens=False))+1 for t in toks]);assert min(r['token_lengths'])>1, (row['id'],lo,repr(text[:100]),[(e['id'],n) for e,n in zip(entries,r['token_lengths']) if n<=1])
 prepared.append(r)
 if len(prepared)%20==0:print('TEXTS_PREPARED',len(prepared),flush=True)
for name,data in [('models.json',entries),('texts_main.json',prepared),('texts_smoke.json',[r for c in sorted({r['category'] for r in prepared}) for r in [q for q in prepared if q['category']==c][:2]])]:
 (a.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False))
config=dict(seed=20261022,models=100,n=300,smoke_models=[1,15,34],max_tokens=512,pca_dim=16,k=20,alpha=1e-3,capture_reference_likelihood=True,dtype='float16',reference_tokenization='tokenizer.encode(add_special_tokens=True); shifted next-token likelihood; no forced BOS',reference_clipping='one global 2nd percentile before centering',reference_commit='194c5436ed01d16381b1b903d3e2feec916a534c',source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),texts_changed_from_15_model_run=sum(x['text']!=y['text'] for x,y in zip(rows,prepared)),protocol='method replication on shared 300 texts; not full 1018-model/10000-text paper reproduction')
(a.output/'config.json').write_text(json.dumps(config,indent=2));print('PREPARED',len(entries),len(prepared),'changed texts',config['texts_changed_from_15_model_run'],flush=True)
