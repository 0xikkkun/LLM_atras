"""Audit real Value projections and the official likelihood summation on smoke models."""
import argparse,gc,json,os,hashlib,importlib.util
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from llm_atras.extraction import extract_z_w
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-code',type=Path);a=p.parse_args();models=json.loads((a.output/'models.json').read_text());cfg=json.loads((a.output/'config.json').read_text());row=json.loads((a.output/'texts_smoke.json').read_text())[0];results=[]
source=a.reference_code or Path(__file__).with_name('official_likelihood.py')
if not source.exists():raise FileNotFoundError('Provide the pinned author likelihood.py using --reference-code')
spec=importlib.util.spec_from_file_location('author_likelihood',source);author_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(author_module)
for index in cfg['smoke_models']:
 e=models[index];t=AutoTokenizer.from_pretrained(e['snapshot'],local_files_only=True);m=AutoModelForCausalLM.from_pretrained(e['snapshot'],local_files_only=True,dtype=torch.float16,attn_implementation='eager').to('cuda').eval();m.config.use_cache=False
 typ=m.config.model_type
 if typ=='gpt2':target=m.transformer.h[-1].attn.c_attn
 elif typ=='gpt_neox':target=m.gpt_neox.layers[-1].attention.query_key_value
 else:target=m.model.layers[-1].self_attn.v_proj
 captured=[];handle=target.register_forward_hook(lambda mod,args,out:captured.append(out.detach().clone()))
 ids=torch.tensor([[t.bos_token_id if t.bos_token_id is not None else t.eos_token_id]+t.encode(row['text'],add_special_tokens=False)],device='cuda')
 with torch.inference_mode():
  V,w,out=extract_z_w(m,ids);handle.remove();actual=captured[0].squeeze(0)
  if typ=='gpt2':actual=actual[:,2*m.config.hidden_size:]
  elif typ=='gpt_neox':
   hs=m.gpt_neox.layers[-1].attention.head_size;actual=actual.view(len(actual),-1,3*hs)[...,2*hs:].reshape(len(actual),-1)
  else:
   nh=m.config.num_attention_heads;nk=getattr(m.config,'num_key_value_heads',nh)
   if nk<nh:actual=actual.view(len(actual),nk,-1).repeat_interleave(nh//nk,1).reshape(len(actual),-1)
  error=float((V-actual.float()).abs().max());assert error==0,(e['id'],error)
  default=torch.tensor([t.encode(row['text'],add_special_tokens=True)],device='cuda');ref=m(default,output_attentions=False,output_hidden_states=False);lp=torch.log_softmax(ref.logits,dim=-1)
  official=sum(float(lp[0,i,default[0,i+1]]) for i in range(default.shape[1]-1))
  calculator=author_module.LikelihoodCalculator.__new__(author_module.LikelihoodCalculator);calculator.model=m;calculator.input_device=torch.device('cuda');calculator._use_cache=False
  author_ll=calculator._calculate_likelihood([(0,default)])[0]['log-likelihood'];assert official==author_ll
  saved=np.load(a.output/'smoke/features'/e['id'].replace('/','__')/'raw/000.npz');delta=abs(official-float(saved['ll_reference']));assert delta==0,(e['id'],delta)
  results.append(dict(id=e['id'],official_code_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),value_max_abs_error=error,reference_likelihood_abs_error=delta,default_tokens=int(default.shape[1]),uniform_bos_tokens=int(ids.shape[1])))
 del calculator,m,ref,out,V,w,actual,captured,saved,lp;gc.collect();torch.cuda.empty_cache()
(a.output/'smoke/reference_validation.json').write_text(json.dumps(results,indent=2));print('REFERENCE_VALIDATION_OK',results,flush=True)
