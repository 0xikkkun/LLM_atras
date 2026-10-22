"""Reproducible LLM maps: extract once, compare distances, compare reductions.

All model loads are offline and use complete local safetensors snapshots.
"""
import argparse
import csv
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np

from .core import (upper, normalize, pairwise_from_features, cka_distance,
                   graph_distances, rank_distance, knn_distance, netlsd_vector,
                   classical, align, assess, metric_audit)

ROOT = Path(os.environ.get('LLM_ATRAS_OUTPUT', 'artifacts'))
SOURCE = None
CACHES = [Path(os.environ.get('HF_HUB_CACHE',
              str(Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface'))) / 'hub')))]
MODELS = [
    ('distilgpt2', 'GPT2'), ('gpt2', 'GPT2'), ('gpt2-medium', 'GPT2'),
    ('gpt2-large', 'GPT2'), ('gpt2-xl', 'GPT2'),
    ('NousResearch/Llama-2-7b-hf', 'Llama2'),
    ('NousResearch/Llama-2-7b-chat-hf', 'Llama2'),
    ('NousResearch/Meta-Llama-3-8B', 'Llama3'),
    ('NousResearch/Meta-Llama-3-8B-Instruct', 'Llama3'),
    ('NousResearch/Meta-Llama-3.1-8B', 'Llama3.1'),
    ('NousResearch/Meta-Llama-3.1-8B-Instruct', 'Llama3.1'),
    ('mistralai/Mistral-7B-v0.1', 'Mistral'),
    ('mistralai/Mistral-7B-Instruct-v0.1', 'Mistral'),
    ('mistralai/Mistral-7B-v0.3', 'Mistral'),
    ('mistralai/Mistral-7B-Instruct-v0.3', 'Mistral'),
]
SMOKE = [1, 5, 11]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def table(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def slug(name):
    return name.replace('/', '__')


def snapshot(name):
    for root in CACHES:
        repo = root / ('models--' + name.replace('/', '--'))
        ref = repo / 'refs/main'
        try:
            preferred = [repo / 'snapshots' / ref.read_text().strip()] if ref.is_file() else []
            for p in preferred + sorted((repo / 'snapshots').glob('*')):
                index = p / 'model.safetensors.index.json'
                complete = (p / 'model.safetensors').is_file()
                if index.is_file():
                    shards = set(json.loads(index.read_text())['weight_map'].values())
                    complete |= bool(shards) and all((p / s).is_file() for s in shards)
                if complete and (p / 'config.json').is_file():
                    return str(p)
        except PermissionError:
            continue
    raise FileNotFoundError('No readable complete safetensors snapshot: ' + name)


def prepare():
    from transformers import AutoTokenizer
    import sklearn
    import transformers
    import torch
    source = Path(SOURCE) if SOURCE is not None else ROOT/'balanced_source.json'
    original = json.loads(source.read_text())
    labels = original['labels']
    cats = sorted(set(labels))
    if len(cats) != 15:
        raise ValueError(('Expected 15 categories', cats))
    entries = []
    toks = []
    for i, (name, family) in enumerate(MODELS):
        p = snapshot(name)
        tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
        toks.append(tok)
        entries.append(dict(index=i, id=name, label=name.split('/')[-1],
                            family=family, snapshot=p, revision=Path(p).name))
    rng = np.random.default_rng(20261022)
    rows = []
    for cat in cats:
        ix = np.flatnonzero(np.array(labels) == cat)
        rng.shuffle(ix)
        if len(ix) < 20:
            raise ValueError((cat, len(ix)))
        for idx in ix[:20]:
            full = original['texts'][int(idx)]
            lo, hi = 0, len(full)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if all(len(t.encode(full[:mid], add_special_tokens=False)) <= 511 for t in toks):
                    lo = mid
                else:
                    hi = mid - 1
            text = full[:lo]
            lengths = [len(t.encode(text, add_special_tokens=False)) + 1 for t in toks]
            assert max(lengths) <= 512 and min(lengths) >= 2
            rows.append(dict(id=f'text_{len(rows):03d}', category=cat,
                             source_index=int(idx), text=text,
                             sha256=hashlib.sha256(text.encode()).hexdigest(),
                             bytes=len(text.encode()), token_lengths=lengths))
    ROOT.mkdir(parents=True, exist_ok=True)
    dump(ROOT/'models.json', entries)
    dump(ROOT/'texts_main.json', rows)
    smoke = [r for cat in cats for r in [q for q in rows if q['category'] == cat][:2]]
    dump(ROOT/'texts_smoke.json', smoke)
    dump(ROOT/'config.json', dict(seed=20261022, models=len(entries), n=len(rows), smoke_models=SMOKE,
         max_tokens=512, pca_dim=16, k=20, alpha=1e-3, source=str(source),
         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         likelihood='BOS (or EOS fallback) then identical raw text; no EOS target',
         covariance='attention-weighted, same weights as eta',
         graph_normalization='mean upper-triangle shortest-path distance',
         alignment='center + orthogonal Procrustes; rotations-only and O(2) both saved; no scaling',
         versions=dict(torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__)))
    print('PREPARED', len(entries), len(rows), cats, flush=True)


def extract(stage, idx):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.decomposition import PCA
    from .extraction import extract_z_w
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(4)
    torch.manual_seed(20261022)
    entry = json.loads((ROOT/'models.json').read_text())[idx]
    config = json.loads((ROOT/'config.json').read_text())
    inference_dtype = config.get('dtype_overrides', {}).get(entry['id'], config.get('dtype', 'bfloat16'))
    capture_reference = config.get('capture_reference_likelihood', False)
    rows = json.loads((ROOT/f'texts_{stage}.json').read_text())
    dest = ROOT/stage/'features'/slug(entry['id'])
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        print('SKIP COMPLETE', stage, entry['id'], flush=True)
        return
    start = time.time()
    load_path = entry['snapshot']
    staged_path = None
    if config.get('local_stage_dir'):
        from .staging import stage_snapshot
        print('STAGING', entry['id'], flush=True)
        staged_path = stage_snapshot(load_path, config['local_stage_dir'], entry['id'])
        load_path = str(staged_path)
        print('STAGED', entry['id'], flush=True)
    tok = AutoTokenizer.from_pretrained(load_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(load_path,
        local_files_only=True, dtype=getattr(torch, inference_dtype),
        attn_implementation='eager').to('cuda').eval()
    model.config.use_cache = False
    eta, hidden, ll, ess, lengths, ll_reference = [], [], [], [], [], []
    raw = dest/'raw'
    raw.mkdir(exist_ok=True)
    for a, row in enumerate(rows):
        rp = raw/f'{a:03d}.npz'
        if rp.exists():
            d = np.load(rp)
            if capture_reference and 'll_reference' not in d:
                raise ValueError('Cached extraction lacks reference likelihood: ' + str(rp))
            eta.append(d['eta']); hidden.append(d['hidden']); ll.append(float(d['ll']))
            ess.append(float(d['ess'])); lengths.append(int(d['length']))
            if capture_reference: ll_reference.append(float(d['ll_reference']))
            continue
        ids = tok.encode(row['text'], add_special_tokens=False)
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        assert bos is not None and len(ids)+1 <= 512
        ids = torch.tensor([[bos]+ids], device='cuda')
        with torch.inference_mode():
            V, w, out = extract_z_w(model, ids)
            e = (w[:, None]*V).sum(0).cpu().numpy()
            h = out.hidden_states[-1][0, -1].float().cpu().numpy()
            loss = F.cross_entropy(out.logits[0, :-1].float(), ids[0, 1:], reduction='sum')
            l = -float(loss)
            extra = {}
            if capture_reference:
                # Follow the official calculator's default tokenizer special tokens.
                ref_ids = tok.encode(row['text'], add_special_tokens=True)
                # The author's shifted-token sum is zero for one-token sequences.
                assert 1 <= len(ref_ids) <= 512
                ref_ids = torch.tensor([ref_ids], device='cuda')
                ref_out = out if torch.equal(ids, ref_ids) else model(
                    input_ids=ref_ids, output_hidden_states=False, output_attentions=False)
                # Official code applies log_softmax in the model's float16 dtype,
                # then sums Python floats. Accumulate the gathered values in float64.
                logp = ref_out.logits[0, :-1].log_softmax(-1)
                ref_ll = float(logp.gather(1, ref_ids[0, 1:, None]).double().sum())
                assert np.isfinite(ref_ll)
                extra['ll_reference'] = ref_ll
                ll_reference.append(ref_ll)
                del ref_out, ref_ids, logp
            es = float(1/(w.square().sum()))
            assert np.isfinite(e).all() and np.isfinite(h).all() and np.isfinite(l)
            np.savez_compressed(rp, V=V.cpu().numpy(), w=w.cpu().numpy(), eta=e,
                                hidden=h, ll=l, ess=es, length=ids.shape[1], **extra)
            eta.append(e); hidden.append(h); ll.append(l); ess.append(es); lengths.append(ids.shape[1])
            del V, w, out, loss
        if (a+1) % 10 == 0:
            print(stage, entry['label'], a+1, '/',len(rows), 'seconds',round(time.time()-start), flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()
    eta, hidden = np.asarray(eta, dtype=np.float64), np.asarray(hidden, dtype=np.float64)
    pca = PCA(n_components=16, svd_solver='full').fit(eta)
    P = pca.components_.T
    coords = pca.transform(eta)
    cov, metrics, eigmin, conditions, regs = [], [], [], [], []
    for a in range(len(rows)):
        d = np.load(raw/f'{a:03d}.npz')
        z = d['V'].astype(np.float64) @ P
        w = d['w'].astype(np.float64); w /= w.sum()
        center = w @ z
        delta = z - center
        C = (delta*w[:, None]).T @ delta
        reg = 1e-3*np.trace(C)/16
        assert reg > 0
        G = np.linalg.inv(C+reg*np.eye(16))
        assert np.allclose(G,G.T,atol=1e-8)
        cov.append(C); metrics.append(G); eigmin.append(float(np.linalg.eigvalsh(G)[0]))
        conditions.append(float(np.linalg.cond(C+reg*np.eye(16)))); regs.append(reg)
    extra_features = {'ll_reference': ll_reference} if capture_reference else {}
    np.savez_compressed(dest/'features.npz', eta=eta, hidden=hidden, coords=coords,
        covariance=cov, metric=metrics, ll=ll, ess=ess, lengths=lengths,
        pca_projection=P, pca_mean=pca.mean_, pca_evr=pca.explained_variance_ratio_, regularization=regs,
        **extra_features)
    dump(dest/'complete.json', dict(**entry,n=len(rows), seconds=time.time()-start,
        torch_version=torch.__version__, dtype=inference_dtype,
        reference_likelihood=capture_reference,
        min_metric_eigenvalue=min(eigmin), max_covariance_condition=max(conditions),
        ess_min=min(ess),ess_median=float(np.median(ess)), pca_evr=float(pca.explained_variance_ratio_.sum())))
    print('EXTRACT_COMPLETE', stage, entry['label'], flush=True)
    if staged_path is not None:
        import shutil
        shutil.rmtree(staged_path)


















def make_distances(features, graphs, include_gw=True):
    from scipy.spatial.distance import cdist
    ll=np.array([f['ll'] for f in features]); ll-=ll.mean(1,keepdims=True)
    D={'loglik_centered':pairwise_from_features(ll)/np.sqrt(ll.shape[1]),
       'hidden_cka':cka_distance([f['hidden'] for f in features]),
       'eta_cka':cka_distance([f['eta'] for f in features]),
       'eta_knn':knn_distance([cdist(f['eta'],f['eta']) for f in features])}
    if all('ll_reference' in f for f in features):
        from .shimodaira import likelihood_coordinates
        reference = np.array([f['ll_reference'] for f in features])
        for clipping in [False, True]:
            L, Q, threshold = likelihood_coordinates(reference, clip=clipping)
            suffix = 'clipped' if clipping else 'raw'
            D['shimodaira_L_' + suffix] = pairwise_from_features(L)/np.sqrt(L.shape[1])
            D['shimodaira_Q_' + suffix] = pairwise_from_features(Q)/np.sqrt(Q.shape[1])
    for weight in ['euclidean','fisher']:
        ss=[g[weight]['S'] for g in graphs]
        ns=[normalize(s) for s in ss]
        D[weight+'_frob']=pairwise_from_features([upper(s) for s in ns])/np.sqrt(len(upper(ns[0])))
        D[weight+'_frob_raw']=pairwise_from_features([upper(s) for s in ss])/np.sqrt(len(upper(ss[0])))
        D[weight+'_rank']=rank_distance(ss)
        D[weight+'_knn']=knn_distance(ss)
        D[weight+'_netlsd']=pairwise_from_features([netlsd_vector(g[weight]) for g in graphs])
        if include_gw:
            import ot
            cached = CURRENT_OUT / f'gw_solver_{weight}.json'
            if cached.exists():
                diagnostics = json.loads(cached.read_text())
                if len(diagnostics) == len(ss)*(len(ss)-1)//2:
                    out=np.zeros((len(ss),len(ss)))
                    for row in diagnostics:
                        i,j=row['i'],row['j']
                        out[i,j]=out[j,i]=np.sqrt(max(0,(row['forward']+row['reverse'])/2))
                    D[weight+'_gw_approx']=out
                    continue
            out=np.zeros((len(ss),len(ss)))
            diagnostics=[]
            p=np.ones(len(ss[0]))/len(ss[0])
            for i,j in itertools.combinations(range(len(ss)),2):
                # Both argument orders: report and symmetrize solver discrepancies.
                values=[]
                for u,v in [(i,j),(j,i)]:
                    cost,log=ot.gromov.gromov_wasserstein2(ns[u],ns[v],p,p,
                        loss_fun='square_loss',symmetric=True,log=True,
                        max_iter=100,tol_rel=1e-7,tol_abs=1e-8)
                    values.append(float(cost))
                out[i,j]=out[j,i]=np.sqrt(max(0,np.mean(values)))
                diagnostics.append(dict(i=i,j=j,forward=values[0],reverse=values[1]))
            D[weight+'_gw_approx']=out
            dump(CURRENT_OUT/f'gw_solver_{weight}.json',diagnostics)
    return D










def plot_points(ax,Y,labels,categories,title,edges=None,text=False):
    import matplotlib.pyplot as plt
    cats=sorted(set(categories)); palette=plt.get_cmap('tab20')
    if edges is not None:
        from matplotlib.collections import LineCollection
        a,b=edges
        pairs=np.array([Y[[i,j]] for i,j in zip(a,b) if i<j])
        ax.add_collection(LineCollection(pairs,colors='#9ca3af',alpha=.20,linewidths=.35))
    for c,cat in enumerate(cats):
        ix=np.flatnonzero(np.asarray(categories)==cat)
        ax.scatter(Y[ix,0],Y[ix,1],s=18 if edges is not None else 48,
                   color=palette(c),label=cat,marker=['o','s','^','D','v'][c%5],edgecolors='white',linewidths=.3)
    if text:
        display_labels=[f'{i+1:03d}' for i in range(len(labels))] if len(labels)>=50 else labels
        annotations=[ax.text(x,y,row,fontsize=7) for row,(x,y) in zip(display_labels,Y)]
        from adjustText import adjust_text
        adjust_text(annotations,x=Y[:,0],y=Y[:,1],ax=ax,
                    arrowprops=dict(arrowstyle='-',color='#6b7280',lw=.5),iter_lim=80)
    ax.set_title(title,fontsize=10)
    ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('Axis 1');ax.set_ylabel('Axis 2')
    ax.spines[['top','right']].set_visible(False)


def analyze(stage, quick=False):
    global CURRENT_OUT
    np.random.seed(20261022)
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    from sklearn.manifold import MDS
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'figure.dpi':150,'savefig.dpi':180})
    entries=json.loads((ROOT/'models.json').read_text())
    if stage=='smoke':
        smoke_indices=json.loads((ROOT/'config.json').read_text())['smoke_models']
        entries=[entries[i] for i in smoke_indices]
    texts=json.loads((ROOT/f'texts_{stage}.json').read_text())
    cats=[r['category'] for r in texts]
    out=ROOT/stage/'analysis';CURRENT_OUT=out
    out.mkdir(parents=True,exist_ok=True)
    (out/'run_used.py').write_bytes(Path(__file__).read_bytes())
    from importlib.metadata import version
    dump(out/'provenance.json',dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={p:version(p) for p in ['numpy','scipy','scikit-learn','matplotlib','umap-learn','adjustText']},
        models=entries,text_ids=[r['id'] for r in texts],text_hashes=[r['sha256'] for r in texts]))
    for folder in ['graphs','distances','maps','heatmaps','csv','evaluation']:(out/folder).mkdir(exist_ok=True)
    features=[dict(np.load(ROOT/stage/'features'/slug(e['id'])/'features.npz')) for e in entries]
    if all('ll_reference' in f for f in features):
        from .shimodaira import likelihood_coordinates
        likelihood = np.array([f['ll_reference'] for f in features])
        reference_provenance = {}
        for clipped in [False, True]:
            L, Q, threshold = likelihood_coordinates(likelihood, clip=clipped)
            suffix = 'clipped' if clipped else 'raw'
            np.savez_compressed(out/'distances'/f'shimodaira_features_{suffix}.npz', L=L, Q=Q)
            qdist = pairwise_from_features(Q)/np.sqrt(Q.shape[1])
            np.save(out/'distances'/f'shimodaira_KL_approx_{suffix}.npy', qdist**2/2)
            reference_provenance[suffix] = dict(global_clip_threshold=threshold,
                fraction_clipped=float(np.mean(likelihood < threshold)) if threshold is not None else 0)
        dump(out/'shimodaira_provenance.json', reference_provenance)
    # Same global k across all models and both edge-weight conditions.
    requested=min(20,len(texts)-1); chosen=None; connectivity=[]
    for k in range(requested,len(texts)):
        gs=[]
        for e,f in zip(entries,features):
            g,nc=graph_distances(f['coords'],f['metric'],k)
            connectivity.append(dict(model=e['id'],k=k,components=nc))
            gs.append(g)
        if all(g is not None for g in gs):chosen=k;graphs=gs;break
    assert chosen is not None
    table(out/'evaluation/connectivity.csv',connectivity)
    dump(out/'analysis_config.json',dict(stage=stage,n=len(texts),m=len(entries),k_requested=requested,k_used=chosen,
         graph_layout='classical MDS of mean-normalized shortest-path distances',
         reference=entries[0]['id']+'/euclidean',reflection_allowed_for_primary_display=True))
    print('CONNECTED',stage,'k',chosen,flush=True)
    reference=classical(normalize(graphs[0]['euclidean']['S']))
    panels=[]; alignment=[]
    graph_quality=[]
    from sklearn.metrics import silhouette_score
    for e,g in zip(entries,graphs):
        for weight in ['euclidean','fisher']:
            a,b=g[weight]['a'],g[weight]['b'];S=g[weight]['S']
            Y=classical(normalize(S));rot,Rr,rr=align(Y,reference,True);Z,R,res=align(Y,reference)
            name=slug(e['id'])+'_'+weight
            np.savez_compressed(out/'graphs'/f'{name}.npz',S=S,source=a,target=b,
                weights=g[weight]['weights'],coords_original=Y,coords_aligned=Z,
                coords_rotation_only=rot,orthogonal_matrix=R,rotation_matrix=Rr)
            alignment.append(dict(model=e['id'],weight=weight,determinant=float(np.linalg.det(R)),
                residual=res,rotation_only_residual=rr,layout_stress=assess(normalize(S),Y)['stress_scale_fitted']))
            nearest=S.copy();np.fill_diagonal(nearest,np.inf)
            ix=np.argsort(nearest,axis=1)[:,:10]
            hit=float(np.mean(np.asarray(cats)[ix]==np.asarray(cats)[:,None]))
            graph_quality.append(dict(model=e['id'],weight=weight,
                category_silhouette=float(silhouette_score(S,np.asarray(cats),metric='precomputed')),
                category_knn10_hit=hit,random_category_hit=float((len(texts)/len(set(cats))-1)/(len(texts)-1))))
            table(out/'csv'/f'graph_{name}.csv',[dict(id=t['id'],category=t['category'],x=float(z[0]),y=float(z[1])) for t,z in zip(texts,Z)])
            panels.append((e,weight,Z,a,b))
    table(out/'evaluation/alignment.csv',alignment)
    table(out/'evaluation/internal_graph_quality.csv',graph_quality)
    limit=max(abs(p[2]).max() for p in panels)*1.1
    for e,w,Z,a,b in panels:
        fig,ax=plt.subplots(figsize=(7,6))
        plot_points(ax,Z,[],cats,e['label']+' | '+w,(a,b))
        ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit);ax.set_aspect('equal',adjustable='box')
        ax.legend(fontsize=6,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.12))
        fig.savefig(out/'graphs'/f'{slug(e["id"])}_{w}.png',bbox_inches='tight');plt.close(fig)
    page_size=10 if len(entries)>20 else len(entries)
    for start in range(0,len(entries),page_size):
        subset=panels[2*start:2*(start+page_size)]
        fig,axes=plt.subplots(len(subset)//2,2,figsize=(12,4*(len(subset)//2)),squeeze=False)
        for ax,(e,w,Z,a,b) in zip(axes.flat,subset):
            plot_points(ax,Z,[],cats,e['label']+' | '+w,(a,b));ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit)
            ax.set_aspect('equal',adjustable='box')
        filename='all_models_aligned'+(f'_page{start//page_size+1:02d}' if len(entries)>20 else '')+'.png'
        fig.tight_layout();fig.savefig(out/'graphs'/filename);plt.close(fig)
    table(out/'evaluation/model_key.csv',[dict(number=i+1,id=e['id'],label=e['label'],
        family=e['family'],revision=e['revision']) for i,e in enumerate(entries)])
    D=make_distances(features,graphs,include_gw=not quick)
    # Length-normalized likelihood is a sensitivity control, not the KL proxy.
    perbyte=np.array([f['ll'] for f in features])/np.array([r['bytes'] for r in texts])[None,:]
    perbyte-=perbyte.mean(1,keepdims=True)
    D['loglik_perbyte_control']=pairwise_from_features(perbyte)/np.sqrt(len(texts))
    metrics=[];audits=[]
    names=[e['label'] for e in entries];families=[e['family'] for e in entries]
    map_reference=None
    for method,d in D.items():
        assert np.isfinite(d).all() and np.allclose(d,d.T) and np.allclose(np.diag(d),0)
        np.save(out/'distances'/f'{method}.npy',d)
        table(out/'distances'/f'{method}.csv',[dict(model=e['id'],**{q['id']:float(v) for q,v in zip(entries,row)}) for e,row in zip(entries,d)])
        audits.append(dict(method=method,**metric_audit(d)))
        ticks=[f'{i+1:03d}' for i in range(len(names))] if len(names)>=50 else names
        fig,ax=plt.subplots(figsize=(14,12) if len(names)>=50 else (10,8));im=ax.imshow(d,cmap='viridis')
        ax.set_xticks(range(len(names)),ticks,rotation=55,ha='right',fontsize=5 if len(names)>=50 else 7)
        ax.set_yticks(range(len(names)),ticks,fontsize=5 if len(names)>=50 else 7);ax.set_title(method);fig.colorbar(im,ax=ax,label='Model dissimilarity')
        fig.tight_layout();fig.savefig(out/'heatmaps'/f'{method}.png');plt.close(fig)
        dnorm=d/upper(d).mean()
        seeds=[0] if stage=='smoke' else list(range(10))
        reductions=[('mds',None)]
        if stage=='main': reductions += [('isomap',3),('isomap',5),('umap',3),('umap',5),('umap',10)]
        if stage=='main' and len(entries)>=50:
            reductions=[('mds',None),('isomap',10),('isomap',30),
                        ('umap',5),('umap',15),('umap',30),('tsne',30)]
        for reducer,nneighbors in reductions:
            for seed in (seeds if reducer!='isomap' else [0]):
                if reducer=='mds':
                    Y=MDS(n_components=2,metric=True,dissimilarity='precomputed',random_state=seed,n_init=4,max_iter=500,eps=1e-6,normalized_stress=False).fit_transform(dnorm)
                elif reducer=='isomap':
                    from sklearn.manifold import Isomap
                    from scipy.sparse import csr_matrix
                    from scipy.sparse.csgraph import connected_components
                    T=dnorm.copy();np.fill_diagonal(T,np.inf);ix=np.argsort(T,axis=1)[:,:nneighbors]
                    A=np.zeros(T.shape,dtype=bool);A[np.arange(len(T))[:,None],ix]=True;A|=A.T
                    nc=connected_components(csr_matrix(A),directed=False)[0]
                    if nc>1:
                        metrics.append(dict(method=method,reducer=reducer,neighbors=nneighbors,seed=seed,status='disconnected',components=nc));continue
                    Y=Isomap(n_neighbors=nneighbors,n_components=2,metric='precomputed').fit_transform(dnorm)
                elif reducer=='tsne':
                    from sklearn.manifold import TSNE
                    Y=TSNE(n_components=2,metric='precomputed',init='random',
                        perplexity=nneighbors,random_state=seed,learning_rate='auto').fit_transform(dnorm)
                else:
                    import umap
                    Y=umap.UMAP(n_components=2,metric='precomputed',n_neighbors=nneighbors,min_dist=.1,random_state=seed,n_jobs=1).fit_transform(dnorm)
                if map_reference is None:map_reference=Y.copy()
                Z,R,res=align(Y,map_reference)
                tag=f'{method}_{reducer}'+(f'_k{nneighbors}' if nneighbors else '')+f'_seed{seed}'
                np.save(out/'maps'/f'{tag}.npy',Z)
                table(out/'csv'/f'{tag}.csv',[dict(id=e['id'],label=e['label'],family=e['family'],method=method,reducer=reducer,seed=seed,x=float(z[0]),y=float(z[1])) for e,z in zip(entries,Z)])
                metrics.append(dict(method=method,reducer=reducer,neighbors=nneighbors,seed=seed,status='ok',**assess(dnorm,Z)))
                if seed==0:
                    fig,ax=plt.subplots(figsize=(11,8));plot_points(ax,Z,names,families,tag,text=True)
                    ax.legend(loc='best',fontsize=8);fig.tight_layout();fig.savefig(out/'maps'/f'{tag}.png');plt.close(fig)
        print('MAP_COMPLETE',stage,method,flush=True)
    # Paper Figure 1 embeds clipped L (not Q), using feature-space t-SNE.
    # Preserve that setting separately from the common-distance reducer study.
    if stage=='main' and all('ll_reference' in f for f in features):
        from sklearn.manifold import TSNE
        from .shimodaira import likelihood_coordinates
        L, Q, _ = likelihood_coordinates(np.array([f['ll_reference'] for f in features]))
        for label, X in [('L',L),('Q',Q)]:
            d=pairwise_from_features(X);dnorm=d/upper(d).mean()
            for seed in range(10):
                Y=TSNE(n_components=2,init='pca',perplexity=min(30,len(entries)-1),
                    random_state=seed,learning_rate='auto').fit_transform(X)
                tag=f'paper_features_{label}_tsne_seed{seed}'
                np.save(out/'maps'/f'{tag}.npy',Y)
                table(out/'csv'/f'{tag}.csv',[dict(id=e['id'],label=e['label'],family=e['family'],x=float(z[0]),y=float(z[1])) for e,z in zip(entries,Y)])
                metrics.append(dict(method='paper_features_'+label,reducer='tsne_pca_init',
                    neighbors=min(30,len(entries)-1),seed=seed,status='ok',**assess(dnorm,Y)))
                if seed==0:
                    fig,ax=plt.subplots(figsize=(14,10));plot_points(ax,Y,names,families,tag,text=True)
                    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'maps'/f'{tag}.png');plt.close(fig)
    table(out/'evaluation/reductions.csv',metrics)
    table(out/'evaluation/metric_axioms.csv',audits)
    relations=[]
    from sklearn.metrics import silhouette_score
    for method,d in D.items():
        nn=d.copy();np.fill_diagonal(nn,np.inf)
        nearest=np.argmin(nn,axis=1)
        row=dict(method=method,nearest_same_family=float(np.mean(np.array(families)[nearest]==np.array(families))))
        if len(set(families))<len(families):row['family_silhouette']=float(silhouette_score(d,np.array(families),metric='precomputed'))
        relations.append(row)
    table(out/'evaluation/family_association.csv',relations)
    keys=list(D);corr=np.array([[spearmanr(upper(D[a]),upper(D[b])).statistic for b in keys] for a in keys])
    table(out/'evaluation/distance_correlations.csv',[dict(method=k,**{q:float(v) for q,v in zip(keys,row)}) for k,row in zip(keys,corr)])
    # Stratified subsampling without replacement: induced-submatrix stability,
    # not a claim to refit PCA or reconstruct kNN graphs on each replicate.
    rng=np.random.default_rng(20261022);stability=[]
    reference_subsample_features = {}
    if all('ll_reference' in f for f in features):
        from .shimodaira import likelihood_coordinates
        for clipped in [False, True]:
            L, _, _ = likelihood_coordinates(np.array([f['ll_reference'] for f in features]),clip=clipped)
            reference_subsample_features['shimodaira_Q_'+('clipped' if clipped else 'raw')] = L
    for rep in range(100 if stage=='main' else 10):
        ids=np.sort(np.concatenate([rng.choice(np.flatnonzero(np.array(cats)==c),max(1,int(.8*cats.count(c))),replace=False) for c in sorted(set(cats))]))
        for weight in ['euclidean','fisher']:
            ss=[normalize(g[weight]['S'][np.ix_(ids,ids)]) for g in graphs]
            dd=pairwise_from_features([upper(s) for s in ss])/np.sqrt(len(upper(ss[0])))
            rho=float(spearmanr(upper(D[weight+'_frob']),upper(dd)).statistic)
            stability.append(dict(replicate=rep,method=weight+'_frob',n=len(ids),spearman=rho))
        for method,L in reference_subsample_features.items():
            X=L[:,ids];X=X-X.mean(1,keepdims=True)
            dd=pairwise_from_features(X)/np.sqrt(len(ids))
            stability.append(dict(replicate=rep,method=method,n=len(ids),
                spearman=float(spearmanr(upper(D[method]),upper(dd)).statistic)))
    table(out/'evaluation/induced_submatrix_stability.csv',stability)
    if stage=='main':
        rebuilt=[]
        for rep in range(20):
            ids=np.sort(np.concatenate([rng.choice(np.flatnonzero(np.array(cats)==c),16,replace=False) for c in sorted(set(cats))]))
            for kk in range(chosen,len(ids)):
                gg=[graph_distances(f['coords'][ids],f['metric'][ids],kk)[0] for f in features]
                if all(g is not None for g in gg):break
            assert all(g is not None for g in gg)
            for weight in ['euclidean','fisher']:
                ss=[normalize(g[weight]['S']) for g in gg]
                dd=pairwise_from_features([upper(s) for s in ss])/np.sqrt(len(upper(ss[0])))
                base=D[weight+'_frob'];rho=float(spearmanr(upper(base),upper(dd)).statistic)
                aa=base.copy();bb=dd.copy();np.fill_diagonal(aa,np.inf);np.fill_diagonal(bb,np.inf)
                ia=np.argsort(aa,axis=1)[:,:3];ib=np.argsort(bb,axis=1)[:,:3]
                overlap=float(np.mean([len(set(a)&set(b))/3 for a,b in zip(ia,ib)]))
                rebuilt.append(dict(replicate=rep,method=weight+'_frob',n=len(ids),k=kk,spearman=rho,model_knn_overlap=overlap,
                    protocol='stratified subsample; rebuild kNN and shortest paths; PCA and local metrics fixed'))
            for method,L in reference_subsample_features.items():
                X=L[:,ids];X=X-X.mean(1,keepdims=True)
                dd=pairwise_from_features(X)/np.sqrt(len(ids))
                aa=D[method].copy();bb=dd.copy();np.fill_diagonal(aa,np.inf);np.fill_diagonal(bb,np.inf)
                ia=np.argsort(aa,axis=1)[:,:3];ib=np.argsort(bb,axis=1)[:,:3]
                rebuilt.append(dict(replicate=rep,method=method,n=len(ids),k=None,
                    spearman=float(spearmanr(upper(D[method]),upper(dd)).statistic),
                    model_knn_overlap=float(np.mean([len(set(a)&set(b))/3 for a,b in zip(ia,ib)])),
                    protocol='same stratified subsample; recenter likelihood; global clipping threshold fixed'))
        table(out/'evaluation/rebuilt_graph_stability.csv',rebuilt)
    # One map per graph comparison in a single review image.
    selected=[m for m in D if m.startswith('fisher_') or m.startswith('euclidean_')]
    fig,axes=plt.subplots((len(selected)+2)//3,3,figsize=(18,5*((len(selected)+2)//3)),squeeze=False)
    for ax,method in zip(axes.flat,selected):
        Y=np.load(out/'maps'/f'{method}_mds_seed0.npy')
        plot_points(ax,Y,names,families,method,text=True)
    for ax in list(axes.flat)[len(selected):]:ax.set_visible(False)
    fig.tight_layout();fig.savefig(out/'maps/graph_comparison_overview.png');plt.close(fig)
    aggregates=[]
    groups=sorted(set((r['method'],r['reducer'],str(r['neighbors'])) for r in metrics if r['status']=='ok'))
    for method,reducer,nneighbors in groups:
        subset=[r for r in metrics if r['status']=='ok' and (r['method'],r['reducer'],str(r['neighbors']))==(method,reducer,nneighbors)]
        row=dict(method=method,reducer=reducer,neighbors=nneighbors,runs=len(subset))
        for field in ['stress_scale_fitted','spearman','model_knn_overlap']:
            values=[r[field] for r in subset]
            row[field+'_mean']=float(np.mean(values));row[field+'_std']=float(np.std(values))
        aggregates.append(row)
    table(out/'evaluation/reduction_summary.csv',aggregates)
    dump(out/'complete.json',dict(stage=stage,models=len(entries),texts=len(texts),methods=list(D),
        map_runs=sum(r['status']=='ok' for r in metrics),disconnected_isomap_runs=sum(r['status']!='ok' for r in metrics)))
    print('ANALYSIS_COMPLETE',stage,flush=True)


def selftest():
    rng=np.random.default_rng(4)
    X=rng.normal(size=(20,2));Q=np.array([[0,-1],[1,0]])
    Z,R,res=align(X@Q+3,X,True)
    assert res<1e-10 and np.linalg.det(R)>0
    S=pairwise_from_features(X)
    assert metric_audit(S)['triangle_violations']==0
    assert np.allclose(rank_distance([S,S]),0)
    assert np.allclose(knn_distance([S,S],5),0)
    assert np.allclose(cka_distance([X,X@Q]),0,atol=1e-10)
    print('SELFTEST_OK')


