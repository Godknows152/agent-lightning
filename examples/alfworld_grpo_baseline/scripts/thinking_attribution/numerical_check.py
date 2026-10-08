"""FP32 derivative and attribution precision checks, with a causal-mask control."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from core import load, forward_scores, gradient

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
rows={json.loads(l)['sample_id']:json.loads(l) for name in ['samples.jsonl','samples_enriched.jsonl'] for l in (root/name).open()}
attrs={}
for f in root.glob('attribution_*.jsonl'):
    for l in f.open():
        d=json.loads(l);attrs[d['sample_id']]=d
ids=json.loads((root/'ig_design.json').read_text())
model,tok=load(root/'model_step666');model.float()
out=[]
for sid in ids:
    r=rows[sid];loc,emb,scores,_=forward_scores(model,tok,r)
    g=torch.autograd.grad(scores['full'],emb)[0]
    think_indices=[len(loc['prompt'])+i for i in loc['thinking']]
    norms=g[0,think_indices].norm(dim=-1)
    peak=int(norms.argmax());absolute=think_indices[peak]
    direction=torch.zeros_like(emb);direction[0,absolute]=g[0,absolute]/g[0,absolute].norm()
    analytic=float((g*direction).sum())
    fd=[]
    for eps in [.001,.005,.02]:
        vals=[]
        for sign in [-1,1]:
            with torch.no_grad():
                _,_,ss,_=forward_scores(model,tok,r,need_grad=False,embedding_override=emb.detach()+sign*eps*direction)
                vals.append(float(ss['full']))
        finite=(vals[1]-vals[0])/(2*eps)
        fd.append({'epsilon':eps,'analytic':analytic,'finite_difference':finite,
                   'relative_error':abs(finite-analytic)/max(abs(analytic),1e-8)})
    future=len(loc['prompt'])+loc['target'][-1]
    original=np.array(attrs[sid]['grad_norms']['full']);now=norms.cpu().numpy()
    cosine=float(np.dot(original,now)/(np.linalg.norm(original)*np.linalg.norm(now)+1e-30))
    entry={'sample_id':sid,'float32_score':float(scores['full'].detach()),
           'bf16_score':attrs[sid]['scores']['full'],'gradient_norm_cosine_bf16_fp32':cosine,
           'peak_same':int(original.argmax())==peak,'causal_future_grad_max':float(g[:,future:].abs().max()),
           'finite_differences':fd,'float32_grad_norms':now.tolist()}
    out.append(entry);print(sid,cosine,fd,flush=True)
    (root/'numerical_checks.json').write_text(json.dumps(out,indent=2))
