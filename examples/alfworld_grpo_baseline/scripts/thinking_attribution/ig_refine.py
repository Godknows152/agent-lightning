"""FP32 path integration with endpoint-focused quadrature for saturated baselines."""
import json
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from core import load, layout, forward_scores

parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
args=parser.parse_args();root=args.root
rows={json.loads(l)['sample_id']:json.loads(l) for name in ['samples.jsonl','samples_enriched.jsonl'] for l in (root/name).open()}
ids=json.loads((root/'ig_design.json').read_text())[args.shard::args.shards]
path=root/('integrated_gradients_fp32.jsonl' if args.shards==1 else f'integrated_gradients_fp32_shard{args.shard}.jsonl')
done={(r['sample_id'],r['baseline']) for prior in root.glob('integrated_gradients_fp32*.jsonl') for l in prior.open() if (r:=json.loads(l))}
model,tok=load(root/'model_step666');model.float()
with path.open('a',buffering=1) as f:
 for sid in ids:
  row=rows[sid];loc=layout(tok,row)
  actual=model.get_input_embeddings()(torch.tensor([loc['full_ids']],device=model.device)).detach()
  indices=torch.tensor([len(loc['prompt'])+i for i in loc['thinking']],device=model.device)
  for baseline in ['zero','space']:
   if (sid,baseline) in done:continue
   start=time.time();base=actual.clone()
   base[:,indices]=0 if baseline=='zero' else model.get_input_embeddings()(torch.tensor(tok.encode(' ',add_special_tokens=False),device=model.device)).mean(0)
   with torch.no_grad():
    _,_,s1,_=forward_scores(model,tok,row,need_grad=False,include_thinking_entropy=False)
    _,_,s0,_=forward_scores(model,tok,row,need_grad=False,embedding_override=base,include_thinking_entropy=False)
    delta=float(s1['full']-s0['full'])
   attempts=[]
   for steps in [64,256]:
    nodes,weights=np.polynomial.legendre.leggauss(steps);nodes=(nodes+1)/2;weights=weights/2
    total=torch.zeros_like(actual[:,indices])
    for u,w in zip(nodes,weights):
     alpha=float(u**4)
     _,emb,scores,_=forward_scores(model,tok,row,baseline_embeddings=base,alpha=alpha,include_thinking_entropy=False)
     g=torch.autograd.grad(scores['full'],emb)[0][:,indices]
     total+=g*float(w*4*u**3)
    values=((actual[:,indices]-base[:,indices])*total).sum(-1)[0]
    absolute=abs(float(values.sum())-delta);relative=absolute/max(abs(delta),1e-3)
    attempts.append({'nodes':steps,'absolute_error':absolute,'relative_error':relative})
    if relative<=.1 or absolute<=.02:break
   entry={'sample_id':sid,'baseline':baseline,'precision':'float32','quadrature':'Gauss-Legendre, alpha=u^4',
          'attempts':attempts,'score_delta':delta,'contributions':values.detach().cpu().tolist(),
          'completeness_absolute_error':absolute,'completeness_relative_error':relative,
          'accepted':relative<=.1 or absolute<=.02,'elapsed_seconds':time.time()-start}
   f.write(json.dumps(entry)+'\n');print('REFINED',sid,baseline,attempts,flush=True)
