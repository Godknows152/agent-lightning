"""Paired fixed-suffix interventions and integrated-gradient robustness checks."""
import argparse
import json
import time
from pathlib import Path

import torch
from core import load, layout, forward_scores, replace_action


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['fixed','ig'],required=True)
    p.add_argument('--model',type=Path)
    a=p.parse_args();root=a.root
    rows={json.loads(l)['sample_id']:json.loads(l) for name in ['samples.jsonl','samples_enriched.jsonl'] for l in (root/name).open()}
    model,tok=load(a.model or root/'model_step666')
    if a.mode=='fixed':
        cases=json.loads((root/'intervention_design.json').read_text())
        path=root/'fixed_interventions.jsonl'
        done={(d['sample_id'],d['condition']) for l in path.open() if (d:=json.loads(l))} if path.exists() else set()
        with path.open('a',buffering=1) as f:
            for case in cases:
                row=rows[case['sample_id']]
                for name,response in case['variants'].items():
                    if (row['sample_id'],name) in done:continue
                    with torch.inference_mode():
                        loc,_,scores,metrics=forward_scores(model,tok,row,response,need_grad=False)
                        result={'sample_id':row['sample_id'],'condition':name,
                                'scores':{k:float(v) for k,v in scores.items()},**metrics}
                        if case['alternative']:
                            _,_,alt_scores,_=forward_scores(model,tok,row,replace_action(loc,case['alternative']),need_grad=False)
                            result['alternative_score']=float(alt_scores['full'])
                            result['margin']=float(scores['full']-alt_scores['full'])
                    f.write(json.dumps(result)+'\n')
                print('fixed',row['sample_id'],flush=True)
    else:
        ids=json.loads((root/'ig_design.json').read_text());path=root/'integrated_gradients.jsonl'
        done={(d['sample_id'],d['baseline']) for l in path.open() if (d:=json.loads(l))} if path.exists() else set()
        with path.open('a',buffering=1) as f:
            for sid in ids:
                row=rows[sid];loc=layout(tok,row)
                input_ids=torch.tensor([loc['full_ids']],device=model.device)
                actual=model.get_input_embeddings()(input_ids).detach()
                indices=torch.tensor([len(loc['prompt'])+i for i in loc['thinking']],device=model.device)
                for baseline in ['zero','space']:
                    if (sid,baseline) in done:continue
                    started=time.time();base=actual.clone()
                    base[:,indices]=0 if baseline=='zero' else model.get_input_embeddings()(torch.tensor(tok.encode(' ',add_special_tokens=False),device=model.device)).mean(0)
                    with torch.no_grad():
                        _,_,s1,_=forward_scores(model,tok,row,need_grad=False)
                        _,_,s0,_=forward_scores(model,tok,row,need_grad=False,embedding_override=base)
                        score_delta=float(s1['full']-s0['full'])
                    def integrate(steps):
                        total=torch.zeros_like(actual[:,indices],dtype=torch.float32)
                        for k in range(steps):
                            _,emb,scores,_=forward_scores(model,tok,row,baseline_embeddings=base,alpha=(k+.5)/steps)
                            grad=torch.autograd.grad(scores['full'],emb)[0][:,indices].float()
                            total+=grad
                        return ((actual[:,indices]-base[:,indices]).float()*total/steps).sum(-1)[0]
                    contributions=integrate(32);steps=32
                    error=abs(float(contributions.sum())-score_delta)/max(abs(score_delta),1e-3)
                    if error>.1:
                        contributions=integrate(128);steps=128
                        error=abs(float(contributions.sum())-score_delta)/max(abs(score_delta),1e-3)
                    result={'sample_id':sid,'baseline':baseline,'integration_steps':steps,
                            'score_delta':score_delta,'contributions':contributions.cpu().tolist(),
                            'completeness_relative_error':error,'elapsed_seconds':time.time()-started}
                    f.write(json.dumps(result)+'\n');print('IG',sid,baseline,steps,error,flush=True)


if __name__=='__main__':main()
