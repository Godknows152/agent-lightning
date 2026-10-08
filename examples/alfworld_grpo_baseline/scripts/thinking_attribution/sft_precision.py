"""FP32 model with FP64 log-softmax to audit saturated SFT action scores."""
import argparse
import json
from pathlib import Path

import torch
from core import load, layout

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
root = args.root.resolve()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model, tokenizer = load(root/'model_merged')
model.float()
rows = [r for line in (root/'samples_self.jsonl').open() if (r := json.loads(line))['eligible']]
selected = {r['sample_id'] for category in sorted({r['category'] for r in rows})
            for r in [r for r in rows if r['category'] == category][:2]}
path = root/'precision_self.jsonl'
done = {json.loads(line)['sample_id'] for line in path.open()} if path.exists() else set()


def scores(loc, emb):
    p = len(loc['prompt'])
    positions = torch.tensor([p+i-1 for i in loc['target']], device=model.device)
    target = torch.tensor([loc['response_ids'][i] for i in loc['target']], device=model.device)
    logits = model(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=model.device),
                   use_cache=False, logits_to_keep=positions).logits[0].double()
    lp = logits.log_softmax(-1)[torch.arange(len(target), device=model.device), target]
    return {'full': lp.sum(), 'first': lp[0]}


with path.open('a', buffering=1) as output:
    for row in rows:
        if row['sample_id'] in done:
            continue
        loc = layout(tokenizer, row)
        tensor = torch.tensor([loc['full_ids']], device=model.device)
        emb = model.get_input_embeddings()(tensor).detach().requires_grad_(True)
        indices = [len(loc['prompt'])+i for i in loc['thinking']]
        objective = scores(loc, emb)
        result = {'sample_id': row['sample_id'], 'model_dtype': 'float32', 'score_dtype': 'float64',
                  'tf32': False, 'scores': {k: float(v.detach()) for k, v in objective.items()},
                  'grad_norms': {}, 'grad_input': {}}
        full_gradient = None
        for name in ['full', 'first']:
            gradient = torch.autograd.grad(objective[name], emb, retain_graph=name == 'full')[0]
            result['grad_norms'][name] = gradient[0, indices].norm(dim=-1).cpu().tolist()
            result['grad_input'][name] = (gradient[0, indices]*emb[0, indices].detach()).sum(-1).cpu().tolist()
            if name == 'full':
                full_gradient = gradient
        if row['sample_id'] in selected:
            peak = max(range(len(indices)), key=result['grad_norms']['full'].__getitem__)
            direction = torch.zeros_like(emb)
            direction[0, indices[peak]] = full_gradient[0, indices[peak]]/full_gradient[0, indices[peak]].norm()
            analytic = float((full_gradient*direction).sum())
            finite = []
            for epsilon in [.001, .005, .02]:
                with torch.no_grad():
                    minus = float(scores(loc, emb.detach()-epsilon*direction)['full'])
                    plus = float(scores(loc, emb.detach()+epsilon*direction)['full'])
                fd = (plus-minus)/(2*epsilon)
                finite.append({'epsilon': epsilon, 'analytic': analytic, 'finite_difference': fd,
                               'relative_error': abs(fd-analytic)/max(abs(analytic), 1e-12)})
            result['finite_differences'] = finite
        output.write(json.dumps(result)+'\n')
        print(row['sample_id'], result['scores']['full'], flush=True)
print('PRECISION_COMPLETE', len(rows), flush=True)
