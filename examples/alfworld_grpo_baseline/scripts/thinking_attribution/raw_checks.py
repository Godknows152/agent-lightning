"""Numerical and serving-backend checks for the original-weight comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import requests
import torch
from transformers import AutoTokenizer
from core import load, layout, forward_scores

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--mode', choices=['backend', 'numerical'], required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--samples', type=Path)
parser.add_argument('--attribution', type=Path)
parser.add_argument('--output-name')
args = parser.parse_args()
root = args.root
sources = [args.samples] if args.samples else [root/'samples.jsonl', root/'samples_enriched.jsonl']
rows = {r['sample_id']: r for p in sources for l in p.open() if (r := json.loads(l))}
sources = [args.attribution] if args.attribution else [root/'controlled_main.jsonl', root/'controlled_enriched.jsonl']
attrs = {r['sample_id']: r for p in sources for l in p.open() if (r := json.loads(l))}
if args.samples:
    selected = []
    for category in sorted({r['category'] for r in attrs.values()}):
        selected.extend([r['sample_id'] for r in attrs.values() if r['category'] == category][:2])
else:
    selected = json.loads((root/'ig_design.json').read_text())
out = []
path = root/(args.output_name or ('backend_consistency.json' if args.mode == 'backend' else 'numerical_checks.json'))
if args.mode == 'backend':
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = json.loads((root/'intervention_design.json').read_text())
    selected = [r['sample_id'] for r in cases[::3]]
    for sid in selected:
        loc = layout(tokenizer, rows[sid]); start = len(loc['prompt'])+loc['target'][0]
        result = requests.post('http://127.0.0.1:30408/generate', json={
            'input_ids': loc['full_ids'], 'sampling_params': {'temperature': 0, 'max_new_tokens': 0},
            'return_logprob': True, 'logprob_start_len': start-1}, timeout=600)
        result.raise_for_status()
        values = result.json()['meta_info']['input_token_logprobs'][1:1+len(loc['target'])]
        assert [v[1] for v in values] == [loc['response_ids'][i] for i in loc['target']]
        hf = attrs[sid]['action_token_logprobs']; sg = [v[0] for v in values]
        out.append({'sample_id': sid, 'hf_logprobs': hf, 'sglang_logprobs': sg,
                    'max_token_abs_difference': max(abs(a-b) for a, b in zip(hf, sg))})
else:
    model, tokenizer = load(args.model)
    model.float()
    for sid in selected:
        row = rows[sid]
        loc, emb, scores, _ = forward_scores(model, tokenizer, row, include_thinking_entropy=False)
        gradient = torch.autograd.grad(scores['full'], emb)[0]
        indices = [len(loc['prompt'])+i for i in loc['thinking']]
        norms = gradient[0, indices].norm(dim=-1)
        peak = int(norms.argmax()); absolute = indices[peak]
        direction = torch.zeros_like(emb)
        direction[0, absolute] = gradient[0, absolute]/gradient[0, absolute].norm()
        analytic = float((gradient*direction).sum())
        finite = []
        for epsilon in [.001, .005, .02]:
            values = []
            for sign in [-1, 1]:
                with torch.no_grad():
                    _, _, perturbed, _ = forward_scores(model, tokenizer, row, need_grad=False,
                        embedding_override=emb.detach()+sign*epsilon*direction, include_thinking_entropy=False)
                    values.append(float(perturbed['full']))
            fd = (values[1]-values[0])/(2*epsilon)
            finite.append({'epsilon': epsilon, 'analytic': analytic, 'finite_difference': fd,
                           'relative_error': abs(fd-analytic)/max(abs(analytic), 1e-8)})
        previous = np.array(attrs[sid]['grad_norms']['full']); current = norms.cpu().numpy()
        cosine = float(np.dot(previous, current)/(np.linalg.norm(previous)*np.linalg.norm(current)))
        future = len(loc['prompt'])+loc['target'][-1]
        out.append({'sample_id': sid, 'float32_score': float(scores['full'].detach()),
                    'gradient_norm_cosine_bf16_fp32': cosine, 'peak_same': int(previous.argmax()) == peak,
                    'causal_future_grad_max': float(gradient[:, future:].abs().max()),
                    'finite_differences': finite, 'float32_grad_norms': current.tolist()})
        print('CHECKED', sid, cosine, flush=True)
        path.write_text(json.dumps(out, indent=2))
path.write_text(json.dumps(out, indent=2))
print('DONE', args.mode, len(out), flush=True)
