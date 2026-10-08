"""Check inference log-probability alignment with the attribution forward path."""
import json
import sys
from pathlib import Path
import requests
from transformers import AutoTokenizer
from core import layout

root=Path(sys.argv[1]);tok=AutoTokenizer.from_pretrained(root/'model_step666',local_files_only=True)
rows={json.loads(l)['sample_id']:json.loads(l) for name in ['samples.jsonl','samples_enriched.jsonl'] for l in (root/name).open()}
attrs={}
for path in root.glob('attribution_*.jsonl'):
    for l in path.open():
        a=json.loads(l);attrs[a['sample_id']]=a
cases=json.loads((root/'intervention_design.json').read_text())
results=[]
for case in cases[::3]:
    r=rows[case['sample_id']];loc=layout(tok,r);start=len(loc['prompt'])+loc['target'][0]
    req={'input_ids':loc['full_ids'],'sampling_params':{'temperature':0,'max_new_tokens':0},
         'return_logprob':True,'logprob_start_len':start-1}
    response=requests.post('http://127.0.0.1:30366/generate',json=req,timeout=600)
    response.raise_for_status();values=response.json()['meta_info']['input_token_logprobs']
    selected=values[1:1+len(loc['target'])]
    assert [v[1] for v in selected]==[loc['response_ids'][i] for i in loc['target']]
    hf=attrs[r['sample_id']]['action_token_logprobs'];sg=[v[0] for v in selected]
    results.append({'sample_id':r['sample_id'],'hf_logprobs':hf,'sglang_logprobs':sg,
                    'max_token_abs_difference':max(abs(x-y) for x,y in zip(hf,sg)),
                    'full_score_difference':sum(sg)-sum(hf)})
(root/'backend_consistency.json').write_text(json.dumps(results,indent=2))
print('CHECKED',len(results),'max token difference',max(r['max_token_abs_difference'] for r in results))
