"""Seeded batched counterfactual continuations through a local SGLang server."""
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer
from core import prompt_ids
from alfworld_baseline.xml_actions import parse_xml_decision


async def run(a):
    root=a.root
    sources=[a.samples] if a.samples else [root/'samples.jsonl',root/'samples_enriched.jsonl']
    rows={json.loads(l)['sample_id']:json.loads(l) for source in sources for l in source.open()}
    cases=json.loads((a.design or root/'intervention_design.json').read_text())
    tok=AutoTokenizer.from_pretrained(a.model or root/'model_step666',local_files_only=True)
    path=root/a.output_name
    done={(d['sample_id'],d['mode'],d['condition']) for l in path.open() if (d:=json.loads(l))} if path.exists() else set()
    jobs=[]
    for case in cases:
        for mode in ['prefixes','continuations']:
            for name,prefix in case[mode].items():
                if (case['sample_id'],mode,name) not in done:jobs.append((case,mode,name,prefix))
    if a.limit:jobs=jobs[:a.limit]
    semaphore=asyncio.Semaphore(a.concurrency)
    print('PENDING',len(jobs),flush=True)
    with path.open('a',buffering=1) as f:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as client:
            async def one(case,mode,name,prefix):
                async with semaphore:
                    row=rows[case['sample_id']]
                    prefix_token_ids=tok.encode(prefix,add_special_tokens=False)
                    input_ids=prompt_ids(tok,row)+prefix_token_ids
                    seed=int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8],16)%100000000
                    if name.endswith('_replicate'):
                        seed+=1000000
                    remaining=a.decision_budget-len(prefix_token_ids)
                    assert remaining>0
                    payload={'input_ids':[input_ids]*a.repeats,'sampling_params':[
                        {'temperature':1.0,'top_p':1.0,'top_k':-1,'max_new_tokens':remaining,
                         'stop_token_ids':[tok.eos_token_id],'sampling_seed':seed+j}
                        for j in range(a.repeats)],'return_logprob':False}
                    started=time.time()
                    async with client.post(a.server+'/generate',json=payload) as resp:
                        body=await resp.text()
                        if resp.status!=200:raise RuntimeError(f'{resp.status}: {body[:1000]}')
                        generated=json.loads(body)
                    assert len(generated)==a.repeats
                    outputs=[]
                    for item in generated:
                        text=item['text'];decision=parse_xml_decision(prefix+text)
                        valid=decision.status=='valid' and decision.action in row['admissible_actions']
                        outputs.append({'text':text,'action':decision.action,'parser_status':decision.status,
                                        'reason':decision.reason,'legal':valid,'finish_reason':item['meta_info'].get('finish_reason'),
                                        'completion_tokens':item['meta_info'].get('completion_tokens')})
                    result={'sample_id':row['sample_id'],'mode':mode,'condition':name,'prefix':prefix,
                            'seed_base':seed,'max_new_tokens':remaining,'outputs':outputs,'elapsed_seconds':time.time()-started}
                    f.write(json.dumps(result,ensure_ascii=False)+'\n')
                    print(row['sample_id'],mode,name,'legal',sum(o['legal'] for o in outputs),'seconds',round(result['elapsed_seconds'],2),flush=True)
            await asyncio.gather(*(one(*job) for job in jobs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--server',default='http://127.0.0.1:30366');p.add_argument('--repeats',type=int,default=16)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--model',type=Path)
    p.add_argument('--samples',type=Path)
    p.add_argument('--design',type=Path)
    p.add_argument('--output-name',default='generations.jsonl')
    p.add_argument('--decision-budget',type=int,default=768)
    p.add_argument('--concurrency',type=int,default=4)
    asyncio.run(run(p.parse_args()))
