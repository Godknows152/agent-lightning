"""Frozen checkpoint attribution; one output row per decision with full provenance."""
import argparse
import json
import re
import time
from pathlib import Path

from core import load, gradient, alternative_actions, replace_action


def run():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--cohort',default='main',choices=['main','enriched'])
    p.add_argument('--model',type=Path)
    p.add_argument('--samples',type=Path)
    p.add_argument('--output-name')
    a=p.parse_args()
    source=a.samples or a.root/('samples.jsonl' if a.cohort=='main' else 'samples_enriched.jsonl')
    rows=[json.loads(l) for l in source.open()]
    rows=[r for i,r in enumerate(rows) if i%a.shards==a.shard and r['eligible']]
    if a.limit: rows=rows[:a.limit]
    out=a.root/f'attribution_{a.shard}.jsonl' if a.cohort=='main' else a.root/f'attribution_enriched_{a.shard}.jsonl'
    if a.output_name: out=a.root/a.output_name
    done={json.loads(l)['sample_id'] for l in out.open()} if out.exists() else set()
    model_path=a.model or a.root/'model_step666'
    model,tokenizer=load(model_path)
    with out.open('a',buffering=1) as f:
        for r in rows:
            if r['sample_id'] in done: continue
            started=time.time()
            loc,m,matrices=gradient(model,tokenizer,r)
            think=loc['text'][:loc['think_end']]
            m.update(sample_id=r['sample_id'], category=r['category'], success=r['success'],
                     model_path=str(model_path.resolve()),
                     phase=r['phase'], game_file=r['gts']['game_file'],session=r['session'],
                     thinking=think, action=r['action'],
                     thinking_tokens=[tokenizer.decode([loc['response_ids'][i]]) for i in loc['thinking']],
                     thinking_offsets=[loc['offsets'][i] for i in loc['thinking']],
                     prompt_length=len(loc['prompt']),
                     exact_mentions=[list(x.span()) for x in re.finditer(r'(?<!\w)'+re.escape(r['action'])+r'(?!\w)',think,re.I)],
                     verb_mentions=[list(x.span()) for x in re.finditer(r'\b'+re.escape(r['action'].split()[0])+r'\b',think,re.I)])
            contrasts={}
            for kind,alt in alternative_actions(r).items():
                alt_loc,alt_m,alt_g=gradient(model,tokenizer,r,replace_action(loc,alt),objectives=('full',))
                assert [alt_loc['response_ids'][i] for i in alt_loc['thinking']] == [loc['response_ids'][i] for i in loc['thinking']]
                contrasts[kind]={'alternative':alt,'margin':m['scores']['full']-alt_m['scores']['full'],
                                 'grad_norms':(matrices['full']-alt_g['full']).norm(dim=-1).tolist()}
            m['contrasts']=contrasts
            m['elapsed_seconds']=time.time()-started
            f.write(json.dumps(m,ensure_ascii=False)+'\n')
            print(r['sample_id'],m['scores']['full'],round(m['elapsed_seconds'],2),flush=True)


if __name__=='__main__':run()
