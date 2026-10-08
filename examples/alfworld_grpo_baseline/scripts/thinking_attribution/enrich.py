"""Conditional literal-mention cohort, kept separate from the representative sample."""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

root=Path(sys.argv[1])
manifest=json.loads((root/'sample_manifest.json').read_text())
original={json.loads(l)['sample_id'] for l in (root/'samples.jsonl').open()}
groups=defaultdict(list)
for line in Path(manifest['source']).open():
    r=json.loads(line)
    if r['uid'] in original or '</think>' not in r['output']:continue
    m=re.search(r'<parameter=action>\s*(.*?)\s*</parameter>',r['output'],re.S)
    if not m:continue
    action=m.group(1).strip()
    if len(action.split())<2:continue
    think=r['output'].split('</think>')[0]
    mentions=list(re.finditer(r'(?<!\w)'+re.escape(action)+r'(?!\w)',think,re.I))
    if not mentions:continue
    cat=Path(r['gts']['game_file']).parts[-3].split('-')[0]
    ma=re.search(r'Your admissible actions of the current situation are: \[(.*?)\]\.\n',r['input'],re.S)
    actions=re.findall(r"'([^']*)'",ma.group(1)) if ma else []
    if action not in actions:continue
    session,index=r['uid'].rsplit('_',1)
    r.update(sample_id=r['uid'],session=session,decision_index=int(index),category=cat,
             success=r['score']>0,phase='enriched_literal',action=action,admissible_actions=actions,
             eligible=True,mention_count=len(mentions))
    groups[cat].append(r)
rng=random.Random(20261009)
selected=[]
for cat,rows in sorted(groups.items()):
    rng.shuffle(rows)
    rows.sort(key=lambda r: r['mention_count']<2)
    seen=set();chosen=[]
    for r in rows:
        if r['session'] in seen:continue
        chosen.append(r);seen.add(r['session'])
        if len(chosen)==10:break
    selected.extend(chosen)
(root/'samples_enriched.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
(root/'enrichment_manifest.json').write_text(json.dumps({'cohort':'conditional, not population representative',
    'selection':'10 decisions/category where available; multiple mentions prioritized; at most one decision/session/category',
    'seed':20261009,'n':len(selected),'multiple_mentions':sum(r['mention_count']>1 for r in selected),
    'population_eligible_by_category':{k:len(v) for k,v in groups.items()}},indent=2))
print((root/'enrichment_manifest.json').read_text())
