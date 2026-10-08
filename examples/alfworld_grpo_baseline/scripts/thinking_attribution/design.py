"""Freeze paired interventions before querying their outcomes."""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from core import alternative_actions

root=Path(sys.argv[1])
attributions={}
for f in root.glob('attribution_*.jsonl'):
    for l in f.open():
        d=json.loads(l);attributions[d['sample_id']]=d
samples={}
for f in ['samples.jsonl','samples_enriched.jsonl']:
    for l in (root/f).open():
        r=json.loads(l);samples[r['sample_id']]=r
rng=random.Random(20261010)
groups=defaultdict(list)
for sid,d in attributions.items():
    cohort='enriched' if d['phase']=='enriched_literal' else 'main'
    groups[cohort,d['category']].append(sid)
selected=[]
for (cohort,cat),ids in sorted(groups.items()):
    rng.shuffle(ids)
    chosen=[];sessions=set()
    for sid in ids:
        if samples[sid]['session'] in sessions:continue
        chosen.append(sid);sessions.add(samples[sid]['session'])
        if len(chosen)==(6 if cohort=='main' else 4):break
    selected.extend(chosen)
cases=[]
for sid in selected:
    r=samples[sid];d=attributions[sid];think=d['thinking'];original=r['output']
    offset=d['thinking_offsets'];norms=d['grad_norms']['full']
    words=[]
    for m in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?",think):
        idx=[i for i,(s,e) in enumerate(offset) if e>m.start() and s<m.end()]
        if idx:words.append((max(norms[i] for i in idx),m.start(),m.end(),len(idx)))
    peak=max(words)
    controls=[w for w in words if (w[1],w[2])!=(peak[1],peak[2]) and abs(w[3]-peak[3])<=1]
    low=min(controls,key=lambda w:(w[0],abs(w[1]-peak[1]))) if controls else min(words)
    # Main variants delete a lexical span; they do not inject an instruction into a random word.
    variants={'original':original,
        'delete_high':original[:peak[1]]+original[peak[2]:],
        'delete_low':original[:low[1]]+original[low[2]:],
        'empty_thinking':original[len(think):]}
    paraphrase=think
    for old,new in [('I will','I plan to'),('look for','search for'),('allows me to','lets me'),('I need to','I should')]:
        if old in paraphrase:
            paraphrase=paraphrase.replace(old,new,1);break
    if paraphrase!=think:variants['paraphrase']=paraphrase+original[len(think):]
    alternatives=alternative_actions(r)
    alternative=alternatives.get('same_verb',alternatives.get('different_verb'))
    if alternative:
        variants['alternative_intent']=f'I will {alternative}.\n'+original[len(think):]
    mentions=d['exact_mentions']
    if mentions and alternative:
        for label,(s,e) in [('first',mentions[0]),('last',mentions[-1])]:
            variants[f'replace_{label}_mention']=original[:s]+alternative+original[e:]
    schema_match=re.search(r'<parameter=action>\s*',original)
    prefixes={'before_high':think[:peak[1]],'after_high':think[:peak[2]],
              'after_thinking':original[:original.index('</think>')+len('</think>')],
              'before_schema_action':original[:schema_match.end()]}
    if mentions:
        prefixes.update(before_first_mention=think[:mentions[0][0]],
                        after_first_mention=think[:mentions[0][1]],
                        after_last_mention=think[:mentions[-1][1]])
    # Regenerate everything after the edit, retaining no later original thinking.
    def deletion_anchor(span):
        later=[w for w in words if w[1]>=span[2]]
        return min(later,key=lambda w:w[1])[2] if later else span[2]
    def deletion_prefix(span):
        anchor=deletion_anchor(span)
        return think[:span[1]]+think[span[2]:anchor]
    prefixes['after_high_anchor']=think[:deletion_anchor(peak)]
    prefixes['after_low_anchor']=think[:deletion_anchor(low)]
    continuations={'original_thinking':think,'empty_thinking':'',
                   'alternative_intent':f'I will {alternative}.\n' if alternative else think,
                   'delete_high':deletion_prefix(peak),'delete_low':deletion_prefix(low)}
    if 'paraphrase' in variants:continuations['paraphrase']=paraphrase
    continuations['original_thinking_replicate']=think
    if mentions and alternative:
        for label,(s,e) in [('first',mentions[0]),('last',mentions[-1])]:
            continuations[f'replace_{label}_mention']=think[:s]+alternative
    cases.append({'sample_id':sid,'cohort':'enriched' if r['phase']=='enriched_literal' else 'main',
                  'category':r['category'],'success':r['success'],'game_file':r['gts']['game_file'],
                  'high_span':list(peak[1:3]),'low_span':list(low[1:3]),'alternative':alternative,
                  'variants':variants,'prefixes':prefixes,'continuations':continuations})
(root/'intervention_design.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
# IG subset selected independently of scores within each cohort/category.
ig=[]
for cohort in ['main','enriched']:
    for cat in sorted({c['category'] for c in cases}):
        ig.extend(c['sample_id'] for c in cases if c['cohort']==cohort and c['category']==cat)
ig=[next(c['sample_id'] for c in cases if c['cohort']==cohort and c['category']==cat)
    for cohort in ['main','enriched'] for cat in sorted({c['category'] for c in cases})]
(root/'ig_design.json').write_text(json.dumps(ig,indent=2))
print('cases',len(cases),'fixed conditions',sum(len(c['variants']) for c in cases),
      'generation conditions',sum(len(c['continuations'])+len(c['prefixes']) for c in cases),'IG',len(ig))
