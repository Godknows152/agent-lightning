"""Summarize paired experiments, retaining task-clustered uncertainty."""
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]);rng=np.random.default_rng(20261008)
def read(name):return [json.loads(l) for l in (root/name).open()]
cases={c['sample_id']:c for c in json.loads((root/'intervention_design.json').read_text())}
samples={r['sample_id']:r for name in ['samples.jsonl','samples_enriched.jsonl'] for r in read(name)}
attr=read('attribution_0.jsonl')+read('attribution_1.jsonl')+read('attribution_enriched_0.jsonl')
fixed=read('fixed_interventions.jsonl');gen=read('generations.jsonl')
assert len(attr)==417 and len({r['sample_id'] for r in attr})==417
assert len(fixed)==407
assert len(gen)==899 and len({(r['sample_id'],r['mode'],r['condition']) for r in gen})==899
assert all(len(g['outputs'])==16 for g in gen)
summary={'n_attribution':len(attr),'n_fixed_conditions':len(fixed),'n_generation_conditions':len(gen),'n_generations':sum(len(r['outputs']) for r in gen),'cohorts':{}}
def estimate(pairs):
    # Sample clusters of game instances, retaining every sampled decision in a cluster.
    clusters=defaultdict(list)
    for sid,value in pairs:clusters[cases[sid]['game_file']].append(value)
    sums=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()])
    idx=rng.integers(0,len(sums),(10000,len(sums)))
    boot=sums[idx].sum(1)/counts[idx].sum(1)
    return {'mean':float(sums.sum()/counts.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'n':int(counts.sum()),'tasks':len(clusters)}
fmap=defaultdict(dict)
for r in fixed:fmap[r['sample_id']][r['condition']]=r
gmap=defaultdict(dict)
for r in gen:
    sid=r['sample_id'];outputs=r['outputs'];valid=[o for o in outputs if o['legal']];ct=Counter(o['action'] for o in valid)
    value={'p_original':sum(o['legal'] and o['action']==samples[sid]['action'] for o in outputs)/16,
           'p_alternative':sum(o['legal'] and o['action']==cases[sid]['alternative'] for o in outputs)/16,
           'valid_rate':len(valid)/16,'unique_actions':len(ct),
           'entropy':-sum(n/len(valid)*math.log(n/len(valid)) for n in ct.values()) if valid else 0,
           'length_exhausted_rate':sum(o['finish_reason'].get('type')=='length' for o in outputs)/16}
    gmap[sid][r['mode']+'/'+r['condition']]=value
comparisons={
    'high_deletion_matched':('continuations/delete_high','prefixes/after_high_anchor'),
    'low_deletion_matched':('continuations/delete_low','prefixes/after_low_anchor'),
    'first_replacement_matched':('continuations/replace_first_mention','prefixes/after_first_mention'),
    'last_replacement_matched':('continuations/replace_last_mention','prefixes/after_last_mention'),
    'alternative_intent':('continuations/alternative_intent','continuations/original_thinking'),
    'paraphrase':('continuations/paraphrase','continuations/original_thinking'),
    'empty_thinking':('continuations/empty_thinking','continuations/original_thinking'),
    'first_mention_commitment':('prefixes/after_first_mention','prefixes/before_first_mention'),
    'last_vs_first_commitment':('prefixes/after_last_mention','prefixes/after_first_mention'),
    'schema_vs_end_thinking':('prefixes/before_schema_action','prefixes/after_thinking'),
    'independent_seed_replicate':('continuations/original_thinking_replicate','continuations/original_thinking'),
}
for cohort in ['main','enriched']:
    attrs=[r for r in attr if (r['phase']=='enriched_literal')==(cohort=='enriched')]
    ids=[sid for sid,c in cases.items() if c['cohort']==cohort]
    info={'attribution_n':len(attrs),'intervention_n':len(ids),'attribution':{},'fixed':{},'generation_conditions':{},'paired_effects':{}}
    for objective in ['first','full','object','location','contrast_same_verb','contrast_different_verb']:
        stats=Counter();peaks=Counter();ranks=[]
        for r in attrs:
            if objective.startswith('contrast_'):
                key=objective.removeprefix('contrast_')
                if key not in r['contrasts']:continue
                gs=r['contrasts'][key]['grad_norms']
            else:
                if objective not in r['grad_norms']:continue
                gs=r['grad_norms'][objective]
            assert np.isfinite(gs).all()
            peak=int(np.argmax(gs));lo,hi=r['thinking_offsets'][peak];stats['n']+=1
            peaks[r['thinking_tokens'][peak].strip()]+=1
            mentions=r['exact_mentions'];stats['with_literal']+=bool(mentions)
            if mentions:
                stats['peak_first']+=int(hi>mentions[0][0] and lo<mentions[0][1])
                stats['peak_last']+=int(hi>mentions[-1][0] and lo<mentions[-1][1])
                stats['peak_any']+=int(any(hi>s and lo<e for s,e in mentions))
        info['attribution'][objective]={**stats,'top_peak_tokens':peaks.most_common(8)}
    info['entropy']={'thinking_mean':st.mean(st.mean(r['thinking_token_entropies']) for r in attrs),
                     'schema_first_mean':st.mean(r['action_token_entropies'][0] for r in attrs)}
    for condition in sorted({k for sid in ids for k in fmap[sid]}):
        if condition=='original':continue
        pairs=[(sid,fmap[sid][condition]['scores']['full']-fmap[sid]['original']['scores']['full']) for sid in ids if condition in fmap[sid]]
        info['fixed'][condition]=estimate(pairs)
    pairs=[(sid,abs(fmap[sid]['delete_high']['scores']['full']-fmap[sid]['original']['scores']['full'])-abs(fmap[sid]['delete_low']['scores']['full']-fmap[sid]['original']['scores']['full'])) for sid in ids]
    info['fixed']['high_minus_low_absolute_effect']=estimate(pairs)
    for key in sorted({k for sid in ids for k in gmap[sid]}):
        vs=[gmap[sid][key] for sid in ids if key in gmap[sid]]
        info['generation_conditions'][key]={'n':len(vs),**{m:st.mean(v[m] for v in vs) for m in vs[0]}}
    for name,(x,y) in comparisons.items():
        pairs=[sid for sid in ids if x in gmap[sid] and y in gmap[sid]]
        if not pairs:continue
        info['paired_effects'][name]={metric:estimate([(sid,gmap[sid][x][metric]-gmap[sid][y][metric]) for sid in pairs]) for metric in ['p_original','p_alternative','valid_rate','entropy']}
    if cohort=='enriched':
        values=[]
        for sid in ids:
            a,b=comparisons['first_replacement_matched'];c,d=comparisons['last_replacement_matched']
            values.append((sid,(gmap[sid][a]['p_original']-gmap[sid][b]['p_original'])-(gmap[sid][c]['p_original']-gmap[sid][d]['p_original'])))
        info['first_minus_last_effect']=estimate(values)
    summary['cohorts'][cohort]=info
ig=read('integrated_gradients.jsonl');summary['ig']={}
for baseline in ['zero','space']:
    rs=[r for r in ig if r['baseline']==baseline]
    summary['ig'][baseline]={'n':len(rs),'pass_relative_error_10pct':sum(r['completeness_relative_error']<=.1 for r in rs),'median_relative_error':st.median(r['completeness_relative_error'] for r in rs)}
(root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
(root/'generation_summary_per_sample.json').write_text(json.dumps(gmap,indent=2,ensure_ascii=False))
for cohort,info in summary['cohorts'].items():
    print(cohort,'ATtribution',info['attribution']['full'])
    for key,metrics in info['paired_effects'].items():print(key,metrics['p_original'])
    if 'first_minus_last_effect' in info:print('first_minus_last',info['first_minus_last_effect'])
print('IG',summary['ig'])
