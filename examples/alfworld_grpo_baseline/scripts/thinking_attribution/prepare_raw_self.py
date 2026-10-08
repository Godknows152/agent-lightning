"""Build the raw-model self-output cohort without selecting a convenient resample."""
import json
import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--generation-file', default='fresh_generations.jsonl')
parser.add_argument('--suffix', default='')
args = parser.parse_args()
root = args.root
original = {r['sample_id']: r for line in (root/'samples.jsonl').open() if (r := json.loads(line))}
generated = [json.loads(line) for line in (root/args.generation_file).open()]
assert len(generated) == len({r['sample_id'] for r in generated}) == 360
assert len({len(r['outputs']) for r in generated}) == 1
stats = Counter()
rows = []
for generation in generated:
    source = original[generation['sample_id']]
    for draw, output in enumerate(generation['outputs']):
        stats['generations'] += 1
        stats['legal'] += output['legal']
        stats['closed_thinking'] += '</think>' in output['text']
        stats['length_exhausted'] += output['finish_reason'].get('type') == 'length'
        stats['reason/'+output['reason']] += 1
        if output['parser_status'] == 'valid' and not output['legal']:
            stats['not_admissible'] += 1
        thinking = output['text'].split('</think>')[0]
        mentions = list(re.finditer(r'(?<!\w)'+re.escape(output['action'])+r'(?!\w)', thinking, re.I)) if output['legal'] else []
        stats['legal_with_literal'] += bool(mentions)
        stats['legal_with_multiple_literals'] += len(mentions) >= 2
        if draw == 0:
            row = dict(source)
            row.update(sample_id='raw_self'+args.suffix+'::'+source['sample_id'], context_id=source['sample_id'],
                       output=output['text'], action=output['action'], success=None,
                       context_source_success=source['success'], score=None,
                       eligible=bool(output['legal'] and '</think>' in output['text'] and thinking.strip()),
                       model_path='/home/LXJ/Python_Projects/Models/Qwen3.5-2B', draw_index=0,
                       output_metadata=output, exact_mentions=[list(m.span()) for m in mentions],
                       raw_seed=generation['seed_base'], output_origin='raw model; no environment execution')
            rows.append(row)
rows.sort(key=lambda row: list(original).index(row['context_id']))
(root/f'samples_self{args.suffix}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
stats['self_attribution_eligible'] = sum(r['eligible'] for r in rows)
stats['self_attribution_literal'] = sum(bool(r['exact_mentions']) and r['eligible'] for r in rows)
stats['self_attribution_multiple'] = sum(len(r['exact_mentions']) >= 2 and r['eligible'] for r in rows)
(root/f'fresh_summary{args.suffix}.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))

# Freeze action-mention interventions before reading their model outcomes.
groups = defaultdict(list)
for row in rows:
    if row['eligible'] and len(row['exact_mentions']) >= 2 and len(row['action'].split()) > 1:
        groups[row['category']].append(row)
rng = random.Random(20261009)
selected = []
for category, candidates in sorted(groups.items()):
    rng.shuffle(candidates)
    chosen = []
    seen_sessions = set()
    for row in candidates:
        if row['session'] in seen_sessions:
            continue
        chosen.append(row)
        seen_sessions.add(row['session'])
        if len(chosen) == 4:
            break
    selected.extend(chosen)
import os.path
cases = []
for row in selected:
    text = row['output']; end = text.index('</think>'); thinking = text[:end]
    others = [a for a in row['admissible_actions'] if a != row['action']]
    same = [a for a in others if a.split()[0] == row['action'].split()[0]]
    alternative = max(same, key=lambda a: len(os.path.commonprefix([a, row['action']]))) if same else sorted(others)[0]
    first, last = row['exact_mentions'][0], row['exact_mentions'][-1]
    cases.append({'sample_id': row['sample_id'], 'cohort': 'raw_self_multiple_mentions',
                  'category': row['category'], 'game_file': row['gts']['game_file'],
                  'original_action': row['action'], 'alternative': alternative,
                  'prefixes': {'before_first_mention': thinking[:first[0]],
                               'after_first_mention': thinking[:first[1]],
                               'after_last_mention': thinking[:last[1]],
                               'after_thinking': text[:end+len('</think>')]},
                  'continuations': {'replace_first_mention': thinking[:first[0]]+alternative,
                                    'replace_last_mention': thinking[:last[0]]+alternative}})
(root/f'self_generation_design{args.suffix}.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
print(json.dumps(stats, ensure_ascii=False, indent=2))
print('Self intervention cases', len(cases), 'by category', dict(Counter(c['category'] for c in cases)))
