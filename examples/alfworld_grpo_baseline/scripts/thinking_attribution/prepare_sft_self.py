"""Select first-draw SFT outputs and freeze matched mention interventions."""
import argparse
import json
import os.path
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
root = args.root
original = {r['sample_id']: r for line in (root/'samples.jsonl').open() if (r := json.loads(line))}
generated = {r['sample_id']: r for line in (root/'fresh_generations.jsonl').open() if (r := json.loads(line))}
assert len(original) == len(generated) == 360 and original.keys() == generated.keys()
stats = Counter()
rows = []
for sid, source in original.items():
    generation = generated[sid]
    assert len(generation['outputs']) == 2
    for draw, output in enumerate(generation['outputs']):
        text, action = output['text'], output['action']
        thinking = text.split('</think>')[0]
        mentions = list(re.finditer(r'(?<!\w)'+re.escape(action)+r'(?!\w)', thinking, re.I)) if output['legal'] else []
        stats['generations'] += 1
        stats['legal'] += output['legal']
        stats['closed_thinking'] += '</think>' in text
        stats['length_exhausted'] += output['finish_reason'].get('type') == 'length'
        stats['reason/'+output['reason']] += 1
        stats['legal_with_literal'] += bool(mentions)
        stats['legal_with_multiple_literals'] += len(mentions) >= 2
        stats['not_admissible'] += output['parser_status'] == 'valid' and not output['legal']
        if draw == 0:
            row = dict(source)
            row.update(sample_id='sft_self::'+sid, context_id=sid, output=text, action=action,
                       success=None, context_source_success=source['success'], score=None,
                       eligible=bool(output['legal'] and '</think>' in text and thinking.strip()),
                       model_path=str((root/'model_merged').resolve()), draw_index=0,
                       output_metadata=output, exact_mentions=[list(m.span()) for m in mentions],
                       seed=generation['seed_base'], output_origin='new SFT generation; no environment execution')
            rows.append(row)
stats['self_attribution_eligible'] = sum(r['eligible'] for r in rows)
stats['self_attribution_literal'] = sum(r['eligible'] and bool(r['exact_mentions']) for r in rows)
stats['self_attribution_multiple'] = sum(r['eligible'] and len(r['exact_mentions']) >= 2 for r in rows)
(root/'samples_self.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
(root/'fresh_summary.json').write_text(json.dumps(stats, indent=2)+'\n')
groups = defaultdict(list)
for row in rows:
    if row['eligible'] and row['exact_mentions'] and len(row['action'].split()) > 1:
        groups[row['category']].append(row)
rng = random.Random(20261009)
selected = []
for category, candidates in sorted(groups.items()):
    rng.shuffle(candidates)
    sessions = set()
    for row in candidates:
        if row['session'] in sessions:
            continue
        selected.append(row)
        sessions.add(row['session'])
        if len(sessions) == 4:
            break
cases = []
for row in selected:
    text = row['output']; end = text.index('</think>'); thinking = text[:end]
    others = [a for a in row['admissible_actions'] if a != row['action']]
    same = [a for a in others if a.split()[0] == row['action'].split()[0]]
    alternative = max(same, key=lambda a: len(os.path.commonprefix([a, row['action']]))) if same else sorted(others)[0]
    first, last = row['exact_mentions'][0], row['exact_mentions'][-1]
    case = {'sample_id': row['sample_id'], 'category': row['category'], 'game_file': row['gts']['game_file'],
            'original_action': row['action'], 'alternative': alternative,
            'mention_count': len(row['exact_mentions']),
            'prefixes': {'before_first_mention': thinking[:first[0]],
                         'after_first_mention': thinking[:first[1]],
                         'after_thinking': text[:end+len('</think>')]},
            'continuations': {'replace_first_mention': thinking[:first[0]]+alternative}}
    if len(row['exact_mentions']) > 1:
        case['prefixes']['after_last_mention'] = thinking[:last[1]]
        case['continuations']['replace_last_mention'] = thinking[:last[0]]+alternative
    cases.append(case)
(root/'self_generation_design.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(stats, indent=2), flush=True)
print('INTERVENTION_CASES', len(cases), 'MULTIPLE', sum(c['mention_count'] > 1 for c in cases), flush=True)
