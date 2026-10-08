"""Build a stratified, auditable sample of the completed ALFWorld validation."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261008)
    sessions = defaultdict(list)
    for line in a.source.open():
        row = json.loads(line)
        session, step = row['uid'].rsplit('_', 1)
        row['decision_index'] = int(step)
        sessions[session].append(row)
    groups = defaultdict(list)
    population = Counter()
    mention_counts = Counter()
    for session, rows in sessions.items():
        rows.sort(key=lambda x: x['decision_index'])
        category = Path(rows[0]['gts']['game_file']).parts[-3].split('-')[0]
        success = rows[-1]['score'] > 0
        groups[category, success].append((session, rows))
        population[category, success] += 1
        for row in rows:
            text = row['output']
            m = re.search(r'<parameter=action>\s*(.*?)\s*</parameter>', text, re.S)
            mention_counts['decisions'] += 1
            if m and '</think>' in text:
                think = text.split('</think>')[0]
                action = m.group(1).strip()
                mention_counts['parseable'] += 1
                mention_counts['exact_action_mentioned'] += bool(re.search(r'(?<!\w)' + re.escape(action) + r'(?!\w)', think, re.I))
                mention_counts['action_verb_mentioned'] += bool(re.search(r'\b' + re.escape(action.split()[0]) + r'\b', think, re.I))
    selected = []
    for category in sorted({g[0] for g in groups}):
        pools = {s: groups[category, s][:] for s in [False, True]}
        for pool in pools.values():
            rng.shuffle(pool)
            # Favor distinct task instances within each outcome stratum.
            seen = set()
            pool.sort(key=lambda pair: (pair[1][0]['gts']['game_file'] in seen or seen.add(pair[1][0]['gts']['game_file'])) is True)
        success_n = min(10, len(pools[True]))
        for session, rows in pools[True][:success_n] + pools[False][:20-success_n]:
            for phase, index in [('early', 0), ('middle', len(rows)//2), ('late', len(rows)-1)]:
                row = dict(rows[index])
                m = re.search(r'<parameter=action>\s*(.*?)\s*</parameter>', row['output'], re.S)
                match = re.search(r'Your admissible actions of the current situation are: \[(.*?)\]\.\n', row['input'], re.S)
                actions = re.findall(r"'([^']*)'", match.group(1)) if match else []
                row.update(sample_id=f'{session}_{index}', session=session, category=category,
                           success=rows[-1]['score'] > 0, phase=phase, trajectory_length=len(rows),
                           action=m.group(1).strip() if m else None, admissible_actions=actions)
                row['eligible'] = bool(m and '</think>' in row['output'] and row['action'] in actions)
                selected.append(row)
    assert len(selected) == 360
    assert len({r['sample_id'] for r in selected}) == 360
    with (a.out/'samples.jsonl').open('w') as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    summary = {'seed': 20261008, 'source': str(a.source.resolve()),
               'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
               'population': {str(k): v for k, v in population.items()},
               'population_mentions': dict(mention_counts), 'selected_decisions': len(selected),
               'selected_trajectories': len({r['session'] for r in selected}),
               'selected_tasks': len({r['gts']['game_file'] for r in selected}),
               'eligible_decisions': sum(r['eligible'] for r in selected),
               'selected_strata': dict(Counter(f"{r['category']}/{r['success']}" for r in selected))}
    (a.out/'sample_manifest.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
