"""Audit paired SFT attribution, task-clustered estimates, and conditional interventions."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
root = args.root
protocol = json.loads((root/'protocol.json').read_text())
raw = Path(protocol['raw_reference_experiment'])
old = Path(protocol['reference_experiment'])


def read(path):
    rows = [json.loads(line) for line in path.open()]
    return rows


def estimate(values, tasks):
    groups = defaultdict(list)
    for sid, value in values:
        groups[tasks[sid]].append(float(value))
    if not groups:
        return None
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(20261009)
    indices = rng.integers(0, len(sums), (10000, len(sums)))
    boot = sums[indices].sum(1)/counts[indices].sum(1)
    return {'mean': float(sums.sum()/counts.sum()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'n': int(counts.sum()), 'tasks': len(groups)}


def token_values(row, objective):
    if objective.startswith('contrast_'):
        item = row['contrasts'].get(objective.removeprefix('contrast_'))
        return item['grad_norms'] if item else None
    if objective == 'abs_grad_input_full':
        return np.abs(row['grad_input']['full'])
    return row['grad_norms'].get(objective)


def indicators(row, objective='full'):
    values = token_values(row, objective)
    if values is None:
        return None
    values = np.asarray(values)
    assert len(values) == len(row['thinking_tokens']) and np.isfinite(values).all()
    peak = int(np.argmax(values)); lo, hi = row['thinking_offsets'][peak]
    mentions = row['exact_mentions']
    hit = lambda span: hi > span[0] and lo < span[1]
    result = {'peak_first_3_tokens': peak < 3, 'peak_any_mention': any(hit(m) for m in mentions),
              'with_literal': bool(mentions), 'multiple_literals': len(mentions) > 1,
              'peak_first_mention': bool(mentions) and hit(mentions[0]),
              'peak_last_mention': bool(mentions) and hit(mentions[-1]),
              'peak_first_mention_first_token': False, 'peak_token': row['thinking_tokens'][peak]}
    if mentions:
        first = [i for i, (s, e) in enumerate(row['thinking_offsets']) if e > mentions[0][0] and s < mentions[0][1]]
        result['peak_first_mention_first_token'] = peak == first[0]
        result['mention_token_fraction'] = len(first)/len(values)
        result['mention_gradient_mass'] = float(values[first].sum()/max(values.sum(), 1e-30))
    return result


def attribution(rows):
    assert len(rows) == len({r['sample_id'] for r in rows})
    result = {'n': len(rows), 'objectives': {}}
    tasks = {r['sample_id']: r['game_file'] for r in rows}
    for objective in ['full', 'first', 'object', 'location', 'contrast_same_verb',
                      'contrast_different_verb', 'abs_grad_input_full']:
        pairs = [(r, indicators(r, objective)) for r in rows]
        pairs = [(r, i) for r, i in pairs if i is not None]
        counts = Counter(); peaks = Counter()
        for row, flags in pairs:
            counts['n'] += 1; peaks[flags['peak_token']] += 1
            for name, value in flags.items():
                if isinstance(value, (bool, np.bool_)):
                    counts[name] += int(value)
        stats = dict(counts, top_peak_tokens=peaks.most_common(10))
        literal = [(r, i) for r, i in pairs if i['with_literal']]
        stats['first_mention_rate_ci'] = estimate([(r['sample_id'], i['peak_first_mention']) for r, i in literal], tasks)
        if literal:
            stats['mean_mention_token_fraction'] = float(np.mean([i['mention_token_fraction'] for _, i in literal]))
            stats['mean_mention_gradient_mass'] = float(np.mean([i['mention_gradient_mass'] for _, i in literal]))
        result['objectives'][objective] = stats
    lengths = [len(r['thinking_tokens']) for r in rows]
    result['thinking_length'] = {'mean': float(np.mean(lengths)), 'median': float(np.median(lengths)),
                                 'min': min(lengths), 'max': max(lengths)}
    result['mean_action_logp'] = float(np.mean([r['scores']['full'] for r in rows]))
    result['mean_first_action_entropy'] = float(np.mean([r['action_token_entropies'][0] for r in rows]))
    result['by_category'] = {}
    for category in sorted({r['category'] for r in rows}):
        selected = [r for r in rows if r['category'] == category]
        result['by_category'][category] = dict(Counter(n=len(selected),
            with_literal=sum(bool(r['exact_mentions']) for r in selected),
            peak_first_mention=sum(indicators(r)['peak_first_mention'] for r in selected),
            peak_first_3_tokens=sum(indicators(r)['peak_first_3_tokens'] for r in selected)))
    result['action_distribution'] = Counter(r['action'].split()[0] for r in rows)
    result['multiword_actions'] = {}
    for objective in ['full', 'first', 'contrast_same_verb', 'contrast_different_verb']:
        selected = [r for r in rows if len(r['action'].split()) > 1 and token_values(r, objective) is not None]
        flags = [indicators(r, objective) for r in selected]
        result['multiword_actions'][objective] = {
            'n': len(selected), 'with_literal': sum(i['with_literal'] for i in flags),
            'peak_first_mention': sum(i['peak_first_mention'] for i in flags),
            'peak_first_mention_first_token': sum(i['peak_first_mention_first_token'] for i in flags)}
    return result


def interventions(filename, designfile, samplefiles):
    rows = read(root/filename)
    cases = {r['sample_id']: r for r in json.loads((root/designfile).read_text())}
    samples = {r['sample_id']: r for p in samplefiles for r in read(root/p)}
    expected = {(sid, mode, name) for sid, c in cases.items() for mode in ['prefixes', 'continuations'] for name in c[mode]}
    actual = {(r['sample_id'], r['mode'], r['condition']) for r in rows}
    assert len(actual) == len(rows) and actual == expected
    tasks = {sid: c['game_file'] for sid, c in cases.items()}
    metrics = defaultdict(dict)
    for row in rows:
        sid = row['sample_id']; outputs = row['outputs']; assert len(outputs) == 16
        metrics[sid][row['mode']+'/'+row['condition']] = {
            'p_original': sum(o['legal'] and o['action'] == samples[sid]['action'] for o in outputs)/16,
            'p_alternative': sum(o['legal'] and o['action'] == cases[sid]['alternative'] for o in outputs)/16,
            'valid_rate': sum(o['legal'] for o in outputs)/16,
        }
    result = {'cases': len(cases), 'conditions': len(rows), 'draws': len(rows)*16,
              'conditions_summary': {}, 'effects': {}}
    for condition in sorted({c for m in metrics.values() for c in m}):
        result['conditions_summary'][condition] = {
            metric: estimate([(sid, m[condition][metric]) for sid, m in metrics.items() if condition in m], tasks)
            for metric in ['p_original', 'p_alternative', 'valid_rate']}
    contrasts = {'first_replacement': ('continuations/replace_first_mention', 'prefixes/after_first_mention'),
                 'last_replacement': ('continuations/replace_last_mention', 'prefixes/after_last_mention'),
                 'first_commitment': ('prefixes/after_first_mention', 'prefixes/before_first_mention')}
    for label, (x, y) in contrasts.items():
        pairs = [sid for sid in metrics if x in metrics[sid] and y in metrics[sid]]
        if pairs:
            result['effects'][label] = {metric: estimate([(sid, metrics[sid][x][metric]-metrics[sid][y][metric])
                for sid in pairs], tasks) for metric in ['p_original', 'p_alternative', 'valid_rate']}
    (root/filename.replace('.jsonl', '_per_sample.json')).write_text(json.dumps(metrics, indent=2)+'\n')
    return result


def checks(filename):
    rows = json.loads((root/filename).read_text())
    return {'n': len(rows), 'same_peak': sum(r['peak_same'] for r in rows),
            'min_bf16_fp32_cosine': min(r['gradient_norm_cosine_bf16_fp32'] for r in rows),
            'max_future_gradient': max(r['causal_future_grad_max'] for r in rows),
            'finite_difference_error_by_epsilon': {str(eps): {
                'median': float(np.median(v)), 'max': max(v), 'above_10pct': sum(x > .1 for x in v)}
                for eps in [.001, .005, .02]
                for v in [[d['relative_error'] for r in rows for d in r['finite_differences'] if d['epsilon'] == eps]]}}


assert (root/'suite_complete.json').is_file()
cohorts = {
    'grpo_main': read(old/'attribution_0.jsonl')+read(old/'attribution_1.jsonl'),
    'grpo_enriched': read(old/'attribution_enriched_0.jsonl'),
    'raw_controlled_main': read(raw/'controlled_main.jsonl'),
    'raw_controlled_enriched': read(raw/'controlled_enriched.jsonl'),
    'sft_controlled_main': read(root/'controlled_main.jsonl'),
    'sft_controlled_enriched': read(root/'controlled_enriched.jsonl'),
    'raw_self_768': read(raw/'attribution_self.jsonl'),
    'sft_self_768': read(root/'attribution_self.jsonl'),
}
for name in ['main', 'enriched']:
    sftrows = cohorts['sft_controlled_'+name]
    assert len(sftrows) == (357 if name == 'main' else 60)
    assert {r['sample_id'] for r in sftrows} == {r['sample_id'] for r in cohorts['raw_controlled_'+name]}
samples = read(root/'samples_self.jsonl')
assert {r['sample_id'] for r in samples if r['eligible']} == {r['sample_id'] for r in cohorts['sft_self_768']}
overlap = json.loads((root/'dataset_overlap.json').read_text())
overlap_ids = set(overlap['overlap']['train']['exact_prompt_sample_ids']) | set(overlap['overlap']['validation']['exact_prompt_sample_ids'])
cohorts['sft_self_nonoverlap_prompts'] = [r for r in cohorts['sft_self_768'] if r['sample_id'].removeprefix('sft_self::') not in overlap_ids]
if (root/'raw_on_sft_self.jsonl').exists():
    cohorts['raw_on_sft_self'] = read(root/'raw_on_sft_self.jsonl')
    assert {r['sample_id'] for r in cohorts['raw_on_sft_self']} == {r['sample_id'] for r in cohorts['sft_self_768']}
result = {'protocol': protocol, 'dataset_overlap': overlap,
          'attribution': {k: attribution(v) for k, v in cohorts.items()}, 'paired_weight_changes': {}}
for name in ['main', 'enriched', 'sft_self']:
    if name == 'sft_self':
        if 'raw_on_sft_self' not in cohorts:
            continue
        new, previous = cohorts['sft_self_768'], cohorts['raw_on_sft_self']
    else:
        new, previous = cohorts['sft_controlled_'+name], cohorts['raw_controlled_'+name]
    previous = {r['sample_id']: r for r in previous}
    tasks = {r['sample_id']: r['game_file'] for r in new}
    result['paired_weight_changes'][name] = {}
    for metric in ['peak_first_3_tokens', 'peak_first_mention', 'peak_first_mention_first_token']:
        chosen = [r for r in new if metric == 'peak_first_3_tokens' or r['exact_mentions']]
        result['paired_weight_changes'][name][metric] = estimate([
            (r['sample_id'], int(indicators(r)[metric])-int(indicators(previous[r['sample_id']])[metric])) for r in chosen], tasks)
result['fresh_sft'] = json.loads((root/'fresh_summary.json').read_text())
result['fresh_raw'] = json.loads((raw/'fresh_summary.json').read_text())
result['controlled_interventions'] = interventions('controlled_generations.jsonl', 'controlled_generation_design.json',
                                                 ['samples.jsonl', 'samples_enriched.jsonl'])
result['self_interventions'] = interventions('self_intervention_generations.jsonl', 'self_generation_design.json', ['samples_self.jsonl'])
result['numerical_controlled'] = checks('numerical_checks.json')
result['numerical_self'] = checks('numerical_checks_self.json')
backend = json.loads((root/'backend_consistency.json').read_text())
result['backend'] = {'n': len(backend), 'max_token_logp_difference': max(r['max_token_abs_difference'] for r in backend),
                     'median_case_max_token_logp_difference': float(np.median([r['max_token_abs_difference'] for r in backend]))}
if (root/'precision_self.jsonl').exists():
    precise = read(root/'precision_self.jsonl')
    previous = {r['sample_id']: r for r in cohorts['sft_self_768']}
    assert len(precise) == len(previous) and {r['sample_id'] for r in precise} == previous.keys()
    merged = [{**previous[r['sample_id']], **r, 'contrasts': {}} for r in precise]
    result['precision'] = {'attribution': attribution(merged), 'model_dtype': 'float32', 'score_dtype': 'float64',
        'bf16_fp32_full_peak_agreement': sum(np.argmax(r['grad_norms']['full']) ==
            np.argmax(previous[r['sample_id']]['grad_norms']['full']) for r in precise).item(),
        'finite_difference_error_by_epsilon': {str(eps): {'n': len(v), 'median': float(np.median(v)), 'max': max(v)}
            for eps in [.001, .005, .02]
            for v in [[d['relative_error'] for r in precise for d in r.get('finite_differences', []) if d['epsilon'] == eps]]}}
fixed = read(root/'fixed_interventions.jsonl')
assert len(fixed) == len({(r['sample_id'], r['condition']) for r in fixed}) == 407
assert all(np.isfinite(list(r['scores'].values())).all() for r in fixed)
result['fixed_conditions'] = len(fixed)
result['audit_complete'] = True
(root/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
for name, item in result['attribution'].items():
    print(name, item['objectives']['full'])
print('FRESH', result['fresh_sft'])
print('SELF_INTERVENTIONS', result['self_interventions'])
