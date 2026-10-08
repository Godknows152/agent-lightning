"""Audit and compare frozen original-weight attribution and conditional interventions."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
root = args.root
protocol = json.loads((root/'protocol.json').read_text())
old = Path(protocol['reference_experiment'])
rng = np.random.default_rng(20261010)


def read(path):
    return [json.loads(line) for line in path.open()]


def attribution(rows):
    result = {'n': len(rows), 'objectives': {}}
    for objective in ['full', 'first', 'object', 'location', 'contrast_same_verb',
                      'contrast_different_verb', 'abs_grad_input_full']:
        counts = Counter()
        peaks = Counter()
        for row in rows:
            if objective.startswith('contrast_'):
                item = row['contrasts'].get(objective.removeprefix('contrast_'))
                if not item:
                    continue
                values = item['grad_norms']
            elif objective == 'abs_grad_input_full':
                values = np.abs(row['grad_input']['full'])
            else:
                values = row['grad_norms'].get(objective)
                if values is None:
                    continue
            assert np.isfinite(values).all() and len(values) == len(row['thinking_tokens'])
            peak = int(np.argmax(values))
            lo, hi = row['thinking_offsets'][peak]
            mentions = row['exact_mentions']
            counts['n'] += 1
            counts['with_literal'] += bool(mentions)
            counts['peak_first_3_tokens'] += peak < 3
            peaks[row['thinking_tokens'][peak]] += 1
            if mentions:
                counts['peak_first_mention'] += hi > mentions[0][0] and lo < mentions[0][1]
                counts['peak_last_mention'] += hi > mentions[-1][0] and lo < mentions[-1][1]
                counts['peak_any_mention'] += any(hi > start and lo < end for start, end in mentions)
                first_token = next(i for i, (s, e) in enumerate(row['thinking_offsets'])
                                   if e > mentions[0][0] and s < mentions[0][1])
                counts['peak_first_mention_first_token'] += peak == first_token
        result['objectives'][objective] = {**counts, 'top_peak_tokens': peaks.most_common(8)}
    result['schema_first_token_entropy_mean'] = float(np.mean([r['action_token_entropies'][0] for r in rows]))
    matched = [r for r in rows if 'same_verb' in r['contrasts'] and r['exact_mentions']]
    result['matched_same_verb_literal'] = {'n': len(matched)}
    for key in ['full', 'contrast']:
        hits = 0
        for row in matched:
            values = row['grad_norms']['full'] if key == 'full' else row['contrasts']['same_verb']['grad_norms']
            lo, hi = row['thinking_offsets'][int(np.argmax(values))]
            start, end = row['exact_mentions'][0]
            hits += hi > start and lo < end
        result['matched_same_verb_literal'][key+'_peak_first_mention'] = hits
    return result


def estimate(values, cases):
    clusters = defaultdict(list)
    for sid, value in values:
        clusters[cases[sid]['game_file']].append(value)
    sums = np.array([sum(v) for v in clusters.values()])
    counts = np.array([len(v) for v in clusters.values()])
    indices = rng.integers(0, len(sums), (10000, len(sums)))
    boot = sums[indices].sum(1)/counts[indices].sum(1)
    return {'mean': float(sums.sum()/counts.sum()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'n': int(counts.sum()), 'tasks': len(clusters)}


def generations(filename, designfile, samplefiles):
    rows = read(root/filename)
    cases = {r['sample_id']: r for r in json.loads((root/designfile).read_text())}
    samples = {r['sample_id']: r for p in samplefiles for r in read(root/p)}
    expected = {(sid, mode, name) for sid, c in cases.items() for mode in ['prefixes', 'continuations']
                for name in c[mode]}
    actual = {(r['sample_id'], r['mode'], r['condition']) for r in rows}
    assert len(actual) == len(rows) and actual == expected
    metrics = defaultdict(dict)
    for row in rows:
        sid = row['sample_id']
        outputs = row['outputs']
        assert len(outputs) == 16
        valid = [r for r in outputs if r['legal']]
        counts = Counter(r['action'] for r in valid)
        metrics[sid][row['mode']+'/'+row['condition']] = {
            'p_original': sum(r['legal'] and r['action'] == samples[sid]['action'] for r in outputs)/16,
            'p_alternative': sum(r['legal'] and r['action'] == cases[sid]['alternative'] for r in outputs)/16,
            'valid_rate': len(valid)/16,
            'entropy_valid_only': -sum(n/len(valid)*math.log(n/len(valid)) for n in counts.values()) if valid else 0,
        }
    contrasts = {
        'first_replacement': ('continuations/replace_first_mention', 'prefixes/after_first_mention'),
        'last_replacement': ('continuations/replace_last_mention', 'prefixes/after_last_mention'),
        'first_commitment': ('prefixes/after_first_mention', 'prefixes/before_first_mention'),
    }
    result = {'cases': len(cases), 'conditions': len(rows), 'draws': len(rows)*16, 'conditions_summary': {}, 'effects': {}}
    for condition in sorted(next(iter(metrics.values()))):
        result['conditions_summary'][condition] = {
            metric: estimate([(sid, metrics[sid][condition][metric]) for sid in cases], cases)
            for metric in next(iter(metrics.values()))[condition]}
    for name, (x, y) in contrasts.items():
        result['effects'][name] = {
            metric: estimate([(sid, metrics[sid][x][metric]-metrics[sid][y][metric]) for sid in cases], cases)
            for metric in next(iter(metrics.values()))[x]}
    result['first_minus_last'] = estimate([
        (sid, (metrics[sid]['continuations/replace_first_mention']['p_original']-
               metrics[sid]['prefixes/after_first_mention']['p_original'])-
              (metrics[sid]['continuations/replace_last_mention']['p_original']-
               metrics[sid]['prefixes/after_last_mention']['p_original'])) for sid in cases], cases)
    (root/filename.replace('.jsonl', '_summary.json')).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return result


def checks(filename):
    rows = json.loads((root/filename).read_text())
    return {'n': len(rows), 'same_peak': sum(r['peak_same'] for r in rows),
            'min_bf16_fp32_cosine': min(r['gradient_norm_cosine_bf16_fp32'] for r in rows),
            'max_future_gradient': max(r['causal_future_grad_max'] for r in rows),
            'finite_difference_error_by_epsilon': {
                str(eps): {'median': float(np.median(v)), 'max': max(v)}
                for eps in [.001, .005, .02]
                for v in [[d['relative_error'] for r in rows for d in r['finite_differences'] if d['epsilon'] == eps]]}}


oldmain = read(old/'attribution_0.jsonl')+read(old/'attribution_1.jsonl')
oldextra = read(old/'attribution_enriched_0.jsonl')
main = read(root/'controlled_main.jsonl')
extra = read(root/'controlled_enriched.jsonl')
assert len(main) == 357 and len(extra) == 60
assert {r['sample_id'] for r in main} == {r['sample_id'] for r in oldmain}
assert {r['sample_id'] for r in extra} == {r['sample_id'] for r in oldextra}
result = {'protocol': protocol, 'attribution': {}}
for label, rows in [('grpo_main', oldmain), ('grpo_enriched', oldextra), ('raw_controlled_main', main),
                    ('raw_controlled_enriched', extra), ('raw_self_768', read(root/'attribution_self.jsonl')),
                    ('raw_self_1536', read(root/'attribution_self_1536.jsonl'))]:
    assert len(rows) == len({r['sample_id'] for r in rows})
    result['attribution'][label] = attribution(rows)
for suffix in ['', '_1536']:
    samples = read(root/f'samples_self{suffix}.jsonl')
    attrs = read(root/f'attribution_self{suffix}.jsonl')
    assert {r['sample_id'] for r in samples if r['eligible']} == {r['sample_id'] for r in attrs}
result['fresh_768'] = json.loads((root/'fresh_summary.json').read_text())
result['fresh_1536'] = json.loads((root/'fresh_summary_1536.json').read_text())
result['controlled_interventions'] = generations('controlled_generations.jsonl', 'controlled_generation_design.json',
                                               ['samples.jsonl', 'samples_enriched.jsonl'])
result['self_interventions'] = generations('self_intervention_generations.jsonl', 'self_generation_design.json',
                                         ['samples_self.jsonl'])
result['numerical_controlled'] = checks('numerical_checks.json')
result['numerical_self'] = checks('numerical_checks_self.json')
result['finite_difference_refinement'] = json.loads((root/'finite_difference_refinement.json').read_text())
backend = json.loads((root/'backend_consistency.json').read_text())
result['backend'] = {'n': len(backend), 'max_token_logp_difference': max(r['max_token_abs_difference'] for r in backend),
                     'median_case_max_token_logp_difference': float(np.median([r['max_token_abs_difference'] for r in backend]))}
fixed = read(root/'fixed_interventions.jsonl')
assert len(fixed) == 407 and len({(r['sample_id'], r['condition']) for r in fixed}) == 407
assert all(np.isfinite(list(r['scores'].values())).all() for r in fixed)
result['fixed_conditions'] = len(fixed)
result['audit_complete'] = True
(root/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
for key, value in result['attribution'].items():
    print(key, value['objectives']['full'])
for key in ['controlled_interventions', 'self_interventions']:
    print(key, {k: v['p_original'] for k, v in result[key]['effects'].items()}, result[key]['first_minus_last'])
