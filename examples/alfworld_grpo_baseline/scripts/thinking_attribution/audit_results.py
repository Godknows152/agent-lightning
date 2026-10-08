"""Audit saved experiment outputs and add sensitivity/uncertainty summaries (CPU only)."""
import hashlib
import importlib.metadata
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()


def read(name):
    return [json.loads(line) for line in (root / name).open()]


summary = json.loads((root / 'summary.json').read_text())
cases = {r['sample_id']: r for r in json.loads((root / 'intervention_design.json').read_text())}
samples = {r['sample_id']: r for name in ['samples.jsonl', 'samples_enriched.jsonl'] for r in read(name)}
attrs = [r for name in ['attribution_0.jsonl', 'attribution_1.jsonl', 'attribution_enriched_0.jsonl'] for r in read(name)]
generations = read('generations.jsonl')
fixed = read('fixed_interventions.jsonl')
assert len(attrs) == len({r['sample_id'] for r in attrs}) == 417
assert len(fixed) == len({(r['sample_id'], r['condition']) for r in fixed}) == 407
actual_keys = {(r['sample_id'], r['mode'], r['condition']) for r in generations}
planned_keys = {(sid, mode, condition) for sid, c in cases.items()
                for mode in ['prefixes', 'continuations'] for condition in c[mode]}
assert actual_keys == planned_keys and len(generations) == 899
for row in generations:
    c = cases[row['sample_id']]
    assert row['prefix'] == c[row['mode']][row['condition']]
    assert len(row['outputs']) == 16 and 0 < row['max_new_tokens'] <= 768
    for output in row['outputs']:
        assert output['legal'] == (output['parser_status'] == 'valid'
                                   and output['action'] in samples[row['sample_id']]['admissible_actions'])
        assert output['completion_tokens'] <= row['max_new_tokens']
for row in attrs:
    for field in ['grad_norms', 'grad_input']:
        for values in row[field].values():
            assert len(values) == len(row['thinking_tokens']) and np.isfinite(values).all()
    assert np.isfinite(list(row['scores'].values())).all()
for row in fixed:
    assert np.isfinite(list(row['scores'].values())).all()

rng = np.random.default_rng(20261020)


def estimate(pairs):
    clusters = defaultdict(list)
    for sid, value in pairs:
        clusters[cases[sid]['game_file']].append(value)
    totals = np.array([sum(v) for v in clusters.values()])
    counts = np.array([len(v) for v in clusters.values()])
    draws = rng.integers(0, len(totals), (10000, len(totals)))
    bootstrap = totals[draws].sum(1) / counts[draws].sum(1)
    return {'mean': float(totals.sum() / counts.sum()),
            'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'n': int(counts.sum()), 'tasks': len(counts)}


gmap = json.loads((root / 'generation_summary_per_sample.json').read_text())
for cohort, info in summary['cohorts'].items():
    ids = [sid for sid, c in cases.items() if c['cohort'] == cohort]
    info['condition_intervals'] = {}
    for key in info['generation_conditions']:
        info['condition_intervals'][key] = {
            metric: estimate([(sid, gmap[sid][key][metric]) for sid in ids if key in gmap[sid]])
            for metric in ['p_original', 'entropy']}
    differences = []
    for sid in ids:
        r = gmap[sid]
        delta = (r['continuations/delete_high']['p_original'] - r['prefixes/after_high_anchor']['p_original']
                 - r['continuations/delete_low']['p_original'] + r['prefixes/after_low_anchor']['p_original'])
        differences.append((sid, delta))
    info['high_minus_low_deletion_effect'] = estimate(differences)
    selected = [r for r in attrs if (r['phase'] == 'enriched_literal') == (cohort == 'enriched')]
    peaks = []
    for row in selected:
        index = int(np.argmax(np.abs(row['grad_input']['full'])))
        start, end = row['thinking_offsets'][index]
        mentions = row['exact_mentions']
        peaks.append({'sample_id': row['sample_id'], 'index': index,
                      'in_first': bool(mentions and end > mentions[0][0] and start < mentions[0][1]),
                      'in_any': any(end > s and start < e for s, e in mentions)})
    info['grad_times_input_abs_peak'] = {'n': len(peaks), 'in_first': sum(r['in_first'] for r in peaks),
                                         'in_any': sum(r['in_any'] for r in peaks)}

numeric = json.loads((root / 'numerical_checks.json').read_text())
assert len(numeric) == 12
summary['numerical_checks'] = {
    'n': 12, 'peak_same': sum(r['peak_same'] for r in numeric),
    'minimum_gradient_cosine': min(r['gradient_norm_cosine_bf16_fp32'] for r in numeric),
    'maximum_causal_future_gradient': max(r['causal_future_grad_max'] for r in numeric),
    'finite_difference_errors_by_epsilon': {}}
for epsilon in [.001, .005, .02]:
    errors = [e['relative_error'] for r in numeric for e in r['finite_differences'] if e['epsilon'] == epsilon]
    summary['numerical_checks']['finite_difference_errors_by_epsilon'][str(epsilon)] = {
        'n': len(errors), 'median': statistics.median(errors), 'maximum': max(errors)}

backend = json.loads((root / 'backend_consistency.json').read_text())
diffs = [abs(a - b) for r in backend for a, b in zip(r['hf_logprobs'], r['sglang_logprobs'])]
summary['backend_check'] = {'n_cases': len(backend), 'n_tokens': len(diffs),
                            'median_absolute_logprob_difference': statistics.median(diffs),
                            'mean_absolute_logprob_difference': statistics.mean(diffs),
                            'maximum_absolute_logprob_difference': max(diffs)}

refined = [json.loads(line) for path in sorted(root.glob('integrated_gradients_fp32*.jsonl'))
           for line in path.open()]
assert len(refined) == len({(r['sample_id'], r['baseline']) for r in refined}) == 24
attr_by_id = {r['sample_id']: r for r in attrs}
summary['ig_refined'] = {}
for baseline in ['zero', 'space']:
    rows = [r for r in refined if r['baseline'] == baseline]
    accepted = [r for r in rows if r['accepted']]
    peaks = []
    for row in accepted:
        attr = attr_by_id[row['sample_id']]
        assert len(row['contributions']) == len(attr['thinking_tokens'])
        index = int(np.argmax(np.abs(row['contributions'])))
        start, end = attr['thinking_offsets'][index]
        mentions = attr['exact_mentions']
        peaks.append({'sample_id': row['sample_id'], 'token': attr['thinking_tokens'][index],
                      'in_first': bool(mentions and end > mentions[0][0] and start < mentions[0][1])})
    summary['ig_refined'][baseline] = {
        'n': len(rows), 'accepted': len(accepted),
        'pass_relative_error_10pct': sum(r['completeness_relative_error'] <= .1 for r in rows),
        'median_relative_error_all': statistics.median(r['completeness_relative_error'] for r in rows),
        'accepted_peaks': peaks}

source_manifest = json.loads((root / 'sample_manifest.json').read_text())
assert hashlib.sha256(Path(source_manifest['source']).read_bytes()).hexdigest() == source_manifest['source_sha256']
summary['audit'] = {'unique_generation_conditions': 899, 'total_generations': 14384,
                    'attributions': 417, 'fixed_conditions': 407, 'refined_ig_conditions': 24,
                    'source_sha256_verified': True, 'all_checks_passed': True}
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
protocol = json.loads((root / 'protocol.json').read_text())
protocol['sampling']['top_k'] = -1
protocol['actual_design']['ig_refinement'] = 'FP32; Gauss-Legendre alpha=u^4; 64 then 256 nodes'
protocol['actual_design']['ig_refinement_acceptance'] = 'relative completeness error <= 0.10 OR absolute <= 0.02 nats'
protocol['amendments'].append({
    'reason': 'BF16 integrated gradients frequently failed completeness checks.',
    'change': 'Repeat the same 12 decisions / 2 baselines in FP32 with endpoint-focused quadrature; retain all failed results.'})
# Keep this audit idempotent.
protocol['amendments'] = list({json.dumps(x, sort_keys=True): x for x in protocol['amendments']}.values())
(root / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
versions = {}
for package in ['torch', 'transformers', 'sglang', 'flashinfer-python', 'sgl-kernel', 'numpy', 'scipy']:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
manifest = {'versions': versions, 'physical_gpus_used': [0, 3], 'model_step': 666,
            'source_sha256': source_manifest['source_sha256'], 'files': {}}
for path in sorted(root.glob('*.json*')):
    if path.name != 'artifact_manifest.json':
        manifest['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
for path in sorted(Path(__file__).resolve().parent.glob('*.py')):
    manifest['files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({k: summary[k] for k in ['audit', 'numerical_checks', 'ig_refined']}, ensure_ascii=False, indent=2))
