"""Freeze a paired SFT attribution protocol and export its completed adapter."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sft-run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--resume-setup', action='store_true')
    args = parser.parse_args()
    root, run, reference = args.root.resolve(), args.sft_run.resolve(), args.reference.resolve()
    if args.resume_setup:
        previous = json.loads((root/'protocol.json').read_text())
        assert previous['sft_run'] == str(run) and not (root/'model_merged').exists()
    root.mkdir(parents=True, exist_ok=args.resume_setup)
    (root/'execution_logs').mkdir(exist_ok=args.resume_setup)
    adapter = run/'adapter/checkpoint-100'
    state = json.loads((adapter/'trainer_state.json').read_text())
    assert state['global_step'] == state['max_steps'] == 100
    assert digest(adapter/'adapter_model.safetensors') == digest(run/'adapter/adapter_model.safetensors')
    cfg = json.loads((adapter/'adapter_config.json').read_text())
    base = Path(cfg['base_model_name_or_path']).resolve()
    for name in ['samples.jsonl', 'samples_enriched.jsonl', 'sample_manifest.json',
                 'enrichment_manifest.json', 'intervention_design.json', 'ig_design.json',
                 'fresh_design.json', 'controlled_generation_design.json']:
        shutil.copy2(reference/name, root/name)
    protocol = {
        'started_at': datetime.now().astimezone().isoformat(),
        'model_kind': 'Qwen3.5-2B + new action_explicit_v1 SFT only; no project RL',
        'sft_run': str(run), 'adapter': str(adapter), 'base_model': str(base),
        'model_path': str(root/'model_merged'), 'step': 100,
        'adapter_sha256': digest(adapter/'adapter_model.safetensors'),
        'training_results': json.loads((run/'adapter/all_results.json').read_text()),
        'reference_experiment': json.loads((reference/'protocol.json').read_text())['reference_experiment'],
        'raw_reference_experiment': str(reference), 'physical_gpus': [0, 3],
        'controlled_texts': {'main': 357, 'enriched': 60},
        'fresh_generation': {'contexts': 360, 'draws': 2, 'budget': 768, 'temperature': 1.0,
                             'top_p': 1.0, 'top_k': -1, 'seeds': 'same context hash as raw experiment'},
        'self_selection': 'first draw only; parsed admissible XML, closed nonempty thinking; retain all failures',
        'objectives': ['full action logP', 'first action token logP', 'object/location logP',
                       'same/different verb logP margins', 'absolute gradient times input'],
        'self_interventions': 'up to 4 distinct sessions per category; select by seed 20261009 before outcomes; '
                              'first/last mention replacement each paired with its own unchanged prefix; 16 draws',
        'limitations': ['Logged GRPO contexts, not fresh SFT environment trajectories.',
                       'New SFT and old SFT+RL use different SFT datasets; not a pure RL ablation.',
                       'Attribution and action admissibility do not measure environment success.',
                       'No integrated-gradient claim in this run.'],
    }
    (root/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    torch.set_num_threads(8)
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16,
        device_map='cpu', local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model = model.merge_and_unload(safe_merge=True)
    assert not any('lora_' in name for name, _ in model.named_parameters())
    model.eval().requires_grad_(False)
    destination = root/'model_merged'
    model.save_pretrained(destination, safe_serialization=True, max_shard_size='4GB')
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    original = AutoTokenizer.from_pretrained(base, local_files_only=True)
    assert tokenizer.get_vocab() == original.get_vocab()
    tokenizer.save_pretrained(destination)
    AutoProcessor.from_pretrained(base, local_files_only=True).save_pretrained(destination)
    # Processor save may include a tokenizer; retain the SFT tokenizer explicitly.
    tokenizer.save_pretrained(destination)
    files = list(destination.glob('*')) + [adapter/'adapter_model.safetensors', adapter/'adapter_config.json']
    manifest = {str(p): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files if p.is_file()}
    (root/'model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('PREPARED', root, flush=True)


if __name__ == '__main__':
    main()
