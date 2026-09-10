#!/usr/bin/env python3
"""GPU sampling diagnostic, not PPO. Stores raw IDs/logprobs and releases Engine.

Uses the same GenerateReqInput token-in/token-out interface as the VERL SGLang
server; only adds the native custom_logit_processor request field. Base 9B
weights are used (LoRA runtime enabled, no trained adapter loaded).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'config/diagnostics/thinking_budget_qwen35_9b.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'results.jsonl').exists():
        raise FileExistsError('Choose a fresh output directory; refusing to append to previous results')
    cfg = json.loads(args.config.read_text())
    import pandas as pd
    import sglang as sgl
    from transformers import AutoTokenizer
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.sampling.custom_logit_processor import Qwen3ThinkingBudgetLogitProcessor
    from alfworld_baseline.thinking_budget import (Qwen35TokenIdsOnlyBudget, ALFWorldThinkingBudgetLogitProcessor,
        ALFWorldNewlineThinkingBudgetLogitProcessor)
    from alfworld_baseline.prompts_qwen35 import PROMPT_VERSION, QWEN35_ALFWORLD_CHAT_TEMPLATE, build_user_prompt
    from alfworld_baseline.text_actions import parse_text_action
    from alfworld_baseline.alfworld_tool import ALFWorldTool
    from alfworld_baseline.tool_registry import ALFWorldToolRegistry
    import asyncio
    from verl.tools.schemas import OpenAIFunctionToolSchema

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    start = tokenizer.convert_tokens_to_ids('<think>')
    end = tokenizer.convert_tokens_to_ids('</think>')
    assert (start, end) == (248068, 248069)
    api_probe = {'sglang_version': sgl.__version__, 'sglang_source':sgl.__file__, 'prompt_version': PROMPT_VERSION,
                 'thinking_start_id': start, 'thinking_end_id': end,
                 'native_qwen3_ids': [151667, 151668], 'uses_trained_adapter': False}
    try:
        SamplingParams(thinking_budget=32)
    except TypeError as error:
        api_probe['direct_parameter_error'] = str(error)
    (args.output/'api_probe.json').write_text(json.dumps(api_probe, indent=2))
    rows = pd.read_parquet(ROOT/'data/qwen35_2b/train.parquet').iloc[:cfg['tasks']]
    tool = ALFWorldTool({'max_steps': 50}, OpenAIFunctionToolSchema.model_validate(ALFWorldToolRegistry.static_tool_schema()))
    prompts = []
    async def get_prompts():
        for _, row in rows.iterrows():
            game = row['extra_info']['game_file']
            instance, _ = await tool.create(create_kwargs={'game_file': game})
            try:
                obs, actions = tool.get_state(instance)
                mission = obs.split('Your task is to:', 1)[1].split('\n', 1)[0].strip()
                history = []
                for state_index in range(2):
                    text = build_user_prompt(mission=mission, observation=obs, admissible_actions=actions, history=history)
                    ids = tokenizer.apply_chat_template([{'role':'user','content':text}], tools=[],
                        chat_template=QWEN35_ALFWORLD_CHAT_TEMPLATE, enable_thinking=True,
                        add_generation_prompt=True, tokenize=True)
                    if hasattr(ids, 'keys'):
                        ids = ids['input_ids']
                    if ids and isinstance(ids[0], list):
                        ids = ids[0]
                    ids = list(map(int, ids))
                    prompts.append({'game_file':game, 'state_index':state_index, 'history':list(history),
                                    'prompt_ids':ids, 'prompt':tokenizer.decode(ids),
                                    'admissible_actions':list(actions)})
                    if state_index == 0:
                        command = next((a for a in actions if a.startswith('go to ')), actions[0])
                        _, _, metrics = await tool.execute(instance, {'action':command})
                        assert not metrics.get('error')
                        obs, actions = tool.get_state(instance)
                        history.append(f'{command} [executed]')
            finally:
                await tool.release(instance)
    asyncio.run(get_prompts())
    (args.output/'prompts.json').write_text(json.dumps(prompts,ensure_ascii=False,indent=2))
    variants = [('baseline', None, None), ('custom_params_only', 32, None),
                ('stock_qwen3',32,Qwen3ThinkingBudgetLogitProcessor.to_str()),
                ('qwen35_ids_only',32,Qwen35TokenIdsOnlyBudget.to_str())]
    variants += [(f'last_open_budget_{b}', b, ALFWorldThinkingBudgetLogitProcessor.to_str()) for b in cfg['budgets']]
    variants += [(f'newline_budget_{b}', b, ALFWorldNewlineThinkingBudgetLogitProcessor.to_str()) for b in (32,64,128)]
    if args.variants:
        unknown = set(args.variants) - {v[0] for v in variants}
        if unknown:
            raise ValueError(f'Unknown variants: {unknown}')
        variants = [v for v in variants if v[0] in args.variants]
    (args.output/'selected_variants.json').write_text(json.dumps([v[0] for v in variants]))
    engine_args = {k:v for k,v in cfg.items() if k not in ('sampling','budgets','tasks')}
    engine_args.update(trust_remote_code=True, random_seed=0, log_level='warning', skip_server_warmup=True)
    (args.output/'configuration.json').write_text(json.dumps(cfg,indent=2))
    print('Launching Qwen3.5-9B SGLang TP=2, base weights, LoRA runtime enabled',flush=True)
    # Use precisely the environment/kernel compatibility setup installed by
    # the project's VERL SGLang server, not standalone SGLang defaults.
    from verl.workers.rollout.sglang_rollout.sglang_rollout import _set_envs_and_config
    import sglang.srt.entrypoints.engine as engine_module
    engine_module._set_envs_and_config = _set_envs_and_config
    engine = sgl.Engine(**engine_args)
    results = []
    try:
        for name, budget, processor in variants:
            params = dict(cfg['sampling'])
            params['stop_token_ids'] = [248046, 248044]  # production VERL request stop IDs
            if budget is not None:
                params['custom_params'] = {'thinking_budget':budget}
            begin = time.monotonic()
            outputs = engine.generate(input_ids=[p['prompt_ids'] for p in prompts],sampling_params=params,
                custom_logit_processor=processor,return_logprob=True, logprob_start_len=-1)
            elapsed = time.monotonic()-begin
            for i, out in enumerate(outputs):
                meta = out['meta_info']
                token_logprobs = meta['output_token_logprobs']
                ids = out.get('output_ids')
                if ids is None:
                    ids = [int(entry[1]) for entry in token_logprobs]
                text = tokenizer.decode(ids,skip_special_tokens=False)
                close = ids.index(end) if end in ids else None
                action = parse_text_action(text)
                row = {'variant':name, 'budget':budget, 'task_index':i,'elapsed_batch_s':elapsed,
                       'thinking_tokens_before_close':close, 'closed':close is not None,
                       'state_index':prompts[i]['state_index'],
                       'output_tokens':len(ids), 'finish_reason':meta.get('finish_reason'),
                       'action':action,'valid_action':action in prompts[i]['admissible_actions'],
                       'close_logprob':token_logprobs[close][0] if close is not None else None,
                       'output_ids':ids,'output_token_logprobs':token_logprobs,'text':text}
                results.append(row)
                with (args.output/'results.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            selected=results[-len(prompts):]
            print(name,'thinking=',[r['thinking_tokens_before_close'] for r in selected],
                  'valid=',sum(r['valid_action'] for r in selected),'seconds=',round(elapsed,2),flush=True)
        summary=[]
        for name,budget,_ in variants:
            rs=[r for r in results if r['variant']==name]
            summary.append({'variant':name,'budget':budget,'n':len(rs),
                'closed':sum(r['closed'] for r in rs),'valid_actions':sum(r['valid_action'] for r in rs),
                'thinking_tokens':[r['thinking_tokens_before_close'] for r in rs],
                'budget_satisfied':None if budget is None else all(r['closed'] and r['thinking_tokens_before_close']<=budget + int(name.startswith('newline_')) for r in rs),
                'closing_prefix_allowance':int(name.startswith('newline_')),
                'close_logprobs':[r['close_logprob'] for r in rs]})
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        # Score the forced closing token using the SAME base model, but without
        # the logits processor. These are raw model conditional probabilities,
        # not probabilities from the forced sampling distribution.
        checks = [r for r in results if r['variant']=='last_open_budget_64'
                  and r['thinking_tokens_before_close']==64][:4]
        if checks:
            score_inputs = [prompts[r['task_index']]['prompt_ids'] + r['output_ids'][:65] for r in checks]
            scored = engine.generate(input_ids=score_inputs,
                sampling_params={'max_new_tokens':1,'temperature':1.0,'top_p':1.0,'top_k':-1},
                return_logprob=True,logprob_start_len=[len(prompts[r['task_index']]['prompt_ids'])-1 for r in checks])
            comparisons=[]
            for r, score in zip(checks,scored,strict=True):
                raw = score['meta_info']['input_token_logprobs']
                assert raw[-1][1] == end
                comparisons.append({'task_index':r['task_index'], 'state_index':r['state_index'],
                    'forced_sampling_close_logprob':r['close_logprob'],
                    'unconstrained_model_close_logprob':raw[-1][0]})
            (args.output/'logprob_comparison.json').write_text(json.dumps(comparisons,indent=2))
            print('Forced vs raw logprobs:',comparisons,flush=True)
        print('RESULTS',args.output,flush=True)
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
