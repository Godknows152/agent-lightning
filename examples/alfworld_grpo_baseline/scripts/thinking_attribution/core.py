"""Shared token alignment and frozen-model operations for attribution experiments."""
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from alfworld_baseline.prompts_gigpo import QWEN3_ALFWORLD_CHAT_TEMPLATE


def load(model_path):
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map='cuda:0', local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    print('MODEL_READY', os.environ.get('CUDA_VISIBLE_DEVICES'), time.time(), flush=True)
    return model, tokenizer


def prompt_ids(tokenizer, row):
    prefix, suffix = 'user\n', '\nassistant\n<think>\n'
    text = row['input']
    assert text.startswith(prefix) and text.endswith(suffix)
    content = text[len(prefix):-len(suffix)]
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
        chat_template=QWEN3_ALFWORLD_CHAT_TEMPLATE, enable_thinking=True,
        add_generation_prompt=True, tokenize=True)
    if hasattr(ids, 'keys'):
        ids = ids['input_ids']
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    assert tokenizer.decode(ids, skip_special_tokens=True) == text, 'Prompt reconstruction differs from logged input'
    return ids


def layout(tokenizer, row, response=None):
    text = row['output'] if response is None else response
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc['input_ids'], enc['offset_mapping']
    assert tokenizer.decode(ids, skip_special_tokens=False) == text
    think_end = text.find('</think>')
    assert think_end >= 0
    # A raw model may quote XML examples inside its thinking. Attribute only
    # the actual call after the thinking block, not the first quoted schema.
    match = re.compile(r'<parameter=action>\s*(.*?)\s*</parameter>', re.S).search(
        text, think_end + len('</think>'))
    assert match
    action = match.group(1).strip()
    start = match.start(1)
    end = start + len(action)
    prompt = prompt_ids(tokenizer, row)
    # Token spans may include neighboring whitespace, but never target XML itself.
    target = [i for i, (s, e) in enumerate(offsets) if e > start and s < end]
    thinking = [i for i, (s, e) in enumerate(offsets) if s < think_end and e > 0]
    assert target
    return dict(prompt=prompt, response_ids=ids, offsets=offsets, target=target,
                thinking=thinking, text=text, action=action, action_start=start,
                action_end=end, think_end=think_end, full_ids=prompt+ids)


def forward_scores(model, tokenizer, row, response=None, need_grad=True, embedding_override=None,
                   baseline_embeddings=None, alpha=None, include_thinking_entropy=True):
    loc = layout(tokenizer, row, response)
    tensor = torch.tensor([loc['full_ids']], device=model.device)
    emb = model.get_input_embeddings()(tensor).detach()
    if embedding_override is not None:
        emb = embedding_override
    if baseline_embeddings is not None:
        emb = baseline_embeddings + alpha * (emb - baseline_embeddings)
    emb = emb.detach().requires_grad_(need_grad)
    p = len(loc['prompt'])
    target_positions = [p+i-1 for i in loc['target']]
    think_positions = [p+i-1 for i in loc['thinking']] if include_thinking_entropy else []
    positions = sorted(set(target_positions+think_positions))
    indexes = {pos: i for i, pos in enumerate(positions)}
    out = model(inputs_embeds=emb, attention_mask=torch.ones_like(tensor), use_cache=False,
                logits_to_keep=torch.tensor(positions, device=model.device))
    logits = out.logits[0].float()
    logprobs = logits.log_softmax(-1)
    lp = torch.stack([logprobs[indexes[pos], loc['response_ids'][i]] for pos, i in zip(target_positions, loc['target'])])
    scores = {'full': lp.sum(), 'first': lp[0]}
    action = loc['action']
    # Verb/object/location character spans, evaluated conditional on sampled preceding action tokens.
    verb_end = len(action.split()[0])
    slot_spans = {'verb': (0, verb_end)}
    relation = re.search(r'\s(?:from|to|with)\s', action)
    if relation:
        slot_spans['object'] = (verb_end, relation.start())
        slot_spans['location'] = (relation.end(), len(action))
    elif len(action) > verb_end:
        slot_spans['object'] = (verb_end, len(action))
    for key, (lo, hi) in slot_spans.items():
        select = [k for k, i in enumerate(loc['target'])
                  if loc['offsets'][i][1] > loc['action_start']+lo
                  and loc['offsets'][i][0] < loc['action_start']+hi]
        if select:
            scores[key] = lp[select].sum()
    with torch.no_grad():
        entropy = -(logprobs.exp()*logprobs).sum(-1)
        summaries = {'action_token_logprobs': lp.detach().cpu().tolist(),
                     'action_token_entropies': [float(entropy[indexes[pos]]) for pos in target_positions],
                     'thinking_token_entropies': [float(entropy[indexes[pos]]) for pos in think_positions]}
    return loc, emb, scores, summaries


def replace_action(loc, action):
    return loc['text'][:loc['action_start']]+action+loc['text'][loc['action_end']:]


def gradient(model, tokenizer, row, response=None, objectives=('full', 'first', 'object', 'location')):
    loc, emb, scores, metrics = forward_scores(model, tokenizer, row, response)
    p = len(loc['prompt'])
    indices = torch.tensor([p+i for i in loc['thinking']], device=model.device)
    matrices = {}
    for name in objectives:
        if name not in scores:
            continue
        g = torch.autograd.grad(scores[name], emb, retain_graph=True)[0][0, indices].float()
        matrices[name] = g.detach().cpu()
    metrics['scores'] = {k: float(v.detach()) for k,v in scores.items()}
    metrics['grad_norms'] = {k: g.norm(dim=-1).tolist() for k,g in matrices.items()}
    metrics['grad_input'] = {k: (g*emb[0, indices].detach().float().cpu()).sum(-1).tolist() for k,g in matrices.items()}
    return loc, metrics, matrices


def alternative_actions(row):
    action = row['action']
    candidates = [a for a in row['admissible_actions'] if a != action]
    # Match textual prefix to isolate differences in object or location.
    import os.path
    same = [a for a in candidates if a.split()[0] == action.split()[0]]
    different = [a for a in candidates if a.split()[0] != action.split()[0]]
    result = {}
    if same:
        result['same_verb'] = max(same, key=lambda a: len(os.path.commonprefix([a, action])))
    if different:
        result['different_verb'] = sorted(different)[0]
    return result
