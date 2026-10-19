"""Replay conservative list filtering on dumped ALFWorld generations (CPU only)."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from alfworld_baseline.action_mentions import select_action_mention
from alfworld_baseline.xml_actions import parse_xml_decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    changed = []
    for line_number, line in enumerate(args.rollout.open(), 1):
        row = json.loads(line)
        if row['uid'].startswith('pad'):
            continue
        counts['outputs'] += 1
        decision = parse_xml_decision(row['output'], enable_thinking=True)
        if decision.status != 'valid':
            continue
        counts['valid_xml'] += 1
        legal = re.search(r'Your admissible actions of the current situation are: \[(.*?)\]\.\n', row['input'], re.S)
        if legal is None:
            raise ValueError(f'Missing legal action list at line {line_number}')
        actions = re.findall(r"'([^']*)'", legal[1])
        if decision.action not in actions:
            continue
        text = row['output'].split('</think>')[0]
        match, info = select_action_mention(text, decision.action, actions, 'first_non_listing')
        if not info['mention_count']:
            continue
        counts['admissible_with_mention'] += 1
        if info['skipped_listing_count']:
            counts['rows_skipping_list'] += 1
            counts['skipped_mentions'] += info['skipped_listing_count']
            counts['later_mention_selected' if match else 'only_listing_mentions'] += 1
            changed.append({'uid': row['uid'], 'line': line_number, 'action': decision.action, **info,
                            'selected_span': list(match.span()) if match else None, 'thinking': text})
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'source': str(args.rollout.resolve()), 'counts': dict(counts),
              'note': 'Text-selection replay only; original rollout token IDs are absent, so trie alignment is not measured.'}
    (args.output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (args.output/'changed.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in changed))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
