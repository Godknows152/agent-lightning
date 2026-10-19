"""Conservative, deterministic filters for action-list repetition in thinking.

These rules identify textual lists, not latent decision points. Ambiguous prose
is retained. No model calls or changes to sampled token IDs are involved.
"""
from __future__ import annotations

import re
from typing import Any

MENTION_SELECTIONS = {"first", "first_non_listing"}
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_AVAILABILITY = re.compile(
    r"\b(?:admissible|available|options|action list|actions? (?:include|are|exist)|"
    r"actions?,? I (?:can see|could)|list of actions)\b", re.I,
)
_CHOICE = re.compile(
    r"\b(?:I (?:will|should|choose|select|prefer)|I'll|let me|best|most logical|"
    r"most relevant|next step|plan|sequence|first do|then do)\b", re.I,
)
_CONNECTORS = re.compile(r"\b(?:and|or|also|are|available|etc)\b", re.I)


def action_pattern(action: str) -> re.Pattern[str]:
    """Match the same case-sensitive, word-bounded command as the original rule."""
    return re.compile(r"(?<![\w])" + re.escape(action) + r"(?![\w])")


def _commands(text: str, actions: list[str]) -> list[re.Match[str]]:
    # Longest first prevents a shorter command from shadowing its extension.
    if not actions:
        return []
    return list(re.finditer(
        r"(?<![\w])(?:" + "|".join(re.escape(a) for a in sorted(set(actions), key=len, reverse=True))
        + r")(?![\w])", text,
    ))


def _only_separators(text: str) -> bool:
    return not re.sub(r"[\s\[\](){}'\"`*,;:/\.\-–—]", "", _CONNECTORS.sub("", text))


def _bare_item(item: str, actions: list[str]) -> bool:
    """Only literal commands and separators; explanations remain ambiguous."""
    spans = _commands(item, actions)
    if not spans:
        return False
    cursor = 0
    for m in spans:
        if not _only_separators(item[cursor:m.start()]):
            return False
        cursor = m.end()
    return _only_separators(item[cursor:])


def listing_reason(text: str, begin: int, end: int, actions: list[str]) -> str | None:
    """Return a high-confidence list rule, or None to retain this mention.

    A bullet alone or an availability keyword alone is insufficient. Bullet
    blocks need an availability heading and multiple action-bearing items. Inline
    lists need multiple distinct legal commands separated only by punctuation or
    conjunctions. Task-specific explanations deliberately fail the bare-item rule.
    """
    line_start = text.rfind("\n", 0, begin) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    bullet = _BULLET.match(line)
    if bullet and _bare_item(line[bullet.end():], actions):
        before = text[:line_start].splitlines()
        after = text[line_end:].splitlines()
        items = [line]
        # Walk just the contiguous list, tolerating blank lines between items.
        while before and (not before[-1].strip() or _BULLET.match(before[-1])):
            items.append(before.pop())
        for following in after:
            if following.strip() and not _BULLET.match(following):
                break
            items.append(following)
        heading = before[-1] if before else ""
        action_items = sum(bool(_commands(i, actions)) for i in items if _BULLET.match(i))
        if _AVAILABILITY.search(heading) and action_items >= 2:
            # "I need to select one" in a heading announces a candidate list,
            # whereas a plan/sequence heading announces actions to execute.
            if not re.search(r"\b(?:plan|sequence|steps to|I will|I'll)\b", heading, re.I):
                return "bare_bullet_list"

    # Limit inline inspection to the current sentence. Explicit brackets can
    # delimit a list embedded inside otherwise evaluative prose.
    local_begin = begin - line_start
    left = max(line.rfind(c, 0, local_begin) for c in ".!?") + 1
    right_candidates = [line.find(c, end-line_start) for c in ".!?"]
    right = min((i for i in right_candidates if i >= 0), default=len(line))
    segment = line[left:right]
    contexts = [(segment, "")]
    for opening, closing in [("[", "]"), ("(", ")")]:
        a = line.rfind(opening, 0, local_begin)
        b = line.find(closing, end-line_start)
        if a >= 0 and b >= 0:
            contexts.insert(0, (line[a+1:b], line[left:a]))
    for segment, outside in contexts:
        spans = _commands(segment, actions)
        if len({m.group() for m in spans}) < 2:
            continue
        prefix = outside + segment[:spans[0].start()]
        if not _AVAILABILITY.search(prefix) or _CHOICE.search(prefix):
            continue
        if _bare_item(segment[spans[0].start():], actions):
            return "inline_action_list"
    return None


def select_action_mention(
    text: str, action: str, actions: list[str], selection: str = "first",
) -> tuple[re.Match[str] | None, dict[str, Any]]:
    """Skip consecutive recognized listings; never fall back into XML or a list."""
    if selection not in MENTION_SELECTIONS:
        raise ValueError(f"Unsupported branch entropy mention_selection: {selection}")
    matches = list(action_pattern(action).finditer(text))
    skipped = []
    for index, match in enumerate(matches, 1):
        reason = listing_reason(text, match.start(), match.end(), actions) if selection == "first_non_listing" else None
        if reason is None:
            return match, {"mention_count": len(matches), "selected_mention_index": index,
                           "skipped_listing_count": len(skipped), "skipped_listing_rules": skipped}
        skipped.append(reason)
    return None, {"mention_count": len(matches), "selected_mention_index": 0,
                  "skipped_listing_count": len(skipped), "skipped_listing_rules": skipped}
