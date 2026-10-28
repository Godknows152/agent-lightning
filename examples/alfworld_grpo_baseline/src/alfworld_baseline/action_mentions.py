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
    """Match a complete command, ignoring capitalization but preserving boundaries."""
    return re.compile(r"(?<![\w])" + re.escape(action) + r"(?![\w])", re.I)


def _commands(text: str, actions: list[str]) -> list[re.Match[str]]:
    # Longest first prevents a shorter command from shadowing its extension.
    if not actions:
        return []
    return list(re.finditer(
        r"(?<![\w])(?:" + "|".join(re.escape(a) for a in sorted(set(actions), key=len, reverse=True))
        + r")(?![\w])", text, re.I,
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


def _object_forms(obj: str) -> set[str]:
    """Allow articles, compound spacing and an optional index in a literal gloss."""
    normalized = re.sub(r"\s+", "", obj.lower())
    forms = {normalized, re.sub(r"\d+$", "", normalized)}
    words = obj.lower().split()
    ordinals = {"1": "first", "2": "second", "3": "third"}
    if words and words[-1] in ordinals:
        forms.add(ordinals[words[-1]] + "".join(words[:-1]))
    return forms


def _same_object(description: str, obj: str) -> bool:
    description = re.sub(r"^(?:the|a|an|this|that) ", "", description)
    return re.sub(r"\s+", "", description) in _object_forms(obj)


def _generic_gloss(tail: str, action: str) -> bool:
    """Accept only whole, whitelisted descriptions of the command's function.

    Full matching is intentional: an otherwise generic opening followed by a
    goal, priority or comparison must remain a possible decision point.
    """
    gloss = tail.strip(" \t'\"`*:-–—()[]{}.,;")
    gloss = re.sub(r"\s+", " ", gloss).lower()
    gloss = re.sub(
        r"^(?:(?:which|this|it) (?:would|could|can|should)|i (?:would|could|can)|would|could|can|to) "
        r"(?:(?:likely|explicitly|potentially) )?(?:(?:try|allow me|enable me) to |let me )?",
        "", gloss,
    )
    gloss = re.sub(r" is (?:also )?a valid action$", "", gloss)
    informational = {
        "get more information", "reveal more information", "examine the situation",
        "do a more detailed inspection", "check general status",
        "help me understand what's around me", "look around", "search by looking around",
    }
    inventory = {
        "see what i currently have", "show me what i currently have", "check available options",
        "show me the current inventory", "check my inventory", "see my inventory",
        "show me what objects are available", "check what i am carrying", "show me all items",
        "check what items are available",
    }
    if (action in {"look", "inventory"} or action.startswith("examine ")) and gloss in informational:
        return True
    if action == "inventory" and gloss in inventory:
        return True
    patterns = [
        ("go to ", r"(?:go to|head to|take me to|move me to|navigate to) (.+)"),
        ("examine ", r"(?:examine the items on|examine|examining|inspect|look at|check the contents of|"
         r"look inside|get more information about) (.+)"),
        ("open ", r"(?:open|check inside) (.+)"),
        ("close ", r"close (.+)"),
        ("take ", r"(?:take|pick up) (.+)"),
    ]
    # Two-object operations must agree with both arguments, not just their verb.
    for verb, separator, pattern in [
        ("take ", " from ", r"(?:take|pick up) (.+?) (?:from|out of) (.+)"),
        ("move ", " to ", r"(?:move|moving|put|place) (.+?) (?:to|on|in|onto|into) (.+)"),
        ("clean ", " with ", r"clean (.+?) with (.+)"),
        ("heat ", " with ", r"heat (.+?) with (.+)"),
        ("cool ", " with ", r"cool (.+?) with (.+)"),
    ]:
        if action.startswith(verb) and separator in action[len(verb):]:
            obj, target = action[len(verb):].split(separator, 1)
            match = re.fullmatch(pattern, gloss)
            if match:
                return _same_object(match[1], obj) and _same_object(match[2], target)
    for verb, pattern in patterns:
        if not action.startswith(verb):
            continue
        obj = action[len(verb):]
        if verb == "take ":
            obj = obj.split(" from ", 1)[0]
        match = re.fullmatch(pattern, gloss)
        if match:
            return _same_object(match[1], obj)
    return False


def _gloss_item(item: str, actions: list[str]) -> bool:
    spans = _commands(item, actions)
    if len(spans) != 1 or not _only_separators(item[:spans[0].start()]):
        return False
    match = spans[0]
    return _generic_gloss(item[match.end():], match.group().lower())


def render_action_case(candidate: str, observed: str) -> str:
    """Apply observed casing to every trie candidate, without changing rollout IDs.

    Normal sentence/title/all-caps styles generalize to all candidate lengths.
    Mixed casing follows the observed word/letter positions, with extra letters
    lowercased. This also reproduces an arbitrarily cased sampled command exactly.
    """
    if observed.islower():
        return candidate.lower()
    if observed.isupper():
        return candidate.upper()
    if observed == observed.capitalize():
        return candidate.capitalize()
    if observed.istitle():
        return candidate.title()
    templates = observed.split(" ")
    words = candidate.split(" ")
    for i, word in enumerate(words):
        template = templates[i] if i < len(templates) else ""
        if template.isupper():
            words[i] = word.upper()
        elif template.istitle():
            words[i] = word.capitalize()
        else:
            words[i] = "".join(
                c.upper() if j < len(template) and template[j].isupper() else c.lower()
                for j, c in enumerate(word)
            )
    return " ".join(words)


def listing_reason(text: str, begin: int, end: int, actions: list[str]) -> str | None:
    """Return a high-confidence list rule, or None to retain this mention.

    A bullet alone or an availability keyword alone is insufficient. Bullet
    blocks need an availability heading and multiple action-bearing items. Inline
    lists need multiple distinct legal commands separated only by punctuation or
    conjunctions, or whitelisted generic function glosses. Task-specific
    explanations fail the whole-gloss rule.
    """
    line_start = text.rfind("\n", 0, begin) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    bullet = _BULLET.match(line)
    bare = bool(bullet and _bare_item(line[bullet.end():], actions))
    if bullet and (bare or _gloss_item(line[bullet.end():], actions)):
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
                return "bare_bullet_list" if bare else "generic_gloss_bullet_list"

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
        if len({m.group().lower() for m in spans}) < 2:
            continue
        prefix = outside + segment[:spans[0].start()]
        if not _AVAILABILITY.search(prefix) or _CHOICE.search(prefix):
            continue
        if _bare_item(segment[spans[0].start():], actions):
            return "inline_action_list"
        # Every item must be bare or a complete generic gloss. Do not suppress a
        # task-specific comparison just because another item is generic.
        items = []
        for index, span in enumerate(spans):
            stop = spans[index + 1].start() if index + 1 < len(spans) else len(segment)
            item = segment[span.start():stop].rstrip(" \t'\"`")
            item = re.sub(r"\s+(?:and|or)(?: there's also)?\s*$", "", item, flags=re.I)
            items.append(item)
        if all(_bare_item(item, actions) or _gloss_item(item, actions) for item in items):
            return "generic_gloss_inline_list"
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
