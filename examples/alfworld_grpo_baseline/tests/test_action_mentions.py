"""Regression cases for list repetition versus actual choice or evaluation."""
import pytest

from alfworld_baseline.action_mentions import select_action_mention

ACTIONS = ['go to cabinet 1', 'go to cabinet 2', 'look', 'inventory']
ACTION = ACTIONS[0]


@pytest.mark.parametrize('listing', [
    "Available actions:\n- 'go to cabinet 1'\n- 'look'\n",
    'My options are:\n\n1. `go to cabinet 1`\n2. inventory\n',
    "Looking at the admissible actions, 'go to cabinet 1' and 'look' are available.\n",
    "My options: ['go to cabinet 1', 'go to cabinet 2', 'inventory'].\n",
    "I need a container (the action list shows: go to cabinet 1, go to cabinet 2).\n",
    "Available actions:\n- go to cabinet 1\n- look\n\nAvailable actions:\n- go to cabinet 1\n- inventory\n",
])
def test_skip_listings_and_keep_next_mention(listing):
    text = listing + '\nI should go to cabinet 1 because the object may be there.'
    match, info = select_action_mention(text, ACTION, ACTIONS, 'first_non_listing')
    assert match.start() == text.rfind(ACTION)
    assert info['selected_mention_index'] == info['mention_count']
    assert info['skipped_listing_count'] == info['mention_count'] - 1
    assert info['skipped_listing_count'] > 0


@pytest.mark.parametrize('text', [
    'I should go to cabinet 1. Then go to cabinet 1.',
    'Available actions:\n- go to cabinet 1 because it may contain the target.\n- look\n',
    'Available actions:\n- go to cabinet 1 — this is the best option.\n- look\n',
    'My plan:\n1. go to cabinet 1\n2. look\n',
    'The available action is go to cabinet 1.',
    'My options include go to cabinet 1 which seems best, or look.',
    'Available actions exist. I choose go to cabinet 1 or look depending on the result.',
    'I will execute [go to cabinet 1, look] in that order.',
    'Available actions:\n- go to cabinet 1\n\nI should look.',
])
def test_keep_choices_plans_singletons_and_ambiguous_prose(text):
    match, info = select_action_mention(text, ACTION, ACTIONS, 'first_non_listing')
    assert match.start() == text.find(ACTION)
    assert info['skipped_listing_count'] == 0


def test_all_listing_mentions_have_no_fallback():
    text = 'Available actions:\n- go to cabinet 1\n- look\n'
    match, info = select_action_mention(text, ACTION, ACTIONS, 'first_non_listing')
    assert match is None
    assert info['mention_count'] == info['skipped_listing_count'] == 1
    assert info['selected_mention_index'] == 0


def test_default_is_legacy_first_mention():
    text = 'Available actions: [go to cabinet 1, look]. I choose go to cabinet 1.'
    match, info = select_action_mention(text, ACTION, ACTIONS)
    assert match.start() == text.find(ACTION)
    assert info['skipped_listing_count'] == 0


def test_case_and_command_boundaries_remain_exact():
    text = 'GO TO CABINET 1; go to cabinet 10; I choose go to cabinet 1.'
    match, info = select_action_mention(text, ACTION, ACTIONS, 'first_non_listing')
    assert match.start() == text.rfind(ACTION)
    assert info['mention_count'] == 1


def test_invalid_selection_fails_closed():
    with pytest.raises(ValueError, match='mention_selection'):
        select_action_mention(ACTION, ACTION, ACTIONS, 'typo')
