"""Regression cases for list repetition versus actual choice or evaluation."""
import pytest

from alfworld_baseline.action_mentions import render_action_case, select_action_mention

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


def test_case_is_ignored_but_command_boundaries_remain_exact():
    text = 'GO TO CABINET 1; go to cabinet 10; I choose go to cabinet 1.'
    match, info = select_action_mention(text, ACTION, ACTIONS, 'first_non_listing')
    assert match.start() == 0
    assert info['mention_count'] == 2


@pytest.mark.parametrize('action,gloss', [
    ('go to coffeetable 2', 'This would take me to the second coffee table'),
    ('go to dresser 1', 'I could head to the dresser'),
    ('go to cabinet 1', 'to go to the cabinet'),
    ('inventory', 'could see what I currently have'),
    ('inventory', 'to check available options'),
    ('examine cabinet 1', 'to get more information'),
    ('examine shelf 13', 'to examine the items on this shelf'),
    ('open cabinet 2', 'This would allow me to open the cabinet'),
    ('open drawer 4', 'I can explicitly try to open the drawer'),
    ('take keychain 2 from shelf 13', 'to take the keychain'),
    ('take pot 1 from sinkbasin 1', 'I could take pot 1 out of the sinkbasin'),
    ('move dishsponge 1 to shelf 2', 'moving the dish sponge to shelf 2 is also a valid action'),
    ('examine dishsponge 1', 'examining the dish sponge is a valid action'),
    ('open cabinet 12', 'This should let me check inside cabinet 12'),
    ('heat potato 1 with microwave 1', 'to heat the potato with the microwave'),
    ('look', 'Could reveal more information'),
])
def test_generic_function_gloss_in_candidate_list_is_skipped(action, gloss):
    actions = [action, 'go to cabinet 2']
    text = f'Looking at the available actions:\n- "{action}" - {gloss}\n- go to cabinet 2\n'
    text += f'I should {action} because it helps with the task.'
    match, info = select_action_mention(text, action, actions, 'first_non_listing')
    assert match.start() == text.rfind(action)
    assert info['skipped_listing_rules'] == ['generic_gloss_bullet_list']


@pytest.mark.parametrize('action,gloss', [
    ('open drawer 12', 'this would directly help since I need the drawer to be open'),
    ('go to drawer 1', 'typically drawers contain packaged ingredients'),
    ('examine handtowelholder 1', 'could help me find the candle or get more details'),
    ('examine pencil 2', 'makes sense to examine what I just picked up'),
    ('open cabinet 2', 'This would allow me to access the cabinet and put the soapbar inside it'),
    ('open drawer 2', 'I could open the drawer first'),
    ('go to cabinet 1', 'This would take me to the cabinet because the target is there'),
    ('go to cabinet 1', 'This would take me to the cabinet to find a mug'),
    ('go to cabinet 1', 'This would take me to the drawer'),
    ('go to cabinet 1', 'This would take me to the second cabinet'),
    ('inventory', 'could see what I currently have and decide which mug to put down'),
    ('move dishsponge 1 to shelf 2', 'move the dish sponge to the cabinet'),
    ('take pot 1 from sinkbasin 1', 'take the pot from sinkbasin 2'),
    ('move dishsponge 1 to shelf 2', 'move the dish sponge to shelf 2 so I can clean the ladle'),
])
def test_task_reason_priority_or_different_destination_is_kept(action, gloss):
    text = f'Available actions:\n- "{action}" - {gloss}\n- look\n'
    match, info = select_action_mention(text, action, [action, 'look'], 'first_non_listing')
    assert match.start() == text.find(action)
    assert info['skipped_listing_count'] == 0


def test_generic_inline_gloss_is_skipped():
    text = ("I have command options including 'take keychain 2 from armchair 1' and there's also "
            "'inventory' (which would likely show me what objects are available).\n"
            "I should use inventory to find out whether I have a pillow.")
    match, info = select_action_mention(
        text, 'inventory', ['take keychain 2 from armchair 1', 'inventory'], 'first_non_listing',
    )
    assert match.start() == text.rfind('inventory')
    assert info['skipped_listing_rules'] == ['generic_gloss_inline_list']


def test_inline_task_comparison_is_kept_even_with_one_generic_gloss():
    text = ("Available actions: 'go to cabinet 1' (to get the mug I need) "
            "or 'inventory' (to see what I currently have).")
    match, info = select_action_mention(text, 'inventory', ACTIONS, 'first_non_listing')
    assert match is not None
    assert info['skipped_listing_count'] == 0


@pytest.mark.parametrize('heading', ['My plan:', 'Available actions I will execute in sequence:'])
def test_generic_gloss_does_not_override_plan_heading(heading):
    text = f'{heading}\n- go to cabinet 1 - to go to the cabinet\n- look\n'
    match, info = select_action_mention(text, ACTION, ACTIONS, 'first_non_listing')
    assert match is not None
    assert info['skipped_listing_count'] == 0


def test_only_generic_gloss_mentions_have_no_fallback():
    text = 'Available actions:\n- GO TO CABINET 1 - to go to the cabinet\n- LOOK\n'
    match, info = select_action_mention(text, ACTION, ACTIONS, 'first_non_listing')
    assert match is None
    assert info['skipped_listing_rules'] == ['generic_gloss_bullet_list']


def test_case_variants_do_not_make_two_distinct_candidates():
    text = "Available actions: 'look', 'LOOK'."
    match, info = select_action_mention(text, 'look', ['look', 'inventory'], 'first_non_listing')
    assert match is not None
    assert info['skipped_listing_count'] == 0


def test_case_insensitive_list_then_capitalized_plan():
    text = "Available actions:\n- GO TO CABINET 1\n- LOOK\nI should:\n1. Go to cabinet 1\n"
    match, info = select_action_mention(text, ACTION, ACTIONS, 'first_non_listing')
    assert match.group() == 'Go to cabinet 1'
    assert info['skipped_listing_count'] == 1


@pytest.mark.parametrize('observed', [
    'go to cabinet 1', 'Go to cabinet 1', 'GO TO CABINET 1', 'Go To Cabinet 1',
    'gO tO caBINet 1', 'go to CABINET 1',
])
def test_case_rendering_reproduces_sampled_spelling(observed):
    assert render_action_case(ACTION, observed) == observed


def test_invalid_selection_fails_closed():
    with pytest.raises(ValueError, match='mention_selection'):
        select_action_mention(ACTION, ACTION, ACTIONS, 'typo')
