"""Entry dispatch and backend provenance checks without starting Ray or GPUs."""
from pathlib import Path
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest

from alfworld_baseline.backend import assert_native_verl, native_verl_root
from alfworld_baseline import main_ppo as entry

ROOT = Path(__file__).resolve().parents[1]


def config_for(profile):
    with initialize_config_dir(config_dir=str(ROOT / 'config/alfworld' / profile / 'v1'), version_base=None):
        return compose(config_name='alfworld_config_2gpu')


def test_imports_use_native_checkout():
    from verl.experimental.agent_loop import tool_agent_loop
    assert Path(entry._base_main.__file__).resolve().is_relative_to(native_verl_root())
    assert Path(tool_agent_loop.__file__).resolve().is_relative_to(native_verl_root())
    assert_native_verl()


def test_cached_foreign_submodule_is_rejected(monkeypatch, tmp_path):
    foreign = ModuleType('verl.foreign_backend_probe')
    foreign.__file__ = str(tmp_path / 'foreign.py')
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    with pytest.raises(RuntimeError, match='backend mismatch'):
        assert_native_verl()


@pytest.mark.parametrize('profile', ['qwen25_1_5b', 'qwen35_2b', 'qwen35_9b'])
def test_hydra_cli_resolves_without_starting_ray(profile):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([
        str(ROOT / 'src'), str(native_verl_root()), os.environ.get('PYTHONPATH', '')]))
    result = subprocess.run([
        sys.executable, '-m', 'alfworld_baseline.main_ppo',
        '--config-path', str(ROOT / 'config/alfworld' / profile / 'v1'),
        '--config-name', 'alfworld_config_2gpu', '--cfg', 'job', '--resolve',
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    cfg = OmegaConf.create(result.stdout)
    assert cfg.trainer.use_v1
    assert 'ray_kwargs' not in cfg.trainer
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.CXX == '/usr/bin/g++-10'


def test_dispatch_passes_custom_actor_explicitly(monkeypatch, tmp_path):
    cfg = config_for('qwen35_2b')
    cfg.trainer.default_local_dir = str(tmp_path)
    calls = []
    monkeypatch.setattr(entry, 'auto_set_device', lambda config: None)
    monkeypatch.setattr(entry, 'install_ray_agent_port_guard', lambda: None)
    monkeypatch.setattr(entry.ray, 'remote', lambda cls: ('actor', cls))
    monkeypatch.setattr(entry._base_main, 'run_ppo', lambda config, task_runner_class: calls.append(task_runner_class))
    entry.main.__wrapped__(cfg)
    assert calls == [('actor', entry.ALFWorldTaskRunner)]
    assert not hasattr(entry._base_main, 'TaskRunner')


def test_rejects_legacy_trainer_before_dispatch():
    cfg = config_for('qwen35_2b')
    cfg.trainer.use_v1 = False
    with pytest.raises(ValueError, match='requires trainer.use_v1'):
        entry.main.__wrapped__(cfg)


@pytest.mark.parametrize('key,value', [
    ('multi_turn.max_generated_response_length', None),
    ('agent.num_gpus_per_worker', 0.0),
])
def test_rejects_unsupported_rollout_fields_before_ray_dispatch(monkeypatch, key, value):
    from hydra.errors import InstantiationException

    cfg = config_for('qwen35_2b')
    OmegaConf.update(cfg, f'actor_rollout_ref.rollout.{key}', value, force_add=True)
    monkeypatch.setattr(entry._base_main, 'run_ppo', lambda *args, **kwargs: pytest.fail('Ray was started'))
    with pytest.raises(InstantiationException, match=key.rsplit('.', 1)[-1]):
        entry.main.__wrapped__(cfg)


def test_metrics_read_v1_extra_fields_and_exclude_padding(monkeypatch):
    calls = []
    rows = np.array([
        {'alfworld_terminal_reason': 'success', 'alfworld_valid_tool_call_count': 3},
        {'alfworld_terminal_reason': 'no_tool_call', 'alfworld_no_tool_call_penalty_count': 1},
    ], dtype=object)
    def get(**kwargs):
        calls.append(kwargs)
        return {'extra_fields': rows}
    monkeypatch.setitem(sys.modules, 'transfer_queue', SimpleNamespace(kv_batch_get=get))
    class Parent:
        def _compute_metrics(self, batch, metrics, *args):
            metrics['native_metric'] = 7
    class Trainer(entry.ALFWorldMetricsMixin, Parent):
        pass
    batch = SimpleNamespace(keys=['a', 'pad', 'b'], tags=[{}, {'is_padding': True}, {}], partition_id='train')
    metrics = {}
    Trainer()._compute_metrics(batch, metrics, {}, 1, 0)
    assert calls == [dict(keys=['a', 'b'], partition_id='train', select_fields=['extra_fields'])]
    assert metrics['native_metric'] == 7
    assert metrics['alfworld_termination/total_trajectories'] == 2
    assert metrics['alfworld_penalty/no_action_count'] == 1
    assert metrics['alfworld/valid_action_count/mean'] == 1.5


@pytest.mark.parametrize('fail', [False, True])
def test_runner_lifecycle_closes_queue_on_success_and_failure(monkeypatch, fail):
    from contextlib import nullcontext
    from alfworld_baseline import tracking
    monkeypatch.setattr(tracking, "swanlab_resume", lambda config: nullcontext())
    calls = []
    fake_v1 = ModuleType('verl.trainer.ppo.v1')
    class Trainer:
        logger = SimpleNamespace(finish=lambda **kw: calls.append(('finish', kw['exit_code'])))
        def __init__(self, config):
            assert config.transfer_queue.enable
        def init(self):
            calls.append('init')
        def get_llm_client(self):
            return 'llm'
        def get_teacher_client(self):
            return None
        def get_reward_handles(self):
            return []
        def fit(self, manager):
            assert manager == 'manager'
            calls.append('fit')
            if fail:
                raise RuntimeError('training failed')
    def create(**kwargs):
        assert kwargs['llm_client'] == 'llm'
        calls.append('manager')
        return 'manager'
    fake_v1.get_trainer_cls = lambda mode: Trainer
    fake_v1.AgentLoopManagerTQ = SimpleNamespace(create=create)
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.v1', fake_v1)
    monkeypatch.setitem(sys.modules, 'transfer_queue', SimpleNamespace(
        init=lambda config: calls.append('queue_init'), close=lambda: calls.append('queue_close')))
    if fail:
        with pytest.raises(RuntimeError, match='training failed'):
            entry.ALFWorldTaskRunner().run(config_for('qwen35_2b'))
    else:
        entry.ALFWorldTaskRunner().run(config_for('qwen35_2b'))
    assert calls == ['queue_init', 'init', 'manager', 'fit', ('finish', int(fail)), 'queue_close']


def test_metrics_count_only_final_steps(monkeypatch):
    monkeypatch.setitem(sys.modules, 'transfer_queue', SimpleNamespace(kv_batch_get=lambda **kw: {
        'extra_fields': np.array([
            {'alfworld_is_final_step': False, 'alfworld_terminal_reason': 'success'},
            {'alfworld_is_final_step': True, 'alfworld_terminal_reason': 'success'},
            {'alfworld_is_final_step': True, 'alfworld_terminal_reason': 'no_tool_call'},
        ], dtype=object)}))
    class Parent:
        def _compute_metrics(self, *args):
            pass
    class Trainer(entry.ALFWorldMetricsMixin, Parent):
        pass
    metrics = {}
    Trainer()._compute_metrics(SimpleNamespace(keys=['a', 'b', 'c'], tags=[{}, {}, {}], partition_id='train'), metrics, {}, 1, 0)
    assert metrics['alfworld_termination/total_trajectories'] == 2
    assert metrics['alfworld_termination/success_count'] == 1


@pytest.mark.parametrize('override', ['algorithm.adv_estimator=gae', 'algorithm.use_kl_in_reward=true',
    'trainer.v1.trainer_mode=colocate_async', 'distillation.enabled=true', 'reward.reward_model.enable=true'])
def test_rejects_modes_outside_step_contract(override):
    from alfworld_baseline.budget import validate_native_step_config
    cfg = config_for('qwen35_2b')
    key, value = override.split('=')
    OmegaConf.update(cfg, key, OmegaConf.create({'value': value}).value)
    with pytest.raises(ValueError):
        validate_native_step_config(cfg)
