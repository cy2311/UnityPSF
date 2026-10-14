from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import torch


def load_trace_module():
    source = Path(__file__).parents[2] / 'scripts/train/qualification_trace.py'
    spec = importlib.util.spec_from_file_location('qualification_trace', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_all_repeats_python_numpy_and_torch():
    from unity_psf.training.checkpoint_state import capture_rng_state, restore_rng_state

    module = load_trace_module()
    original = capture_rng_state()
    try:
        module.seed_all(45)
        first = (random.random(), np.random.rand(), torch.rand(3))
        module.seed_all(45)
        second = (random.random(), np.random.rand(), torch.rand(3))
        assert module.state_hash(first) == module.state_hash(second)
    finally:
        restore_rng_state(original)


def test_state_hash_handles_scalar_tensor_dataclass_and_mapping_order():
    module = load_trace_module()

    @dataclass
    class State:
        tensor: torch.Tensor

    assert module.state_hash({'value': State(torch.tensor(1.)), 'other': None}) == module.state_hash(
        {'other': None, 'value': State(torch.tensor(1.))},
    )
    assert module.state_hash(torch.tensor(1.)) != module.state_hash(torch.tensor(2.))
    assert module.state_hash(torch.tensor(0.)) != module.state_hash(torch.tensor(-0.))


def test_trace_identifies_first_resumed_batch_difference(tmp_path):
    module = load_trace_module()
    initial = {'checkpoint_sha256': 'same', 'state': {'rng': 'same'}}
    continuous = [
        {'epoch': 1, 'step': 1, 'batch': 'first', 'before': {}, 'loss': 2.0},
        {'epoch': 2, 'step': 1, 'batch': 'second', 'before': {}, 'loss': 1.0},
    ]
    interrupted = [dict(row) for row in continuous]
    interrupted[1]['batch'] = 'different'
    for label, rows in [('continuous', continuous), ('interrupted', interrupted)]:
        (tmp_path / f'initial-{label}-astigmatism.json').write_text(json.dumps(initial))
        (tmp_path / f'trace-{label}-astigmatism.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in rows),
        )
    result = module.compare_traces(tmp_path, ['astigmatism'])['astigmatism']
    assert result['initial_state_status'] == 'passed'
    assert result['step_trace_status'] == 'failed'
    assert result['first_sampled_difference'] == {'epoch': 2, 'step': 1, 'fields': ['batch']}
    assert result['resume_boundary']['pre_step_state_equal']
    assert not result['resume_boundary']['first_batch_equal']


def test_state_hash_does_not_consume_rng():
    from unity_psf.training.checkpoint_state import capture_rng_state

    module = load_trace_module()
    before = capture_rng_state()
    module.state_hash({'tensor': torch.ones(2, 3), 'scalar': 2.3})
    assert module.state_hash(before) == module.state_hash(capture_rng_state())


def test_gate_cannot_pass_when_final_equal_but_trace_failed():
    module = load_trace_module()
    traces = {'modality': {'initial_state_status': 'passed', 'step_trace_status': 'failed'}}
    epochs = {'modality': {1: {'strict_byte_status': 'passed'}}}
    final = {'modality': {'strict_byte_status': 'passed'}}
    assert module.gate_status(traces, epochs, final) == 'failed'
    traces['modality']['step_trace_status'] = 'passed'
    assert module.gate_status(traces, epochs, final) == 'passed'
    assert module.gate_status({}, {}, {}) == 'failed'
