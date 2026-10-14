from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_hash(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif is_dataclass(item):
            visit({field.name: getattr(item, field.name) for field in fields(item)})
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                visit(child)
        else:
            digest.update(str((type(item).__name__, item)).encode())

    visit(value)
    return digest.hexdigest()


def runtime_hashes(runtime) -> dict:
    from unity_psf.training.checkpoint_state import capture_rng_state
    from unity_psf.training.modality_physical_state import snapshot_channel_physical_state

    rng = capture_rng_state()
    return {
        'model': state_hash(runtime.model.state_dict()),
        'optimizer': state_hash(runtime.optimizer.state_dict()),
        'scheduler': state_hash(None if runtime.scheduler is None else runtime.scheduler.state_dict()),
        'scaler': state_hash(None if runtime.amp_scaler is None else runtime.amp_scaler.state_dict()),
        'rng': state_hash(rng),
        'rng_sources': {key: state_hash(value) for key, value in rng.items()},
        'physical_state': state_hash({key: snapshot_channel_physical_state(stream)
                                     for key, stream in runtime.channels.items()}),
    }


def install_trace(worker, *, receipt_dir: Path, label: str, resumed: bool) -> None:
    from unity_psf.training import modality_epoch
    from unity_psf.training.modality_progress import channel_progress, empty_metrics

    original_build = worker.build_modality_runtime
    original_step = modality_epoch.run_optimizer_step
    original_epoch = worker.train_modality_epoch
    context = {}

    def build(*args, **kwargs):
        built = original_build(*args, **kwargs)
        runtime = built[0]
        modality = runtime.modality.value
        runtime_scaler = torch.amp.GradScaler(
            device='cuda', enabled=runtime.amp_enabled, init_scale=128.0,
        )
        object.__setattr__(runtime, 'amp_scaler', runtime_scaler)
        initial = receipt_dir / 'initial' / f'{modality}.ckpt'
        if not resumed:
            if label == 'continuous':
                initial.parent.mkdir(parents=True, exist_ok=True)
                worker.save_modality_training_shard(
                    initial, runtime=runtime, epoch=0, optimizer_steps=0,
                    channel_progress=channel_progress(empty_metrics(runtime)), status='running',
                )
            else:
                manifest = json.loads((receipt_dir / 'initial-manifest.json').read_text())
                if hashlib.sha256(initial.read_bytes()).hexdigest() != manifest[modality]:
                    raise ValueError('common initial checkpoint hash changed')
            worker.restore_modality_training_shard(initial, runtime=runtime)
            payload = {'checkpoint_sha256': hashlib.sha256(initial.read_bytes()).hexdigest(),
                       'state': runtime_hashes(runtime)}
            (receipt_dir / f'initial-{label}-{modality}.json').write_text(json.dumps(payload, indent=2))
        channels = {}
        for channel, stream in runtime.channels.items():
            def batches(epoch, provider=stream.batches, channel=channel):
                for batch in provider(epoch):
                    context['channel'] = channel
                    context['batch'] = batch
                    yield batch
            channels[channel] = replace(stream, batches=batches)
        object.__setattr__(runtime, 'channels', channels)
        def forward_trace(model, inputs, output):
            if context.get('tracing', False):
                context['forward_output'] = state_hash(output)
        runtime.model.register_forward_hook(forward_trace)
        context['runtime'] = runtime
        context['trace_path'] = receipt_dir / f'trace-{label}-{modality}.jsonl'
        return built

    def epoch(*, runtime, epoch):
        context['epoch'] = epoch
        context['step'] = 0
        return original_epoch(runtime=runtime, epoch=epoch)

    def step(**kwargs):
        context['step'] += 1
        tracing = context['step'] <= 3
        context['tracing'] = tracing
        if tracing:
            before = runtime_hashes(context['runtime'])
            batch = context['batch']
            target = batch.target.inputs
            batch_values = {
                'images': batch.images, 'conditions': batch.conditions,
                'targets': {name: getattr(target, name) for name in
                            ('detect_tar', 'bkg_tar', 'pxyz_tar', 'mask_tar')},
            }
            batch_hash = state_hash(batch_values)
            batch_parts = {key: state_hash(value) for key, value in batch_values.items()}
        result = original_step(**kwargs)
        context['tracing'] = False
        if tracing:
            row = {'epoch': context['epoch'], 'step': context['step'],
                   'channel': context['channel'], 'batch': batch_hash,
                   'batch_parts': batch_parts,
                   'before': before, 'after': runtime_hashes(context['runtime']),
                   'loss': result.loss, 'performed': result.optimizer_step_performed,
                   'forward_output': context['forward_output'],
                   'gradient_stage': 'after_unscale_clip_and_optimizer_step',
                   'gradients': state_hash({name: parameter.grad
                                           for name, parameter in kwargs['model'].named_parameters()})}
            with context['trace_path'].open('a') as handle:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
        return result

    worker.build_modality_runtime = build
    worker.train_modality_epoch = epoch
    modality_epoch.run_optimizer_step = step


def compare_traces(receipt_dir: Path, modalities, *, epochs: int | None = None, steps_per_epoch: int | None = None) -> dict:
    results = {}
    for modality in modalities:
        initial = [json.loads((receipt_dir / f'initial-{label}-{modality}.json').read_text())
                   for label in ('continuous', 'interrupted')]
        traces = [[json.loads(line) for line in (receipt_dir / f'trace-{label}-{modality}.jsonl').read_text().splitlines()]
                  for label in ('continuous', 'interrupted')]
        keys = [[(row['epoch'], row['step']) for row in rows] for rows in traces]
        expected = None if epochs is None else [
            (epoch, step) for epoch in range(1, epochs + 1)
            for step in range(1, min(3, int(steps_per_epoch)) + 1)
        ]
        covered = all(rows and len(set(items)) == len(items) and (expected is None or items == expected)
                      for rows, items in zip(traces, keys))
        mismatches = []
        for left, right in zip(*traces):
            if left != right:
                mismatches.append({'epoch': left['epoch'], 'step': left['step'],
                                   'fields': [key for key in left if left[key] != right.get(key)]})
        results[modality] = {'initial_state_status': 'passed' if initial[0] == initial[1] else 'failed',
                             'step_trace_status': 'passed' if covered and traces[0] == traces[1] else 'failed',
                             'coverage_status': 'passed' if covered else 'failed',
                             'first_sampled_difference': mismatches[0] if mismatches else None,
                             'trace_lengths': [len(rows) for rows in traces]}
        boundary = [next((row for row in rows if row['epoch'] == 2 and row['step'] == 1), None)
                    for rows in traces]
        results[modality]['resume_boundary'] = {
            'first_batch_equal': boundary[0] is not None and boundary[1] is not None
            and boundary[0]['batch'] == boundary[1]['batch'],
            'pre_step_state_equal': boundary[0] is not None and boundary[1] is not None
            and boundary[0]['before'] == boundary[1]['before'],
            'first_loss_equal': boundary[0] is not None and boundary[1] is not None
            and boundary[0]['loss'] == boundary[1]['loss'],
        }
    return results


def gate_status(trace_results, epoch_results, comparisons) -> str:
    traces_passed = all(item['initial_state_status'] == 'passed'
                        and item['step_trace_status'] == 'passed' for item in trace_results.values())
    epochs_passed = all(item['strict_byte_status'] == 'passed'
                        for epochs in epoch_results.values() for item in epochs.values())
    final_passed = all(item['strict_byte_status'] == 'passed' for item in comparisons.values())
    return 'passed' if trace_results and epoch_results and comparisons and traces_passed and epochs_passed and final_passed else 'failed'
