"""Checkpoint, accounting and validation selection helpers (not inference assets)."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess

import torch


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def update_ema(ema, model, decay):
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema[name].lerp_(value.detach(), 1 - decay)
        else:
            ema[name].copy_(value)


def select_candidate(best, state, validation, step, weights):
    """Only validation BPB selects a frozen weight set; ties keep the earlier one."""
    if best is None or validation['bpb'] < best['validation']['bpb']:
        return dict(model={k: v.detach().cpu().clone() for k, v in state.items()},
                    validation=dict(validation), step=step, weights=weights)
    return best


def inference_asset_bytes(model, source_paths=()):
    # Count shared tensor storage once (the baseline ties embedding/head weights).
    storages = {}
    for value in model.state_dict().values():
        storage = value.untyped_storage()
        storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values()) + sum(Path(p).stat().st_size for p in source_paths)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def source_metadata(root):
    paths = sorted(root.glob('*.py')) + sorted((root / 'configs').glob('*.json'))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return 'unavailable'
    return dict(code_commit=git('rev-parse', 'HEAD'), code_dirty=bool(git('status', '--porcelain')),
                source_sha256=hashes)


def hardware_metadata():
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), cpu)
    return dict(cpu=cpu, platform=platform.platform(), python=platform.python_version(),
                cuda_version=torch.version.cuda,
                peak_cpu_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2))


def append_run_log(path, metrics, config_path, run_dir):
    fields = ['run_id','code_commit','method','config_path','seed','parent_checkpoints','steps',
              'batch_size','context','processed_targets_including_ancestry','parameters','hardware',
              'threads','train_precision','train_seconds','validation_seconds','test_seconds',
              'peak_ram_gb','peak_gpu_allocated_gb','validation_bpb','test_bpb','checkpoint_sha256',
              'selected_on','notes']
    row = dict(run_id=str(run_dir), code_commit=metrics['code_commit'], method=metrics['implementation'],
               config_path=str(config_path), seed=metrics['seed'], parent_checkpoints=json.dumps(metrics['ancestry']),
               steps=metrics['updates'], batch_size=metrics['recipe']['batch_size'] * metrics['recipe']['grad_accum'],
               context=256, processed_targets_including_ancestry=metrics['train_tokens'],
               parameters=metrics['parameters'], hardware=metrics['cpu'] + ' / ' + metrics['device_name'],
               threads=metrics['threads'], train_precision=metrics['precision'], train_seconds=metrics['train_seconds'],
               validation_seconds=metrics['validation']['seconds'], peak_ram_gb=metrics['peak_cpu_ram_gib'] * 2**30 / 1e9,
               peak_gpu_allocated_gb=metrics['peak_allocated_gb'], validation_bpb=metrics['validation']['bpb'],
               checkpoint_sha256=metrics['checkpoint_sha256'], selected_on='validation',
               notes=json.dumps({'metrics': str(run_dir / 'metrics.json'), 'selected_weights': metrics['selected_weights'],
                                 'selected_step': metrics['selected_step'], 'config': metrics['config']}))
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
