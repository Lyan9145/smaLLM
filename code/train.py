"""Reproducible random-span training; original baseline defaults remain available."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from distillation import distillation_loss
from training_utils import (append_run_log, atomic_save, cpu_state, hardware_metadata,
                            inference_asset_bytes, initialize_from_checkpoint, select_candidate,
                            source_metadata, update_ema)


def learning_rate(step, recipe):
    return (recipe['lr'] * min(1., (step + 1) / max(1, recipe['warmup'])) *
            (recipe['min_lr_ratio'] + (1 - recipe['min_lr_ratio']) * .5 *
             (1 + math.cos(math.pi * step / recipe['steps']))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    budget = p.add_mutually_exclusive_group()
    budget.add_argument('--steps', '--updates', type=int, default=None)
    budget.add_argument('--target-budget', type=int, help='Must be divisible by targets per update.')
    p.add_argument('--batch-size', '--micro-batch-size', type=int, default=32)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--betas', type=float, nargs=2, default=[.9, .999])
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--dropout', type=float, help='Overrides student config; baseline does not support dropout.')
    p.add_argument('--eval-every', '--validation-interval', type=int, default=0)
    p.add_argument('--ema', type=float, default=0., help='EMA decay; 0 disables EMA.')
    p.add_argument('--save-every', '--save-interval', type=int, default=0)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument('--resume', type=Path)
    initialization.add_argument('--init-checkpoint', type=Path,
                                help='Warm-start selected weights with a NEW optimizer/schedule; not exact recovery.')
    p.add_argument('--stop-after', type=int, help='Stop this segment at an update, preserving the full schedule.')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--run-log', type=Path, default=ROOT/'RUN_LOG_TEMPLATE.csv')
    p.add_argument('--tf32', action='store_true', help='Permit TF32 for CUDA training only.')
    p.add_argument('--teacher-checkpoint', type=Path,
                   help='Optional frozen checkpoint used only for training-time distillation.')
    p.add_argument('--distill-temperature', type=float, default=2.0)
    p.add_argument('--distill-alpha', type=float, default=0.0,
                   help='Soft-teacher loss weight; 0 disables distillation.')
    args = p.parse_args()
    if min(args.batch_size, args.grad_accum, args.threads, args.log_every) < 1:
        p.error('Batch size, accumulation, threads and log interval must be positive.')
    targets_per_update = args.batch_size * args.grad_accum * 256
    if args.target_budget is not None:
        if args.target_budget < 1 or args.target_budget % targets_per_update:
            p.error('--target-budget must be a positive multiple of targets per update.')
        args.steps = args.target_budget // targets_per_update
    elif args.steps is None:
        args.steps = 1200
    if args.steps < 1 or min(args.warmup, args.eval_every, args.save_every) < 0:
        p.error('Steps must be positive; intervals and warmup must be nonnegative.')
    if not (args.lr > 0 and math.isfinite(args.lr) and 0 <= args.min_lr_ratio <= 1 and
            0 <= args.ema < 1 and all(0 <= b < 1 for b in args.betas) and
            math.isfinite(args.weight_decay) and args.weight_decay >= 0):
        p.error('Invalid optimizer, scheduler or EMA settings.')
    if args.dropout is not None and not 0 <= args.dropout < 1:
        p.error('Dropout must be in [0, 1).')
    if args.implementation == 'model' and args.dropout not in (None, 0.):
        p.error('The preserved baseline has no dropout; use --implementation student.')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        p.error('--stop-after must be between 1 and the total update budget.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Use a new --run-dir, including for resumed segments; prior evidence is immutable.')
    return args


def main():
    total_started = time.perf_counter()
    args = parse_args()
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    data = load_data()
    config = json.loads(args.config.read_text())
    if args.dropout is not None and args.implementation != 'model':
        config['dropout'] = args.dropout
    if not (0 <= args.distill_alpha <= 1 and args.distill_temperature > 0 and
            math.isfinite(args.distill_temperature)):
        raise ValueError('Distillation alpha must be in [0, 1] and temperature positive.')
    if args.distill_alpha > 0 and args.teacher_checkpoint is None:
        raise ValueError('--distill-alpha > 0 requires --teacher-checkpoint')
    if args.teacher_checkpoint is not None and not args.teacher_checkpoint.is_file():
        raise ValueError('--teacher-checkpoint must point to a checkpoint file')
    model, implementation_sha = make_model(args.implementation, config, device)
    initialization_metadata = None
    inherited_targets = 0
    if args.init_checkpoint:
        initialization_metadata = initialize_from_checkpoint(
            model, args.init_checkpoint, args.implementation, config)
        inherited_targets = initialization_metadata['processed_targets_including_ancestry']
    teacher = None
    teacher_metadata = None
    if args.teacher_checkpoint is not None:
        teacher_state = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=True)
        if teacher_state.get('protocol') != PROTOCOL:
            raise ValueError('Teacher checkpoint belongs to a different course protocol.')
        teacher, teacher_sha = make_model(teacher_state['implementation'], teacher_state['config'], device)
        teacher.load_state_dict(teacher_state['model'])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher_metadata = dict(path=str(args.teacher_checkpoint.resolve()),
                                checkpoint_sha256=sha(args.teacher_checkpoint),
                                implementation=teacher_state['implementation'],
                                implementation_sha256=teacher_sha,
                                config=teacher_state['config'],
                                processed_targets_including_ancestry=teacher_state.get(
                                    'processed_targets_including_ancestry', teacher_state.get('train_tokens')))
    recipe = {key: getattr(args, key) for key in ('steps', 'batch_size', 'grad_accum', 'lr', 'warmup',
              'min_lr_ratio', 'betas', 'weight_decay', 'ema', 'seed', 'eval_every', 'tf32',
              'distill_temperature', 'distill_alpha')}
    recipe.update(teacher_checkpoint_sha256=(teacher_metadata or {}).get('checkpoint_sha256'),
                  teacher_implementation=(teacher_metadata or {}).get('implementation'))
    recipe.update(precision=precision, device_type=device.type, threads=args.threads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema else None
    ema_model = make_model(args.implementation, config, device)[0] if args.ema else None
    metadata = source_metadata(ROOT)
    start, history, validation_history, best, ancestry = 0, [], [], None, []
    if initialization_metadata is not None:
        ancestry.append(dict(kind='warm_start', **initialization_metadata))
    previous_train_seconds = previous_process_seconds = intermediate_validation_seconds = 0.
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if (saved['protocol'] != PROTOCOL or saved['recipe'] != recipe or saved['config'] != config or
                saved['implementation'] != args.implementation):
            raise ValueError('Resume must retain protocol, implementation, config and full training recipe.')
        if saved['source_sha256'] != metadata['source_sha256']:
            raise ValueError('Resume source/config hashes differ; use the exact recovery code.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['update']
        if saved['scheduler'] != {'next_update': start, 'recipe': recipe}:
            raise ValueError('Inconsistent scheduler state')
        rng.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        ema = {k: v.to(device) for k, v in saved['ema'].items()} if args.ema else None
        history, validation_history, best = saved['history'], saved['validation_history'], saved['best']
        previous_train_seconds = saved['train_seconds']
        previous_process_seconds = saved['process_seconds']
        intermediate_validation_seconds = saved['validation_seconds']
        initialization_metadata = saved.get('initialization')
        inherited_targets = saved.get('inherited_targets', 0)
        ancestry = saved['ancestry'] + [dict(path=str(args.resume.resolve()), sha256=sha(args.resume),
                                            updates=start, train_tokens=start * args.batch_size * args.grad_accum * 256)]
    end = args.stop_after or args.steps
    if end <= start:
        raise ValueError('The requested segment must advance beyond the resume update.')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - total_started
    train_seconds = previous_train_seconds
    offsets = torch.arange(257, device=device)

    def validate(update):
        nonlocal best, intermediate_validation_seconds
        # Scoring always uses full FP32, including when TF32 training is enabled.
        torch.set_float32_matmul_precision('highest')
        for label, candidate in [('raw', model)] + ([('ema', ema_model)] if ema is not None else []):
            if label == 'ema':
                candidate.load_state_dict(ema)
            result = score(candidate, *data['validation'], device, 'fp32')
            result.pop('window_nll_nats')
            intermediate_validation_seconds += result['seconds']
            validation_history.append(dict(step=update, weights=label, **result))
            best = select_candidate(best, candidate.state_dict(), result, update, label)
            print(json.dumps({'validation': validation_history[-1]}), flush=True)

    def save_resume(update):
        atomic_save(dict(protocol=PROTOCOL, implementation=args.implementation, config=config,
                         recipe=recipe, source_sha256=metadata['source_sha256'], model=cpu_state(model),
                         optimizer=optimizer.state_dict(), ema=ema, best=best, update=update,
                         teacher=teacher_metadata,
                         initialization=initialization_metadata, inherited_targets=inherited_targets,
                         scheduler=dict(next_update=update, recipe=recipe), sampler_rng=rng.get_state(),
                         torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                         history=history, validation_history=validation_history, ancestry=ancestry,
                         train_seconds=train_seconds, validation_seconds=intermediate_validation_seconds,
                         process_seconds=previous_process_seconds + time.perf_counter() - total_started),
                    args.run_dir/'resume.pt')

    # Keep the parent as an eligible candidate: a continuation need not improve.
    if args.init_checkpoint:
        validate(0)

    for step in range(start, end):
        tick = time.perf_counter()
        torch.set_float32_matmul_precision('high' if args.tf32 and device.type == 'cuda' else 'highest')
        lr = learning_rate(step, recipe)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for _ in range(args.grad_accum):
            # Keep the supplied baseline sampler exactly, including its upper bound.
            starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
            batch = tokens[starts[:, None] + offsets]
            with autocast(device, precision):
                student_logits = model(batch[:, :-1])
                if teacher is not None and args.distill_alpha > 0:
                    with torch.no_grad():
                        teacher_logits = teacher(batch[:, :-1])
                    loss, hard_loss, soft_loss = distillation_loss(
                        student_logits, teacher_logits, batch[:, 1:],
                        args.distill_temperature, args.distill_alpha)
                else:
                    hard_loss = F.cross_entropy(student_logits.flatten(0, 1).float(), batch[:, 1:].flatten())
                    soft_loss = torch.zeros_like(hard_loss)
                    loss = hard_loss
                aux_loss = getattr(model, 'aux_loss', student_logits.new_zeros(()))
                loss = loss + aux_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at update {step + 1}')
            (loss / args.grad_accum).backward()
            loss_sum += loss.detach() / args.grad_accum
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if ema is not None:
            update_ema(ema, model, args.ema)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - tick
        if step == start or (step + 1) % args.log_every == 0 or step + 1 == end:
            row = dict(step=step + 1, loss=loss_sum.item(), lr=lr, gradient_norm=norm.item(), seconds=train_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        if (args.eval_every and (step + 1) % args.eval_every == 0) or step + 1 == end:
            validate(step + 1)
        if (args.save_every and (step + 1) % args.save_every == 0) or step + 1 == end:
            save_resume(step + 1)

    train_tokens = end * args.batch_size * args.grad_accum * 256
    total_targets = inherited_targets + train_tokens
    checkpoint = args.run_dir/'checkpoint.pt'
    atomic_save(dict(protocol=PROTOCOL, implementation=args.implementation, config=config,
                     model=best['model'], seed=args.seed, train_tokens=total_targets,
                     processed_targets_including_ancestry=total_targets,
                     initialization=initialization_metadata,
                     selected_step=best['step'], selected_weights=best['weights']), checkpoint)
    model.load_state_dict(best['model'])
    if args.implementation == 'student':
        source_paths = [ROOT/'student.py', ROOT/'student_model.py']
    else:
        source_paths = [ROOT/'model.py']
    asset_bytes = inference_asset_bytes(model, source_paths) + len(json.dumps(config).encode())
    result = dict(protocol=PROTOCOL, implementation=args.implementation, config=config, seed=args.seed,
                  recipe=recipe, updates=end, completed_budget=end == args.steps,
                  parameters=sum(p.numel() for p in model.parameters()), precision=precision,
                  train_tokens=train_tokens, processed_targets=train_tokens, preparation_seconds=preparation_seconds,
                  processed_targets_including_ancestry=total_targets, inherited_targets=inherited_targets,
                  train_seconds=train_seconds, validation=best['validation'], history=history,
                  validation_history=validation_history, selected_step=best['step'], selected_weights=best['weights'],
                  intermediate_validation_seconds=intermediate_validation_seconds,
                  process_seconds=previous_process_seconds + time.perf_counter() - total_started,
                  segment_process_seconds=time.perf_counter() - total_started,
                  torch_version=str(torch.__version__), threads=args.threads, ancestry=ancestry,
                  teacher=teacher_metadata,
                  initialization=initialization_metadata,
                  parent_checkpoint=(str(args.resume.resolve()) if args.resume else
                                     (initialization_metadata or {}).get('path')),
                  checkpoint_sha256=sha(checkpoint), implementation_sha256=implementation_sha,
                  inference_asset_bytes=asset_bytes,
                  serialized_inference_asset_bytes=checkpoint.stat().st_size + sum(p.stat().st_size for p in source_paths) + len(json.dumps(config).encode()),
                  **metadata, **hardware_metadata(), **device_metrics(device))
    (args.run_dir/'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    append_run_log(args.run_log, result, args.config, args.run_dir)
    print(json.dumps({k: v for k, v in result.items() if k not in ('history', 'validation_history', 'source_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
