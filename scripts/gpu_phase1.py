"""Run Plan.md's baseline, LR pilots, matched comparison and local ablation.

Launch from code/ with nohup; all candidate selection uses full validation.
The next scale/long-training phase requires reviewing these results.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
commands = a.output / 'commands.jsonl'


def run(args, log):
    command = [sys.executable, '-u', *map(str, args)]
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), shlex.join(command), flush=True)
    with commands.open('a') as f:
        f.write(json.dumps({'time': time.time(), 'argv': command, 'log': str(log)}) + '\n')
    with log.open('x') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)


def train(name, implementation, config, extra):
    directory = a.output / name
    run(['train.py', '--implementation', implementation, '--config', config,
         '--device', 'cuda', '--threads', '4', '--seed', '17',
         '--micro-batch-size', '32', '--save-every', '100', '--log-every', '25',
         '--run-log', a.output / 'RUN_LOG.csv', '--run-dir', directory, *extra],
        a.output / (name + '.log'))
    metrics = json.loads((directory / 'metrics.json').read_text())
    print(json.dumps({'completed': name, 'validation_bpb': metrics['validation']['bpb'],
                      'processed_targets': metrics['processed_targets']}), flush=True)
    return directory, metrics


def cpu_score(name, directory):
    run(['evaluate.py', '--checkpoint', directory / 'checkpoint.pt', '--device', 'cpu',
         '--precision', 'fp32', '--threads', '4', '--split', 'validation'],
        a.output / (name + '-cpu-validation.log'))


try:
    baseline, _ = train('baseline-s17', 'model', 'configs/baseline.json',
                        ['--precision', 'fp32', '--updates', '1200'])
    cpu_score('baseline-s17', baseline)
    common = ['--precision', 'bf16', '--tf32', '--grad-accum', '4',
              '--min-lr-ratio', '.1', '--betas', '.9', '.95', '--weight-decay', '.1',
              '--ema', '.999', '--eval-every', '100']
    pilots = []
    for lr in ['0.0005', '0.001', '0.002']:
        _, metrics = train('pilot-lr-' + lr, 'model', 'configs/baseline.json',
                           [*common, '--updates', '305', '--warmup', '12', '--lr', lr])
        pilots.append({'lr': lr, 'validation_bpb': metrics['validation']['bpb']})
    winner = min(pilots, key=lambda x: x['validation_bpb'])
    selection = {'pilots': pilots, 'selected_lr': winner['lr'], 'dropout': 0,
                 'selection_split': 'validation',
                 'reason': 'Lowest full-validation BPB; baseline has no dropout support.'}
    (a.output / 'schedule_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(selection), flush=True)
    for name, implementation, config in [
        ('matched-baseline-s17', 'model', 'configs/baseline.json'),
        ('matched-student-s17', 'student', 'configs/student_matched.json'),
        ('matched-no-local-s17', 'student', 'configs/student_matched_no_local.json'),
    ]:
        directory, _ = train(name, implementation, config,
                             [*common, '--updates', '916', '--warmup', '36',
                              '--lr', winner['lr'], '--dropout', '0'])
        cpu_score(name, directory)
    (a.output / 'COMPLETE').write_text('Phase 1 complete. Review ablation and CPU results before scaling.\n')
    print('PHASE 1 COMPLETE: review validation evidence before scale and long runs.', flush=True)
except BaseException as error:
    (a.output / 'FAILED').write_text(repr(error) + '\n')
    raise
