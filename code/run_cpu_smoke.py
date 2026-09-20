"""Reproduce local smoke evidence, not the long GPU experiment sequence in Plan.md."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT/'runs/cpu-smoke')
    parser.add_argument('--evidence', type=Path, default=ROOT/'evidence')
    args = parser.parse_args()
    if args.run_root.exists():
        parser.error('Choose a fresh --run-root.')
    args.evidence.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(arguments, log):
        command = [sys.executable, *arguments]
        commands.append(command)
        (args.evidence/'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(' '.join(command), flush=True)
        with (args.evidence/log).open('w') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    run(['-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'tests.log')
    run(['smoke_check.py', '--output', str(args.evidence/'cpu_gate.json')], 'cpu_gate.log')
    gate = json.loads((args.evidence/'cpu_gate.json').read_text())
    results = []
    for implementation, name, steps, batch in [('model', 'baseline', 200, 8),
            ('student', 'student_matched', 200, 8), ('student', 'student_matched_no_local', 200, 8),
            ('student', 'student_8', 40, 2), ('student', 'student_10', 40, 2)]:
        run_dir = args.run_root/name
        command = ['train.py', '--implementation', implementation, '--config', f'configs/{name}.json',
                   '--run-dir', str(run_dir), '--steps', str(steps), '--micro-batch-size', str(batch),
                   '--grad-accum', '2', '--lr', '.001', '--warmup', '10', '--betas', '.9', '.95',
                   '--ema', '.9', '--eval-every', str(steps // 2), '--save-every', str(steps // 2),
                   '--threads', '4', '--precision', 'fp32', '--seed', '17', '--log-every', '20']
        if steps == 200:
            command += ['--dropout', '0']
        run(command, f'{name}.log')
        metrics = json.loads((run_dir/'metrics.json').read_text())
        shutil.copyfile(run_dir/'metrics.json', args.evidence/f'{name}_metrics.json')
        run(['evaluate.py', '--checkpoint', str(run_dir/'checkpoint.pt'), '--split', 'validation',
             '--device', 'cpu', '--precision', 'fp32', '--threads', '4',
             '--output', str(run_dir/'validation_cpu_fp32.json')], f'{name}_evaluation.log')
        evaluation = json.loads((run_dir/'validation_cpu_fp32.json').read_text())
        shutil.copyfile(run_dir/'validation_cpu_fp32.json', args.evidence/f'{name}_validation.json')
        assert abs(evaluation['bpb'] - metrics['validation']['bpb']) < 1e-6, 'Checkpoint reproduction failed'
        assert metrics['history'][-1]['loss'] < metrics['history'][0]['loss'] - .2, 'Loss failed to decrease'
        assert evaluation['bpb'] < 3.5, 'No material improvement from random weights (~3.62 BPB)'
        assert metrics['train_tokens'] == steps * batch * 2 * 256
        assert metrics['serialized_inference_asset_bytes'] < 56 * 2**20
        assert metrics['peak_cpu_ram_gib'] < 4
        results.append(dict(config=name, validation_bpb=evaluation['bpb'],
                            initial_loss=metrics['history'][0]['loss'], final_loss=metrics['history'][-1]['loss'],
                            train_tokens=metrics['train_tokens'], train_seconds=metrics['train_seconds'],
                            selected_weights=metrics['selected_weights'], selected_step=metrics['selected_step'],
                            cpu_scoring_seconds=evaluation['seconds'],
                            inference_assets_mib=metrics['serialized_inference_asset_bytes'] / 2**20))
    for row in results:
        row['baseline_time_ratio'] = row['cpu_scoring_seconds'] / results[0]['cpu_scoring_seconds']
        row['cpu_gate_passed'] = row['baseline_time_ratio'] <= 4
    summary = dict(scope='CPU smoke only; no candidate test evaluation or long-run quality claim.',
                   results=results, random_weight_resource_gate=gate['models'])
    (args.evidence/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
