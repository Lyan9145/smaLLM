"""Bounded parallel GPU search, followed by isolated CPU/FP32 validation only.

Run from a frozen code snapshot. All commands/costs are retained. The unchanged
evaluator is executed via runpy solely to measure whole-process peak CPU RAM.
"""
import argparse
import concurrent.futures
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def experiment_plan(old_teacher, new_teacher):
    def job(name, **extra):
        return dict(name=name, config='student_8_no_local.json', teacher=str(new_teacher),
                    steps=3052, lr=.0015, alpha=.25, temperature=2., seed=17, **extra)
    # dict.update avoids duplicate default keys and makes overrides explicit.
    def variant(name, **extra):
        item = job(name)
        item.update(extra)
        return item
    return [
        variant('continue-lr2e4', init=str(new_teacher), teacher=str(old_teacher),
                steps=2000, lr=.0002, warmup=40, seed=1701),
        variant('continue-lr5e4', init=str(new_teacher), teacher=str(old_teacher),
                steps=2000, lr=.0005, warmup=40, seed=1701),
        variant('oldteacher-long6000', teacher=str(old_teacher), steps=6000),
        variant('gen2-long6000', steps=6000),
        variant('gen2-control3052'),
        variant('gen2-alpha10-t2', alpha=.10),
        variant('gen2-alpha40-t2', alpha=.40),
        variant('gen2-alpha25-t15', temperature=1.5),
        variant('gen2-alpha25-t3', temperature=3.),
        variant('dense10-gen2-long6000', config='student_10_no_local.json', steps=6000),
        variant('dense12-gen2-long6000', config='student_12_no_local.json', steps=6000),
    ]


class Search:
    def __init__(self, args):
        self.args, self.out, self.code = args, args.output.resolve(), args.code.resolve()
        self.lock = threading.Lock()
        self.children = {}
        self.plan = experiment_plan(args.old_teacher.resolve(), args.teacher.resolve())
        self.started = time.time()

    def event(self, status, **values):
        row = dict(time=datetime.now().astimezone().isoformat(), status=status, **values)
        with self.lock:
            with (self.out/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            with (self.out/'STATUS').open('a') as stream:
                stream.write(row['time'] + ' ' + status + ' ' + values.get('name', '') + '\n')
            print(json.dumps(row), flush=True)

    def run(self, args, name):
        if (self.out/'STOP').exists():
            raise RuntimeError('STOP requested')
        command = [sys.executable, '-u', *map(str, args)]
        with self.lock:
            with (self.out/'commands.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(), name=name, argv=command,
                                            cwd=str(self.code))) + '\n')
        with (self.out/'logs'/f'{name}.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=self.code, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            with self.lock:
                self.children[process.pid] = process
                write_json(self.out/'ACTIVE.json', {str(pid): p.args for pid, p in self.children.items()})
            try:
                while process.poll() is None:
                    if (self.out/'STOP').exists():
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                        raise RuntimeError('STOP requested')
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError(f'{name} exited {process.returncode}; see logs/{name}.log')
            finally:
                with self.lock:
                    self.children.pop(process.pid, None)
                    write_json(self.out/'ACTIVE.json', {str(pid): p.args for pid, p in self.children.items()})

    def train_args(self, item, directory, steps=None):
        updates = steps or item['steps']
        result = ['train.py', '--implementation', 'student',
                  '--config', self.code/'configs'/item['config'], '--run-dir', directory,
                  '--device', 'cuda', '--precision', 'bf16', '--tf32', '--threads', 4,
                  '--seed', item['seed'], '--batch-size', 32, '--grad-accum', 4,
                  '--updates', updates, '--lr', item['lr'],
                  '--warmup', item.get('warmup', round(updates*.04)),
                  '--min-lr-ratio', .1, '--betas', .9, .999, '--weight-decay', .1,
                  '--ema', .99, '--eval-every', 100, '--save-every', 500, '--log-every', 25,
                  '--teacher-checkpoint', item['teacher'],
                  '--distill-alpha', item['alpha'], '--distill-temperature', item['temperature'],
                  '--run-log', directory.parent/(directory.name + '-RUN_LOG.csv')]
        if item.get('init'):
            result += ['--init-checkpoint', item['init']]
        return result

    def provenance(self):
        costs = {}
        pending = [self.args.teacher.resolve(), self.args.old_teacher.resolve(),
                   self.args.baseline.resolve()]
        while pending:
            path = pending.pop()
            if str(path) in costs:
                continue
            record = dict(checkpoint_sha256=digest(path))
            metrics = path.parent/'metrics.json'
            if metrics.exists():
                data = json.loads(metrics.read_text())
                if data.get('checkpoint_sha256') != record['checkpoint_sha256']:
                    raise ValueError(f'Parent checkpoint hash disagrees with metrics: {path}')
                record.update(metrics=data, metrics_path=str(metrics))
                teacher = (data.get('teacher') or {}).get('path')
                if teacher:
                    pending.append(Path(teacher))
            costs[str(path)] = record
        write_json(self.out/'parent_provenance.json', costs)
        # Include immutable evaluator/data fingerprints without modifying them.
        paths = [p for p in self.code.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.txt')]
        write_json(self.out/'source_hashes.json', {str(p.relative_to(self.code)): digest(p) for p in paths})
        (self.out/'packages.txt').write_text(subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'], text=True))
        (self.out/'gpu.txt').write_text(subprocess.check_output(['nvidia-smi', '-q'], text=True))

    def telemetry(self, done):
        while not done.is_set():
            try:
                text = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                    '--format=csv,noheader'], text=True).strip()
                with (self.out/'gpu_usage.csv').open('a') as stream:
                    stream.write(text + '\n')
            except Exception as error:
                self.event('TELEMETRY_ERROR', error=str(error))
            done.wait(30)

    def train(self, item):
        self.event('TRAIN_START', name=item['name'])
        try:
            self.run(self.train_args(item, self.out/item['name']), item['name']+'-train')
            self.event('TRAIN_DONE', name=item['name'])
            return dict(name=item['name'], ok=True)
        except Exception as error:
            self.event('TRAIN_FAIL', name=item['name'], error=str(error))
            return dict(name=item['name'], ok=False, error=str(error))

    def cpu_score(self, checkpoint, name):
        directory = self.out/'cpu'/name
        directory.mkdir(parents=True)
        results = []
        for repetition in range(self.args.cpu_repeats):
            output = directory/f'validation_cpu_fp32_{repetition + 1}.json'
            ram = directory/f'ram_{repetition + 1}.json'
            # Whole-process RAM includes imports, loading and evaluation; the
            # unchanged evaluator retains its own scoring-only time metric.
            wrapper = ("import json,resource,runpy; "
                       "runpy.run_path('evaluate.py',run_name='__main__'); "
                       f"open({str(ram)!r},'w').write(json.dumps(dict(peak_ram_gib="
                       "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)))")
            self.run(['-c', wrapper, '--checkpoint', checkpoint, '--device', 'cpu',
                      '--precision', 'fp32', '--threads', 4, '--split', 'validation',
                      '--output', output], f'{name}-cpu-{repetition + 1}')
            results.append(json.loads(output.read_text()) | json.loads(ram.read_text()))
        if max(r['bpb'] for r in results)-min(r['bpb'] for r in results) > 1e-5:
            raise ValueError(f'CPU BPB repeats disagree: {name}')
        return dict(name=name, checkpoint=str(checkpoint), bpb=results[0]['bpb'],
                    cpu_seconds_median=statistics.median(r['seconds'] for r in results),
                    cpu_seconds_repeats=[r['seconds'] for r in results],
                    peak_ram_gib=max(r['peak_ram_gib'] for r in results))

    def main(self):
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out/'logs').mkdir()
        write_json(self.out/'plan.json', dict(jobs=self.plan, workers=self.args.workers,
            processed_student_targets=sum(j['steps'] * 32768 for j in self.plan),
            purpose='Validation-only selection; never test; teachers evaluated on training spans only.',
            accounting='Per-student ancestry plus separate unique teacher provenance; smoke runs are extra.'))
        (self.out/'supervisor.pid').write_text(str(os.getpid()) + '\n')
        self.event('START', workers=self.args.workers)
        self.provenance()
        self.run(['-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'tests')
        # Three real BF16 training jobs exercise EMA, teacher loading, backward,
        # checkpoint serialization and warm-start on the actual CUDA software.
        self.event('PREFLIGHT_START')
        (self.out/'preflight').mkdir()
        smoke_items = [self.plan[0], self.plan[-2], self.plan[-1]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            futures = [pool.submit(self.run, self.train_args(j, self.out/'preflight'/j['name'], steps=3),
                                   'preflight-'+j['name']) for j in smoke_items]
            for future in futures:
                future.result()
        self.event('PREFLIGHT_DONE')
        done = threading.Event()
        telemetry = threading.Thread(target=self.telemetry, args=(done,), daemon=True)
        telemetry.start()
        training = []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                futures = [pool.submit(self.train, item) for item in self.plan]
                for future in concurrent.futures.as_completed(futures):
                    training.append(future.result())
                    write_json(self.out/'training_status.json', training)
        finally:
            done.set()
            telemetry.join()
        if (self.out/'STOP').exists():
            raise RuntimeError('STOP requested; CPU measurement not started')
        self.event('CPU_VALIDATION_START')
        baseline = self.cpu_score(self.args.baseline.resolve(), 'baseline')
        incumbent = self.cpu_score(self.args.teacher.resolve(), 'incumbent')
        write_json(self.out/'baseline.json', baseline)
        write_json(self.out/'incumbent.json', incumbent)
        results, failures = [], [r for r in training if not r['ok']]
        for item in self.plan:
            if not any(r['name'] == item['name'] and r['ok'] for r in training):
                continue
            name = item['name']
            try:
                row = self.cpu_score(self.out/name/'checkpoint.pt', name)
                metrics = json.loads((self.out/name/'metrics.json').read_text())
                if abs(row['bpb'] - metrics['validation']['bpb']) > 1e-5:
                    raise ValueError(f'CPU/GPU BPB mismatch: {name}')
                row.update(assets_bytes=metrics['serialized_inference_asset_bytes'],
                    processed_targets=metrics['processed_targets'],
                    processed_targets_including_ancestry=metrics['processed_targets_including_ancestry'],
                    train_seconds=metrics['train_seconds'], selected_step=metrics['selected_step'],
                    selected_weights=metrics['selected_weights'])
                row['time_ratio'] = row['cpu_seconds_median']/baseline['cpu_seconds_median']
                row['eligible'] = row['time_ratio'] <= 5 and row['peak_ram_gib'] <= 4 and row['assets_bytes'] <= 64*2**20
                row['beats_incumbent'] = row['bpb'] < incumbent['bpb']
                row['practical_24s_guardrail'] = row['cpu_seconds_median'] <= 24
                results.append(row)
                self.event('CPU_VALIDATION_DONE', name=name, **{k: row[k] for k in ('bpb','eligible')})
                write_json(self.out/'results.json', results)
            except Exception as error:
                failures.append(dict(name=name, phase='cpu', error=str(error)))
                self.event('CPU_VALIDATION_FAIL', name=name, error=str(error))
        eligible = [r for r in results if r['eligible'] and r['beats_incumbent']]
        selection = min(eligible, key=lambda r: r['bpb']) if eligible else incumbent
        write_json(self.out/'selection.json', dict(candidate=selection, selection_split='validation',
                   frozen_for_test=False, failures=failures))
        write_json(self.out/'costs.json', dict(wall_seconds=time.time()-self.started,
                   completed_run_train_seconds_sum=sum(r['train_seconds'] for r in results),
                   planned_student_targets=sum(j['steps']*32768 for j in self.plan),
                   smoke_student_targets=3*3*32768, workers=self.args.workers,
                   note='Parallel run seconds overlap; sum is not GPU-exclusive time. Parent costs in parent_provenance.json.'))
        marker = 'FINISHED_WITH_FAILURES' if failures else 'COMPLETE'
        (self.out/marker).write_text('Validation-only search finished. Review before freezing/testing.\n')
        self.event(marker, best_bpb=selection['bpb'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--old-teacher', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3, choices=(1, 2, 3))
    p.add_argument('--cpu-repeats', type=int, default=3, choices=(1, 2, 3))
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory; prior runs are immutable.')
    search = Search(args)
    try:
        search.main()
    except BaseException as error:
        if search.out.exists():
            (search.out/'FAILED').write_text(repr(error) + '\n')
            search.event('FAILED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
