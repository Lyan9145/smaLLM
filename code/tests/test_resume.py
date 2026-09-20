"""End-to-end recovery test on synthetic tokens, including dropout and accumulation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


class ResumeTests(unittest.TestCase):
    def test_recovery_matches_uninterrupted_training(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config = temporary / 'config.json'
            config.write_text(json.dumps(dict(vocab=2048, context=256, width=16, depth=1, heads=2,
                                             hidden_width=32, dropout=.1, local_mixing=True)))
            launch = ("import torch, train; "
                      "train.load_data = lambda: {'train': (torch.arange(1024)%100, 1024), "
                      "'validation': (torch.arange(65)%100, 65)}; train.main()")
            shared = [sys.executable, '-c', launch, '--config', str(config), '--steps', '4',
                      '--batch-size', '1', '--grad-accum', '2', '--warmup', '1', '--eval-every', '2',
                      '--ema', '.9', '--threads', '1', '--run-log', str(temporary / 'runs.csv')]
            for name, extra in [('full', []), ('part', ['--stop-after', '2']),
                                ('resumed', ['--resume', str(temporary / 'part/resume.pt')])]:
                result = subprocess.run(shared + ['--run-dir', str(temporary / name)] + extra,
                                        cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            full = torch.load(temporary/'full/resume.pt', weights_only=True)
            resumed = torch.load(temporary/'resumed/resume.pt', weights_only=True)
            for key in ('model', 'ema'):
                for name, tensor in full[key].items():
                    torch.testing.assert_close(tensor, resumed[key][name], rtol=0, atol=0)
            for key in ('sampler_rng', 'torch_rng'):
                torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
            self.assertEqual(full['best']['step'], resumed['best']['step'])
            self.assertEqual(full['best']['weights'], resumed['best']['weights'])
            for name, state in full['optimizer']['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, resumed['optimizer']['state'][name][key], rtol=0, atol=0)
            metrics = json.loads((temporary/'resumed/metrics.json').read_text())
            self.assertEqual(metrics['train_tokens'], 4 * 1 * 2 * 256)
            self.assertEqual(metrics['ancestry'][0]['train_tokens'], 2 * 1 * 2 * 256)
            self.assertEqual(len(metrics['validation_history']), 4)
            final = torch.load(temporary/'resumed/checkpoint.pt', weights_only=True)
            self.assertNotIn('optimizer', final)


if __name__ == '__main__':
    unittest.main()
