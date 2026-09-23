"""Warm-start is a new training phase, never a silent exact-resume override."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from common import PROTOCOL
from student import build_model
from training_utils import initialize_from_checkpoint


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = dict(vocab=2048, context=256, width=16, depth=1, heads=2,
                           hidden_width=32, dropout=.05, local_mixing=False)

    def test_selected_weights_and_validation_of_parent(self):
        model = build_model(self.config)
        saved = dict(protocol=PROTOCOL, implementation='student', config=self.config,
                     model=model.state_dict(), train_tokens=1024, selected_step=2,
                     selected_weights='ema')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save(saved, path)
            child = build_model(self.config)
            metadata = initialize_from_checkpoint(child, path, 'student', self.config)
            self.assertEqual(metadata['processed_targets_including_ancestry'], 1024)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, child.state_dict()[name], rtol=0, atol=0)
            for changes in [dict(protocol='wrong'), dict(implementation='model'),
                            dict(config=self.config | {'dropout': .1}),
                            dict(optimizer={}), dict(train_tokens=None)]:
                with self.subTest(changes=changes):
                    torch.save(saved | changes, path)
                    with self.assertRaises(ValueError):
                        initialize_from_checkpoint(child, path, 'student', self.config)

    def test_continuation_and_recovery_preserve_ancestry(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config = temporary / 'config.json'
            config.write_text(json.dumps(self.config))
            parent = temporary / 'parent.pt'
            torch.save(dict(protocol=PROTOCOL, implementation='student', config=self.config,
                            model=build_model(self.config).state_dict(), train_tokens=1024,
                            selected_step=4, selected_weights='ema'), parent)
            launch = ("import torch, train; "
                      "train.load_data = lambda: {'train': (torch.arange(1024)%100, 1024), "
                      "'validation': (torch.arange(65)%100, 65)}; train.main()")
            shared = [sys.executable, '-c', launch, '--config', str(config), '--steps', '2',
                      '--batch-size', '1', '--ema', '.9', '--threads', '1',
                      '--run-log', str(temporary/'runs.csv')]
            for name, extra in [
                ('full', ['--init-checkpoint', str(parent)]),
                ('part', ['--init-checkpoint', str(parent), '--stop-after', '1']),
                ('resumed', ['--resume', str(temporary/'part/resume.pt')]),
            ]:
                run = subprocess.run(shared + ['--run-dir', str(temporary/name)] + extra,
                                     cwd=root, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            full = torch.load(temporary/'full/resume.pt', weights_only=True)
            resumed = torch.load(temporary/'resumed/resume.pt', weights_only=True)
            for key in ('model', 'ema'):
                for name, value in full[key].items():
                    torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)
            metrics = json.loads((temporary/'resumed/metrics.json').read_text())
            self.assertEqual(metrics['processed_targets'], 512)
            self.assertEqual(metrics['processed_targets_including_ancestry'], 1536)
            self.assertEqual(metrics['inherited_targets'], 1024)
            checkpoint = torch.load(temporary/'resumed/checkpoint.pt', weights_only=True)
            self.assertEqual(checkpoint['train_tokens'], 1536)
            # A second-generation continuation must retain both prior phases.
            metadata = initialize_from_checkpoint(build_model(self.config),
                temporary/'resumed/checkpoint.pt', 'student', self.config)
            self.assertEqual(metadata['processed_targets_including_ancestry'], 1536)
            self.assertLessEqual(metrics['validation']['bpb'], metrics['validation_history'][0]['bpb'])
            invalid = subprocess.run(shared + ['--resume', str(temporary/'part/resume.pt'),
                '--init-checkpoint', str(parent)], cwd=root, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == '__main__':
    unittest.main()
