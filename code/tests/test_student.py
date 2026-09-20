"""Mechanism and training-state tests beyond the fixed contract."""
import tempfile
import unittest
from pathlib import Path

import torch

from student_model import CausalLocalMix
from student import build_model
from training_utils import inference_asset_bytes, select_candidate, update_ema, atomic_save
from train import learning_rate


class StudentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_local_impulse_is_left_padded_and_channel_independent(self):
        local = CausalLocalMix(2, 3)
        with torch.no_grad():
            local.conv.weight.fill_(1)
            local.gate.zero_()
        x = torch.zeros(1, 8, 2)
        x[0, 3, 0] = 2
        expected = torch.zeros_like(x)
        expected[0, 3:6, 0] = 1
        torch.testing.assert_close(local(x), expected)
        local(x).sum().backward()
        self.assertGreater(local.gate.grad.abs().sum().item(), 0)
        self.assertGreater(local.conv.weight.grad.abs().sum().item(), 0)

    def test_ablation_removes_only_local_parameters(self):
        config = dict(vocab=2048, width=32, heads=4, depth=2, context=256)
        full = build_model(config)
        ablated = build_model(config | {'local_mixing': False})
        removed = set(full.state_dict()) - set(ablated.state_dict())
        self.assertEqual(removed, {f'blocks.{i}.local.{key}' for i in range(2) for key in ('gate', 'conv.weight')})
        self.assertEqual(sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in ablated.parameters()), 2 * 32 * 4)

    def test_asset_size_counts_tied_storage_once_and_sources(self):
        from model import GPT
        model = GPT(dict(vocab=2048, width=32, heads=4, depth=2, context=256))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.py'
            source.write_bytes(b'1234567')
            self.assertEqual(inference_asset_bytes(model, [source]), sum(p.numel() * p.element_size() for p in model.parameters()) + 7)

    def test_ema_selection_freezes_winner_and_safe_loads(self):
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(2)
        ema = {'weight': torch.zeros_like(model.weight)}
        update_ema(ema, model, .75)
        torch.testing.assert_close(ema['weight'], torch.full_like(model.weight, .5))
        best = select_candidate(None, model.state_dict(), {'bpb': 3.}, 1, 'raw')
        best = select_candidate(best, ema, {'bpb': 2.}, 1, 'ema')
        ema['weight'].zero_()
        best = select_candidate(best, model.state_dict(), {'bpb': 2.5}, 2, 'raw')
        self.assertEqual((best['weights'], best['step']), ('ema', 1))
        torch.testing.assert_close(best['model']['weight'], torch.full_like(model.weight, .5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            atomic_save(best, path)
            saved = torch.load(path, weights_only=True)
            torch.testing.assert_close(saved['model']['weight'], best['model']['weight'])

    def test_baseline_schedule_is_preserved(self):
        import math
        recipe = dict(lr=.001, warmup=100, min_lr_ratio=.1, steps=1200)
        for step in (0, 99, 500, 1199):
            expected = .001 * min(1., (step + 1) / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / 1200)))
            self.assertAlmostEqual(learning_rate(step, recipe), expected)


if __name__ == '__main__':
    unittest.main()
