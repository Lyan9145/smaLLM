import unittest

import torch

from distillation import distillation_loss
from moe_model import MoEStudentLM
from speculative import speculative_decode


class _TinyLM(torch.nn.Module):
    context = 16

    def __init__(self, vocab=17):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, 8)
        self.head = torch.nn.Linear(8, vocab)

    def forward(self, ids):
        return self.head(self.embedding(ids))

    def predict_log_probs(self, ids):
        return self(ids).log_softmax(-1)


class DistillationMoETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_distillation_has_student_gradients_and_detached_teacher(self):
        student = torch.randn(2, 5, 17, requires_grad=True)
        teacher = torch.randn(2, 5, 17, requires_grad=True)
        targets = torch.randint(17, (2, 5))
        loss, hard, soft = distillation_loss(student, teacher, targets, 2.0, .4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(soft), 0.)
        self.assertGreater(float(hard), 0.)

    def test_moe_is_causal_normalized_and_example_independent(self):
        config = dict(vocab=2048, context=256, width=32, heads=4, depth=2,
                      hidden_width=48, dropout=0., num_experts=2,
                      moe_aux_weight=.01)
        model = MoEStudentLM(config).eval()
        x = torch.randint(2048, (2, 12))
        changed = x.clone()
        changed[:, 7:] = (changed[:, 7:] + 19) % 2048
        with torch.no_grad():
            first = model.predict_log_probs(x)
            second = model.predict_log_probs(changed)
            alone = model.predict_log_probs(x[:1])
        torch.testing.assert_close(first[:, :7], second[:, :7], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first[:1], alone, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first.logsumexp(-1), torch.zeros(2, 12), atol=1e-6, rtol=1e-6)

    def test_moe_router_receives_task_gradient(self):
        config = dict(vocab=2048, context=256, width=32, heads=4, depth=1,
                      hidden_width=48, dropout=0., num_experts=2,
                      moe_aux_weight=0.)
        model = MoEStudentLM(config)
        ids = torch.randint(2048, (2, 8))
        loss = torch.nn.functional.cross_entropy(model(ids).flatten(0, 1), torch.randint(2048, (16,)))
        loss.backward()
        self.assertIsNotNone(model.blocks[0].moe.router.weight.grad)
        self.assertGreater(model.blocks[0].moe.router.weight.grad.abs().sum().item(), 0.)

    def test_speculative_decode_respects_context_and_shape(self):
        draft, target = _TinyLM(), _TinyLM()
        prefix = torch.randint(17, (1, 3))
        result = speculative_decode(draft, target, prefix, 4)
        self.assertEqual(tuple(result.shape), (1, 7))
        with self.assertRaises(ValueError):
            speculative_decode(draft, target, torch.randint(17, (1, 14)), 3)


if __name__ == '__main__':
    unittest.main()
