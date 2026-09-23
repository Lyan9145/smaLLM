"""Training-only teacher--student distillation helpers.

The evaluator never imports this module.  A teacher is used only while
training on the supplied training stream; the submitted checkpoint contains
the student weights and does not depend on the teacher at inference time.
"""

import torch
from torch.nn import functional as F


def distillation_loss(student_logits, teacher_logits, targets, temperature=2.0,
                      alpha=0.5):
    """Blend hard next-token CE with temperature-scaled teacher KL.

    The reduction is per token, so the soft term has a comparable scale for
    every vocabulary size and sequence length.  Teacher tensors are detached
    by the caller (and are detached again here as a safety boundary).
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError('student and teacher logits must have the same shape')
    if student_logits.ndim != 3 or targets.shape != student_logits.shape[:2]:
        raise ValueError('logits must be [batch, time, vocab] and targets [batch, time]')
    if not (0.0 <= alpha <= 1.0):
        raise ValueError('alpha must be in [0, 1]')
    if not (temperature > 0.0 and torch.isfinite(torch.as_tensor(temperature))):
        raise ValueError('temperature must be positive and finite')

    student = student_logits.float()
    teacher = teacher_logits.detach().float()
    hard = F.cross_entropy(student.flatten(0, 1), targets.flatten())
    if alpha == 0.0:
        return hard, hard.detach(), torch.zeros_like(hard.detach())

    scaled = float(temperature)
    soft = F.kl_div(
        F.log_softmax(student / scaled, dim=-1),
        F.softmax(teacher / scaled, dim=-1),
        reduction='none',
    ).sum(dim=-1).mean() * (scaled * scaled)
    total = (1.0 - alpha) * hard + alpha * soft
    return total, hard.detach(), soft.detach()
