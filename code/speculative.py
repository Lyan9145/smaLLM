"""Training/research utility for speculative autoregressive generation.

The fixed MP1 scorer does not use this function: it requests probabilities at
every position in independent windows. Speculation is therefore unsuitable
for benchmark scoring, but this utility is useful for separate generation
latency experiments.
"""

import torch


@torch.no_grad()
def speculative_decode(draft, target, prefix, max_new_tokens, temperature=1.0,
                       eos_token=None):
    """Generate with draft proposals verified by a target model.

    Both models must implement predict_log_probs. This implementation uses
    one-token proposals, preserves the target distribution by rejection
    sampling, and rejects sequences that exceed either model context.
    """
    if prefix.ndim != 2 or prefix.shape[0] != 1:
        raise ValueError('prefix must have shape [1, time]')
    if max_new_tokens < 0 or temperature <= 0:
        raise ValueError('max_new_tokens must be nonnegative and temperature positive')
    if prefix.shape[1] + max_new_tokens > min(draft.context, target.context):
        raise ValueError('speculative sequence exceeds model context')
    result = prefix.clone()
    draft.eval()
    target.eval()
    for _ in range(max_new_tokens):
        draft_probs = (draft.predict_log_probs(result)[:, -1] / temperature).softmax(-1)
        target_probs = (target.predict_log_probs(result)[:, -1] / temperature).softmax(-1)
        proposed = torch.distributions.Categorical(probs=draft_probs).sample()
        q = draft_probs.gather(-1, proposed[:, None]).squeeze(-1)
        p = target_probs.gather(-1, proposed[:, None]).squeeze(-1)
        accept = torch.rand_like(q) < torch.minimum(torch.ones_like(q), p / q.clamp_min(1e-12))
        residual = (target_probs - draft_probs).clamp_min(0)
        residual = residual / residual.sum(-1, keepdim=True).clamp_min(1e-12)
        replacement = torch.distributions.Categorical(probs=residual).sample()
        next_token = torch.where(accept, proposed, replacement)
        result = torch.cat((result, next_token[:, None]), dim=1)
        if eos_token is not None and bool((next_token == eos_token).all()):
            break
    return result
