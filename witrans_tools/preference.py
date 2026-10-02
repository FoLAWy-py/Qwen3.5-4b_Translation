"""Reference-free CPO primitives; prompt and padding never enter answer loss."""


def answer_scores(logits, labels, token_weights=None):
    import torch
    import torch.nn.functional as F
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    if not mask.any():
        raise ValueError("偏好答案没有有效token")
    # Select supervised positions before FP32 log-softmax to limit vocabulary memory.
    losses = F.cross_entropy(logits[:, :-1][mask].float(), shifted_labels[mask], reduction="none")
    if labels.shape[0] != 1:
        raise ValueError("本机偏好训练仅支持microbatch1")
    if token_weights is None:
        nll = losses.mean()
    else:
        if token_weights.shape != labels.shape:
            raise ValueError('Token weights must match labels')
        weights = token_weights[:, 1:][mask].float()
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Supervised weights must be positive and finite')
        nll = (losses * weights).sum() / weights.sum()
    # Preference margin remains summed likelihood. Only chosen-answer NLL
    # may be weighted; prompt/padding are still excluded and EOS retained.
    return -losses.sum(), nll


def cpo_loss(chosen_logp, rejected_logp, chosen_nll, beta=0.1):
    import torch.nn.functional as F
    if beta <= 0:
        raise ValueError("beta 必须为正数")
    return -F.logsigmoid(beta * (chosen_logp - rejected_logp)) + chosen_nll
