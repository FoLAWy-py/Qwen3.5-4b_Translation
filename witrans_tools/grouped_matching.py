"""Experimental two-source/two-answer assignment loss; no efficacy claim."""


def grouped_fact_loss(logps, supervised_counts, *, temperature=.1, matching_weight=1.):
    """Inputs: summed answer/EOS log probabilities for all four source/answer pairs.

    Rows are sources; columns are fixed candidate answers. Correct pairs are the
    diagonal. Column answer token counts are identical across source rows.
    Assignment contrast cancels additive source-difficulty and answer-fluency
    priors in the normalized score matrix. It does not prove semantic fidelity.
    """
    import torch
    import torch.nn.functional as F
    if logps.shape != (2,2) or supervised_counts.shape != (2,):
        raise ValueError('Two source rows and two answer columns required')
    if (not torch.isfinite(logps).all() or not torch.isfinite(supervised_counts).all()
            or (supervised_counts <= 0).any()):
        raise ValueError('Finite scores and positive answer/EOS counts required')
    if not 0 < temperature < float('inf') or not 0 <= matching_weight < float('inf'):
        raise ValueError('Positive finite temperature and nonnegative finite weight required')
    scores = logps / supervised_counts.to(device=logps.device, dtype=logps.dtype)[None,:]
    margin = scores[0,0] + scores[1,1] - scores[0,1] - scores[1,0]
    sft = -(scores[0,0]+scores[1,1])/2
    matching = F.softplus(-margin/temperature)
    return {'loss':sft+matching_weight*matching, 'sft':sft,'matching':matching,'assignment_margin':margin}
