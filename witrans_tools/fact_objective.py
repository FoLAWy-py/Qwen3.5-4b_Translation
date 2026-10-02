"""Common four-score objectives for a small source-conditioned training pilot."""
from .grouped_matching import grouped_fact_loss
from .preference import cpo_loss


def matching_gradient_norm_ratio(margin, *, temperature, matching_weight):
    """Matching-only/SFT L2 score-gradient norm, including both off-diagonals."""
    import math
    if not all(math.isfinite(v) for v in (margin,temperature,matching_weight)) or temperature<=0 or matching_weight<0:
        raise ValueError('Finite margin, positive temperature and nonnegative weight required')
    scaled = margin/temperature
    if scaled>=0:
        inverse = math.exp(-scaled)
        probability = inverse/(1+inverse)
    else:
        probability = 1/(1+math.exp(scaled))
    return 2*math.sqrt(2)*matching_weight/temperature*probability


def fact_objective(logps, counts, mode, *, beta=.1, temperature=.1, matching_weight=.1):
    common = grouped_fact_loss(logps, counts, temperature=temperature, matching_weight=matching_weight)
    if mode == 'matching':
        return common
    if mode == 'sft':
        return {**common, 'loss':common['sft']}
    if mode == 'cpo':
        loss = (cpo_loss(logps[0,0],logps[0,1],-logps[0,0]/counts[0],beta)
                +cpo_loss(logps[1,1],logps[1,0],-logps[1,1]/counts[1],beta))/2
        return {**common,'loss':loss}
    raise ValueError('Unsupported fact objective')
