import pytest
import torch
from witrans_tools.fact_objective import fact_objective,matching_gradient_norm_ratio


@pytest.mark.parametrize('margin',[-1.,0.,.5,1.5])
def test_analytic_strength_matches_actual_objective_autograd(margin):
    counts = torch.tensor([7.,17.],dtype=torch.float64)
    scores = torch.tensor([[-.5,-.5-margin/2],[-.5-margin/2,-.5]],dtype=torch.float64)
    logps = (scores*counts[None,:]).requires_grad_()
    result = fact_objective(logps,counts,'matching',temperature=.1,matching_weight=.1)
    g_sft, = torch.autograd.grad(result['sft'],logps,retain_graph=True)
    g_matching, = torch.autograd.grad(.1*result['matching'],logps)
    observed = float(g_matching.norm()/g_sft.norm())
    assert matching_gradient_norm_ratio(margin,temperature=.1,matching_weight=.1)==pytest.approx(observed,rel=1e-8)


def test_extreme_margin_diagnostic_remains_finite():
    assert matching_gradient_norm_ratio(1000.,temperature=.1,matching_weight=.1)==0.
    assert matching_gradient_norm_ratio(-1000.,temperature=.1,matching_weight=.1)==pytest.approx(2*2**.5)
