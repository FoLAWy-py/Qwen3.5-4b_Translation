import copy
import pytest
import torch
from witrans_tools.fact_objective import fact_objective
from witrans_tools.recomputed_matrix import recomputed_matrix_backward


@pytest.mark.parametrize('mode',['matching','sft','cpo'])
def test_dropout_recomputation_matches_full_graph_and_rng(mode):
    torch.manual_seed(15)
    direct = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.3))
    recomputed = copy.deepcopy(direct)
    sources = torch.tensor([[1.,2.,3.],[-1.,2.,1.]])
    targets = torch.tensor([[0.,1.,2.,3.],[3.,2.,1.,0.]])
    counts = torch.tensor([2.,3.])
    def score(model,i,j):
        return -(model(sources[i])-targets[j]).square().sum()
    torch.manual_seed(25)
    matrix = torch.stack([torch.stack([score(direct,i,j) for j in range(2)]) for i in range(2)])
    result = fact_objective(matrix,counts,mode)
    (result['loss']*.125).backward()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(25)
    actual = recomputed_matrix_backward(lambda i,j:score(recomputed,i,j),
                 lambda values:fact_objective(values,counts,mode),scale=.125)
    assert actual['loss']==pytest.approx(float(result['loss'].detach()),rel=1e-6)
    assert torch.equal(expected_rng,torch.get_rng_state())
    for left,right in zip(direct.parameters(),recomputed.parameters()):
        assert torch.allclose(left.grad,right.grad,rtol=1e-5,atol=1e-5)


def test_changed_recomputed_score_rejected():
    value = torch.tensor(1.,requires_grad=True)
    calls = []
    def score(i,j):
        calls.append((i,j))
        return value*(-1. if len(calls)<=4 else -2.)
    with pytest.raises(ValueError,match='dropout scores differ'):
        recomputed_matrix_backward(score,lambda matrix:fact_objective(matrix,torch.tensor([1.,1.]),'matching'))
