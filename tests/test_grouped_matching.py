import pytest


def test_assignment_cancels_source_and_answer_priors():
    import torch
    from witrans_tools.grouped_matching import grouped_fact_loss
    counts = torch.tensor([5.,7.])
    scores = torch.tensor([[-1.,-2.],[-2.,-1.]])
    original = grouped_fact_loss(scores*counts,counts)
    shifted = scores + torch.tensor([[.3],[-.4]]) + torch.tensor([[.7,-.8]])
    result = grouped_fact_loss(shifted*counts,counts)
    assert result['assignment_margin'].item() == pytest.approx(original['assignment_margin'].item())
    assert result['matching'].item() == pytest.approx(original['matching'].item(),abs=1e-7)


def test_wrong_assignment_drives_all_four_conditional_scores():
    import torch
    from witrans_tools.grouped_matching import grouped_fact_loss
    logps = torch.tensor([[-8.,-2.],[-2.,-8.]],requires_grad=True)
    result = grouped_fact_loss(logps,torch.tensor([4.,4.]))
    result['matching'].backward()
    assert logps.grad[0,0] < 0 and logps.grad[1,1] < 0
    assert logps.grad[0,1] > 0 and logps.grad[1,0] > 0


def test_zero_matching_weight_recovers_equal_pair_sft():
    import torch
    from witrans_tools.grouped_matching import grouped_fact_loss
    logps = torch.tensor([[-8.,-99.],[-99.,-15.]],requires_grad=True)
    result = grouped_fact_loss(logps,torch.tensor([4.,5.]),matching_weight=0.)
    assert result['loss'].item() == pytest.approx(2.5)
    result['loss'].backward()
    assert logps.grad[0,1] == 0 and logps.grad[1,0] == 0


def test_invalid_answer_counts_are_rejected():
    import torch
    from witrans_tools.grouped_matching import grouped_fact_loss
    with pytest.raises(ValueError,match='positive'):
        grouped_fact_loss(torch.zeros(2,2),torch.tensor([4.,0.]))
