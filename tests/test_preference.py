import pytest


def test_answer_score_excludes_prompt_and_retains_eos_gradient():
    import torch
    from witrans_tools.preference import answer_scores
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 4]])
    lp, nll = answer_scores(logits, labels)
    assert float(nll.detach()) == pytest.approx(1.6094379)
    assert float(lp.detach()) == pytest.approx(-2 * 1.6094379)
    nll.backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 2].abs().sum() > 0
    assert logits.grad[0, 3].abs().sum() == 0


def test_preference_loss_rewards_correct_answer_margin():
    import torch
    from witrans_tools.preference import cpo_loss
    chosen = torch.tensor(-3.0, requires_grad=True)
    wrong = torch.tensor(-2.0, requires_grad=True)
    loss = cpo_loss(chosen, wrong, torch.tensor(1.0))
    loss.backward()
    assert chosen.grad < 0 and wrong.grad > 0


def test_weighted_nll_focuses_selected_answer_without_changing_margin():
    import torch
    from witrans_tools.preference import answer_scores
    logits = torch.tensor([[[0.,0.,0.], [2.,0.,0.], [0.,0.,1.], [0.,0.,0.]]], requires_grad=True)
    labels = torch.tensor([[-100,-100,1,2]])
    plain_lp, plain_nll = answer_scores(logits, labels)
    weights = torch.tensor([[float('nan'),float('nan'),3.,1.]])
    weighted_lp, weighted_nll = answer_scores(logits, labels, weights)
    assert float(weighted_lp.detach()) == pytest.approx(float(plain_lp.detach()))
    assert weighted_nll > plain_nll
    weighted_nll.backward()
    assert logits.grad[0,0].abs().sum() == 0
    # Unweighted EOS receives gradient as well as selected answer position.
    assert logits.grad[0,1].abs().sum() > 0 and logits.grad[0,2].abs().sum() > 0


@pytest.mark.parametrize('invalid', [0.,-1.,float('nan'),float('inf')])
def test_weighted_nll_rejects_invalid_supervised_weights(invalid):
    import torch
    from witrans_tools.preference import answer_scores
    with pytest.raises(ValueError, match='positive and finite'):
        answer_scores(torch.zeros(1,3,4), torch.tensor([[-100,1,2]]), torch.tensor([[0.,invalid,1.]]))
