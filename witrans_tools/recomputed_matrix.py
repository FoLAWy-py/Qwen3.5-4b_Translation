"""Exact chain-rule score recomputation with bounded graph memory and saved RNG."""


def recomputed_matrix_backward(score, objective, *, scale=1.):
    import torch

    def rng():
        return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

    def restore(state):
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state_all(state[1])

    states, values = [], []
    with torch.no_grad():
        for i in range(2):
            row = []
            for j in range(2):
                states.append(rng())
                row.append(score(i,j).detach())
            values.append(torch.stack(row))
    matrix = torch.stack(values).requires_grad_(True)
    diagnostics = objective(matrix)
    if not torch.isfinite(diagnostics['loss']):
        raise ValueError('Nonfinite matrix objective')
    coefficients, = torch.autograd.grad(diagnostics['loss'],matrix)
    final_rng = rng()
    try:
        for i in range(2):
            for j in range(2):
                restore(states[2*i+j])
                value = score(i,j)
                if not torch.allclose(value.detach(),matrix[i,j].detach(),rtol=1e-5,atol=1e-4):
                    raise ValueError('Recomputed dropout scores differ; gradient would not bind objective')
                (value*coefficients[i,j].detach()*scale).backward()
    finally:
        restore(final_rng)
    return {key:float(value.detach()) for key,value in diagnostics.items()}
