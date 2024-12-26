import torch
from torchdiffeq import odeint


class CFM:
    """Conditional Flow Matching"""

    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()

    def training_losses(self, flow, x, t, model_kwargs):
        """
        x: (B, T, C)
        t: (B,)
        """
        x_1 = x
        x_0 = torch.randn_like(x_1)
        # unsqueeze t to (B, 1, 1)
        t_unsqueezed = t.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - t_unsqueezed) * x_0 + t_unsqueezed * x_1
        dx_t = x_1 - x_0
        loss = self.loss_fn(flow(t=t, x_t=x_t, **model_kwargs), dx_t)
        return {"loss": loss}

    @torch.no_grad()
    def solve_ode(
        self,
        flow,
        shape,
        z,
        time_grid: torch.Tensor,
        model_kwargs,
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        return_intermediates: bool = False,
        device: torch.device = None,
        **model_extras,
    ):

        def ode_func(t, x):
            B = x.shape[0]
            # repeat interleaved t
            t_interleaved = torch.repeat_interleave(t[None], B, dim=0)
            return flow(x_t=x, t=t_interleaved, **model_kwargs)

        ode_opts = {}
        time_grid = time_grid.to(device)

        # Approximate ODE solution with numerical ODE solver
        x_0 = torch.randn(shape, device=device)
        torch.nn.init.trunc_normal_(x_0, mean=0.0, std=1.0, a=-2.0, b=2.0)
        sol = odeint(
            ode_func,
            x_0,
            time_grid,
            method=method,
            options=ode_opts,
            atol=atol,
            rtol=rtol,
        )

        if return_intermediates:
            return sol
        else:
            return sol[-1]


def create_cfm():
    return CFM()
