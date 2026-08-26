"""Flow matching sampling.

Time runs from noise to data: ``z_t = (1 - t) z_noise + t z_data``, with the model
predicting the velocity ``z_data - z_noise``. Sampling starts from Gaussian noise at
``t = 0`` and integrates the velocity field to ``t = 1`` with an adaptive ODE solver
(Dormand-Prince, as in the paper).

Two integrators live here:

* :func:`sample_ode` -- the standard single-schedule solve, used by the naive and
  cascaded variants.
* :func:`decoupled_ode_sample` -- the joint variant, where content and position
  channels advance on independently shifted time schedules within one solve.
"""

import torch


def sample_ode(z, model_fn, *, t0=0.0, t1=1.0, num_steps=50, atol=1e-6, rtol=1e-3,
               method="dopri5", **model_kwargs):
    """Integrate the velocity field from noise to data.

    Args:
        z: ``(B, L, C)`` initial Gaussian noise.
        model_fn: called as ``model_fn(x, t, **model_kwargs)`` returning the velocity.
        num_steps: number of interpolation points reported by the solver. The step
            size itself is adaptive; this sets the output grid.

    Returns:
        A list of states along the trajectory; the last element is the sample.
    """
    from torchdiffeq import odeint

    B = z.shape[0]
    t = torch.linspace(t0, t1, num_steps, device=z.device)

    def drift(t_scalar, x):
        return model_fn(x, torch.ones(B, device=z.device) * t_scalar, **model_kwargs)

    sol = odeint(drift, z, t, method=method, atol=[atol], rtol=[rtol])
    return list(sol)


def shift_timesteps(t, mu, sigma):
    """Logit-normal time shift ``t' = sigmoid(logit(t) * sigma + mu)``.

    ``(mu, sigma) = (0, 1)`` is the identity. Negative ``mu`` biases a channel group
    toward finishing earlier, which is how the joint variant resolves spatial layout
    before fine content detail.
    """
    t = t.clamp(1e-6, 1 - 1e-6)
    return torch.sigmoid(torch.logit(t) * sigma + mu)


def shift_derivative(t_raw, t_shifted, sigma):
    """``dt'/dt`` for :func:`shift_timesteps`, used to rescale the velocity."""
    t_raw = t_raw.clamp(1e-6, 1 - 1e-6)
    t_shifted = t_shifted.clamp(1e-6, 1 - 1e-6)
    return t_shifted * (1 - t_shifted) * sigma / (t_raw * (1 - t_raw))


def decoupled_ode_sample(z, model_fn, *, content_dim, t0=0.0, t1=1.0, num_steps=50,
                         content_mu=0.0, content_sigma=1.0, pos_mu=0.0, pos_sigma=1.0,
                         atol=1e-6, rtol=1e-3, method="dopri5", **model_kwargs):
    """Integrate content and position channels on separate time schedules.

    The whole state is solved in a single ODE call over a uniform raw time grid.
    Inside the drift, the raw time is mapped to a content time and a position time,
    and each channel group's velocity is rescaled by its own ``dt'/dt`` so that each
    effectively follows its own schedule.

    Args:
        z: ``(B, L, content_dim + 3)`` noise over content and position channels.
        content_dim: number of leading channels that carry content.
        content_mu / content_sigma, pos_mu / pos_sigma: logit-normal shift per group.
    """
    from torchdiffeq import odeint

    device = z.device
    B = z.shape[0]
    extra_dims = z.ndim - 1

    raw_t = torch.linspace(t0, t1, num_steps, device=device)

    def drift(t_scalar, x):
        t_raw = torch.ones(B, device=device) * t_scalar
        t_c = shift_timesteps(t_raw, content_mu, content_sigma)
        t_p = shift_timesteps(t_raw, pos_mu, pos_sigma)

        v = model_fn(x, t_c, t_pos=t_p, **model_kwargs)

        dtc = shift_derivative(t_raw, t_c, content_sigma).view(B, *([1] * extra_dims))
        dtp = shift_derivative(t_raw, t_p, pos_sigma).view(B, *([1] * extra_dims))
        return torch.cat([v[..., :content_dim] * dtc, v[..., content_dim:] * dtp], dim=-1)

    sol = odeint(drift, z, raw_t, method=method, atol=[atol], rtol=[rtol])
    return [sol[-1]]
