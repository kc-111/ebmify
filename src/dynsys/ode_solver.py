"""Tsitouras 5(4) Runge-Kutta solver for batched ODEs in PyTorch.

Single class ``Tsit5SolverTorch`` adaptively integrates ``dy/dt = f(t, y)``
for a batch of independent systems sharing the same RHS. Step acceptance
is governed by an embedded 4th-order error estimator, dense output is
produced via cubic Hermite interpolation between accepted endpoints, and
all per-step buffers are preallocated so the inner loop allocates no new
memory.

Public API:
    * :class:`Tsit5SolverTorch` — the solver class.
    * :func:`_hermite_interp_out` — module-private cubic Hermite kernel
      used for dense output between accepted RK steps.
"""

import torch

class Tsit5SolverTorch:
    """
    An adaptive step-size Tsitouras 5(4) Runge-Kutta solver with dense output
    via cubic Hermite interpolation, implemented in PyTorch.

    Dense output: The solver takes natural adaptive steps and interpolates the
    solution at requested output times using the FSAL (First Same As Last)
    property — Tsit5's last stage ``ks[6]`` evaluated at ``t + h`` is reused
    as the first stage ``ks[0]`` of the next step, giving the right-endpoint
    derivative for free and saving one ``fun`` evaluation per accepted step.

    All intermediate buffers (``ks``, ``dy``, ``y_stage``, ``y_new``,
    ``error_estimate``, ``abs_max``, ``scaled_err``, ``results_y``,
    ``f_current``) are preallocated in ``solve`` to eliminate per-step
    memory allocation overhead — critical for large spatial systems
    (millions of state variables per batch).

    Step control follows Hairer/Wanner: the new step is
    ``h_new = h * safety_factor * err^{-alpha} * err_prev^{beta}`` with
    ``alpha = 0.7/(p+1)``, ``beta = 0.4/(p+1)``, ``p=4`` (Hairer/Wanner
    PI controller; on reject falls back to I-only ``(1/err)^{1/(p+1)}``)
    (the order of the embedded error estimator), capped at ``h_max`` on
    accept and floored at ``h_min`` on reject.

    Tsit5 Solver:
    Tsitouras, C. (2011). Runge-Kutta pairs of order 5 (4) satisfying only the
    first column simplifying assumption. Computers & mathematics with applications,
    62(2), 770-775.

    Error Control:
    Hairer, E., Wanner, G., & Norsett, S. P. (1993). Solving ordinary
    differential equations I: Nonstiff problems. Berlin, Heidelberg:
    Springer Berlin Heidelberg.

    Attributes:
        atol (float): Absolute tolerance per state variable.
        rtol (float): Relative tolerance per state variable.
        h_min (float): Minimum step size (rejection floor).
        h_max (float): Maximum step size (acceptance cap).
        maxiters (int): Hard cap on combined accepted + rejected steps.
        p (int): Order of the embedded error estimator (``4``); used
            in the step-control exponent ``1/(p+1) = 1/5``.
        safety_factor (float): Step-control safety multiplier
            (``0.9``).
        A (list[list[float]]): Lower-triangular Tsit5 stage matrix —
            ``A[j][k]`` is the coefficient applied to stage ``ks[k]``
            when forming stage ``ks[j]`` (``k < j``). Stored as Python
            floats to avoid ``.item()`` calls in the hot loop.
        c (list[float]): Stage time offsets (``t + c[j] * h``).
        b (list[float]): 5th-order accepted-step weights.
        e (list[float]): 4th-order embedded-error weights (computed as
            ``b - b_hat``).
        b_nz (list[tuple[int, float]]): Sparse ``(index, weight)``
            pairs of nonzero ``b`` entries — used to skip zero-weight
            stages in the linear combination. FSAL for Tsit5 sets
            ``b[1] = 0`` and ``b[6] = 0``.
        e_nz (list[tuple[int, float]]): Sparse nonzero ``e`` entries
            (``e[1] = 0``).

    Notes:
        FSAL ("First Same As Last"): the final stage ``ks[6]`` of an
        accepted step is the RHS evaluated at ``t + h``; the next step
        copies it into ``ks[0]``, saving one ``fun`` evaluation per
        accepted step.

        Error norm: ``err = sqrt(mean(scaled_err**2))`` over all state
        variables, where ``scaled_err_i = error_estimate_i /
        (atol + rtol * max(|y_i|, |y_new_i|))``. The "max across the
        batch dimension" inside ``solve`` couples all batched systems
        to the worst-case error, so a single hard sample drags the
        whole batch's step size down (consistent with running multiple
        simulations under a single shared step controller).
    """
    def __init__(self, atol=1e-6, rtol=1e-6, h_min=1e-8, h_max=10.0, maxiters=1000000):
        """
        Configure solver tolerances and step bounds.

        Args:
            atol (float): Absolute tolerance per state variable.
                Combined with ``rtol`` into a per-component scale
                ``sc_i = atol + rtol * max(|y_i|, |y_new_i|)`` against
                which the embedded error estimate is normalized.
            rtol (float): Relative tolerance. Per
                ``src/implementation_notes.md`` item 9, the simulator
                drives ``f(t, y)`` through a pH bisection sub-solver
                whose noise floor sits below ``1e-3`` — calling code
                that uses pH should keep ``rtol`` above that floor
                (e.g. ``1e-3``) to avoid rejection cascades where the
                stepper interprets pH jitter as integration error.
            h_min (float): Lower bound on step size. Rejected steps
                cannot shrink below this; if a step at ``h_min`` still
                fails the error test it is taken anyway (the loop
                simply continues).
            h_max (float): Upper bound on step size. Per
                ``implementation_notes.md`` item 10 the non-spatial
                simulator caps this at ~1.0 hour because adaptive
                growth into stiff regions (fast pH dynamics from acid
                production) wastes evaluations rejecting back down.
                Spatial mode computes its own ``h_max`` from the CFL
                condition.
            maxiters (int): Hard cap on accepted-plus-rejected
                iterations. The solver raises ``ValueError`` if it
                hits this without covering all of ``t_eval``.

        Returns:
            None.

        Notes:
            Sets ``self.p = 4`` (order of the embedded error estimator)
            and ``self.safety_factor = 0.9``. PI controller exponents
            ``self.alpha = 0.7/(p+1) = 0.14`` and ``self.beta =
            0.4/(p+1) = 0.08`` follow Hairer/Wanner II.4. In a steady-
            error regime the effective exponent on err is
            ``(alpha - beta)/(p+1) = 0.3/(p+1)`` — *smaller* than the
            pure-I exponent ``1/(p+1)``, so the step responds less
            aggressively to any single err sample, which is what
            damps step-size oscillations. ``self.err_prev_floor = 1e-4`` floors
            ``err_prev`` so a single ultra-accurate step does not let
            the next step's ``err_prev^{beta}`` factor collapse the
            growth ratio. The Tsit5 Butcher tableau (``A``, ``c``,
            ``b``, ``e``) and its sparse representations (``b_nz``,
            ``e_nz``) are stored as Python floats so the hot loop can
            use them as ``alpha=`` scalars in fused ``add_`` / ``mul_``
            ops without repeated ``.item()`` round-trips.
        """
        self.atol = atol
        self.rtol = rtol
        self.h_min = h_min
        self.h_max = h_max
        self.maxiters = maxiters
        self.p = 4
        self.safety_factor = 0.9
        # PI step controller (Hairer/Wanner II.4 / Gustafsson 1991):
        #   h_{n+1} = h_n * safety * err_n^{-alpha} * err_{n-1}^{beta}
        # with alpha = 0.7/(p+1) and beta = 0.4/(p+1). In steady-error
        # regimes (err_n ~ err_{n-1}) the effective exponent on err is
        # (alpha - beta)/(p+1) = 0.3/(p+1) — smaller than the pure-I
        # exponent 1/(p+1), so the step size reacts less aggressively
        # to a single err sample. That damping is what suppresses the
        # accept/reject ringing the I controller exhibits in stiff
        # regimes.
        self.alpha = 0.7 / (self.p + 1)  # = 0.14
        self.beta = 0.4 / (self.p + 1)   # = 0.08
        # Floor on err_prev so a single ultra-accurate step does not
        # over-amplify the next step via err_prev^beta -> 0.
        self.err_prev_floor = 1e-4

        # Tsitouras 5(4) Butcher Tableau — store as Python floats to avoid
        # repeated .item() calls in the hot loop
        self.A = [
            [],
            [0.2],
            [0.075, 0.225],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84],
        ]
        self.c = [0.0, 0.2, 0.3, 0.8, 8/9, 1.0, 1.0]

        # b and e weights as Python float lists (b[1]=0, b[6]=0, e[1]=0)
        self.b = [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0]
        self.e = [71/57600, 0.0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40]

        # Precompute nonzero indices for b and e to skip zero terms
        self.b_nz = [(i, self.b[i]) for i in range(7) if self.b[i] != 0.0]
        self.e_nz = [(i, self.e[i]) for i in range(7) if self.e[i] != 0.0]

    def solve(self, fun, y0, t_span, t_eval, args=None, h0=0.1,
              progress_callback=None, converge_fn=None):
        """
        Solves a batch of ODEs using dense output with preallocated buffers.

        Output values are clamped to ``>= 0`` (both the post-step state
        and each interpolated eval point). This matches the simulator
        convention that biomass and metabolite concentrations cannot be
        negative; sub-eps negatives produced by RK stages would otherwise
        propagate into ``M**n`` evaluations and produce NaNs (see
        ``src/implementation_notes.md`` item 1).

        Args:
            fun (callable): ``fun(t, y, args) -> dy/dt``. Called with
                ``y`` shaped ``(num_samples, num_vars)``; must return the
                same shape.
            y0 (torch.Tensor): Initial state, ``(num_samples, num_vars)``.
            t_span (tuple): ``(t_start, t_end)``. Only ``t_start`` is
                used for the integration start; ``t_end`` is taken from
                ``t_eval[-1]`` and the ``t_span`` value is currently
                informational.
            t_eval (torch.Tensor): Output time points, monotonically
                increasing within ``[t_start, t_end]``. The first sample
                may equal ``t_start`` (within ``1e-12``) to record the
                initial condition.
            args: Additional arguments forwarded to ``fun``.
            h0 (float): Initial step size. The adaptive controller will
                shrink or grow it as needed.
            progress_callback: Optional ``callable(t_current, t_end)``
                invoked after each accepted step.
            converge_fn: Optional ``callable(t, y, dydt) -> bool``. When
                it returns ``True`` the solver fills remaining eval
                points with the current state (assumed steady) and stops.

        Returns:
            torch.Tensor: ``(num_samples, len(t_eval), num_vars)``

        Raises:
            ValueError: If ``maxiters`` is exhausted before all of
                ``t_eval`` has been covered.

        Notes:
            **Adaptive RK5(4) loop**. Each iteration:

              1. Compute the seven RK stages ``ks[0..6]`` at current
                 ``y`` and step size ``h_current``. Stage ``j`` uses
                 ``ks[k<j]`` weighted by ``A[j][k]``; the stage state
                 ``y_stage = y + h * Σ A[j][k] * ks[k]`` is fed back
                 into ``fun``. Stage 0 is reused via FSAL (see below).
              2. Form the 5th-order solution
                 ``y_new = y + h · Σ b[i] · ks[i]`` (only nonzero
                 ``b_nz`` entries iterated).
              3. Form the embedded error
                 ``error_estimate = h · Σ e[i] · ks[i]`` from the
                 difference between the 5th- and 4th-order weights.
              4. Compute the scaled error norm and accept if
                 ``err <= 1`` else reject.

            **FSAL** (First Same As Last). Tsit5 has ``c[6] = 1`` and
            its 5th-order weights coincide with the next step's first
            stage, so on accept we ``f_current.copy_(ks[6])`` and feed
            it as ``ks[0]`` next iteration — saving one ``fun`` call
            per accepted step. ``f_current`` is its own buffer (never
            aliased to ``ks``) to make the copy explicit and safe.

            **Dense Hermite output**. Output times in ``t_eval`` will
            in general fall *between* accepted RK steps. After each
            accept, every ``t_eval[eval_idx]`` falling in
            ``[t, t_new]`` is interpolated by
            :func:`_hermite_interp_out` using the local cubic Hermite
            basis on ``[0, 1]`` parameterized by
            ``theta = (t_target - t) / h_current``. This uses both
            endpoint values (``y``, ``y_new``) and both endpoint
            derivatives (``ks[0]``, ``ks[6]``) — exactly the data
            already in hand from the FSAL stage.

            **Error norm and step control**. ``scaled_err_i =
            error_estimate_i / (atol + 1e-9 + rtol · max(|y_i|,
            |y_new_i|))``; the global error scalar is
            ``err = sqrt(mean_over_vars(max_over_batch(scaled_err²)))``.
            On accept, the new step is the PI update ``h_new = h *
            safety_factor * err^{-alpha} * err_prev^{beta}`` with
            ``alpha = 0.7/(p+1) = 0.14``, ``beta = 0.4/(p+1) = 0.08``,
            ``p = 4`` (Hairer/Wanner II.4); ``err_prev`` is then set
            to ``max(err, err_prev_floor)``. The first step uses
            ``err_prev = 1`` so its update reduces to I-only. On
            reject, the controller falls back to I-only
            ``(1/err)^{1/(p+1)}`` and ``err_prev`` is *not* updated
            (so a poisoned err does not propagate). On accept ``h_new``
            is capped at ``h_max``; on reject it is floored at
            ``h_min``. The ``beta`` term damps the accept/reject
            ringing the I controller exhibits in stiff regimes (e.g.
            sharp pH-gate transitions).

            **Sub-eps clamp**. Both ``y_new`` and each interpolated
            output are clamped to ``>= 0`` so RK round-off cannot push
            concentrations slightly negative — important because
            downstream code evaluates ``M**n`` and would NaN on
            negatives.

            **Convergence early-exit**. When ``converge_fn`` returns
            ``True`` the remaining ``t_eval`` rows are filled with the
            current ``y`` and the loop breaks (system assumed at
            steady state).
        """
        device, dtype = y0.device, y0.dtype
        num_samples, num_vars = y0.shape
        A, c, b_nz, e_nz = self.A, self.c, self.b_nz, self.e_nz
        atol, rtol = self.atol, self.rtol

        t_eval = torch.as_tensor(t_eval, device=device, dtype=torch.float64)
        t_end = t_eval[-1].item()
        n_eval = len(t_eval)

        # ---- Preallocate ALL buffers ----
        ks = torch.zeros((7, num_samples, num_vars), dtype=dtype, device=device)
        dy = torch.empty((num_samples, num_vars), dtype=dtype, device=device)
        y_stage = torch.empty_like(dy)
        y_new = torch.empty_like(dy)
        error_estimate = torch.empty_like(dy)
        abs_max = torch.empty_like(dy)
        scaled_err = torch.empty_like(dy)
        results_y = torch.zeros((num_samples, n_eval, num_vars), dtype=dtype, device=device)

        inv_sqrt_nv = 1.0 / (num_vars ** 0.5)

        # ---- State ----
        y = y0.clone()
        t = float(t_span[0])
        h = h0

        # Preallocate FSAL buffer (own memory, never aliases ks)
        f_current = torch.empty((num_samples, num_vars), dtype=dtype, device=device)
        f_current.copy_(fun(t, y, args))

        # PI controller: previous accepted error norm. Initialized to 1.0
        # so the first accepted step reduces to I-only (err_prev^beta = 1).
        err_prev = 1.0

        # Store initial point
        eval_idx = 0
        if t_eval[0].item() <= t + 1e-12:
            results_y[:, 0, :] = y
            eval_idx = 1

        iters = 0
        while eval_idx < n_eval and iters < self.maxiters:
            iters += 1
            h_current = min(h, t_end - t)
            if h_current < self.h_min:
                h_current = self.h_min

            # ---- RK stages (FSAL: ks[0] = f_current) ----
            ks[0].copy_(f_current)
            for j in range(1, 7):
                Aj = A[j]
                # dy = sum_k A[j,k] * ks[k] for k < j
                dy.copy_(ks[0]).mul_(Aj[0])
                for k in range(1, j):
                    dy.add_(ks[k], alpha=Aj[k])
                # y_stage = y + h * dy
                torch.add(y, dy, alpha=h_current, out=y_stage)
                ks[j] = fun(t + h_current * c[j], y_stage, args)

            # ---- y_new = y + h * sum(b[i] * ks[i]) ----
            i0, b0 = b_nz[0]
            y_new.copy_(ks[i0]).mul_(b0)
            for i, bi in b_nz[1:]:
                y_new.add_(ks[i], alpha=bi)
            y_new.mul_(h_current).add_(y)

            # ---- error = h * sum(e[i] * ks[i]) ----
            i0, e0 = e_nz[0]
            error_estimate.copy_(ks[i0]).mul_(e0)
            for i, ei in e_nz[1:]:
                error_estimate.add_(ks[i], alpha=ei)
            error_estimate.mul_(h_current)

            # ---- Adaptive step-size control (fused, in-place) ----
            torch.maximum(y.abs(), y_new.abs(), out=abs_max)
            scaled_err.copy_(abs_max).mul_(rtol).add_(atol + 1e-9)
            torch.div(error_estimate, scaled_err, out=scaled_err)
            scaled_err.square_()
            err = scaled_err.sum(dim=1).max().item() ** 0.5 * inv_sqrt_nv

            if err <= 1.0:  # Accept
                t_new = t + h_current

                # Interpolate at t_eval points within [t, t_new]
                # Use ks[0] for f_start (safe copy), ks[6] for f_end
                while eval_idx < n_eval and t_eval[eval_idx].item() <= t_new + 1e-12:
                    t_target = t_eval[eval_idx].item()
                    theta = (t_target - t) / h_current if h_current > 1e-15 else 1.0
                    theta = max(0.0, min(1.0, theta))
                    _hermite_interp_out(theta, y, y_new, ks[0], ks[6],
                                        h_current, results_y[:, eval_idx, :])
                    results_y[:, eval_idx, :].clamp_(min=0.0)
                    eval_idx += 1

                t = t_new
                y.copy_(y_new)
                f_current.copy_(ks[6])  # FSAL: copy into own buffer

                if progress_callback is not None:
                    progress_callback(t, t_end)

                if converge_fn is not None and converge_fn(t, y, f_current):
                    # Steady state: fill remaining eval points
                    if eval_idx < n_eval:
                        results_y[:, eval_idx:, :] = y.unsqueeze(1)
                        eval_idx = n_eval
                    if progress_callback is not None:
                        progress_callback(t_end, t_end)
                    break

                # PI step update: q = err^{-alpha} * err_prev^{beta}.
                q = (1.0 / (err + 1e-9)) ** self.alpha * err_prev ** self.beta
                h = min(self.h_max, h_current * self.safety_factor * q)
                # Update err_prev only on accept; rejects keep last good err.
                err_prev = max(err, self.err_prev_floor)
            else:
                # On reject, fall back to I-only so a bad err does not
                # poison the err_prev history that drives subsequent steps.
                q = (1.0 / err) ** (1.0 / (self.p + 1))
                h = max(self.h_min, h * self.safety_factor * q)

        if eval_idx < n_eval:
            raise ValueError(
                f"Solver reached maxiters ({self.maxiters}) with {n_eval - eval_idx} "
                f"output points remaining at t={t:.6f}."
            )

        return results_y


def _hermite_interp_out(theta, y0, y1, f0, f1, h, out):
    """Cubic Hermite interpolation between two RK endpoints.

    Standard Hermite basis on ``[0, 1]`` with both endpoint values
    (``y0``, ``y1``) and endpoint derivatives (``f0``, ``f1``).
    Derivatives are scaled by ``h`` because the local parameter
    ``theta = (t - t0)/h`` is dimensionless. Writes in-place into
    ``out`` to avoid allocating a fresh tensor each call (the solver
    interpolates many output points per accepted step in the spatial
    case).

    Args:
        theta (float): Local parameter in ``[0, 1]``.
            ``theta=0`` returns ``y0``; ``theta=1`` returns ``y1``.
        y0 (torch.Tensor): State at the left endpoint.
        y1 (torch.Tensor): State at the right endpoint.
        f0 (torch.Tensor): Derivative at the left endpoint
            (typically ``ks[0]`` of the accepted step).
        f1 (torch.Tensor): Derivative at the right endpoint
            (typically ``ks[6]`` via the FSAL property).
        h (float): Step size that bridges ``y0`` to ``y1``.
        out (torch.Tensor): Pre-allocated output buffer, same shape as
            ``y0``. Overwritten in-place.
    """
    h00 = 2 * theta**3 - 3 * theta**2 + 1
    h10 = theta**3 - 2 * theta**2 + theta
    h01 = -2 * theta**3 + 3 * theta**2
    h11 = theta**3 - theta**2
    # out = h00*y0 + h01*y1 + h10*h*f0 + h11*h*f1
    out.copy_(y0).mul_(h00)
    out.add_(y1, alpha=h01)
    out.add_(f0, alpha=h10 * h)
    out.add_(f1, alpha=h11 * h)
