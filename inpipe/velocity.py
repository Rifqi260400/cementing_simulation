"""Module 2 - concentric axial velocity profile (paper Appendix A.1).

Sign convention
---------------
``+z`` is the *flow direction*, i.e. downward inside the casing for a vertical
well.  The inclination angle ``beta`` is measured from vertical, so ``beta = 0``
is a vertical pipe and ``cos(beta) = 1`` there.  With this convention the
frictional pressure gradient (paper Eq. A.7) is

    P = dp/dz - rho * g * cos(beta) = 2 * tau_w / R

and a positive ``tau_w`` corresponds to flow in the ``+z`` direction.  This
convention is used consistently throughout the package; see assumption A-02.

Physics
-------
A force balance on a pipe gives shear stress linear in radius::

    tau(r) = tau_w * r / R
    T(r)   = tau_w * r / R - tau0          (paper Eq. A.4)
    r0     = tau0 * R / tau_w              (plug radius)

Integrating ``-du/dr = ((tau - tau0)/k)**(1/n)`` from ``r`` to ``R`` with
``u(R) = 0`` gives (paper Eqs. A.3, A.5, A.6)::

    u(r) = A * (T(r)/k)**(1/n + 1) + B
    A    = -k / ((1/n + 1) * (tau_w/R))
    B    = (T(R)/k)**(1/n + 1) * k / ((1/n + 1) * (tau_w/R))

Inside the plug (``r <= r0``) we set ``T(r) = 0``, so ``u = B``: B is both the
plug velocity and the centreline velocity ``u(0)``.

Note on the exponent: the PDF text of Eq. A.3/A.6 renders the exponent as
``1/(n+1)``.  That is an OCR corruption.  The correct exponent, recovered by
carrying out the integration above, is ``1/n + 1``; it reproduces the
Poiseuille, power-law and Buckingham-Reiner closed forms exactly, whereas
``1/(n+1)`` does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .fluid import Fluid

__all__ = [
    "VelocityProfile",
    "plug_radius",
    "velocity_profile",
    "flow_rate",
    "flow_rate_quad",
    "solve_tau_w",
    "pressure_gradient",
]

#: Relative offset above tau0 used for the lower bracket of the tau_w root
#: search.  Below the yield stress there is no flow at all.
_TAU_W_BRACKET_EPS = 1.0e-9
#: Growth factor and iteration cap for the geometric expansion of the upper
#: bracket in :func:`solve_tau_w`.
_BRACKET_GROWTH = 2.0
_BRACKET_MAX_ITER = 200


class NoBracketError(RuntimeError):
    """Raised when no ``tau_w`` bracket enclosing the target flow rate exists."""


@dataclass(frozen=True)
class VelocityProfile:
    """An evaluated concentric axial velocity profile.

    Callable: ``profile(r)`` returns the axial velocity [m/s] at radius ``r``.
    """

    fluid: Fluid
    radius: float
    tau_w: float

    @property
    def plug_radius(self) -> float:
        return plug_radius(self.fluid.tau0, self.radius, self.tau_w)

    @property
    def plug_velocity(self) -> float:
        """Velocity of the unyielded plug, equal to the centreline velocity."""
        return _coefficient_B(self.fluid, self.radius, self.tau_w)

    @property
    def u_max(self) -> float:
        return self.plug_velocity

    def __call__(self, r):
        return velocity_profile(r, self.fluid, self.radius, self.tau_w)

    @property
    def flow_rate(self) -> float:
        return flow_rate(self.fluid, self.radius, self.tau_w)

    @property
    def mean_velocity(self) -> float:
        return self.flow_rate / (math.pi * self.radius**2)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def plug_radius(tau0: float, R: float, tau_w: float) -> float:
    """Radius of the unyielded plug, ``r0 = tau0 * R / tau_w`` [m].

    Clipped to ``[0, R]``: if ``tau_w <= tau0`` the whole section is unyielded.
    """
    if tau_w <= 0.0:
        return R
    return min(R, tau0 * R / tau_w)


def _coefficient_A(fluid: Fluid, R: float, tau_w: float) -> float:
    """Paper Eq. A.5."""
    return -fluid.k / ((1.0 / fluid.n + 1.0) * (tau_w / R))


def _coefficient_B(fluid: Fluid, R: float, tau_w: float) -> float:
    """Paper Eq. A.6 - plug (and centreline) velocity."""
    if tau_w <= fluid.tau0:
        return 0.0
    T_R = tau_w - fluid.tau0  # T(R), paper Eq. A.4 evaluated at r = R
    exponent = 1.0 / fluid.n + 1.0
    return (T_R / fluid.k) ** exponent * fluid.k / (exponent * (tau_w / R))


def velocity_profile(r, fluid: Fluid, R: float, tau_w: float):
    """Axial velocity ``u(r)`` [m/s] (paper Eq. A.3).

    Accepts a scalar or an array of radii.  Radii outside ``[0, R]`` are not
    clipped - passing them is a caller error.
    """
    r_arr = np.asarray(r, dtype=float)
    if np.any(r_arr < 0.0) or np.any(r_arr > R * (1.0 + 1e-12)):
        raise ValueError("radius outside [0, R]")

    if tau_w <= fluid.tau0:
        # Wall stress below the yield stress: the whole section is unyielded
        # and there is no flow.
        out = np.zeros_like(r_arr)
        return out if out.ndim else float(out)

    A = _coefficient_A(fluid, R, tau_w)
    B = _coefficient_B(fluid, R, tau_w)
    exponent = 1.0 / fluid.n + 1.0

    # T(r) = tau_w * r / R - tau0, floored at zero inside the plug so that
    # u = B there (paper: "one can simply set T(r) = 0 when T(r) < 0").
    T = np.maximum(tau_w * np.minimum(r_arr, R) / R - fluid.tau0, 0.0)
    u = A * (T / fluid.k) ** exponent + B

    # Enforce the no-slip identity exactly at the wall.  Analytically
    # A*(T(R)/k)**e + B == 0; this only removes round-off.
    u = np.where(np.isclose(r_arr, R, rtol=0.0, atol=1e-15), 0.0, u)
    return u if u.ndim else float(u)


# ---------------------------------------------------------------------------
# Flow rate
# ---------------------------------------------------------------------------


def flow_rate(fluid: Fluid, R: float, tau_w: float) -> float:
    """``Q = int_0^R u(r) 2 pi r dr`` [m^3/s] (paper Eq. A.2), in closed form.

    Integrating Eq. A.2 by parts (``u(R) = 0``) gives ``Q = pi int_0^R r^2
    (-du/dr) dr``.  Substituting ``T = tau_w r / R - tau0`` turns that into an
    elementary integral of ``(T + tau0)^2 (T/k)^(1/n)``, which evaluates to

        Q = pi R^3 * (Tm/k)^(1/n) * a
            * [ a^2 * n/(3n+1) + 2 x a * n/(2n+1) + x^2 * n/(n+1) ]

    with ``Tm = T(R) = tau_w - tau0``, ``x = tau0/tau_w`` and ``a = 1 - x``.
    Writing it in terms of the stress ratio rather than dividing by
    ``tau_w**3`` keeps it well conditioned down to arbitrarily small
    ``tau_w``, which the root-find bracket relies on.  This is exact for Herschel-Bulkley and
    collapses to Hagen-Poiseuille (``n = 1``, ``tau0 = 0``), to the power-law
    result, and to Buckingham-Reiner (``n = 1``).

    [DEVIATION FROM SPEC, A-01] The build spec chose adaptive quadrature split
    at the plug radius.  That is correct, but the ``tau_w`` root find sits
    inside the time loop at every station holding mixed fluids, where the
    quadrature version dominated the field-scale runtime.  It is retained as
    :func:`flow_rate_quad` and the two are asserted equal across rheologies in
    ``tests/test_velocity.py``.
    """
    if tau_w <= fluid.tau0:
        return 0.0

    n, k, tau0 = fluid.n, fluid.k, fluid.tau0
    Tm = tau_w - tau0
    x = tau0 / tau_w  # stress ratio, 0 for a fluid without yield stress
    a = 1.0 - x
    bracket = (
        a * a * n / (3.0 * n + 1.0)
        + 2.0 * x * a * n / (2.0 * n + 1.0)
        + x * x * n / (n + 1.0)
    )
    return math.pi * R**3 * (Tm / k) ** (1.0 / n) * a * bracket


def flow_rate_quad(fluid: Fluid, R: float, tau_w: float) -> float:
    """Reference ``Q(tau_w)`` by adaptive quadrature - the build spec's choice.

    [OUR CHOICE, A-01] The integral is split at the plug radius ``r0``: the
    plug contributes analytically (``B * pi * r0**2``) and the sheared annulus
    ``r in [r0, R]`` is integrated numerically with ``scipy.integrate.quad``.
    Splitting at ``r0`` keeps the integrand smooth on each piece; the
    derivative of ``u`` is discontinuous there.

    Kept as the independent check on the closed form in :func:`flow_rate`.
    """
    if tau_w <= fluid.tau0:
        return 0.0

    r0 = plug_radius(fluid.tau0, R, tau_w)
    B = _coefficient_B(fluid, R, tau_w)

    # Analytic plug contribution: uniform velocity B over a disc of radius r0.
    q_plug = B * math.pi * r0**2

    if r0 >= R:
        return q_plug

    integrand = lambda rr: velocity_profile(rr, fluid, R, tau_w) * 2.0 * math.pi * rr
    q_sheared, _ = quad(integrand, r0, R, epsabs=1e-16, epsrel=1e-13, limit=200)
    return q_plug + q_sheared


def _initial_upper_bracket(fluid: Fluid, R: float, q_target: float) -> float:
    """A first guess for the upper end of the ``tau_w`` bracket.

    Uses the power-law (tau0 = 0) closed form, which is exact for a power-law
    fluid and an under-estimate for a yield-stress fluid, then adds tau0.
    """
    n = fluid.n
    mean_u = q_target / (math.pi * R**2)
    # Power-law: Q = pi R^3 (n/(3n+1)) (tau_w/k)^(1/n)  =>  invert for tau_w.
    tau_guess = fluid.k * (mean_u * (3.0 * n + 1.0) / (n * R)) ** n
    return max(tau_guess, 1.0e-12) + fluid.tau0


def solve_tau_w(
    q_target: float,
    fluid: Fluid,
    R: float,
    xtol: float = 1.0e-12,
    rtol: float = 1.0e-12,
) -> float:
    """Invert ``Q(tau_w) = q_target`` for the wall shear stress [Pa].

    Uses Brent's method (``scipy.optimize.brentq``), matching the paper, which
    cites Brent (1971) for exactly this root find.

    The lower bracket is ``tau0 * (1 + 1e-9)`` - below the yield stress there is
    no flow at all.  The upper bracket is expanded geometrically from an
    analytical guess until the residual changes sign.
    """
    if q_target < 0.0:
        raise ValueError("negative target flow rate is not supported")
    if q_target == 0.0:
        return fluid.tau0

    hi = _initial_upper_bracket(fluid, R, q_target)

    # Lower bracket: just above the yield stress, below which there is no flow
    # at all.  For a fluid without yield stress there is no such floor, so the
    # bracket is scaled off the analytical guess and shrunk until the residual
    # is negative.
    if fluid.tau0 > 0.0:
        lo = fluid.tau0 * (1.0 + _TAU_W_BRACKET_EPS)
    else:
        lo = hi * 1.0e-8
        for _ in range(_BRACKET_MAX_ITER):
            if flow_rate(fluid, R, lo) - q_target < 0.0:
                break
            lo *= 1.0e-4
    f_lo = flow_rate(fluid, R, lo) - q_target
    if f_lo > 0.0:  # pragma: no cover - defensive
        raise NoBracketError(
            f"Q at the lower bracket already exceeds the target "
            f"(tau_w={lo:.6g} Pa, residual={f_lo:.6g} m3/s)"
        )

    for _ in range(_BRACKET_MAX_ITER):
        if flow_rate(fluid, R, hi) - q_target > 0.0:
            break
        hi *= _BRACKET_GROWTH
    else:
        raise NoBracketError(
            f"no upper bracket found for Q_target={q_target:.6g} m3/s with "
            f"fluid {fluid.name!r} at R={R:.6g} m (last tau_w={hi:.6g} Pa)"
        )

    return brentq(
        lambda tw: flow_rate(fluid, R, tw) - q_target,
        lo,
        hi,
        xtol=xtol,
        rtol=rtol,
        maxiter=200,
    )


def solve_profile(q_target: float, fluid: Fluid, R: float, **kwargs) -> VelocityProfile:
    """Convenience wrapper: solve for ``tau_w`` and return the profile."""
    return VelocityProfile(fluid=fluid, radius=R, tau_w=solve_tau_w(q_target, fluid, R, **kwargs))


# ---------------------------------------------------------------------------
# Pressure gradient
# ---------------------------------------------------------------------------


def pressure_gradient(tau_w: float, R: float, rho: float, beta: float, g: float) -> float:
    """Axial pressure gradient ``dp/dz`` [Pa/m] (paper Eq. A.7).

    ``dp/dz = rho * g * cos(beta) + 2 * tau_w / R``

    with ``+z`` the flow direction and ``beta`` the inclination from vertical.
    """
    return rho * g * math.cos(beta) + 2.0 * tau_w / R


def frictional_gradient(tau_w: float, R: float) -> float:
    """Frictional part of the pressure gradient, ``P = 2 tau_w / R`` [Pa/m]."""
    return 2.0 * tau_w / R
