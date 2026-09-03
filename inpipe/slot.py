"""Annular flow by the parallel-plate (slot) approximation.

The source paper solves the annulus this way.  Appendix A.1 states it
explicitly: "Very similar equations as Eqs. (4)-(7) are used for calculating
the flow in annulus (assuming parallel plates) in the work by Dai and Liu
(2018), but note that we have ``tau_w = h/2 * P`` for flow between plates with
a gap width ``h``, yet we have ``tau_w = R/2 * P`` for flow in a circular pipe."

So the annulus of outer radius ``r_o`` and inner radius ``r_i`` is unwrapped
into a slot of

    half-gap   b = (r_o - r_i) / 2
    width      W = pi * (r_o + r_i)        (the mean circumference)

The velocity profile itself is *identical* in form to the pipe profile of
:mod:`inpipe.velocity` with the radius ``R`` replaced by the half-gap ``b`` and
the coordinate measured from the centre of the gap, because both follow from
integrating ``-du/ds = ((tau_w s / S - tau0)/k)**(1/n)``.  Only the flow-rate
integral differs: ``dA = 2 pi r dr`` in the pipe, ``dA = W ds`` in the slot.

Validity
--------
The slot approximation is good when the gap is small compared with the radii.
For a 139.7 mm casing in a 215.9 mm hole the curvature ratio ``r_i/r_o`` is
0.65, where the approximation over-estimates the flow rate by a few per cent
against the exact concentric-annulus solution.  In a washed-out interval it is
worse.  :func:`slot_error_estimate` reports the deviation so it is never
silent, and the error is recorded in the assumption register (A-24).
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from .fluid import Fluid
from .rheology import stress_moment
from .velocity import NoBracketError, velocity_profile

__all__ = [
    "SlotProfile",
    "slot_geometry",
    "slot_flow_rate",
    "solve_slot_tau_w",
    "solve_slot_profile",
    "slot_error_estimate",
]

_TAU_W_BRACKET_EPS = 1.0e-9
_BRACKET_GROWTH = 2.0
_BRACKET_MAX_ITER = 200


def slot_geometry(r_inner: float, r_outer: float) -> tuple[float, float]:
    """Return ``(half_gap, width)`` of the equivalent slot [m, m]."""
    if r_outer <= r_inner:
        raise ValueError(
            f"outer radius {r_outer:.6g} m must exceed inner radius {r_inner:.6g} m"
        )
    return 0.5 * (r_outer - r_inner), math.pi * (r_outer + r_inner)


def slot_flow_rate(fluid: Fluid, b: float, width: float, tau_w: float,
                   gammadot_c: float | None = None) -> float:
    """``Q`` through a slot of half-gap ``b`` and width ``width`` [m^3/s].

    Integrating ``Q = W * int_{-b}^{b} u ds`` by parts with ``u(+-b) = 0`` and
    substituting ``T = tau_w s / b - tau0`` gives the closed form

        Q = 2 W b^2 (Tm/k)^(1/n) * [ a^2 n/(2n+1) + x a n/(n+1) ]

    with ``Tm = tau_w - tau0``, ``x = tau0/tau_w`` and ``a = 1 - x``.  For a
    Newtonian fluid this reduces to plane Poiseuille, ``Q = 2 W b^2 tau_w /
    (3 mu)``, and the peak-to-mean ratio to 3/2.
    """
    if gammadot_c is not None:
        # Q = 2 W (b/tau_w)^2 int_0^{tau_w} tau gammadot(tau) dtau
        if tau_w <= 0.0:
            return 0.0
        return float(2.0 * width * (b / tau_w) ** 2
                     * stress_moment(tau_w, fluid, gammadot_c, 1))

    if tau_w <= fluid.tau0:
        return 0.0
    n, k, tau0 = fluid.n, fluid.k, fluid.tau0
    Tm = tau_w - tau0
    x = tau0 / tau_w
    a = 1.0 - x
    bracket = a * a * n / (2.0 * n + 1.0) + x * a * n / (n + 1.0)
    return 2.0 * width * b * b * (Tm / k) ** (1.0 / n) * bracket


def _initial_upper_bracket(fluid: Fluid, b: float, width: float, q_target: float) -> float:
    """Power-law inversion as a first guess for the upper bracket."""
    n = fluid.n
    mean_u = q_target / (2.0 * b * width)
    # Power-law slot: Q = 2 W b^2 (tau_w/k)^(1/n) n/(2n+1)
    tau_guess = fluid.k * (mean_u * (2.0 * n + 1.0) / n) ** n
    return max(tau_guess, 1.0e-12) + fluid.tau0


def solve_slot_tau_w(
    q_target: float,
    fluid: Fluid,
    b: float,
    width: float,
    xtol: float = 1.0e-12,
    rtol: float = 1.0e-12,
    gammadot_c: float | None = None,
) -> float:
    """Invert ``Q(tau_w) = q_target`` for the slot wall shear stress [Pa]."""
    if q_target < 0.0:
        raise ValueError("negative target flow rate is not supported")
    if q_target == 0.0:
        return fluid.tau0 if gammadot_c is None else 0.0

    hi = _initial_upper_bracket(fluid, b, width, q_target)
    if fluid.tau0 > 0.0 and gammadot_c is None:
        lo = fluid.tau0 * (1.0 + _TAU_W_BRACKET_EPS)
    else:
        lo = hi * 1.0e-8
        for _ in range(_BRACKET_MAX_ITER):
            if slot_flow_rate(fluid, b, width, lo, gammadot_c) - q_target < 0.0:
                break
            lo *= 1.0e-4

    for _ in range(_BRACKET_MAX_ITER):
        if slot_flow_rate(fluid, b, width, hi, gammadot_c) - q_target > 0.0:
            break
        hi *= _BRACKET_GROWTH
    else:
        raise NoBracketError(
            f"no upper bracket for Q_target={q_target:.6g} m3/s in a slot with "
            f"b={b:.6g} m, W={width:.6g} m and fluid {fluid.name!r}"
        )

    return brentq(
        lambda tw: slot_flow_rate(fluid, b, width, tw, gammadot_c) - q_target,
        lo, hi, xtol=xtol, rtol=rtol, maxiter=200,
    )


class SlotProfile:
    """An evaluated slot velocity profile.

    Callable: ``profile(s)`` returns the axial velocity at distance ``s`` from
    the *centre* of the gap, ``s`` in ``[0, b]``.  The profile is symmetric, so
    only the half-gap is parameterised.
    """

    def __init__(self, fluid: Fluid, half_gap: float, width: float, tau_w: float,
                 gammadot_c: float | None = None):
        self.fluid = fluid
        self.half_gap = half_gap
        self.width = width
        self.tau_w = tau_w
        #: Fluent-style regularisation shear rate; ``None`` is the exact law.
        self.gammadot_c = gammadot_c

    def __call__(self, s):
        # Identical functional form to the pipe profile with R -> b.
        return velocity_profile(s, self.fluid, self.half_gap, self.tau_w,
                                gammadot_c=self.gammadot_c)

    @property
    def plug_half_width(self) -> float:
        """Half-width of the rigid plug [m]; zero under regularisation."""
        if self.gammadot_c is not None:
            return 0.0
        if self.tau_w <= 0.0:
            return self.half_gap
        return min(self.half_gap, self.fluid.tau0 * self.half_gap / self.tau_w)

    @property
    def u_max(self) -> float:
        return float(self(0.0))

    @property
    def flow_rate(self) -> float:
        return slot_flow_rate(self.fluid, self.half_gap, self.width, self.tau_w,
                              gammadot_c=self.gammadot_c)

    @property
    def area(self) -> float:
        return 2.0 * self.half_gap * self.width

    @property
    def mean_velocity(self) -> float:
        return self.flow_rate / self.area

    def frictional_gradient(self) -> float:
        """``P = tau_w / b`` [Pa/m] - the paper's ``tau_w = (h/2) P``."""
        return self.tau_w / self.half_gap

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SlotProfile(fluid={self.fluid.name!r}, b={self.half_gap:.4g} m, "
            f"W={self.width:.4g} m, tau_w={self.tau_w:.4g} Pa)"
        )


def solve_slot_profile(
    q_target: float, fluid: Fluid, r_inner: float, r_outer: float,
    gammadot_c: float | None = None, **kwargs
) -> SlotProfile:
    """Solve the slot profile for an annulus of the given radii."""
    b, width = slot_geometry(r_inner, r_outer)
    tau_w = solve_slot_tau_w(q_target, fluid, b, width, gammadot_c=gammadot_c, **kwargs)
    return SlotProfile(fluid, b, width, tau_w, gammadot_c=gammadot_c)


def slot_error_estimate(r_inner: float, r_outer: float) -> float:
    """Relative error of the slot approximation for a *Newtonian* annulus.

    Compares the slot flow rate against the exact concentric-annulus solution

        Q = (pi / (8 mu)) * (-dp/dz) * [ r_o^4 - r_i^4
                                         - (r_o^2 - r_i^2)^2 / ln(r_o/r_i) ]

    at the same pressure gradient.  Returns ``(Q_slot - Q_exact) / Q_exact``,
    so a positive number means the slot over-estimates the flow.  Newtonian
    only - it is a geometry diagnostic, not a correction factor.
    """
    if r_inner <= 0.0:
        raise ValueError("inner radius must be positive")
    b, width = slot_geometry(r_inner, r_outer)
    # Slot at unit pressure gradient: tau_w = b * P, Q = 2 W b^2 tau_w / (3 mu)
    #                                             = 2 W b^3 P / (3 mu)
    q_slot = 2.0 * width * b**3 / 3.0
    q_exact = (math.pi / 8.0) * (
        r_outer**4 - r_inner**4
        - (r_outer**2 - r_inner**2) ** 2 / math.log(r_outer / r_inner)
    )
    return (q_slot - q_exact) / q_exact
