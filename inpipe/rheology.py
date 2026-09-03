"""Herschel-Bulkley with an optional Fluent-style regularisation.

Two treatments of the yield stress are available, and they differ in kind, not
in degree.

``gammadot_c = None`` - the exact law
    ``tau = tau0 + k gammadot^n`` above the yield stress, and *no motion at all*
    below it.  The unyielded core moves as a rigid plug.  This is the law the
    in-pipe paper (Dai et al. 2023) writes and the default everywhere here.

``gammadot_c`` set - the regularised law
    What ANSYS-Fluent actually integrates, and what Tao et al. (2025) used::

        eta = tau0/g + k (g/gc)^(n-1)                    for g >= gc
        eta = tau0 (2 - g/gc)/gc + k[(2-n) + (n-1) g/gc]  for g <  gc

    Below ``gc`` the viscosity is capped rather than infinite, so **there is no
    plug**: every point shears, however small the stress.  The two branches meet
    continuously at ``gc``.

Two things about the published form are worth stating, because both change
numbers rather than presentation.

**The inequalities in Tao et al. Eqs. (15)-(16) are printed the wrong way
round.** As printed, the ``tau0/g`` branch is assigned to ``g < gc``, where it
diverges as ``g -> 0`` and so cannot be a regularisation; and the bi-viscosity
branch is assigned to ``g > gc``, where it turns *negative* above ``2 gc``.
Swapping them, as done here, is the only reading that yields a usable model.

**Whether ``k`` is normalised by ``gc`` changes the fluid.** Eq. (15) prints the
shear-thinning term as ``k(g/gc)^(n-1)``; Fluent's documented model uses
``k g^(n-1)``.  *Both* pair continuously with the bi-viscosity branch - the
choice is not settled by continuity, which is what it first appears.  What
settles it is that only the literal form reproduces the paper's own Eq. (14),
``tau = tau_y + k gammadot^n``, and its Table 1 value ``k = 0.6``:

    literal    ``k g^(n-1)``       ->  tau = 1.4 + 0.600 g^0.4   (Eq. 14, Table 1)
    normalised ``k(g/gc)^(n-1)``   ->  tau = 1.4 + 1.669 g^0.4

So Eqs. (15)-(16) as printed contradict Eq. (14) and Table 1 by a factor of 2.8
in the consistency index.  The default here is the **literal** form, which keeps
Eq. (14) intact and matches what Fluent integrates; pass
``normalise_consistency=True`` to follow Eqs. (15)-(16) literally instead and
see how much it matters.

Only the literal form reduces to the exact law as ``gc -> 0``.  Under the
normalised form ``k gc^(1-n) -> 0`` for ``n < 1``, so shrinking ``gc`` thins the
fluid away rather than sharpening the plug - the regularisation is not a
vanishing perturbation there.
"""

from __future__ import annotations

import numpy as np

from .fluid import Fluid

__all__ = [
    "critical_stress",
    "plateau_viscosity",
    "effective_consistency",
    "apparent_viscosity",
    "stress",
    "shear_rate",
    "stress_moment",
    "velocity_integral",
]

#: Gauss-Legendre nodes per smooth piece when integrating stress moments.
#: Convergence is algebraic rather than spectral, because the yielded branch
#: carries a ``(tau - tau0)^(1/n)`` branch point just outside its interval; 24
#: nodes hold the moments to 2e-7 relative across the fluids and stress range
#: in ``tests/test_rheology.py``, and 48 buys 1e-9 for twice the time.
#: [OUR CHOICE, A-43]
_N_GAUSS = 24
#: Gauss-Legendre nodes and weights, computed once.  ``leggauss`` runs an
#: eigenvalue decomposition, which dominated the regularised path when it was
#: called afresh on every stress moment.
_GAUSS = np.polynomial.legendre.leggauss(_N_GAUSS)


def effective_consistency(fluid: Fluid, gammadot_c: float,
                          normalise_consistency: bool = False) -> float:
    """Consistency index the high-shear branch actually uses [Pa.s^n]."""
    if not normalise_consistency:
        return fluid.k
    return fluid.k * gammadot_c ** (1.0 - fluid.n)


def critical_stress(fluid: Fluid, gammadot_c: float,
                    normalise_consistency: bool = False) -> float:
    """Shear stress at ``gammadot_c``, where the two branches meet [Pa].

    Below this stress the whole flow is in the regularised branch.
    """
    k_eff = effective_consistency(fluid, gammadot_c, normalise_consistency)
    return fluid.tau0 + k_eff * gammadot_c**fluid.n


def _quadratic_coefficients(fluid: Fluid, gammadot_c: float,
                            normalise_consistency: bool = False):
    """``(A, B)`` for ``tau = A x^2 + B x`` on the regularised branch.

    ``x = gammadot / gammadot_c`` in ``[0, 1]``.  Substituting the bi-viscosity
    viscosity into ``tau = eta gammadot`` gives a quadratic in ``x``, so the
    inversion is closed-form rather than iterative.
    """
    k_eff = effective_consistency(fluid, gammadot_c, normalise_consistency)
    scale = k_eff * gammadot_c**fluid.n  # the k term's value at gammadot_c
    a = -fluid.tau0 + scale * (fluid.n - 1.0)
    b = 2.0 * fluid.tau0 + scale * (2.0 - fluid.n)
    return a, b


def plateau_viscosity(fluid: Fluid, gammadot_c: float,
                      normalise_consistency: bool = False) -> float:
    """Zero-shear viscosity the regularisation caps at [Pa.s]."""
    k_eff = effective_consistency(fluid, gammadot_c, normalise_consistency)
    return (2.0 * fluid.tau0 / gammadot_c
            + k_eff * (2.0 - fluid.n) * gammadot_c ** (fluid.n - 1.0))


def apparent_viscosity(gammadot, fluid: Fluid, gammadot_c: float | None = None,
                       normalise_consistency: bool = False):
    """``eta(gammadot)`` [Pa.s].  Unregularised when ``gammadot_c`` is None."""
    g = np.asarray(gammadot, dtype=float)
    if gammadot_c is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            return fluid.tau0 / g + fluid.k * g ** (fluid.n - 1.0)
    k_eff = effective_consistency(fluid, gammadot_c, normalise_consistency)
    x = g / gammadot_c
    with np.errstate(divide="ignore", invalid="ignore"):
        yielded = fluid.tau0 / g + k_eff * g ** (fluid.n - 1.0)
    regularised = (fluid.tau0 * (2.0 - x) / gammadot_c
                   + k_eff * ((2.0 - fluid.n) + (fluid.n - 1.0) * x)
                   * gammadot_c ** (fluid.n - 1.0))
    return np.where(g >= gammadot_c, yielded, regularised)


def stress(gammadot, fluid: Fluid, gammadot_c: float | None = None,
           normalise_consistency: bool = False):
    """``tau(gammadot)`` [Pa]."""
    g = np.asarray(gammadot, dtype=float)
    if gammadot_c is None:
        return np.where(g > 0.0, fluid.tau0 + fluid.k * g**fluid.n, 0.0)
    a, b = _quadratic_coefficients(fluid, gammadot_c, normalise_consistency)
    k_eff = effective_consistency(fluid, gammadot_c, normalise_consistency)
    x = g / gammadot_c
    return np.where(g >= gammadot_c,
                    fluid.tau0 + k_eff * np.maximum(g, 0.0) ** fluid.n,
                    a * x * x + b * x)


def shear_rate(tau, fluid: Fluid, gammadot_c: float | None = None,
               normalise_consistency: bool = False):
    """Invert ``tau -> gammadot`` [1/s].

    Unregularised: zero below the yield stress - the rigid plug.
    Regularised: the quadratic branch below ``gammadot_c``, so nothing is ever
    rigid.
    """
    t = np.asarray(tau, dtype=float)
    if gammadot_c is None:
        excess = np.maximum(t - fluid.tau0, 0.0)
        return (excess / fluid.k) ** (1.0 / fluid.n)

    k_eff = effective_consistency(fluid, gammadot_c, normalise_consistency)
    tau_c = critical_stress(fluid, gammadot_c, normalise_consistency)
    a, b = _quadratic_coefficients(fluid, gammadot_c, normalise_consistency)

    # Each branch is evaluated only where it applies.  Evaluating both
    # everywhere and selecting afterwards costs a fractional power and a square
    # root on every point regardless, and this runs on every quadrature node of
    # every root-find iteration of every station.
    flat = np.atleast_1d(t)
    out = np.zeros(flat.shape, dtype=float)

    yielded = flat >= tau_c
    if yielded.any():
        # High branch: tau = tau0 + k_eff g^n.
        out[yielded] = ((flat[yielded] - fluid.tau0) / k_eff) ** (1.0 / fluid.n)

    low = (flat > 0.0) & ~yielded
    if low.any():
        # Low branch: solve A x^2 + B x = tau on [0, 1].
        tc = flat[low]
        if abs(a) < 1e-300:
            x = tc / b
        else:
            disc = np.maximum(b * b + 4.0 * a * tc, 0.0)
            x = (-b + np.sqrt(disc)) / (2.0 * a)
        out[low] = gammadot_c * np.clip(x, 0.0, 1.0)

    return out.reshape(t.shape)


def velocity_integral(tau_upper, tau_lower, fluid: Fluid, gammadot_c: float,
                     normalise_consistency: bool = False):
    """``int_{tau_lower}^{tau_upper} gammadot(tau) dtau`` - the velocity profile.

    Named separately from :func:`stress_moment` because it is what the velocity
    profile actually is; it is the zeroth moment.
    """
    return stress_moment(tau_upper, fluid, gammadot_c, 0, tau_lower,
                         normalise_consistency)


def stress_moment(tau_upper, fluid: Fluid, gammadot_c: float, power: int,
                  tau_lower=0.0, normalise_consistency: bool = False):
    """``int_{tau_lower}^{tau_upper} tau^power gammadot(tau) dtau``.

    The integrals that give velocity (``power = 0``), slot flow rate
    (``power = 1``) and pipe flow rate (``power = 2``) once the substitution
    ``tau = tau_w s / S`` is made.  Split at the critical stress so each piece
    is smooth, then Gauss-Legendre on each - the regularised branch has no
    closed form worth the algebra.
    """
    lo = np.asarray(tau_lower, dtype=float)
    hi = np.asarray(tau_upper, dtype=float)
    lo, hi = np.broadcast_arrays(lo, hi)
    tau_c = critical_stress(fluid, gammadot_c, normalise_consistency)
    nodes, weights = _GAUSS

    # Where the interval actually crosses the kink.  Clamping into [lo, hi]
    # keeps both pieces non-negative when the interval lies wholly on one side.
    split = np.clip(np.full(np.shape(hi), tau_c), lo, hi)

    total = np.zeros(np.shape(hi), dtype=float)
    for a_edge, b_edge in (
        (lo, split),      # regularised piece
        (split, hi),      # yielded piece
    ):
        width = b_edge - a_edge
        mid = 0.5 * (a_edge + b_edge)
        t = mid[..., None] + 0.5 * width[..., None] * nodes
        g = shear_rate(t, fluid, gammadot_c, normalise_consistency)
        # ``power`` is only ever 0, 1 or 2; spelling those out avoids a general
        # power over every quadrature node, which is not free at this call rate.
        if power == 0:
            weighted = g
        elif power == 1:
            weighted = g * t
        elif power == 2:
            weighted = g * t * t
        else:
            weighted = g * t**power
        total = total + 0.5 * width * (weighted @ weights)
    return total
