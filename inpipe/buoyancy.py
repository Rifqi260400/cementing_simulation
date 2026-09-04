"""Buoyancy - what is modelled, what is not, and a correction to an earlier claim.

Gravity reaches this solver three ways.  Two are in it; one is not.

**1. Axial, at well scale - IN the model.**  Density sets the hydrostatic head
of each column, so a casing full of cement pulls harder than an annulus full of
mud.  That is the U-tube imbalance and the free-fall margin
(:mod:`inpipe.hydraulics`), and on K-GEP-1 it reaches **21.8 bar**.  It is
reported but not fed back into the flow, by choice (assumption A-45).

**2. Transverse, across the section - IN the model, and correctly inert here.**
Dai et al. (2024) drive both their gravity mechanisms - density segregation and
instability-triggered mixing - from an inertial velocity
``v_t = sqrt(At g sin(beta) D)``, their Eq. 4, with ``beta`` measured **from
vertical**.  It is implemented in :mod:`inpipe.segregation`.  In a vertical
well ``sin(beta) = 0``, so ``v_t = 0`` and both mechanisms switch off - not as
an approximation but because gravity has no component across a horizontal
cross-section to stratify it.

    **Correction.**  An earlier version of this module reported a buoyancy
    velocity ``V_b = sqrt(g' h)`` of 0.709 m/s against an imposed 0.277, and
    concluded that buoyancy outweighs the flow by 2.5x.  That used the **full**
    ``g`` across the gap - a horizontal-well formula - on a vertical well.
    Dai et al.'s ``sin(beta)`` is exactly the factor that was missing.  The
    transverse buoyancy on this well is **zero**, and the advice that followed
    from that number was wrong.

**3. Axial, within the section - NOT in the model, and not in either paper.**
What survives in a vertical well is subtler.  Gravity is axial; the composition
gradient across the gap is transverse; the two are perpendicular, so buoyancy
cannot segregate anything.  But the front is *not* flat - the fast core runs
ahead, so inside the mixing zone the core is cement-rich and the wall region
mud-rich.  The dense core then carries a larger body force per unit volume than
the light wall region and is retarded relative to it, which **flattens the
front**.

That is not small:

==========================================================  ==========
differential body force, pure cement against pure mud        6590 Pa/m
frictional pressure gradient driving the flow                 748 Pa/m
ratio                                                           8.8 x
==========================================================  ==========

Inside the mixing zone this is the dominant axial force difference across the
section, and this model has no mechanism for it: one velocity profile is solved
per station from the *mixture* rheology, so a cement-rich core and a mud-rich
wall move at whatever the single profile says.  Neither Dai et al. nor Xue et
al. model it for a vertical well - Dai's criteria degenerate at ``beta = 0``,
and Xue et al.'s case is horizontal, where transverse segregation dominates and
this effect is a side issue.  **ANSYS will capture it**, because it solves the
momentum equation with ``rho g`` in it and does not care which direction the
density gradient points.

So the expected direction of disagreement, for a vertical density-stable job:

    ANSYS should show a SHORTER interface and a HIGHER efficiency than this
    model, because it has a mechanism to flatten the front that this model
    lacks.  Arrival time should be much closer, being set by volume balance.

Why still no drift-flux closure
-------------------------------
The closure this would need is a relative axial velocity between phases at the
same station, and its coefficient is set by inertia rather than viscosity here
(the viscous estimate, 20.9 m/s, is 34x the inertial one, 0.61 m/s, so
viscosity is not what limits it).  The inertial coefficient is of order one and
depends on the shape of the mixing zone - nothing in this study pins it down.
Letting the matched CFD case measure it, then calibrating against that, is the
sound order; it makes the validation runs improve this model instead of only
grading it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["BuoyancyScales", "buoyancy_scales"]


@dataclass(frozen=True)
class BuoyancyScales:
    """How buoyancy compares with the imposed flow in one leg."""

    leg: str
    velocity: float           # imposed mean velocity [m/s]
    buoyancy_velocity: float  # sqrt(g' sin(beta) h) [m/s]
    reduced_gravity: float    # g' [m/s^2]
    atwood: float             # (rho_h - rho_l) / (rho_h + rho_l)
    gap: float                # transverse length scale [m]
    stable: bool              # heavy fluid below light one
    inclination: float = 0.0  # from vertical [rad]

    @property
    def froude(self) -> float:
        """``u / V_b``.  Below one, buoyancy outweighs the imposed flow."""
        if self.buoyancy_velocity <= 0.0:
            return math.inf
        return self.velocity / self.buoyancy_velocity

    @property
    def crossing_time(self) -> float:
        """Time for buoyancy to move a blob across the gap [s]."""
        if self.buoyancy_velocity <= 0.0:
            return math.inf
        return self.gap / self.buoyancy_velocity

    def summary(self) -> str:
        if self.buoyancy_velocity <= 0.0:
            return (f"{self.leg:8s}: no transverse buoyancy - gravity has no "
                    "component across the section at this inclination")
        kind = "stable (heavy below)" if self.stable else "unstable (heavy above)"
        verdict = ("buoyancy outweighs the imposed flow"
                   if self.froude < 1.0 else "the imposed flow dominates")
        return "\n".join([
            f"{self.leg:8s}: u {self.velocity:.3f} m/s, V_b {self.buoyancy_velocity:.3f} m/s, "
            f"Fr {self.froude:.2f} - {verdict}, {kind}",
        ])


def buoyancy_scales(leg, rho_displacing, rho_displaced, velocity, gap,
                    stable, gravity=9.80665, inclination=0.0):
    """Transverse buoyancy scales for one leg.

    ``inclination`` is measured **from vertical**, and it is not optional
    physics: the buoyancy velocity is ``sqrt(g' sin(beta) h)``, so a vertical
    well gives zero.  An earlier version of this function left ``sin(beta)``
    out and reported a large buoyancy velocity for a vertical well, which is a
    horizontal-well answer to a vertical-well question.

    ``stable`` says whether the denser fluid sits below the lighter one, which
    decides the sign of the effect, not its size.
    """
    mean = 0.5 * (rho_displacing + rho_displaced)
    delta = abs(rho_displacing - rho_displaced)
    g_prime = gravity * delta / mean if mean > 0.0 else 0.0
    transverse = g_prime * math.sin(inclination)
    return BuoyancyScales(
        leg=leg, velocity=float(velocity),
        buoyancy_velocity=math.sqrt(max(transverse * gap, 0.0)),
        reduced_gravity=g_prime,
        atwood=delta / (rho_displacing + rho_displaced) if mean > 0.0 else 0.0,
        gap=float(gap), stable=bool(stable), inclination=float(inclination),
    )
