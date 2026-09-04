"""Buoyancy - the largest physics this model leaves out, measured.

The solver imposes the pump rate and advects volume fractions.  Density enters
the hydrostatic head and so the reported pressures, but it never drives the
flow: there is no momentum equation for a density difference to act in
(assumption A-29).  ANSYS does not share that omission - buoyancy is in the
``rho g`` term of the momentum equation the moment gravity is switched on, and
both papers behind this study use it.  Xue et al. (2022) call the "buoying
effect" one of the two strongest influences on the displacement interface.

So the comparison against CFD will be biased, and this module says by how much
and in which direction rather than leaving it to be discovered.

The numbers on K-GEP-1
----------------------
The buoyancy velocity scale is ``V_b = sqrt(g' h)`` with reduced gravity
``g' = g (rho_c - rho_m) / rho_mean`` and ``h`` the gap:

==========  ==========  ==========  =======  =====================
leg         u [m/s]     V_b [m/s]   Fr       stratification
==========  ==========  ==========  =======  =====================
annulus         0.437       0.612    0.71    stable (heavy below)
casing          0.684       0.821    0.83    unstable (heavy above)
==========  ==========  ==========  =======  =====================

``Fr < 1`` in both: **buoyancy is stronger than the imposed flow**, not a
correction to it.  And it has time to act - a blob crosses the annular gap
under buoyancy in 0.14 s, some 5000 times over during the 12 min the front
spends in the annulus.

What that implies for the comparison
------------------------------------
The two legs are in opposite regimes, and both push the same way:

* **Annulus, stable.**  Cement is denser and it is *below* the mud it is
  pushing up.  A finger of cement running ahead into the mud is heavier than
  its surroundings, so buoyancy pulls it back: fingers are suppressed and the
  interface is flattened.  This model has no such mechanism, so it should
  **over-predict the interface length and under-predict the efficiency**.
* **Casing, unstable.**  Cement is denser and *above* the mud, both moving
  down.  Buoyancy drives cement ahead of the bulk, so it reaches the shoe
  earlier.  This is the U-tube imbalance the model already reports, in local
  form, and it pushes arrival the same way free-fall does.

So the expected direction of disagreement is not ambiguous:

    ANSYS should show a SHORTER interface, a HIGHER efficiency, and an
    EARLIER arrival than this model.

That is a falsifiable prediction, and it is the useful way to run the
comparison.  If the CFD instead shows a *longer* interface, something other
than buoyancy is wrong and it is worth finding out what.

Why no drift-flux closure here
------------------------------
The standard reduced-order fix is a drift velocity between the phases.  It
would need a closure coefficient that no data in this study constrains, and a
mis-calibrated buoyancy model is harder to detect than an absent one - it moves
the answer in the right direction for the wrong reason.  The better order is to
let the CFD *measure* the effect on the matched case, then calibrate a drift
term against it.  That turns the validation runs into something that improves
this model rather than only grading it.
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
    buoyancy_velocity: float  # sqrt(g' h) [m/s]
    reduced_gravity: float    # g' [m/s^2]
    atwood: float             # (rho_h - rho_l) / (rho_h + rho_l)
    gap: float                # transverse length scale [m]
    stable: bool              # heavy fluid below light one

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
        kind = "stable (heavy below)" if self.stable else "unstable (heavy above)"
        verdict = ("buoyancy outweighs the imposed flow"
                   if self.froude < 1.0 else "the imposed flow dominates")
        expect = ("suppresses fingering, so CFD should show a shorter interface "
                  "and higher efficiency than this model"
                  if self.stable else
                  "drives the front ahead of the bulk, so CFD should show an "
                  "earlier arrival than this model")
        return "\n".join([
            f"{self.leg:8s}: u {self.velocity:.3f} m/s, V_b {self.buoyancy_velocity:.3f} m/s, "
            f"Fr {self.froude:.2f} - {verdict}",
            f"          {kind}; {expect}",
        ])


def buoyancy_scales(leg, rho_displacing, rho_displaced, velocity, gap,
                    stable, gravity=9.80665):
    """Buoyancy scales for one leg.

    ``stable`` says whether the denser fluid sits below the lighter one, which
    is what decides the sign of the effect, not its size.
    """
    mean = 0.5 * (rho_displacing + rho_displaced)
    delta = abs(rho_displacing - rho_displaced)
    g_prime = gravity * delta / mean if mean > 0.0 else 0.0
    return BuoyancyScales(
        leg=leg, velocity=float(velocity),
        buoyancy_velocity=math.sqrt(max(g_prime * gap, 0.0)),
        reduced_gravity=g_prime,
        atwood=delta / (rho_displacing + rho_displaced) if mean > 0.0 else 0.0,
        gap=float(gap), stable=bool(stable),
    )
