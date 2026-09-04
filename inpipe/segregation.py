"""Gravity in the cross-section - the mechanism of Dai et al. (2024).

The paper this model replicates does carry buoyancy, and it does so without a
momentum equation.  Two mechanisms, both driven by an **inertial velocity**
that measures how fast buoyancy can move fluid *across* the flow::

    v_t = sqrt(At . g . sin(beta) . D)              (their Eq. 4)

with ``At`` the Atwood number, ``beta`` the inclination **from vertical** and
``D`` the passage diameter.  From it come a Froude number ``Fr = u / v_t``
(Eq. A.13) and a Reynolds number on the mean viscosity (Eqs. A.11-A.12).

**Segregation** (Eqs. A.17-A.18).  When ``Re_t = rho v_t D / mu > 1`` buoyancy
beats the imposed flow across the section, the fluids stratify, and the paper
re-orders the cross-section by density - heaviest layer at the bottom -
conserving each fluid's volume.

**Instability-driven mixing** (Eqs. A.10, A.14-A.16).  Four criteria, each from
an experimental study, that trigger instantaneous mixing of the whole cross
section.  The paper is explicit that these apply to *density-unstable* flow:
"This type of instabilities is considered absent for density-stable flow."

The ``sin(beta)`` is the whole story for a vertical well
--------------------------------------------------------
Buoyancy can only segregate a cross-section if gravity has a component *in*
that cross-section.  A vertical well has none - gravity points along the axis -
so ``sin(beta) = 0``, ``v_t = 0``, ``Re_t = 0``, and **both mechanisms switch
themselves off**:

===================  =========  =============  ==================
well                 sin(beta)  v_t [m/s]      segregation
===================  =========  =============  ==================
vertical                 0.000         0.000   inactive
30 deg deviated          0.500         0.306   active (Re_t 205)
horizontal               1.000         0.433   active (Re_t 290)
===================  =========  =============  ==================

So on the vertical K-GEP-1 case this module is deliberately inert, and the
model matches the paper exactly.  It is implemented anyway because it is part
of the paper being replicated, it costs nothing when inactive, and it is what
makes a deviated section modellable without any new physics.

What it is *not*
----------------
This is buoyancy **across** the flow.  Buoyancy **along** the flow - the heavy
cement column pulling the well down - is a separate thing, and it is already in
the model as the hydrostatic head, the U-tube imbalance and the free-fall
margin (:mod:`inpipe.hydraulics`).  On this well that one is large: 21.8 bar.
Neither paper models the axial Rayleigh-Taylor instability of a heavy fluid
sitting on a light one in a vertical pipe; Dai et al.'s criteria degenerate at
``beta = 0`` and are not written for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["atwood", "inertial_velocity", "effective_viscosity", "CrossSectionRegime",
           "cross_section_regime", "segregate", "mix_uniformly"]


def atwood(rho_heavy: float, rho_light: float) -> float:
    """``(rho_h - rho_l) / (rho_h + rho_l)`` - their Eq. 5."""
    total = rho_heavy + rho_light
    return 0.0 if total <= 0.0 else abs(rho_heavy - rho_light) / total


def inertial_velocity(at: float, diameter: float, inclination: float,
                      gravity: float = 9.80665) -> float:
    """``v_t = sqrt(At g sin(beta) D)`` - their Eq. 4.

    ``inclination`` is measured **from vertical**, so a vertical well gives
    zero: gravity has no component across the section to segregate it.
    """
    return math.sqrt(max(at * gravity * math.sin(inclination) * diameter, 0.0))


def effective_viscosity(fluid, velocity: float, diameter: float) -> float:
    """Their Eq. A.11, ``mu = tau0 D/u + k (u/D)^(n-1)``.

    A nominal viscosity built from the bulk velocity and the diameter rather
    than a resolved shear rate, which is what the paper uses in its regime
    criteria.  Kept separate from :mod:`inpipe.regime`, which asks a different
    question and uses the solver's own wall shear stress for it.
    """
    if velocity <= 0.0 or diameter <= 0.0:
        return math.inf
    return fluid.tau0 * diameter / velocity + fluid.k * (velocity / diameter) ** (
        fluid.n - 1.0)


@dataclass(frozen=True)
class CrossSectionRegime:
    """What gravity does to one cross-section, by the paper's criteria."""

    inertial_velocity: float
    froude: float
    reynolds: float
    reynolds_inertial: float
    inclination: float
    density_stable: bool

    @property
    def segregates(self) -> bool:
        """Eq. A.18: buoyancy beats inertia across the section."""
        return self.reynolds_inertial > 1.0

    @property
    def diffusive_mixing(self) -> bool:
        """Eq. A.10."""
        fr = self.froude
        if not math.isfinite(fr):
            return False
        return (self.reynolds * math.cos(self.inclination)
                > 500.0 * fr - 50.0 * fr * fr)

    @property
    def inertial_instability(self) -> bool:
        """Eq. A.14, Taghavi et al. (2011)."""
        fr = self.froude
        if not math.isfinite(fr):
            return False
        return self.reynolds * math.cos(self.inclination) > 58.16 * fr * fr

    @property
    def exchange_instability(self) -> bool:
        """Eq. A.15, Seon et al. (2005); the small-Froude branch."""
        fr = self.froude
        if not math.isfinite(fr) or fr <= 0.0:
            return False
        return self.reynolds * math.cos(self.inclination) / fr > 50.0

    @property
    def turbulent(self) -> bool:
        """Eq. A.16."""
        return self.reynolds > 2100.0

    @property
    def mixes(self) -> bool:
        """Whether the paper would mix the whole cross-section.

        The buoyancy-driven criteria are for density-unstable flow only - the
        paper says so in as many words - so a density-stable section mixes only
        if it is turbulent.
        """
        if self.turbulent:
            return True
        if self.density_stable:
            return False
        return (self.diffusive_mixing or self.inertial_instability
                or self.exchange_instability)


def cross_section_regime(displacing, displaced, velocity, diameter, inclination,
                         density_stable, gravity=9.80665) -> CrossSectionRegime:
    """Evaluate the paper's criteria for one section."""
    at = atwood(max(displacing.rho, displaced.rho), min(displacing.rho, displaced.rho))
    v_t = inertial_velocity(at, diameter, inclination, gravity)
    mu1 = effective_viscosity(displaced, velocity, diameter)
    mu2 = effective_viscosity(displacing, velocity, diameter)
    mu_m = math.sqrt(mu1 * mu2) if math.isfinite(mu1 * mu2) else math.inf
    rho_m = 0.5 * (displacing.rho + displaced.rho)
    re = rho_m * velocity * diameter / mu_m if mu_m > 0.0 else 0.0
    return CrossSectionRegime(
        inertial_velocity=v_t,
        froude=velocity / v_t if v_t > 0.0 else math.inf,
        reynolds=re,
        reynolds_inertial=rho_m * v_t * diameter / mu_m if mu_m > 0.0 else 0.0,
        inclination=inclination,
        density_stable=bool(density_stable),
    )


def mix_uniformly(fractions, cell_volume):
    """Replace a cross-section by its own average, conserving every fluid.

    ``fractions`` is ``(n_fluids, n_layer, n_azimuth)``.
    """
    volume = np.asarray(cell_volume, dtype=float)
    total = volume.sum()
    if total <= 0.0:
        return np.array(fractions, dtype=float)
    mean = np.einsum("ilm,lm->i", fractions, volume) / total
    return np.broadcast_to(mean[:, None, None], np.shape(fractions)).copy()


def segregate(fractions, cell_volume, densities):
    """Re-order a cross-section by density, heaviest at the bottom.

    Layers are horizontal slabs indexed from the bottom of the section upward,
    which is the grid's own convention, so the heaviest fluid fills layer 0
    first and the lightest is left at the top.  Each fluid's volume is
    conserved exactly, which is what makes this safe to apply inside a solver
    whose whole correctness rests on that (their Appendix A.3).
    """
    fractions = np.asarray(fractions, dtype=float)
    volume = np.asarray(cell_volume, dtype=float)

    wanted = np.einsum("ilm,lm->i", fractions, volume)      # volume of each fluid
    layer_volume = volume.sum(axis=1)                        # per horizontal slab
    order = np.argsort(-np.asarray(densities, dtype=float))  # heaviest first

    out = np.zeros_like(fractions)
    remaining = wanted.copy()
    for layer in range(volume.shape[0]):                     # bottom slab upward
        total = layer_volume[layer]
        if total <= 0.0:
            continue
        capacity = total
        for i in order:
            if remaining[i] <= 0.0:
                continue
            take = min(remaining[i], capacity)
            # Fraction of the *layer*, not of what is left of it: dividing by
            # the running capacity gives the second fluid in a shared layer a
            # fraction of a fraction, and the fluid volumes stop balancing.
            out[i, layer, :] += take / total                 # uniform across azimuth
            remaining[i] -= take
            capacity -= take
            if capacity <= 0.0:
                break
    return out
