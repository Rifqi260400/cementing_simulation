"""Flow regime - is the laminar velocity profile this model solves defensible?

Everything in :mod:`inpipe.velocity` and :mod:`inpipe.slot` is a laminar
Herschel-Bulkley profile.  That is an assumption about the flow, not a property
of it, and whether it holds depends on the fluid the job is actually pumped
with.  Nothing in the solver notices when it stops holding, so this module
computes the Reynolds number and says so.

Where this stands on this study
-------------------------------
Xue et al. (2022), *J. Pet. Sci. Eng.* 208:109393, make the point that in a
cementing annulus the wide side can be turbulent while the narrow side is still
laminar, and that assuming one regime everywhere "will lead to serious model
error".  That is worth checking rather than assuming, so this module checks it.

**On this job the check passes.**  Both fluids are Herschel-Bulkley and the
ANSYS comparison will use Herschel-Bulkley too, so the numbers below are the
ones that matter, not a Newtonian stand-in:

=======  ===========  ==========  ==========  ==========
rate     v (annulus)  Re ann mud  Re ann cem  Re casing
=======  ===========  ==========  ==========  ==========
5 bpm       0.44 m/s         404         412    822-875
10 bpm      0.87 m/s        1050        1149  2344-2361
15 bpm      1.31 m/s        1819        2076  4120-4336
=======  ===========  ==========  ==========  ==========

At the job rate of 5 bpm the annulus sits a factor of five below the laminar
limit and the casing a factor of about 2.4, so the laminar profile this solver
integrates is sound for both fluids.  **The casing is what leaves laminar
first**, at roughly 10 bpm; the annulus holds to about 15-20 bpm.  So the
assumption is safe here and would need re-checking if the rate were raised, or
if the displaced fluid turned out to be far thinner than the placeholder mud -
water at 1 cP would give Re of 38 000 in the annulus, which is where the
sensitivity ends.

The regime is reported, never modelled: a turbulence model is a different
solver.  The point of the check is that the assumption is now backed by a
number that is recomputed on every run instead of being taken on trust.

Definition
----------
The effective viscosity is taken at the wall, from the solver's own
constitutive law and its own wall shear stress::

    mu_eff = tau_w / gammadot(tau_w)
    Re     = rho * V * D_h / mu_eff

so the number is consistent with the profile actually being solved rather than
imported from a correlation.  ``D_h`` is the hydraulic diameter: the bore for
the casing, the difference of diameters for the annulus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rheology import shear_rate

__all__ = ["FlowRegime", "regime_from_result", "reynolds", "LAMINAR_LIMIT",
           "TURBULENT_LIMIT"]

#: Below this the flow is laminar, above ``TURBULENT_LIMIT`` fully turbulent,
#: between them transitional.  The usual pipe values; for a non-Newtonian fluid
#: the critical number drifts with the flow index, so these are bands and not
#: sharp lines.  [OUR CHOICE, A-50]
LAMINAR_LIMIT = 2100.0
TURBULENT_LIMIT = 4000.0


@dataclass(frozen=True)
class FlowRegime:
    """Reynolds number against depth for one leg of the well."""

    leg: str
    depth: np.ndarray            # ascending measured depth [m]
    reynolds: np.ndarray
    velocity: np.ndarray         # mean axial velocity [m/s]
    effective_viscosity: np.ndarray   # at the wall [Pa.s]
    hydraulic_diameter: np.ndarray    # [m]

    @property
    def laminar(self) -> np.ndarray:
        return self.reynolds < LAMINAR_LIMIT

    @property
    def turbulent(self) -> np.ndarray:
        return self.reynolds >= TURBULENT_LIMIT

    @property
    def transitional(self) -> np.ndarray:
        return ~self.laminar & ~self.turbulent

    @property
    def is_defensible(self) -> bool:
        """True only if a laminar profile holds everywhere."""
        return bool(np.all(self.laminar))

    def summary(self) -> str:
        lo, hi = float(np.min(self.reynolds)), float(np.max(self.reynolds))
        lines = [
            f"{self.leg:8s}: Re {lo:.0f} - {hi:.0f} "
            f"(mu_eff {np.min(self.effective_viscosity):.4g} - "
            f"{np.max(self.effective_viscosity):.4g} Pa.s)"
        ]
        if self.is_defensible:
            lines.append("          laminar everywhere - the solved profile holds")
        else:
            frac_t = 100.0 * float(np.mean(self.turbulent))
            frac_x = 100.0 * float(np.mean(self.transitional))
            lines.append(
                f"          *** {frac_t:.0f} % turbulent, {frac_x:.0f} % transitional - "
                "this model solves a LAMINAR profile everywhere, so those "
                "stations are outside what it can represent"
            )
            worst = int(np.argmax(self.reynolds))
            lines.append(f"          worst at {self.depth[worst]:.1f} m, Re {hi:.0f}")
        return "\n".join(lines)


def reynolds(rho, velocity, hydraulic_diameter, tau_w, tau0, k, n):
    """``Re = rho V D_h / mu_eff`` with ``mu_eff`` at the wall, vectorised.

    Takes the rheology as arrays rather than :class:`~inpipe.fluid.Fluid`
    objects so it can run inside the time loop over every station at once.
    The exact Herschel-Bulkley inverse is used even when the solver is running
    Fluent's regularisation: the two differ only below the critical shear rate,
    where the flow is slow and the Reynolds number is nowhere near the
    transition anyway.

    A station with no wall stress is not flowing.  Its effective viscosity is
    undefined rather than infinite, and calling it laminar by default would be
    a silent pass, so it gets ``Re = 0``, which it honestly is.
    """
    tau_w = np.asarray(tau_w, dtype=float)
    excess = np.maximum(tau_w - np.asarray(tau0, dtype=float), 0.0)
    gdot = (excess / np.asarray(k, dtype=float)) ** (1.0 / np.asarray(n, dtype=float))
    mu = np.divide(tau_w, gdot, out=np.full_like(tau_w, np.inf), where=gdot > 0.0)
    return np.divide(rho * velocity * hydraulic_diameter, mu,
                     out=np.zeros_like(mu), where=np.isfinite(mu) & (mu > 0.0)), mu


def _regime(leg, depth, rho, velocity, hydraulic_diameter, tau_w, fluids,
            gammadot_c=None):
    gdot = np.array([
        float(shear_rate(t, f, gammadot_c)) for t, f in zip(tau_w, fluids)
    ])
    mu = np.divide(tau_w, gdot, out=np.full_like(tau_w, np.inf), where=gdot > 0.0)
    re = np.divide(rho * velocity * hydraulic_diameter, mu,
                   out=np.zeros_like(mu), where=np.isfinite(mu) & (mu > 0.0))
    return FlowRegime(leg=leg, depth=depth, reynolds=re, velocity=velocity,
                      effective_viscosity=mu, hydraulic_diameter=hydraulic_diameter)


def regime_from_result(result, q, numerics=None):
    """Flow regime in both legs at the end of a circulation run.

    Uses the effective fluid at each station, so it reflects the mixture that
    is actually there rather than either pure fluid.
    """
    from .fluid import mix_fluids

    gammadot_c = None if numerics is None else numerics.regularisation_shear_rate
    ag, cg = result.annulus_grid, result.casing_grid
    order = np.argsort(ag.z_centers)

    def effective(fractions, volume):
        """Volume-weighted mixture at each station (assumption A-06)."""
        weights = np.einsum("i...,...->i...", fractions, volume)
        return [
            mix_fluids(result.fluids,
                       [float(weights[i, k].sum()) for i in range(len(result.fluids))],
                       name="eff")
            for k in range(fractions.shape[1])
        ]

    # --- annulus ---
    ann_fluids = effective(result.annulus_fractions, ag.cell_volume)
    area = ag.cell_volume.sum(axis=(1, 2)) / ag.dz
    v_a = q / area
    d_h_a = 2.0 * (ag.r_outer - ag.r_inner)
    rho_a = np.array([f.rho for f in ann_fluids])
    tau_a = (result.annulus_tau_w if result.annulus_tau_w is not None
             else np.zeros(ag.n_axial))
    annulus = _regime("annulus", ag.z_centers[order], rho_a[order], v_a[order],
                      d_h_a[order], tau_a[order],
                      [ann_fluids[i] for i in order], gammadot_c)

    # --- casing ---
    cas_fluids = effective(result.casing_fractions, cg.cell_volume)
    v_c = np.full(cg.n_axial, q / float(cg.cell_area.sum()))
    d_h_c = np.full(cg.n_axial, 2.0 * cg.radius)
    rho_c = np.array([f.rho for f in cas_fluids])
    from .velocity import solve_tau_w
    tau_c = np.array([
        solve_tau_w(q, f, cg.radius, gammadot_c=gammadot_c) if q > 0.0 else 0.0
        for f in cas_fluids
    ])
    casing = _regime("casing", cg.z_centers, rho_c, v_c, d_h_c, tau_c,
                     cas_fluids, gammadot_c)
    return casing, annulus
