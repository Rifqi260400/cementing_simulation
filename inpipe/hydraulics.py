"""Circulating pressures: hydrostatic head, friction, ECD and the U-tube.

Depth convention: ``z`` is measured depth, increasing downward from surface.

A force balance on a fluid element gives, along that axis,

    dp/dz = rho g cos(beta) - flow_sign * P_fric

with ``flow_sign = +1`` where the fluid moves *down* the axis (inside the
casing) and ``-1`` where it moves *up* it (in the annulus): wall shear always
opposes the motion, so it costs pressure along the flow.

Gravity enters here and only here.  The flow rate is the pump rate - it is not
driven by the density contrast in this phase - so these pressures are a
*report* on the run, not a feedback into it.  The one number that says whether
that simplification is safe is :attr:`HydraulicsReport.utube_imbalance`; see
its docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["HydraulicsReport", "circulation_pressure", "equivalent_mud_weight"]


def equivalent_mud_weight(pressure: float, depth: float, gravity: float) -> float:
    """Density [kg/m^3] whose static head equals ``pressure`` at ``depth``."""
    if depth <= 0.0:
        return float("nan")
    return pressure / (gravity * depth)


@dataclass
class HydraulicsReport:
    """Pressures around the circulating loop at one instant.

    All pressures are gauge, relative to the surface reference.
    """

    #: Depths [m] and pressures [Pa] going *down* the casing.
    casing_depth: np.ndarray
    casing_pressure: np.ndarray
    #: Depths [m] and pressures [Pa] in the annulus, ascending depth.
    annulus_depth: np.ndarray
    annulus_pressure: np.ndarray
    #: Pressure the pump must supply at surface [Pa].
    pump_pressure: float
    #: Pressure at the shoe on the casing side [Pa].
    shoe_pressure: float
    #: Frictional loss in each leg [Pa].
    casing_friction: float
    annulus_friction: float
    #: Static head of each leg [Pa].
    casing_hydrostatic: float
    annulus_hydrostatic: float
    #: Equivalent circulating density at the shoe [kg/m^3].
    ecd_at_shoe: float
    #: Equivalent static density of the annular column [kg/m^3].
    esd_at_shoe: float
    gravity: float

    @property
    def utube_imbalance(self) -> float:
        """``casing_hydrostatic - annulus_hydrostatic`` [Pa].

        Positive means the casing column is the heavier of the two and is
        trying to fall on its own.  Once it exceeds the total friction the
        two legs can supply, a real well free-falls and the returns rate
        exceeds the pump rate - at which point an imposed-rate model
        understates how fast the job actually goes.
        :attr:`free_fall_margin` is that comparison.
        """
        return self.casing_hydrostatic - self.annulus_hydrostatic

    @property
    def free_fall_margin(self) -> float:
        """``total friction - U-tube imbalance`` [Pa].

        Negative means gravity alone would drive the flow faster than the pump
        is imposing, i.e. the well is free-falling and this phase's
        constant-rate assumption no longer holds.  Reported, not acted on.
        """
        return (self.casing_friction + self.annulus_friction) - self.utube_imbalance

    @property
    def is_free_falling(self) -> bool:
        return self.free_fall_margin < 0.0

    def ecd(self) -> np.ndarray:
        """Equivalent circulating density against annular depth [kg/m^3]."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.annulus_pressure / (self.gravity * self.annulus_depth)

    def summary(self) -> str:
        ppg = 0.45359237 / (0.158987294928 / 42.0)  # kg/m^3 per ppg
        lines = [
            f"pump pressure     : {self.pump_pressure / 1e5:8.2f} bar "
            f"({self.pump_pressure / 6894.757:7.1f} psi)",
            f"shoe pressure     : {self.shoe_pressure / 1e5:8.2f} bar "
            f"({self.shoe_pressure / 6894.757:7.1f} psi)",
            f"friction casing   : {self.casing_friction / 1e5:8.2f} bar",
            f"friction annulus  : {self.annulus_friction / 1e5:8.2f} bar",
            f"hydrostatic casing: {self.casing_hydrostatic / 1e5:8.2f} bar",
            f"hydrostatic annul.: {self.annulus_hydrostatic / 1e5:8.2f} bar",
            f"ECD at shoe       : {self.ecd_at_shoe:8.1f} kg/m3 "
            f"({self.ecd_at_shoe / ppg:6.2f} ppg)",
            f"ESD at shoe       : {self.esd_at_shoe:8.1f} kg/m3 "
            f"({self.esd_at_shoe / ppg:6.2f} ppg)",
            f"U-tube imbalance  : {self.utube_imbalance / 1e5:8.2f} bar "
            f"(casing heavier if positive)",
            f"free-fall margin  : {self.free_fall_margin / 1e5:8.2f} bar"
            + ("   *** WELL WOULD FREE-FALL ***" if self.is_free_falling else ""),
        ]
        return "\n".join(lines)


def circulation_pressure(
    casing_z, casing_dz, casing_rho, casing_tau_w, casing_radius,
    annulus_z, annulus_dz, annulus_rho, annulus_tau_w, annulus_half_gap,
    inclination=0.0, gravity=9.80665, surface_pressure=0.0,
) -> HydraulicsReport:
    """Integrate pressure down the casing and back up the annulus.

    ``casing_*`` arrays run in ascending depth; ``annulus_*`` arrays are in
    annular flow order (index 0 at the shoe) and are reordered internally.

    Wall-shear-to-gradient relations are the paper's own (Appendix A.1):
    ``P = 2 tau_w / R`` in the pipe, ``P = tau_w / b`` in the slot, since
    ``tau_w = (h/2) P`` with ``h = 2b``.
    """
    cosb = math.cos(inclination)
    casing_rho = np.asarray(casing_rho, dtype=float)
    casing_tau_w = np.asarray(casing_tau_w, dtype=float)
    annulus_rho = np.asarray(annulus_rho, dtype=float)
    annulus_tau_w = np.asarray(annulus_tau_w, dtype=float)
    annulus_half_gap = np.asarray(annulus_half_gap, dtype=float)

    # --- casing: flow along +z, so friction subtracts -----------------------
    casing_fric_grad = 2.0 * casing_tau_w / casing_radius
    casing_hydro = float((casing_rho * gravity * cosb * casing_dz).sum())
    casing_friction = float((casing_fric_grad * casing_dz).sum())
    dP = (casing_rho * gravity * cosb - casing_fric_grad) * casing_dz
    casing_pressure = surface_pressure + np.cumsum(dP) - 0.5 * dP  # cell centres
    shoe_pressure = surface_pressure + float(dP.sum())

    # --- annulus: flow along -z, so friction adds ---------------------------
    order = np.argsort(np.asarray(annulus_z, dtype=float))
    z_a = np.asarray(annulus_z, dtype=float)[order]
    rho_a = annulus_rho[order]
    fric_grad_a = annulus_tau_w[order] / annulus_half_gap[order]
    annulus_hydro = float((rho_a * gravity * cosb * annulus_dz).sum())
    annulus_friction = float((fric_grad_a * annulus_dz).sum())
    dPa = (rho_a * gravity * cosb + fric_grad_a) * annulus_dz
    # Integrate downward from the surface return, which is at the reference.
    annulus_pressure = surface_pressure + np.cumsum(dPa) - 0.5 * dPa

    # Pump pressure closes the loop: what surface must supply so that the
    # annulus returns to the reference pressure.
    pump_pressure = (
        annulus_hydro + annulus_friction - casing_hydro + casing_friction
    )

    shoe_depth = float(z_a[-1] + 0.5 * annulus_dz)
    return HydraulicsReport(
        casing_depth=np.asarray(casing_z, dtype=float),
        casing_pressure=casing_pressure + pump_pressure,
        annulus_depth=z_a,
        annulus_pressure=annulus_pressure,
        pump_pressure=pump_pressure,
        shoe_pressure=shoe_pressure + pump_pressure,
        casing_friction=casing_friction,
        annulus_friction=annulus_friction,
        casing_hydrostatic=casing_hydro,
        annulus_hydrostatic=annulus_hydro,
        ecd_at_shoe=equivalent_mud_weight(
            surface_pressure + annulus_hydro + annulus_friction, shoe_depth, gravity
        ),
        esd_at_shoe=equivalent_mud_weight(
            surface_pressure + annulus_hydro, shoe_depth, gravity
        ),
        gravity=gravity,
    )
