"""Configuration dataclasses and unit conversion helpers.

SI units are used everywhere internally (m, s, kg, Pa, m^3/s). Field units are
converted only at I/O boundaries, using the helpers in this module.

Reference: Dai, H., Eslami, A., Schneider, J., Liu, G., Schwering, F. (2023),
"Modeling displacement flow inside a full-length casing string for well
cementing", Petroleum Research 9, 1-16.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G_ACCEL = 9.80665  # m/s^2, standard gravity

# ---------------------------------------------------------------------------
# Unit conversion helpers.  Naming convention: <from>_to_<to>.
# ---------------------------------------------------------------------------

_BBL_M3 = 0.158987294928  # 1 US oil barrel in m^3 (exact: 42 US gal)
_FT_M = 0.3048  # exact
_IN_M = 0.0254  # exact
_GAL_M3 = _BBL_M3 / 42.0  # 1 US gallon in m^3
_LB_KG = 0.45359237  # exact
_PSI_PA = 6894.757293168361


def bpm_to_m3s(bpm: float) -> float:
    """Barrels per minute -> m^3/s."""
    return bpm * _BBL_M3 / 60.0


def m3s_to_bpm(q: float) -> float:
    """m^3/s -> barrels per minute."""
    return q * 60.0 / _BBL_M3


def ft_to_m(ft: float) -> float:
    return ft * _FT_M


def m_to_ft(m: float) -> float:
    return m / _FT_M


def inch_to_m(inch: float) -> float:
    return inch * _IN_M


def m_to_inch(m: float) -> float:
    return m / _IN_M


def cp_to_pas(cp: float) -> float:
    """Centipoise -> Pa.s."""
    return cp * 1.0e-3


def pas_to_cp(pas: float) -> float:
    return pas * 1.0e3


def ppg_to_kgm3(ppg: float) -> float:
    """Pounds per US gallon -> kg/m^3."""
    return ppg * _LB_KG / _GAL_M3


def kgm3_to_ppg(rho: float) -> float:
    return rho * _GAL_M3 / _LB_KG


def psi_to_pa(psi: float) -> float:
    return psi * _PSI_PA


def pa_to_psi(pa: float) -> float:
    return pa / _PSI_PA


def lbf100ft2_to_pa(v: float) -> float:
    """lbf/100 ft^2 -> Pa (the field unit for yield stress / API rheology)."""
    return v * 4.44822161526 / (100.0 * _FT_M**2)


def pa_to_lbf100ft2(v: float) -> float:
    return v * (100.0 * _FT_M**2) / 4.44822161526


# ---------------------------------------------------------------------------
# Configuration dataclasses.  No magic numbers in the solver: every numerical
# and geometric parameter is declared here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryConfig:
    """Pipe geometry.

    Attributes
    ----------
    length : total measured depth of the pipe section [m].
    inner_diameter : casing / drill-pipe ID [m].
    inclination : inclination angle beta from vertical [rad]. beta = 0 is a
        vertical pipe (paper Eq. A.7 sign convention).
    """

    length: float
    inner_diameter: float
    inclination: float = 0.0

    @property
    def radius(self) -> float:
        return 0.5 * self.inner_diameter

    @property
    def area(self) -> float:
        import math

        return math.pi * self.radius**2


@dataclass(frozen=True)
class GridConfig:
    """Mesh resolution for the structured n_axial x n_layer x n_azimuth grid."""

    n_axial: int
    n_layer: int
    n_azimuth: int
    #: Rule used to place the horizontal layer boundaries in the chord
    #: coordinate y in [-R, R].  See docs/assumptions.md (A-03).
    layer_rule: str = "uniform_y"
    #: Reference azimuth of the central longitudinal grid line, measured from
    #: the +x axis [rad].  Physically arbitrary in a vertical pipe (paper
    #: Section 2.2); fixed here for reproducibility.  See A-09.
    reference_azimuth: float = 0.0


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical parameters of the transport solver."""

    #: Courant number used to size the explicit Euler timestep.  See A-05.
    cfl: float = 0.4
    #: Face-value reconstruction used by the advection kernel.
    face_scheme: str = "upwind"
    #: Tolerances handed to scipy.optimize.brentq in the tau_w inverse solve.
    brentq_xtol: float = 1.0e-12
    brentq_rtol: float = 1.0e-12
    #: Absolute tolerance on the sum-to-one constraint on volume fractions.
    sum_to_one_atol: float = 1.0e-12
    #: Absolute tolerance on the 0 <= f_i <= 1 boundedness constraint.
    boundedness_atol: float = 1.0e-12
    #: Apply the discrete divergence correction to the VOF update so that
    #: sum_i f_i = 1 is preserved when the axial velocity profile varies with
    #: depth (mixed rheology).  See A-07.
    divergence_correction: bool = True
    #: Velocity mapping strategy from the 1D profile onto 3D cells.  See A-04.
    #: "centroid" evaluates u at the cell centroid radius; it exceeds the 1 %
    #: flow-rate gate for yield-stress fluids at the paper's own 13 x 18
    #: resolution, so "area_average" is the default.
    velocity_mapping: str = "area_average"
    #: Number of steps between diagnostic records.
    diagnostics_every: int = 10


@dataclass
class SimulationConfig:
    """Top-level configuration bundle."""

    geometry: GeometryConfig
    grid: GridConfig
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    gravity: float = G_ACCEL
