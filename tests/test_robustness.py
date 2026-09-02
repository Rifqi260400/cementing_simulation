"""Robustness sweep - the solver must not break outside the curated cases.

Every case here asserts the same four invariants the solver promises:
per-fluid mass budget, sum-to-one, boundedness, and finiteness.  The point is
coverage of the input space, not physics: the physics is pinned down by the
five test gates.
"""

import math

import numpy as np
import pytest

from inpipe.config import GeometryConfig, GridConfig, NumericsConfig, SimulationConfig
from inpipe.fluid import Fluid, PumpSchedule, PumpStage
from inpipe.solver import InPipeSolver

# A deliberately awkward spread of rheologies.
RHEOLOGIES = {
    "newtonian": Fluid("A", 1000.0, 0.0, 1e-3, 1.0),
    "thick": Fluid("A", 1000.0, 0.0, 1.0, 1.0),
    "power_n0.2": Fluid("A", 1000.0, 0.0, 0.8, 0.2),
    "power_n0.4": Fluid("A", 1000.0, 0.0, 0.8, 0.4),
    "bingham": Fluid("A", 1500.0, 5.0, 0.03, 1.0),
    "hb_high_yield": Fluid("A", 1800.0, 30.0, 0.55, 0.5),
    "hb_low_n": Fluid("A", 1800.0, 2.0, 0.9, 0.25),
}


def _displacer(a: Fluid) -> Fluid:
    return Fluid("B", a.rho * 1.2, a.tau0 * 1.5, a.k * 1.3, min(a.n * 1.1, 1.0))


def build(length=4.0, diameter=0.019, n_axial=40, n_layer=7, n_azimuth=8,
          rheology="bingham", q_mult=1.0, stages=1, **numerics):
    area = math.pi * (0.5 * diameter) ** 2
    a = RHEOLOGIES[rheology]
    b = _displacer(a)
    q = 0.05 * area * q_mult
    pv = area * length
    if stages == 1:
        schedule = PumpSchedule([PumpStage(b, 1.2 * pv, q)])
    else:
        schedule = PumpSchedule([
            PumpStage(b, 0.3 * pv, q),
            PumpStage(a, 0.3 * pv, q * 3.0),   # a genuine mid-job rate change
            PumpStage(b, 0.8 * pv, q * 0.5),
        ])
    config = SimulationConfig(
        geometry=GeometryConfig(length=length, inner_diameter=diameter),
        grid=GridConfig(n_axial=n_axial, n_layer=n_layer, n_azimuth=n_azimuth),
        numerics=NumericsConfig(diagnostics_every=10, **numerics),
    )
    return InPipeSolver(config, schedule, initial_fluid=a), schedule


def assert_healthy(result, label):
    d = result.diagnostics
    vols = np.asarray(d.fluid_volumes)
    budget = vols - (vols[0] + np.asarray(d.influx) - np.asarray(d.outflux))
    mass = float(np.max(np.abs(budget)) / vols[0].sum())

    assert np.all(np.isfinite(result.fractions)), f"{label}: non-finite volume fractions"
    assert np.all(np.isfinite(result.velocity)), f"{label}: non-finite velocity"
    assert result.n_steps > 0, f"{label}: no steps taken"
    assert mass < 1e-9, f"{label}: mass budget error {mass:.2e}"
    assert max(d.sum_to_one_error) < 1e-11, f"{label}: sum-to-one {max(d.sum_to_one_error):.2e}"
    assert min(d.f_min) > -1e-11, f"{label}: f went to {min(d.f_min):.2e}"
    assert max(d.f_max) < 1.0 + 1e-11, f"{label}: f went to {max(d.f_max):.2e}"


def run_case(label, t_frac=0.35, **kwargs):
    solver, schedule = build(**kwargs)
    result = solver.run(t_end=t_frac * schedule.total_time)
    assert_healthy(result, label)
    return result


# --- rheology --------------------------------------------------------------


@pytest.mark.parametrize("rheology", list(RHEOLOGIES))
def test_every_rheology_runs_healthily(rheology):
    run_case(f"rheology={rheology}", rheology=rheology)


# --- flow rate -------------------------------------------------------------


@pytest.mark.parametrize("q_mult", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_flow_rate_spans_four_decades(q_mult):
    """Including rates barely above the yield stress of a 30 Pa fluid."""
    run_case(f"q_mult={q_mult}", rheology="hb_high_yield", q_mult=q_mult)


# --- grid, including degenerate shapes -------------------------------------


@pytest.mark.parametrize(
    "n_axial,n_layer,n_azimuth",
    [
        (1, 7, 8),    # a single axial cell
        (5, 1, 1),    # a single cell per cross-section
        (10, 2, 2),
        (20, 3, 5),   # odd azimuth count
        (40, 7, 8),
        (60, 13, 18),  # the paper's own cross-section
    ],
)
def test_degenerate_and_ordinary_grids(n_axial, n_layer, n_azimuth):
    run_case(f"grid={n_axial}x{n_layer}x{n_azimuth}",
             n_axial=n_axial, n_layer=n_layer, n_azimuth=n_azimuth)


# --- schedule --------------------------------------------------------------


@pytest.mark.parametrize("rheology", ["newtonian", "power_n0.4", "hb_high_yield"])
def test_mid_job_rate_changes(rheology):
    """Three stages with a 3x rate step up and a 6x step down."""
    result = run_case(f"3-stage {rheology}", rheology=rheology, stages=3, t_frac=1.0)
    # The timestep must actually have responded to the rate changes.
    dts = np.asarray(result.diagnostics.dt)
    assert dts.max() / dts.min() > 2.0, "timestep did not track the rate change"


# --- geometry scale --------------------------------------------------------


@pytest.mark.parametrize(
    "length,diameter",
    [(0.5, 0.005), (4.0, 0.019), (200.0, 0.127), (1524.0, 0.127)],
)
def test_geometry_spans_lab_to_full_length_casing(length, diameter):
    run_case(f"L={length} D={diameter}", length=length, diameter=diameter,
             rheology="hb_high_yield", t_frac=0.1)


# --- numerics options ------------------------------------------------------


@pytest.mark.parametrize(
    "numerics",
    [
        dict(cfl=0.9),
        dict(cfl=0.05),
        dict(velocity_mapping="centroid"),
        dict(face_scheme="upwind"),
    ],
)
def test_numerics_options_stay_healthy(numerics):
    run_case(f"numerics={numerics}", rheology="power_n0.4", stages=3, **numerics)


# --- the two known-deficient non-default modes -----------------------------


def test_local_closure_is_still_the_documented_tradeoff():
    """A-07: the 'local' closure holds sum-to-one but loses per-fluid volume.

    Kept as a live check that the deficiency is what the register says it is,
    so nobody switches to it expecting conservation.
    """
    solver, schedule = build(rheology="power_n0.4", stages=3,
                             transverse_closure="local")
    result = solver.run(t_end=schedule.total_time)
    d = result.diagnostics
    vols = np.asarray(d.fluid_volumes)
    budget = vols - (vols[0] + np.asarray(d.influx) - np.asarray(d.outflux))
    mass = float(np.max(np.abs(budget)) / vols[0].sum())
    assert max(d.sum_to_one_error) < 1e-11, "sum-to-one should still hold"
    assert mass > 1e-6, f"expected the documented mass loss, got {mass:.2e}"


def test_disabling_continuity_enforcement_degrades_conservation():
    """A-22: without the per-station rescale the closure conserves only to the
    mapping error, not to round-off."""
    solver, schedule = build(rheology="power_n0.4", stages=3,
                             enforce_discrete_continuity=False)
    result = solver.run(t_end=schedule.total_time)
    d = result.diagnostics
    vols = np.asarray(d.fluid_volumes)
    budget = vols - (vols[0] + np.asarray(d.influx) - np.asarray(d.outflux))
    mass = float(np.max(np.abs(budget)) / vols[0].sum())
    assert mass > 1e-11, "expected degraded conservation without the rescale"
    assert mass < 1e-5, f"but not this bad: {mass:.2e}"


# --- fine meshes -----------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("n_axial,n_layer,n_azimuth", [(200, 26, 36), (400, 52, 72)])
def test_fine_meshes(n_axial, n_layer, n_azimuth):
    run_case(f"fine {n_axial}x{n_layer}x{n_azimuth}", n_axial=n_axial,
             n_layer=n_layer, n_azimuth=n_azimuth, t_frac=0.2)
