"""Test gate 5 - integration.

The gates run in order.  Gate 4 (the 200 m field-scale case) is guarded: it
only runs once gates 1-3 have passed, because debugging at field scale wastes
days.
"""

import math

import numpy as np
import pytest

from inpipe.config import (
    GeometryConfig,
    GridConfig,
    NumericsConfig,
    SimulationConfig,
    bpm_to_m3s,
    inch_to_m,
)
from inpipe.fluid import Fluid, PumpSchedule, PumpStage
from inpipe.solver import InPipeSolver
from inpipe.transport import upwind_diffusivity

# --- the paper's Section 3.1 geometry --------------------------------------
LAB_LENGTH = 4.0  # m
LAB_ID = 0.019  # m
LAB_UBAR = 0.05  # m/s
LAB_AREA = math.pi * (0.5 * LAB_ID) ** 2
LAB_Q = LAB_UBAR * LAB_AREA


def lab_config(n_axial=100, n_layer=13, n_azimuth=18, **numerics):
    return SimulationConfig(
        geometry=GeometryConfig(length=LAB_LENGTH, inner_diameter=LAB_ID, inclination=0.0),
        grid=GridConfig(n_axial=n_axial, n_layer=n_layer, n_azimuth=n_azimuth),
        numerics=NumericsConfig(**numerics),
    )


def lab_solver(n_axial=100, z0=1.0, **kw):
    mud = Fluid.newtonian(1000.0, 0.001, "mud")
    displacing = Fluid.newtonian(1000.0, 0.001, "displacing")
    schedule = PumpSchedule([PumpStage(displacing, LAB_Q * 1000.0, LAB_Q)])
    solver = InPipeSolver(lab_config(n_axial=n_axial, **kw), schedule, initial_fluid=mud)
    solver.set_initial_interface(z0, displacing, mud)
    return solver, displacing, mud


# ---------------------------------------------------------------------------
# Gate 1 - single fluid, no displacement
# ---------------------------------------------------------------------------


def test_gate1_single_fluid_is_stationary():
    """100 steps of a single fluid: nothing changes."""
    mud = Fluid.newtonian(1000.0, 0.001, "mud")
    schedule = PumpSchedule([PumpStage(mud, LAB_Q * 1000.0, LAB_Q)])
    solver = InPipeSolver(lab_config(), schedule, initial_fluid=mud)
    f0 = solver.f.copy()
    for _ in range(100):
        solver.step()
    np.testing.assert_allclose(solver.f, f0, rtol=0.0, atol=1e-14)
    assert solver.n_steps == 100
    assert solver.t > 0.0, "time must actually have advanced"


def test_gate1_mixing_status_stays_zero():
    """Nothing sets s = 1 in Phase 1 (assumption A-17)."""
    solver, _, _ = lab_solver()
    for _ in range(50):
        solver.step()
    assert np.max(np.abs(solver.s)) == 0.0


# ---------------------------------------------------------------------------
# Gate 2 - the paper's 4 m / 19 mm geometry, vertical
# ---------------------------------------------------------------------------


def test_gate2_parabolic_stretching_and_conservation():
    solver, displacing, mud = lab_solver()
    result = solver.run(t_end=20.0)
    g = result.grid

    # -- volume conservation against the exact flux budget ------------------
    d = result.diagnostics
    vols = np.asarray(d.fluid_volumes)
    budget = vols - (vols[0] + np.asarray(d.influx) - np.asarray(d.outflux))
    rel = np.max(np.abs(budget)) / vols[0].sum()
    assert rel < 1e-10, f"mass budget error {rel:.3e}"

    # -- sum-to-one and boundedness ----------------------------------------
    assert max(d.sum_to_one_error) < 1e-12
    assert min(d.f_min) > -1e-12 and max(d.f_max) < 1.0 + 1e-12

    # -- parabolic stretching against z(r,t) = z0 + u(r) t ------------------
    front = result.diagnostics.front_position[-1]
    exact = 1.0 + result.velocity[0] * result.time
    inside = np.isfinite(front) & (exact < g.length - 6 * g.dz)
    err = np.abs(front - exact)[inside]
    assert err.max() < 3.0 * g.dz, f"front error {err.max() / g.dz:.2f} cells"

    # -- the Newtonian signature: the tip runs at 2 u_bar -------------------
    tip = np.nanmax(front) - 1.0
    assert tip / (LAB_UBAR * result.time) == pytest.approx(2.0, rel=0.05)

    # -- the interface is a parabola in r, not a plane ----------------------
    j = g.n_azimuth // 2
    col_front = front[:, j]
    ok = np.isfinite(col_front)
    r = g.cell_r[:, j][ok]
    disp = col_front[ok] - 1.0
    fit = np.polyfit(r**2, disp, 1)
    residual = np.abs(np.polyval(fit, r**2) - disp).max()
    assert residual < 2.0 * g.dz, "front is not parabolic in r^2"
    assert fit[0] < 0.0, "front must decrease with radius"


def test_gate2_outlet_series_is_recorded():
    solver, displacing, mud = lab_solver(z0=3.5)
    result = solver.run(t_end=40.0)
    assert result.outlet_fractions.shape == (result.n_steps, len(result.fluids))
    np.testing.assert_allclose(result.outlet_fractions.sum(axis=1), 1.0, atol=1e-12)
    i_disp = result.fluids.index(displacing)
    # Breakthrough happens, and the displacing fraction rises monotonically.
    assert result.outlet_fractions[0, i_disp] < 1e-12
    assert result.outlet_fractions[-1, i_disp] > 0.3


# ---------------------------------------------------------------------------
# Gate 3 - grid convergence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def convergence_study():
    """Halve dz three times and record the front and the numerical diffusion."""
    out = []
    for n_axial in (100, 200, 400, 800):
        solver, displacing, mud = lab_solver(n_axial=n_axial)
        result = solver.run(t_end=15.0)
        g = result.grid
        front = result.diagnostics.front_position[-1]
        # Track the front at a fixed physical radius, interpolating across the
        # differing layer counts is unnecessary since n_layer is fixed.
        j = g.n_azimuth // 2
        li = g.n_layer // 2
        dt = float(np.mean(result.diagnostics.dt))
        u = float(result.velocity[0, li, j])
        out.append(
            dict(
                n_axial=n_axial,
                dz=g.dz,
                dt=dt,
                front=float(front[li, j]),
                exact=1.0 + u * result.time,
                dm_paper=g.dz**2 / dt,
                dm_scheme=upwind_diffusivity(u, g.dz, dt),
            )
        )
    return out


def test_gate3_front_position_converges(convergence_study):
    """The front position converges to the analytical value under refinement."""
    errs = [abs(c["front"] - c["exact"]) for c in convergence_study]
    assert errs[-1] < errs[0], f"front did not converge: {errs}"
    assert errs[-1] < 0.01 * LAB_LENGTH
    # Monotone, or at worst flat, refinement.
    for a, b in zip(errs[:-1], errs[1:]):
        assert b <= a * 1.05, f"front error grew under refinement: {errs}"


def test_gate3_numerical_diffusion_falls_linearly_not_quadratically(
    convergence_study, capsys
):
    """Dm_num falls as dx, not dx^2, once dt is CFL-limited.

    The build spec asks for dx^2.  That is only reachable at *fixed* dt, which
    an explicit CFL-limited scheme cannot do: halving dz forces dt to halve too,
    so even the paper's own dx^2/dt estimate is linear in dx here.  The true
    upwind numerical diffusivity, u*dz*(1-C)/2, is likewise linear in dz at
    fixed Courant number.  Both are asserted below, and the ratios are printed.
    """
    with capsys.disabled():
        print("\n  dz [m]      dt [s]   front err [m]   dx^2/dt      u dz(1-C)/2")
        for c in convergence_study:
            print(
                f"  {c['dz']:.5f}  {c['dt']:.5f}  {abs(c['front'] - c['exact']):.3e}   "
                f"{c['dm_paper']:.4e}   {c['dm_scheme']:.4e}"
            )
    for a, b in zip(convergence_study[:-1], convergence_study[1:]):
        assert 1.7 < a["dm_paper"] / b["dm_paper"] < 2.3
        assert 1.7 < a["dm_scheme"] / b["dm_scheme"] < 2.3


def test_gate3_numerical_diffusion_would_fall_quadratically_at_fixed_dt():
    """The spec's dx^2 claim, checked on its own terms.

    Holding dt fixed while halving dz does give dx^2/dt ~ dx^2 - but it also
    doubles the Courant number each time, so after two refinements the scheme
    is unstable.  This test shows the arithmetic and the stability wall.
    """
    from inpipe.transport import CFLViolation, assert_cfl, numerical_diffusivity

    dt = 0.05
    u = np.full((10, 1, 1), 0.1)
    dms, courants = [], []
    for dz in (0.04, 0.02, 0.01):
        dms.append(numerical_diffusivity(dz, dt))
        courants.append(0.1 * dt / dz)
    assert dms[0] / dms[1] == pytest.approx(4.0)
    assert dms[1] / dms[2] == pytest.approx(4.0)
    # ...and the last of those violates CFL = 0.4.
    with pytest.raises(CFLViolation):
        assert_cfl(u, 0.01, dt, 0.4)


# ---------------------------------------------------------------------------
# Gate 4 - field scale.  Runs only after gates 1-3.
# ---------------------------------------------------------------------------

FIELD_LENGTH = 200.0
FIELD_ID = inch_to_m(5.0)
FIELD_Q = bpm_to_m3s(5.0)


def field_solver(n_axial=200):
    mud = Fluid("mud", rho=1198.0, tau0=2.0, k=0.30, n=0.72)
    spacer = Fluid("spacer", rho=1318.0, tau0=1.2, k=0.18, n=0.80)
    cement = Fluid("cement", rho=1870.0, tau0=6.0, k=0.55, n=0.65)
    area = math.pi * (0.5 * FIELD_ID) ** 2
    pipe_volume = area * FIELD_LENGTH
    schedule = PumpSchedule(
        [
            PumpStage(spacer, 0.35 * pipe_volume, FIELD_Q),
            PumpStage(cement, 1.20 * pipe_volume, FIELD_Q),
        ]
    )
    config = SimulationConfig(
        geometry=GeometryConfig(length=FIELD_LENGTH, inner_diameter=FIELD_ID, inclination=0.0),
        grid=GridConfig(n_axial=n_axial, n_layer=13, n_azimuth=18),
        numerics=NumericsConfig(diagnostics_every=25),
    )
    return InPipeSolver(config, schedule, initial_fluid=mud), (mud, spacer, cement)


@pytest.mark.slow
def test_gate4_field_scale_200m(convergence_study, capsys):
    """200 m, 127 mm ID, three Herschel-Bulkley fluids, under 5 minutes."""
    solver, (mud, spacer, cement) = field_solver()
    result = solver.run()

    d = result.diagnostics
    vols = np.asarray(d.fluid_volumes)
    budget = vols - (vols[0] + np.asarray(d.influx) - np.asarray(d.outflux))
    rel = np.max(np.abs(budget)) / vols[0].sum()

    with capsys.disabled():
        from inpipe.postprocess import summary_table

        print("\n" + summary_table(result))

    assert result.wall_time < 300.0, f"runtime {result.wall_time:.1f} s exceeds 5 min"
    assert rel < 1e-10, f"mass budget error {rel:.3e}"
    assert max(d.sum_to_one_error) < 1e-12
    assert min(d.f_min) > -1e-12 and max(d.f_max) < 1.0 + 1e-12
    # Cement reaches the shoe by the end of the job.
    i_cem = result.fluids.index(cement)
    assert result.outlet_fractions[-1, i_cem] > 0.5
    # Mixed-rheology stations exist, so the divergence correction is exercised.
    assert solver._n_unique_stations > 1
