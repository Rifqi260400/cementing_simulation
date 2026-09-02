"""Test gate 7 - annular grid, hydraulics, and the coupled circulation."""

import math

import numpy as np
import pytest

from inpipe.annulus_grid import AnnulusGrid
from inpipe.caliper import synthetic_caliper
from inpipe.circulation import CirculationSolver, WellConfig
from inpipe.config import GridConfig, NumericsConfig, bpm_to_m3s, inch_to_m
from inpipe.fluid import Fluid, PumpSchedule, PumpStage
from inpipe.hydraulics import circulation_pressure, equivalent_mud_weight
from inpipe.slot import solve_slot_profile

LENGTH = 200.0
CASING_ID = inch_to_m(5.0)
CASING_OD = inch_to_m(5.5)
GAUGE = inch_to_m(8.5)
MUD = Fluid("mud", 1198.0, 2.0, 0.30, 0.72)
CEMENT = Fluid("cement", 1870.0, 6.0, 0.55, 0.65)


@pytest.fixture(scope="module")
def caliper():
    return synthetic_caliper(LENGTH, GAUGE)


@pytest.fixture(scope="module")
def smooth_caliper():
    return synthetic_caliper(LENGTH, GAUGE, washouts=(), tight_zones=(), roughness=0.0)


# ---------------------------------------------------------------------------
# Annulus grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_layer,n_azimuth", [(1, 1), (5, 4), (9, 8), (17, 12)])
def test_ring_areas_reproduce_the_annulus_exactly(caliper, n_layer, n_azimuth):
    g = AnnulusGrid(LENGTH, CASING_OD, caliper, 60, n_layer, n_azimuth)
    np.testing.assert_allclose(g.station_area, g.exact_station_area(), rtol=1e-13)


def test_volume_matches_the_caliper_integral(smooth_caliper):
    """A smooth hole must give exactly the analytic annular volume."""
    g = AnnulusGrid(LENGTH, CASING_OD, smooth_caliper, 200, 7, 6)
    exact = math.pi * ((0.5 * GAUGE) ** 2 - (0.5 * CASING_OD) ** 2) * LENGTH
    assert g.total_volume == pytest.approx(exact, rel=1e-10)


def test_washouts_add_volume(caliper, smooth_caliper):
    rough = AnnulusGrid(LENGTH, CASING_OD, caliper, 200, 7, 6)
    smooth = AnnulusGrid(LENGTH, CASING_OD, smooth_caliper, 200, 7, 6)
    assert rough.total_volume > 1.2 * smooth.total_volume


def test_flow_order_puts_the_shoe_first(caliper):
    g = AnnulusGrid(LENGTH, CASING_OD, caliper, 20, 5, 4)
    assert g.z_centers[0] > g.z_centers[-1]
    assert g.z_faces[0] == pytest.approx(LENGTH)
    assert g.z_faces[-1] == pytest.approx(0.0)
    assert g.dz > 0.0


def test_a_hole_narrower_than_the_casing_is_refused(caliper):
    with pytest.raises(ValueError, match="the annulus closes"):
        AnnulusGrid(LENGTH, GAUGE * 1.1, caliper, 20, 5, 4)


def test_area_average_mapping_beats_centroid(caliper):
    g = AnnulusGrid(LENGTH, CASING_OD, caliper, 60, 9, 6)
    q = bpm_to_m3s(5.0)
    profiles = [solve_slot_profile(q, CEMENT, g.r_inner, ro) for ro in g.r_outer]
    err = {}
    for method in ("centroid", "area_average"):
        u = g.map_velocity(profiles, method)
        err[method] = float(
            np.max(np.abs(np.einsum("klm,klm->k", u, g.cell_area) - q) / q)
        )
    assert err["area_average"] < 1e-5
    assert err["area_average"] < err["centroid"] / 100.0


def test_face_flux_normalisation_is_exact(caliper):
    """Every axial face must pass the imposed rate.

    Normalising at cell centres is not enough where the hole diameter varies:
    the flux crosses face areas, which differ from the cell areas either side.
    """
    from inpipe.transport import face_velocity_stack

    g = AnnulusGrid(LENGTH, CASING_OD, caliper, 60, 9, 6)
    q = bpm_to_m3s(5.0)
    profiles = [solve_slot_profile(q, CEMENT, g.r_inner, ro) for ro in g.r_outer]
    u = g.normalise_to_flow_rate(g.map_velocity(profiles), q)

    raw = face_velocity_stack(u)
    before = np.max(np.abs(np.einsum("klm,klm->k", raw, g.face_area) - q) / q)
    after_uf = g.normalise_face_flux(raw, q)
    after = np.max(np.abs(np.einsum("klm,klm->k", after_uf, g.face_area) - q) / q)

    assert before > 1e-4, "test needs a case where face and cell areas differ"
    assert after < 1e-13


def test_unknown_mapping_raises(caliper):
    g = AnnulusGrid(LENGTH, CASING_OD, caliper, 10, 5, 4)
    profiles = [solve_slot_profile(0.01, CEMENT, g.r_inner, ro) for ro in g.r_outer]
    with pytest.raises(ValueError, match="unknown velocity mapping"):
        g.map_velocity(profiles, "nonsense")


# ---------------------------------------------------------------------------
# Hydraulics
# ---------------------------------------------------------------------------


def static_report(rho_casing, rho_annulus, n=100, tau=0.0):
    dz = LENGTH / n
    z = np.linspace(0.5 * dz, LENGTH - 0.5 * dz, n)
    return circulation_pressure(
        casing_z=z, casing_dz=dz, casing_rho=np.full(n, rho_casing),
        casing_tau_w=np.full(n, tau), casing_radius=0.5 * CASING_ID,
        annulus_z=z[::-1], annulus_dz=dz, annulus_rho=np.full(n, rho_annulus),
        annulus_tau_w=np.full(n, tau), annulus_half_gap=np.full(n, 0.019),
        inclination=0.0, gravity=9.80665, surface_pressure=0.0,
    )


def test_balanced_static_well_needs_no_pump_pressure():
    r = static_report(1200.0, 1200.0)
    assert r.pump_pressure == pytest.approx(0.0, abs=1e-9)
    assert r.utube_imbalance == pytest.approx(0.0, abs=1e-9)
    assert r.ecd_at_shoe == pytest.approx(1200.0, rel=1e-9)
    assert r.esd_at_shoe == pytest.approx(1200.0, rel=1e-9)
    assert not r.is_free_falling


def test_utube_imbalance_is_the_density_contrast_head():
    r = static_report(1870.0, 1198.0)
    assert r.utube_imbalance == pytest.approx(
        (1870.0 - 1198.0) * 9.80665 * LENGTH, rel=1e-12
    )
    assert r.is_free_falling, "a heavy casing column with no friction must free-fall"
    assert r.pump_pressure < 0.0


def test_friction_opposes_the_flow_in_both_legs():
    """Friction adds to the pump duty regardless of which leg it is in."""
    dry = static_report(1200.0, 1200.0, tau=0.0)
    wet = static_report(1200.0, 1200.0, tau=5.0)
    assert wet.pump_pressure > dry.pump_pressure
    assert wet.casing_friction > 0.0 and wet.annulus_friction > 0.0
    assert wet.pump_pressure == pytest.approx(
        wet.casing_friction + wet.annulus_friction, rel=1e-12
    )


def test_ecd_exceeds_esd_only_because_of_annular_friction():
    still = static_report(1200.0, 1200.0, tau=0.0)
    flowing = static_report(1200.0, 1200.0, tau=5.0)
    assert still.ecd_at_shoe == pytest.approx(still.esd_at_shoe, rel=1e-12)
    assert flowing.ecd_at_shoe > flowing.esd_at_shoe


def test_equivalent_mud_weight_round_trips():
    rho, depth, g = 1500.0, 200.0, 9.80665
    assert equivalent_mud_weight(rho * g * depth, depth, g) == pytest.approx(rho)
    assert math.isnan(equivalent_mud_weight(1.0, 0.0, g))


def test_shoe_pressure_is_the_casing_column(caplog):
    r = static_report(1500.0, 1500.0, tau=0.0)
    assert r.shoe_pressure == pytest.approx(1500.0 * 9.80665 * LENGTH, rel=1e-9)


# ---------------------------------------------------------------------------
# Coupled circulation
# ---------------------------------------------------------------------------


def build_solver(caliper, n_axial=60, n_layer=5, n_azimuth=4, excess=1.05,
                 **numerics):
    well = WellConfig(LENGTH, CASING_ID, CASING_OD, caliper)
    v_c = math.pi * (0.5 * CASING_ID) ** 2 * LENGTH
    v_a = AnnulusGrid(LENGTH, CASING_OD, caliper, n_axial, n_layer,
                      n_azimuth).total_volume
    schedule = PumpSchedule(
        [PumpStage(CEMENT, (v_c + v_a) * excess, bpm_to_m3s(5.0))]
    )
    solver = CirculationSolver(
        well, schedule, initial_fluid=MUD,
        grid=GridConfig(n_axial=n_axial, n_layer=n_layer, n_azimuth=n_azimuth),
        numerics=NumericsConfig(diagnostics_every=20, **numerics),
    )
    return solver, schedule


def test_well_config_validates():
    cal = synthetic_caliper(LENGTH, GAUGE)
    with pytest.raises(ValueError):
        WellConfig(LENGTH, CASING_OD, CASING_ID, cal)  # ID > OD


@pytest.fixture(scope="module")
def run(caliper):
    solver, schedule = build_solver(caliper)
    return solver.run(t_end=schedule.total_time, n_snapshots=6)


def test_circulation_conserves_and_stays_bounded(run):
    h = run.history
    assert h["sum_to_one_error"].max() < 1e-12
    assert h["mass_error"].max() < 1e-12
    for f in (run.casing_fractions, run.annulus_fractions):
        assert f.min() > -1e-12 and f.max() < 1.0 + 1e-12
        assert np.all(np.isfinite(f))


def test_cement_reaches_the_annulus_and_displaces_mud(run):
    i_cem = run.fluids.index(CEMENT)
    eff = run.annular_displacement_efficiency(i_cem)
    assert 0.5 < eff < 1.0, f"annular efficiency {eff:.3f}"
    # Monotone progress: the annulus fills as the job runs.
    series = run.history["annular_efficiency"]
    assert series[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(series) >= -1e-9)


def test_cement_is_richest_at_the_shoe_and_leanest_at_surface(run):
    i_cem = run.fluids.index(CEMENT)
    z, prof = run.annulus_profile(i_cem)
    assert np.all(np.diff(z) > 0), "profile must be in ascending depth"
    assert prof[-1] > prof[0], "the shoe end should lead the surface end"


def test_washouts_hold_more_mud_in_absolute_volume(run, caliper):
    """A washout leaves more mud behind simply by being bigger."""
    i_mud = run.fluids.index(MUD)
    g = run.annulus_grid
    order = np.argsort(g.z_centers)
    mud = (run.annulus_fractions[i_mud] * g.cell_volume).sum(axis=(1, 2))[order]
    vol = g.cell_volume.sum(axis=(1, 2))[order]
    hole = g.hole_diameter[order]
    wide = hole > 1.15 * caliper.gauge
    narrow = hole < 1.02 * caliper.gauge
    assert wide.any() and narrow.any(), "test needs both washed-out and gauge hole"
    # Per metre of hole, a washout retains more mud than an in-gauge section.
    assert mud[wide].mean() > mud[narrow].mean()
    assert vol[wide].mean() > vol[narrow].mean()


def test_a_wider_gap_flattens_the_profile(caliper):
    """The mechanism behind the counter-intuitive washout result.

    For the same imposed rate, a wider gap flows more slowly, so the yield
    stress dominates a larger share of the stress budget: the plug grows and
    the profile flattens toward slug flow.  A flatter profile displaces
    *better*, which is why washouts do not hurt efficiency in this model.
    """
    ri = 0.5 * CASING_OD
    q = bpm_to_m3s(5.0)
    shapes, plugs = [], []
    for hole in (GAUGE, 1.4 * GAUGE, 1.75 * GAUGE):
        p = solve_slot_profile(q, CEMENT, ri, 0.5 * hole)
        shapes.append(p.u_max / p.mean_velocity)
        plugs.append(p.plug_half_width / p.half_gap)
    assert shapes == sorted(shapes, reverse=True), shapes
    assert plugs == sorted(plugs), plugs
    assert shapes[0] > 1.2 and shapes[-1] < 1.15


def test_yield_diagnostic_reports_where_the_displaced_fluid_cannot_move(run):
    """The model does not enforce the yield criterion, so it must report it."""
    d = run.yield_diagnostic(MUD)
    assert d["tau_w"].shape == (run.annulus_grid.n_axial,)
    assert np.all(np.diff(d["depth"]) > 0)
    assert 0.0 <= d["volume_fraction_below"] <= 1.0
    # With this fluid pair the cement's own yield stress floors tau_w well
    # above the mud's, so nothing is stranded here.
    assert d["min_tau_w"] > MUD.tau0
    assert d["volume_fraction_below"] == pytest.approx(0.0)
    # A fluid with a very high yield stress would be stranded everywhere.
    stiff = Fluid("stiff", 1200.0, 1e4, 0.3, 0.7)
    assert run.yield_diagnostic(stiff)["volume_fraction_below"] == pytest.approx(1.0)


def test_shoe_handover_composition_sums_to_one(run):
    shoe = run.history["shoe_fraction"]
    # The first record predates any step, so skip it.
    np.testing.assert_allclose(shoe[1:].sum(axis=1), 1.0, atol=1e-12)


def test_returns_reach_surface_and_sum_to_one(run):
    returns = run.history["returns_fraction"]
    np.testing.assert_allclose(returns.sum(axis=1), 1.0, atol=1e-12)
    i_cem = run.fluids.index(CEMENT)
    assert returns[0, i_cem] == pytest.approx(0.0, abs=1e-12)


def test_snapshots_are_recorded_for_animation(run):
    assert len(run.snapshots) >= 5
    times = [s["time"] for s in run.snapshots]
    assert times == sorted(times)
    for s in run.snapshots:
        assert s["casing"].shape == run.casing_fractions.shape
        assert s["annulus"].shape == run.annulus_fractions.shape


def test_free_fall_is_detected_and_reported(run):
    """The honest limit of an imposed-rate model, made visible.

    Heavy cement over lighter mud in a vertical well is U-tube unstable; the
    model does not act on that, but it must say so.
    """
    h = run.history
    assert h["utube_imbalance"].max() > 5e5, "expected a large imbalance"
    assert (h["pump_pressure"] < 0.0).any(), "expected free-fall to be flagged"


def test_ecd_rises_as_cement_fills_the_annulus(run):
    ecd = run.history["ecd_at_shoe"]
    assert ecd[0] == pytest.approx(MUD.rho, rel=0.06)
    assert ecd[-1] > 0.9 * CEMENT.rho
    assert ecd.max() <= CEMENT.rho * 1.05


def test_washouts_do_not_degrade_efficiency_in_this_model(smooth_caliper, caliper):
    """A finding, not an expectation - and a limit of the phase.

    Field experience says washouts cement badly.  This model says the opposite,
    by a small margin, and the reason is instructive: with a *concentric*
    annulus, no buoyancy and no yield check on the displaced fluid, the only
    thing a washout changes is the local shear rate, and lowering it makes a
    yield-stress fluid's profile flatter and therefore better at displacing
    (see test_a_wider_gap_flattens_the_profile).

    Every mechanism that makes real washouts bad is outside this phase:
    eccentricity, density segregation into the cavity, and mud left below its
    yield stress.  This test pins the current behaviour so that adding any of
    those is visible as a change.
    """
    results = {}
    for label, cal in (("smooth", smooth_caliper), ("rough", caliper)):
        solver, schedule = build_solver(cal, n_axial=40, n_layer=5, n_azimuth=4)
        r = solver.run(t_end=schedule.total_time)
        results[label] = r.annular_displacement_efficiency(r.fluids.index(CEMENT))
    assert results["rough"] >= results["smooth"] - 1e-3, results
    # Both are respectable, so the comparison is between sane numbers.
    assert all(0.7 < v < 1.0 for v in results.values()), results


# --- modelling only a sub-interval of the well ------------------------------


def test_top_depth_keeps_true_depths_and_samples_the_caliper_there(caliper):
    """A sub-interval model must not renumber depth from zero.

    The caliper is a function of true depth, and ECD divides by it, so an
    interval starting at 175 m has to know that.
    """
    top = 60.0
    well = WellConfig(LENGTH - top, CASING_ID, CASING_OD, caliper, top_depth=top)
    assert well.shoe_depth == pytest.approx(LENGTH)
    solver = CirculationSolver(
        well, PumpSchedule([PumpStage(CEMENT, 1.0, bpm_to_m3s(5.0))]),
        initial_fluid=MUD,
        grid=GridConfig(n_axial=30, n_layer=5, n_azimuth=4),
    )
    cg, ag = solver.casing_grid, solver.annulus_grid
    assert cg.z_faces[0] == pytest.approx(top)
    assert cg.z_faces[-1] == pytest.approx(LENGTH)
    assert ag.z_centers.min() > top and ag.z_centers.max() < LENGTH
    # The hole diameters must match the caliper at those true depths.
    np.testing.assert_allclose(
        ag.hole_diameter, caliper.diameter_at(ag.z_centers), rtol=1e-12
    )


def test_head_above_the_interval_restores_true_depth_pressures():
    """Without the column above, ECD at the shoe is badly understated."""
    n, top = 60, 175.0
    length = 211.75
    dz = length / n
    z = np.linspace(top + 0.5 * dz, top + length - 0.5 * dz, n)
    rho = np.full(n, 1200.0)
    kwargs = dict(
        casing_z=z, casing_dz=dz, casing_rho=rho, casing_tau_w=np.zeros(n),
        casing_radius=0.5 * CASING_ID, annulus_z=z[::-1], annulus_dz=dz,
        annulus_rho=rho, annulus_tau_w=np.zeros(n),
        annulus_half_gap=np.full(n, 0.022), gravity=9.80665, top_depth=top,
    )
    with_column = circulation_pressure(
        rho_above_casing=1200.0, rho_above_annulus=1200.0, **kwargs
    )
    without = circulation_pressure(**kwargs)
    assert with_column.ecd_at_shoe == pytest.approx(1200.0, rel=1e-9)
    assert with_column.shoe_pressure == pytest.approx(
        1200.0 * 9.80665 * (top + length), rel=1e-9
    )
    assert with_column.pump_pressure == pytest.approx(0.0, abs=1e-9)
    assert without.ecd_at_shoe < 0.7 * with_column.ecd_at_shoe
    assert with_column.top_depth == pytest.approx(top)


def test_auto_column_above_turns_over_as_the_job_runs(caliper):
    """The casing above starts as the in-situ fluid and becomes the pumped one."""
    top = 100.0
    well = WellConfig(LENGTH - top, CASING_ID, CASING_OD, caliper, top_depth=top,
                      rho_above_casing="auto", rho_above_annulus=MUD.rho)
    bore = math.pi * (0.5 * CASING_ID) ** 2 * top
    q = bpm_to_m3s(5.0)
    solver = CirculationSolver(
        well, PumpSchedule([PumpStage(CEMENT, bore * 10.0, q)]), initial_fluid=MUD,
        grid=GridConfig(n_axial=30, n_layer=5, n_azimuth=4),
    )
    assert solver._rho_above_casing() == pytest.approx(MUD.rho, rel=1e-12)
    solver.t = 0.5 * bore / q          # half the column turned over
    mid = solver._rho_above_casing()
    assert MUD.rho < mid < CEMENT.rho
    assert mid == pytest.approx(0.5 * (MUD.rho + CEMENT.rho), rel=1e-9)
    solver.t = 3.0 * bore / q          # fully turned over
    assert solver._rho_above_casing() == pytest.approx(CEMENT.rho, rel=1e-12)


def test_local_efficiency_is_confounded_by_arrival_order(caliper):
    """Why a raw washout-vs-gauge comparison cannot be trusted.

    Annular flow is upward, so a shallow cell is reached last and reads low at
    the end of the job whatever its diameter.  Local efficiency therefore
    correlates strongly with depth, and comparing washouts against gauge hole
    without controlling for that measures where the front happens to be - it
    can and does flip the sign of the conclusion.

    On the K-GEP-1 interval the raw comparison says washouts are worse (0.880
    against 0.939) purely because the big washout sits near the top; comparing
    within depth bands reverses it (0.818 vs 0.775 shallow, 0.995 vs 0.978
    deep).  The case script reports both, and this test pins the confound so
    the raw number is never read on its own.
    """
    solver, schedule = build_solver(caliper, n_axial=60, n_layer=5, n_azimuth=4)
    r = solver.run(t_end=schedule.total_time)
    g = r.annulus_grid
    order = np.argsort(g.z_centers)
    z = g.z_centers[order]
    vol = g.cell_volume.sum(axis=(1, 2))[order]
    i_cem = r.fluids.index(CEMENT)
    eff = (r.annulus_fractions[i_cem] * g.cell_volume).sum(axis=(1, 2))[order] / vol

    assert eff[0] < eff[-1], "the shallow end must lag the deep end"
    assert np.corrcoef(z, eff)[0, 1] > 0.8, "expected a strong depth trend"
    # Deep cells are essentially fully swept, shallow ones are not.
    assert eff[-5:].mean() > 0.95
    assert eff[:5].mean() < eff[-5:].mean()


def test_vectorised_effective_matches_mix_fluids(caliper):
    """The time loop's averaging must equal the reference rule, on both legs."""
    from inpipe.fluid import mix_fluids

    solver, schedule = build_solver(caliper, n_axial=30, n_layer=5, n_azimuth=4)
    for _ in range(20):
        solver.step()
    for leg, fields, volume in (
        ("casing", solver.f_casing, solver.casing_grid.cell_volume),
        ("annulus", solver.f_annulus, solver.annulus_grid.cell_volume),
    ):
        params = solver._effective(fields, volume, solver._params)
        for k in (0, 7, 15, 29):
            ref = solver.effective_fluid(leg, k)
            got = params[k]
            assert got[0] == pytest.approx(ref.rho, rel=1e-12)
            assert got[1] == pytest.approx(ref.tau0, rel=1e-12, abs=1e-15)
            assert got[2] == pytest.approx(ref.k, rel=1e-12)
            assert got[3] == pytest.approx(ref.n, rel=1e-12)
    with pytest.raises(ValueError, match="leg must be"):
        solver.effective_fluid("nonsense", 0)
    assert mix_fluids is not None
