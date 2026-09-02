"""Residual displaced fluid, and how much of it the enlargements hold."""

import math

import numpy as np
import pytest

from inpipe.annulus_grid import AnnulusGrid
from inpipe.caliper import synthetic_caliper
from inpipe.circulation import CirculationSolver, WellConfig
from inpipe.config import GridConfig, NumericsConfig, bpm_to_m3s, inch_to_m
from inpipe.fluid import Fluid, PumpSchedule, PumpStage
from inpipe.mudleft import mud_left_behind

LENGTH = 200.0
CASING_ID = inch_to_m(6.184)
CASING_OD = inch_to_m(7.0)
GAUGE = inch_to_m(10.43)
MUD = Fluid("mud", 1198.0, 2.0, 0.30, 0.72)
CEMENT = Fluid("cement", 1870.0, 6.0, 0.55, 0.65)


def run(caliper, excess=1.05, n_axial=50, cement_volume=None):
    well = WellConfig(LENGTH, CASING_ID, CASING_OD, caliper)
    v_c = math.pi * (0.5 * CASING_ID) ** 2 * LENGTH
    v_a = AnnulusGrid(LENGTH, CASING_OD, caliper, n_axial, 5, 4).total_volume
    pumped = (v_c + v_a) * excess if cement_volume is None else cement_volume
    solver = CirculationSolver(
        well, PumpSchedule([PumpStage(CEMENT, pumped, bpm_to_m3s(5.0))]),
        initial_fluid=MUD,
        grid=GridConfig(n_axial=n_axial, n_layer=5, n_azimuth=4),
        numerics=NumericsConfig(diagnostics_every=50),
    )
    result = solver.run()
    return result, mud_left_behind(result, MUD, gauge=caliper.gauge), pumped


@pytest.fixture(scope="module")
def rough():
    return synthetic_caliper(LENGTH, GAUGE)


@pytest.fixture(scope="module")
def smooth():
    return synthetic_caliper(LENGTH, GAUGE, washouts=(), tight_zones=(), roughness=0.0)


@pytest.fixture(scope="module")
def rough_run(rough):
    return run(rough)


def test_report_shares_are_consistent(rough_run):
    _, rep, _ = rough_run
    assert rep.mud_volume.shape == rep.depth.shape == rep.hole_diameter.shape
    assert np.all(np.diff(rep.depth) > 0)
    assert rep.total_mud == pytest.approx(rep.mud_volume.sum())
    assert rep.efficiency == pytest.approx(1.0 - rep.total_mud / rep.total_volume)
    assert 0.0 <= rep.washout_volume_share <= 1.0
    assert 0.0 <= rep.washout_mud_share <= 1.0
    assert np.all(rep.local_fraction <= 1.0 + 1e-12)
    assert np.all(rep.local_fraction >= -1e-12)


def test_concentration_ratio_is_well_specific_not_a_law(rough_run):
    """Washouts do not universally hold more than their share of the residual.

    On K-GEP-1 they hold 1.55 x their volume share; on this synthetic log,
    whose enlargements sit mid-well rather than at the shallow end, the ratio
    is below 1.  Where the front has got to by the end of the job matters as
    much as the geometry, so the ratio is a property of a *well and a job*,
    never a general statement about washouts.
    """
    _, rep, _ = rough_run
    assert rep.is_washout.any(), "test needs a washed-out interval"
    assert 0.0 < rep.concentration_ratio < 5.0
    # The two shares it is built from must be self-consistent.
    w = rep.is_washout
    assert rep.washout_mud_share == pytest.approx(
        rep.mud_volume[w].sum() / rep.total_mud, rel=1e-12
    )
    assert rep.washout_volume_share == pytest.approx(
        rep.cell_volume[w].sum() / rep.total_volume, rel=1e-12
    )


def test_concentration_ratio_is_stable_across_pumped_volume(rough):
    """It measures the geometry's share, not just how far the front has got.

    If the ratio were purely an arrival-order artefact it would drift as the
    front sweeps past the enlargements.  It does not: the share of the residual
    sitting in washed-out hole is roughly constant while the residual itself
    falls by two orders of magnitude.
    """
    shares = []
    for excess in (1.05, 1.4, 2.0):
        _, rep, _ = run(rough, excess=excess)
        shares.append(rep.washout_mud_share)
    assert max(shares) - min(shares) < 0.15, shares


def test_a_smooth_hole_has_no_washouts_and_no_concentration(smooth):
    _, rep, _ = run(smooth)
    assert not rep.is_washout.any()
    assert rep.washout_volume_share == pytest.approx(0.0)
    assert math.isnan(rep.concentration_ratio)


def test_nothing_is_stranded_permanently(rough):
    """Circulating longer clears the annulus - there is no trapping mechanism.

    This is the limit of the model, and the reason residual mud here always
    means "not yet swept at the volume pumped", never "stuck".
    """
    residuals = [run(rough, excess=e)[1].total_mud for e in (1.05, 1.6, 2.5)]
    assert residuals == sorted(residuals, reverse=True), residuals
    assert residuals[-1] < 0.02 * residuals[0]


def test_ignoring_the_caliper_leaves_far_more_mud(rough, smooth):
    """Sizing a job on bit size instead of the caliper is the real cost."""
    _, rep_gauge, gauge_volume = run(smooth)
    _, rep_sized, _ = run(rough)
    _, rep_under, _ = run(rough, cement_volume=gauge_volume)
    assert rep_under.total_mud > 1.5 * rep_sized.total_mud
    assert rep_under.efficiency < rep_sized.efficiency - 0.1


def test_worst_intervals_are_ordered_and_inside_the_well(rough_run):
    _, rep, _ = rough_run
    worst = rep.worst_intervals(n=3)
    assert worst, "expected at least one flagged interval"
    volumes = [w[2] for w in worst]
    assert volumes == sorted(volumes, reverse=True)
    for top, bottom, mud, dia in worst:
        assert rep.depth[0] - 10 <= top < bottom <= rep.depth[-1] + 10
        assert mud > 0.0 and dia > 0.0


def test_excess_over_gauge_differences(rough, smooth):
    _, rep_r, _ = run(rough)
    _, rep_s, _ = run(smooth)
    assert rep_r.excess_over_gauge(rep_s) == pytest.approx(
        rep_r.total_mud - rep_s.total_mud
    )


def test_summary_mentions_the_key_numbers(rough_run):
    _, rep, _ = rough_run
    text = rep.summary()
    for key in ("residual mud", "washed-out hole", "concentration", "worst intervals"):
        assert key in text


def test_report_handles_a_fully_displaced_annulus(rough):
    _, rep, _ = run(rough, excess=4.0)
    assert rep.total_mud < 1e-6
    assert rep.efficiency > 0.999
    assert math.isnan(rep.washout_mud_share) or rep.washout_mud_share >= 0.0


# --- the bundled field log --------------------------------------------------

from pathlib import Path  # noqa: E402

FIELD_LOG = Path(__file__).resolve().parent.parent / "data" / "K-GEP-1_composite.las"


@pytest.mark.slow
@pytest.mark.skipif(not FIELD_LOG.exists(), reason="field log not bundled")
def test_k_gep_1_washouts_hold_more_than_their_share():
    """The actual result on the actual well, pinned.

    Not a general law - see test_concentration_ratio_is_well_specific_not_a_law
    - but it is what this well does, and it is the number the case reports.
    """
    from cases.circulation import MUD as CASE_MUD
    from cases.circulation import build, load_caliper

    caliper, _ = load_caliper(verbose=False)
    solver, schedule, *_ = build(caliper, n_axial=80, excess=1.05)
    result = solver.run(t_end=schedule.total_time)
    rep = mud_left_behind(result, CASE_MUD, gauge=caliper.gauge)
    assert rep.concentration_ratio > 1.2, rep.summary()
    assert rep.washout_volume_share > 0.3
    assert rep.washout_mud_share > rep.washout_volume_share
