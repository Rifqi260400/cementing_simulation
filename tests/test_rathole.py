"""The rat hole - open hole below the shoe, where the cement turns around."""

import numpy as np
import pytest

from inpipe.caliper import CaliperLog, synthetic_caliper
from inpipe.circulation import WellConfig
from inpipe.config import inch_to_m


def _caliper(length=200.0, diameter=None):
    return synthetic_caliper(length, inch_to_m(8.5) if diameter is None else diameter)


def _well(rat_hole_length, length=100.0):
    return WellConfig(length, inch_to_m(6.184), inch_to_m(7.0), _caliper(),
                      rat_hole_length=rat_hole_length, top_depth=0.0)


# --- geometry ---------------------------------------------------------------


def test_the_shoe_sits_above_total_depth():
    """The casing does not land on bottom; that gap is the rat hole."""
    well = _well(5.0)
    assert well.shoe_depth == pytest.approx(100.0)
    assert well.total_depth == pytest.approx(105.0)
    assert well.total_depth - well.shoe_depth == pytest.approx(well.rat_hole_length)


def test_the_rat_hole_is_full_bore():
    """No casing in it, so it holds the whole hole area, not an annulus.

    Getting this wrong the other way - charging it the annular area - would
    under-count the volume by the casing's own cross-section, which here is
    most of it.
    """
    diameter = inch_to_m(8.5)
    uniform = CaliperLog(np.array([0.0, 200.0]), np.full(2, diameter))
    well = WellConfig(100.0, inch_to_m(6.184), inch_to_m(7.0), uniform,
                      rat_hole_length=5.0, top_depth=0.0)
    expected = 0.25 * np.pi * diameter**2 * 5.0
    assert well.rat_hole_volume == pytest.approx(expected, rel=1e-9)

    annular = 0.25 * np.pi * (diameter**2 - inch_to_m(7.0) ** 2) * 5.0
    assert well.rat_hole_volume > 2.0 * annular


def test_no_rat_hole_is_the_default_and_costs_nothing():
    well = _well(0.0)
    assert well.rat_hole_volume == 0.0
    assert well.total_depth == well.shoe_depth


def test_a_negative_rat_hole_is_refused():
    with pytest.raises(ValueError, match="rat_hole_length"):
        _well(-1.0)


# --- the mixing volume ------------------------------------------------------


@pytest.fixture(scope="module")
def ratted_job():
    from cases.circulation import build

    caliper = synthetic_caliper(60.0, inch_to_m(8.5))
    solver, schedule, *_ = build(caliper, n_axial=30, top_depth=0.0,
                                 rat_hole_length=5.0)
    return solver, solver.run(t_end=schedule.total_time, n_snapshots=0), schedule


def test_the_rat_hole_holds_volume_and_conserves_it(ratted_job):
    """It is part of the well, so the volume audit has to include it.

    Left out, the well appears to hold less than it does and the volume error
    reports the rat hole as a leak.
    """
    solver, result, _ = ratted_job
    assert result.rathole_volume > 0.0
    assert result.rathole_fractions.sum() == pytest.approx(1.0)
    assert result.history["mass_error"].max() < 1e-12
    assert result.history["sum_to_one_error"].max() < 1e-12


def test_the_annulus_is_fed_by_the_rat_hole_not_the_shoe(ratted_job):
    """Cement must pass through the gap before it can rise.

    With a rat hole in the way the annulus cannot see pure cement on the step
    the casing first delivers it - the rat hole is still full of mud and dilutes
    it.  If the annulus inlet ever matches the casing outlet exactly while the
    rat hole is still muddy, the gap has been bypassed.
    """
    solver, result, _ = ratted_job
    shoe = result.history["shoe_fraction"]
    cement = shoe[:, -1]
    arrived = np.argmax(cement > 0.01)
    assert cement[arrived] > 0.0, "cement never reached the shoe in this run"
    # The rat hole ends the job cleaner than it started but still mixed on the
    # way: a dead end purges exponentially, it does not switch over.
    assert 0.0 < result.rathole_fractions[-1] < 1.0


def test_a_rat_hole_delays_the_front_without_being_counted_twice(ratted_job):
    """The delay comes from the geometry now, not from an added offset.

    An earlier version had no rat hole in the mesh and shifted the arrival
    curves by its pumping time.  Now the solver holds the volume, so the front
    waits for it by itself; shifting as well would count it twice.
    """
    from cases.circulation import build

    solver, ratted, schedule = ratted_job
    caliper = synthetic_caliper(60.0, inch_to_m(8.5))
    plain_solver, plain_schedule, *_ = build(caliper, n_axial=30, top_depth=0.0,
                                             rat_hole_length=0.0)
    plain = plain_solver.run(t_end=plain_schedule.total_time, n_snapshots=0)

    delay = ratted.arrival.rat_hole_delay
    assert delay > 0.0 and plain.arrival.rat_hole_delay == 0.0

    # The front is later with the rat hole, by roughly the pumping time of that
    # volume - not by twice it.
    shift = ratted.arrival.shoe_arrival - plain.arrival.shoe_arrival
    assert 0.3 * delay < shift < 1.9 * delay


def test_the_volumetric_curve_still_carries_the_rat_hole(ratted_job):
    """It is a hand calculation with no rat hole of its own, so it needs one.

    This is the one curve that must be offset explicitly - Eq. 2 of Hart et al.
    (2025) puts the stinger and rat hole volume in front of the annulus sum.
    """
    _, result, _ = ratted_job
    report = result.arrival
    assert np.all(report.volumetric > report.rat_hole_delay)


# --- it has to be visible, not just present --------------------------------


def test_the_section_image_draws_the_rat_hole(ratted_job):
    """A picture that stops at the shoe is misleading, not merely incomplete.

    The rat hole is the space the cement turns around in, so leaving it out
    shows cement appearing in the annulus from nowhere.  It must be drawn below
    the shoe, at the full hole width - there is no casing down there.
    """
    from inpipe.wellview import well_section_image

    _, result, _ = ratted_job
    img, extent = well_section_image(result, len(result.fluids) - 1)
    z_min, z_max = extent[3], extent[2]
    assert z_max == pytest.approx(result.rathole_bottom, abs=0.5)
    assert result.rathole_bottom > result.rathole_top

    # The bottom row is fluid across a width wider than the casing bore, since
    # the rat hole is full bore.
    bottom = img[-1]
    filled = np.isfinite(bottom)
    assert filled.sum() > 0
    x = np.linspace(extent[0], extent[1], bottom.size)
    assert np.abs(x[filled]).max() > 0.5 * result.annulus_grid.casing_od

    # And it carries the rat hole's own composition, not the annulus's.
    assert np.allclose(bottom[filled], result.rathole_fractions[-1])


def test_snapshots_carry_the_rat_hole_for_the_animation(ratted_job):
    """The video steps through snapshots, so they need it too.

    Without it every frame would fall back to the final composition and the rat
    hole would appear already cemented from the first frame.
    """
    from cases.circulation import build

    caliper = synthetic_caliper(60.0, inch_to_m(8.5))
    solver, schedule, *_ = build(caliper, n_axial=20, top_depth=0.0,
                                 rat_hole_length=5.0)
    result = solver.run(t_end=schedule.total_time, n_snapshots=6)
    assert all("rathole" in snap for snap in result.snapshots)
    first = result.snapshots[0]["rathole"]
    last = result.snapshots[-1]["rathole"]
    assert first[0] == pytest.approx(1.0)        # starts full of mud
    assert last[-1] > first[-1]                  # and takes cement as it runs
