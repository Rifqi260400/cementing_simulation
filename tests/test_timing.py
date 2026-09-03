"""Cement arrival against depth - the rising-time output."""

from dataclasses import dataclass

import numpy as np
import pytest

from inpipe.caliper import synthetic_caliper
from inpipe.config import inch_to_m
from inpipe.timing import ArrivalTracker


@dataclass
class _Stations:
    """Minimal stand-in for an annulus grid: equal cells, flow order."""

    n_axial: int
    volume: float = 1.0

    def __post_init__(self):
        self.cell_volume = np.full((self.n_axial, 1, 1), self.volume)
        # Flow order: index 0 at the shoe, so depth descends with index.
        self.z_centers = np.linspace(self.n_axial, 1.0, self.n_axial)


def _fractions(values):
    """Shape a per-station cement fraction into a two-fluid field."""
    cement = np.asarray(values, dtype=float).reshape(-1, 1, 1)
    return np.stack([1.0 - cement, cement])


# --- the tracker itself -----------------------------------------------------


def test_crossing_time_is_interpolated_between_steps():
    """Arrival must not be quantised to the timestep.

    A field job runs thousands of steps; rounding each crossing up to the next
    one would put a systematic half-step bias into every arrival time.
    """
    tracker = ArrivalTracker(_Stations(1), fluid_index=1, thresholds=(0.5,))
    tracker.update(0.0, _fractions([0.0]))
    tracker.update(10.0, _fractions([0.25]))     # still below
    tracker.update(20.0, _fractions([0.75]))     # crosses midway
    report = tracker.report(job_time=20.0)
    assert report.at(0.5)[0] == pytest.approx(15.0)


def test_first_crossing_wins():
    """Fractions are not monotonic; arrival is the *first* crossing."""
    tracker = ArrivalTracker(_Stations(1), fluid_index=1, thresholds=(0.5,))
    for t, f in ((0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)):
        tracker.update(t, _fractions([f]))
    assert tracker.report(3.0).at(0.5)[0] == pytest.approx(0.5)


def test_depths_never_reached_are_nan_not_zero():
    """A station cement never reached must be absent, not "arrived at t = 0"."""
    tracker = ArrivalTracker(_Stations(3), fluid_index=1, thresholds=(0.5,))
    tracker.update(0.0, _fractions([0.0, 0.0, 0.0]))
    tracker.update(1.0, _fractions([1.0, 0.0, 0.0]))
    # The report is sorted by ascending depth and the cells are in flow order,
    # so the station that got cement - index 0, at the shoe - sorts last.
    arrival = tracker.report(1.0).at(0.5)
    assert np.isnan(arrival[0]) and np.isnan(arrival[1])
    assert arrival[-1] == pytest.approx(0.5)


def test_report_is_ordered_by_ascending_depth():
    """The CSV is read against a depth axis, so it must be sorted by depth."""
    tracker = ArrivalTracker(_Stations(5), fluid_index=1, thresholds=(0.5,))
    tracker.update(0.0, _fractions(np.zeros(5)))
    report = tracker.report(0.0)
    assert np.all(np.diff(report.depth) > 0.0)
    assert report.shoe_depth > report.top_depth


def test_volumetric_arrival_counts_the_casing_first():
    """Nothing reaches the annulus until the casing has been displaced."""
    tracker = ArrivalTracker(_Stations(2, volume=3.0), fluid_index=1,
                             thresholds=(0.5,), casing_volume=10.0)
    tracker.update(0.0, _fractions([0.0, 0.0]), 0.0)
    tracker.update(1.0, _fractions([0.0, 0.0]), 13.0)   # fills casing + first cell
    tracker.update(2.0, _fractions([0.0, 0.0]), 16.0)   # and the second
    volumetric = tracker.report(2.0).volumetric          # ascending depth
    assert volumetric[-1] == pytest.approx(1.0)          # shoe cell
    assert volumetric[0] == pytest.approx(2.0)


# --- against the solver -----------------------------------------------------


@pytest.fixture(scope="module")
def short_job():
    from cases.circulation import build

    caliper = synthetic_caliper(40.0, inch_to_m(8.5))
    solver, schedule, length, v_casing, v_annulus, shoe = build(
        caliper, n_axial=30, top_depth=0.0)
    result = solver.run(t_end=schedule.total_time, n_snapshots=0)
    return result, schedule, v_casing


def test_volumetric_arrival_reproduces_the_hand_calculation(short_job):
    """It must be exactly volume over rate, or it is not a reference.

    The point of reporting it is that it is the number an engineer computes
    without a simulator.  If it drifts from that it is just a second opinion
    from the same model, and comparing the front against it means nothing.
    """
    result, schedule, v_casing = short_job
    g = result.annulus_grid
    order = np.argsort(g.z_centers)
    below = np.cumsum(g.cell_volume.sum(axis=(1, 2))[order][::-1])[::-1]
    expected = (v_casing + below) / schedule.stages[0].flow_rate

    got = result.arrival.volumetric
    assert np.all(np.isfinite(got))
    assert got == pytest.approx(expected, rel=1e-9)


def test_the_front_runs_ahead_of_the_volume_balance(short_job):
    """And by a margin worth reporting, not round-off.

    Half the annular area moves faster than the mean, so the interface reaches
    a depth before the volume balance says it should.  This is the whole reason
    the model gives a different answer from the hand calculation.
    """
    result, _, _ = short_job
    report = result.arrival
    reached = report.reached
    assert np.all(report.front[reached] < report.volumetric[reached])
    lead = -report.front_lead()[reached]
    assert 0.02 < np.mean(lead) < 0.40


def test_arrival_is_earlier_deeper(short_job):
    """Cement rises, so a deeper station must be reached first."""
    result, _, _ = short_job
    front = result.arrival.front          # ascending depth
    reached = front[np.isfinite(front)]
    assert np.all(np.diff(reached) < 0.0)


def test_thresholds_arrive_in_order(short_job):
    """0.1 before 0.5 before 0.9 - otherwise the mixing zone is meaningless.

    Each contour is masked separately: displacement efficiency is below one, so
    there are depths the 0.5 contour reaches and the 0.9 contour never does.
    That is a result about the job, not a gap to paper over - it means those
    depths end the job still holding more than 10 % mud.
    """
    report = short_job[0].arrival
    low, mid, high = (report.at(t) for t in (0.1, 0.5, 0.9))
    both = np.isfinite(low) & np.isfinite(mid)
    assert np.any(both) and np.all(low[both] <= mid[both])
    both = np.isfinite(mid) & np.isfinite(high)
    assert np.any(both) and np.all(mid[both] <= high[both])
    width = report.mixing_zone_duration()
    assert np.all(width[np.isfinite(width)] >= 0.0)
    assert np.any(np.isnan(high)), "this well should not be fully displaced"


def test_rising_time_is_the_shoe_to_top_travel(short_job):
    report = short_job[0].arrival
    assert report.rising_time == pytest.approx(
        report.top_arrival - report.shoe_arrival)
    assert report.rising_time > 0.0


def test_csv_carries_every_curve(short_job, tmp_path):
    report = short_job[0].arrival
    path = tmp_path / "arrival.csv"
    report.to_csv(path)
    header = path.read_text().splitlines()[0].split(",")
    assert header == ["depth_m", "arrival_f0.1_s", "arrival_f0.5_s",
                      "arrival_f0.9_s", "arrival_volumetric_s",
                      "arrival_in_gauge_s", "front_envelope_s",
                      "front_rise_velocity_m_per_min"]
    body = np.loadtxt(path, delimiter=",", skiprows=1)
    assert body.shape == (report.depth.size, len(header))


def test_tracking_can_be_switched_off(short_job):
    """It costs a reduction per step; a caller must be able to decline it."""
    from cases.circulation import build

    caliper = synthetic_caliper(40.0, inch_to_m(8.5))
    solver, schedule, *_ = build(caliper, n_axial=12, top_depth=0.0)
    result = solver.run(t_end=0.5 * schedule.total_time, track_arrival=False)
    assert result.arrival is None


# --- the Hart et al. (2025) comparison curves -------------------------------


def test_in_gauge_arrival_is_faster_than_the_caliper_one(short_job):
    """The "fast rise" bound of Hart et al. (2025).

    An in-gauge hole holds less than a washed-out one, so the same pumped
    volume carries the interface higher.  The two curves bracket the field
    observation: their cement, being denser than the mud, filled the whole
    caliper volume and tracked the slow curve, while their lighter freshwater
    spacer took the path of least resistance and tracked the fast one.
    """
    report = short_job[0].arrival
    assert report.in_gauge is not None
    # Aggregate: an in-gauge hole holds less in total, so the top is reached
    # sooner.  Station by station it is not a bound, because the caliper reads
    # *under* gauge over part of the interval and the real hole holds less
    # there - which is why a true bound needs the bit diameter, not the log's
    # own median.
    assert report.in_gauge[0] < report.volumetric[0]
    assert np.mean(report.in_gauge < report.volumetric) > 0.5


def test_rise_velocity_is_flat_in_gauge_and_dips_in_washouts(short_job):
    """Their Fig. 2b: geometry read straight off the velocity curve.

    On an in-gauge hole the rise velocity is Q/A, a constant.  The caliper
    curve must be slower wherever the hole is wider, since the same rate has
    more area to fill.
    """
    result = short_job[0]
    report = result.arrival
    flat = report.rise_velocity(report.in_gauge)
    flat = flat[np.isfinite(flat)]
    assert np.ptp(flat) / np.mean(flat) < 1e-9

    # Wherever the hole is wider than gauge the interface must climb slower,
    # and wherever it is narrower, faster.  Both directions, or the curve is
    # not reading the geometry.
    g = result.annulus_grid
    order = np.argsort(g.z_centers)
    wider = g.hole_diameter[order] > 1.05 * g.caliper.gauge
    narrower = g.hole_diameter[order] < 0.95 * g.caliper.gauge
    real = report.rise_velocity(report.volumetric)
    inner = np.zeros(real.shape, dtype=bool)
    inner[2:-2] = True                      # gradient is one-sided at the ends
    assert np.all(real[wider & inner] < flat.mean())
    assert np.all(real[narrower & inner] > flat.mean())


def test_rise_velocity_recovers_rate_over_area(short_job):
    """It must equal Q/A, or it is not the quantity the paper plots."""
    result, schedule, _ = short_job
    report = result.arrival
    g = result.annulus_grid
    gauge_area = 0.25 * np.pi * (g.caliper.gauge**2 - g.casing_od**2)
    expected = 60.0 * schedule.stages[0].flow_rate / gauge_area
    got = report.rise_velocity(report.in_gauge)
    assert got[np.isfinite(got)] == pytest.approx(expected, rel=1e-9)


def test_a_rat_hole_delays_every_curve_by_its_own_volume():
    """Hart et al. had to assume a 10 m3 rat hole to make the times line up.

    It is open hole below the shoe, so it fills before the annulus starts to
    rise.  The mesh does not contain it, so the delay is applied uniformly -
    and it must be the pumping time for that volume, not a fitted offset.
    """
    stations = _Stations(2, volume=1.0)
    rate = 2.0                                   # m^3/s, so 5 m^3 takes 2.5 s
    plain = ArrivalTracker(stations, 1, thresholds=(0.5,), casing_volume=1.0)
    ratted = ArrivalTracker(stations, 1, thresholds=(0.5,), casing_volume=1.0,
                            rat_hole_volume=5.0)
    for tracker in (plain, ratted):
        for step in range(9):
            t = 0.5 * step
            cement = [1.0, 0.0] if t >= 1.0 else [0.0, 0.0]
            tracker.update(t, _fractions(cement), rate * t)
    delay = ratted.report(4.0).rat_hole_delay
    assert delay == pytest.approx(5.0 / rate)
    shifted = ratted.report(4.0).at(0.5) - plain.report(4.0).at(0.5)
    assert shifted[np.isfinite(shifted)] == pytest.approx(delay)


def test_the_front_envelope_is_monotonic_where_the_front_is_not(short_job):
    """Channelling makes the raw front non-monotonic; the envelope must not be.

    Cement reaches a shallower station through the narrow side of a washout
    before the wide station is half displaced, so the raw arrival curve doubles
    back.  Differencing that gives negative rise velocities - on K-GEP-1 it ran
    to -167 m/min.  The envelope is the leading edge, which is what an operator
    traces on a DAS waterfall, and its velocity is non-negative by construction.
    """
    report = short_job[0].arrival
    envelope = report.front_envelope
    finite = np.isfinite(envelope)
    assert np.all(np.diff(envelope[finite]) <= 1e-9)   # earlier as it gets deeper
    speed = report.rise_velocity(envelope)
    assert np.all(speed[np.isfinite(speed)] >= 0.0)
    # The envelope never claims a later arrival than the front itself.
    got = np.isfinite(report.front)
    assert np.all(envelope[got] <= report.front[got] + 1e-9)


def test_overtaking_is_counted_not_smoothed_away():
    """The non-monotonicity is a finding, so it has to be reported as one."""
    tracker = ArrivalTracker(_Stations(3), fluid_index=1, thresholds=(0.5,))
    tracker.update(0.0, _fractions([0.0, 0.0, 0.0]))
    tracker.update(1.0, _fractions([1.0, 0.0, 1.0]))   # skipped the middle
    tracker.update(2.0, _fractions([1.0, 1.0, 1.0]))
    report = tracker.report(2.0)
    assert report.overtaking_depths == 1


def test_rise_velocity_survives_depths_crossed_within_one_interval():
    """Ties in the arrival curve must not divide by zero.

    When the interface crosses several cells between two recordings they share
    an arrival time.  Differencing straight across that is a zero denominator -
    it produced NaNs and RuntimeWarnings on the real caliper, where the front
    jumps through a narrow section.
    """
    tracker = ArrivalTracker(_Stations(6), fluid_index=1, thresholds=(0.5,))
    tracker.update(0.0, _fractions(np.zeros(6)))
    # Cells are in flow order, so cement fills from index 0 upward.  The three
    # deepest cross together and share an arrival time.
    tracker.update(1.0, _fractions([1, 1, 1, 0, 0, 0]))
    tracker.update(2.0, _fractions([1, 1, 1, 1, 0, 0]))
    tracker.update(3.0, _fractions(np.ones(6)))
    report = tracker.report(3.0)
    speed = report.rise_velocity(report.front_envelope)
    finite = np.isfinite(speed)
    assert finite.sum() >= 2
    assert np.all(speed[finite] >= 0.0)
