"""Interface length, swept efficiency and flow regime - Xue et al. (2022)."""

import numpy as np
import pytest

from inpipe.displacement import interface_metrics
from inpipe.regime import LAMINAR_LIMIT, TURBULENT_LIMIT, reynolds


def _stations(fraction, dz=1.0, area=1.0):
    """Flow-order arrays: index 0 at the shoe, depth descending."""
    fraction = np.asarray(fraction, dtype=float)
    depth = np.arange(fraction.size, 0, -1, dtype=float)
    return fraction, np.full(fraction.size, area * dz), depth


# --- interface length -------------------------------------------------------


def test_a_sharp_front_has_almost_no_interface_length():
    """The whole point of the measure: it separates sharp from smeared."""
    fraction, volume, depth = _stations([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    m = interface_metrics(fraction, volume, depth)
    assert m.length == pytest.approx(0.0, abs=1.0)


def test_a_smeared_front_has_a_long_one():
    fraction, volume, depth = _stations(np.linspace(1.0, 0.0, 21))
    m = interface_metrics(fraction, volume, depth)
    sharp = interface_metrics(*_stations([1.0] * 10 + [0.0] * 11))
    assert m.length > 5.0 * max(sharp.length, 1e-9)


def test_the_edges_are_interpolated_between_stations():
    """Snapping to the nearest station would quantise the length to the mesh."""
    fraction, volume, depth = _stations([1.0, 0.5, 0.0])
    m = interface_metrics(fraction, volume, depth)
    # 0.2 lies between stations 1 (0.5) and 2 (0.0); depth descends by 1 m.
    assert m.front_depth == pytest.approx(2.0 - 0.6, abs=1e-9)
    # 0.8 lies between stations 0 (1.0) and 1 (0.5).
    assert m.back_depth == pytest.approx(3.0 - 0.4, abs=1e-9)


def test_the_front_is_the_furthest_crossing_not_the_first():
    """Cement channels past a wide spot, so the profile is not monotonic.

    Taking the first crossing from the shoe would report the front as stuck
    behind the bypassed station.
    """
    fraction, volume, depth = _stations([1.0, 0.05, 0.9, 0.0])
    m = interface_metrics(fraction, volume, depth)
    assert m.front_depth < 2.0        # ahead of the bypassed station


def test_nothing_pumped_yet_is_nan_not_zero():
    fraction, volume, depth = _stations([0.0, 0.0, 0.0])
    m = interface_metrics(fraction, volume, depth)
    assert np.isnan(m.front_depth) and np.isnan(m.length)
    assert np.isnan(m.swept_efficiency)


# --- swept efficiency -------------------------------------------------------


def test_swept_efficiency_ignores_what_the_front_has_not_reached():
    """That is the whole fix: the global figure is dominated by empty annulus.

    Two jobs that cleaned the swept region identically must report the same
    swept efficiency even when one has travelled twice as far.
    """
    near = interface_metrics(*_stations([0.9, 0.9, 0.3, 0.0, 0.0, 0.0]))
    far = interface_metrics(*_stations([0.9, 0.9, 0.9, 0.9, 0.3, 0.0]))
    assert near.swept_efficiency == pytest.approx(0.7, abs=1e-9)
    assert far.swept_efficiency == pytest.approx(0.78, abs=1e-9)
    # Global efficiency over the same six stations would be 0.35 and 0.60 -
    # a factor of 1.7 apart for the same displacement quality.
    assert np.mean([0.9, 0.9, 0.3, 0, 0, 0]) < np.mean([0.9, 0.9, 0.9, 0.9, 0.3, 0])


def test_swept_efficiency_meets_the_global_one_at_breakthrough():
    """Once the front is at the top the swept region is the whole annulus."""
    fraction, volume, depth = _stations([1.0, 0.9, 0.8, 0.7, 0.6, 0.25])
    m = interface_metrics(fraction, volume, depth)
    assert m.swept_efficiency == pytest.approx(float(np.mean(fraction)))


def test_the_metrics_are_volume_weighted_not_station_counted():
    """Annular volume varies by a factor of ten here; counting cells is wrong."""
    fraction = np.array([1.0, 0.0, 0.5])
    depth = np.array([3.0, 2.0, 1.0])
    big_at_the_shoe = interface_metrics(fraction, np.array([10.0, 1.0, 1.0]), depth)
    big_up_top = interface_metrics(fraction, np.array([1.0, 1.0, 10.0]), depth)
    # Cement sits at the shoe: (1.0*10 + 0.0*1 + 0.5*1) / 12.
    assert big_at_the_shoe.swept_efficiency == pytest.approx(10.5 / 12.0)
    # Same fractions, volume up top instead: (1.0*1 + 0.0*1 + 0.5*10) / 12.
    assert big_up_top.swept_efficiency == pytest.approx(6.0 / 12.0)
    # Counting stations would have given 0.5 for both.


# --- flow regime ------------------------------------------------------------


def test_reynolds_recovers_the_newtonian_value():
    """With tau0 = 0 and n = 1 the effective viscosity is just k."""
    re, mu = reynolds(rho=1000.0, velocity=1.0, hydraulic_diameter=0.1,
                      tau_w=10.0, tau0=0.0, k=0.001, n=1.0)
    assert mu == pytest.approx(0.001)
    assert re == pytest.approx(1000.0 * 1.0 * 0.1 / 0.001)


def test_a_station_that_is_not_flowing_reads_zero_not_laminar():
    """Silently calling a stalled station laminar would be a free pass."""
    re, mu = reynolds(rho=1000.0, velocity=0.0, hydraulic_diameter=0.1,
                      tau_w=0.0, tau0=2.0, k=0.3, n=0.72)
    assert re == 0.0 and not np.isfinite(mu)


def test_a_thin_mud_is_flagged_turbulent_and_a_thick_one_is_not():
    """The reason this diagnostic exists.

    The solver integrates a laminar Herschel-Bulkley profile everywhere.  With
    the placeholder mud that is sound; with a water-thin mud - which is how the
    mud on this well has been described - it is not, and nothing else in the
    model would notice.
    """
    v, d_h = 0.437, 0.0872          # K-GEP-1 annulus at 5 bpm
    thick, _ = reynolds(1198.0, v, d_h, tau_w=20.0, tau0=2.0, k=0.30, n=0.72)
    thin, _ = reynolds(998.0, v, d_h, tau_w=20.0, tau0=0.0, k=1.0e-3, n=1.0)
    assert thick < LAMINAR_LIMIT
    assert thin > TURBULENT_LIMIT
    assert thin > 10.0 * thick


def test_reynolds_is_vectorised_over_stations():
    re, mu = reynolds(np.full(4, 1200.0), np.full(4, 0.4), np.full(4, 0.09),
                      np.array([5.0, 10.0, 20.0, 40.0]), 2.0, 0.3, 0.72)
    assert re.shape == (4,)
    assert np.all(np.diff(re) > 0.0)      # thinner at higher shear, so faster
