"""Caliper log reading - and above all, unit resolution.

A silent unit error here scales the whole well, so most of these tests are
about refusing to guess rather than about arithmetic.
"""

import numpy as np
import pytest

from inpipe.caliper import CaliperLog, read_caliper, synthetic_caliper
from inpipe.config import inch_to_m, m_to_ft, m_to_inch

LENGTH = 200.0
GAUGE = inch_to_m(8.5)


@pytest.fixture(scope="module")
def log():
    return synthetic_caliper(LENGTH, GAUGE)


def write_csv(path, log, header, depth_fn, dia_fn):
    rows = [header] + [
        f"{depth_fn(z):.8f},{dia_fn(d):.8f}" for z, d in zip(log.depth, log.diameter)
    ]
    path.write_text("\n".join(rows))
    return path


# --- unit resolution -------------------------------------------------------


@pytest.mark.parametrize(
    "label,header,depth_fn,dia_fn",
    [
        ("field units", "DEPT,CALI", m_to_ft, m_to_inch),
        ("SI units", "DEPT,CALI", lambda x: x, lambda x: x),
        ("named ft/in", "depth_ft,CALI_in", m_to_ft, m_to_inch),
        ("named m", "depth_m,diameter_m", lambda x: x, lambda x: x),
        ("bracketed", "MD (m),HoleDia (m)", lambda x: x, lambda x: x),
        ("mixed m and in", "MD_m,CALI_in", lambda x: x, m_to_inch),
    ],
)
def test_units_resolve_correctly(tmp_path, log, label, header, depth_fn, dia_fn):
    path = write_csv(tmp_path / "c.csv", log, header, depth_fn, dia_fn)
    got = read_caliper(path)
    np.testing.assert_allclose(got.depth, log.depth, atol=1e-6)
    np.testing.assert_allclose(got.diameter, log.diameter, atol=1e-9)


def test_depth_magnitude_is_never_used_to_infer_its_unit(tmp_path, log):
    """A 656 ft well and a 656 m well are indistinguishable by magnitude.

    Inferring from the depth range would silently scale this log by 3.28.
    """
    path = write_csv(tmp_path / "shallow.csv", log, "DEPT,CALI", m_to_ft, m_to_inch)
    got = read_caliper(path)
    assert got.depth[-1] == pytest.approx(LENGTH, rel=1e-6)
    assert "ft" in got.units


def test_explicit_units_override_inference(tmp_path, log):
    path = write_csv(tmp_path / "c.csv", log, "A,B", m_to_ft, m_to_inch)
    got = read_caliper(path, depth_column=0, diameter_column=1,
                       depth_unit="ft", diameter_unit="in")
    np.testing.assert_allclose(got.depth, log.depth, atol=1e-6)
    assert "given" in got.units


def test_implausible_diameter_is_refused_not_accepted(tmp_path):
    """Better a loud failure than a well 200 m wide."""
    (tmp_path / "bad.csv").write_text("DEPT,CALI\n0,8500\n100,8600\n")
    with pytest.raises(ValueError, match="plausible range"):
        read_caliper(tmp_path / "bad.csv")


def test_unknown_unit_names_raise(tmp_path, log):
    path = write_csv(tmp_path / "c.csv", log, "DEPT,CALI", lambda x: x, lambda x: x)
    with pytest.raises(ValueError, match="unknown depth unit"):
        read_caliper(path, depth_unit="fathoms")
    with pytest.raises(ValueError, match="unknown diameter unit"):
        read_caliper(path, diameter_unit="cubits")


# --- column resolution -----------------------------------------------------


def test_short_alias_cannot_steal_the_depth_column(tmp_path, log):
    """'d' is a diameter alias and a substring of 'depth' - it must not match."""
    path = write_csv(tmp_path / "c.csv", log, "depth_ft,CALI_in", m_to_ft, m_to_inch)
    got = read_caliper(path)
    np.testing.assert_allclose(got.diameter, log.diameter, atol=1e-9)
    assert got.diameter.max() < 1.0, "picked the depth column as diameter"


def test_columns_can_be_named_or_indexed(tmp_path, log):
    path = write_csv(tmp_path / "c.csv", log, "a,b", lambda x: x, lambda x: x)
    by_index = read_caliper(path, depth_column=0, diameter_column=1)
    by_name = read_caliper(path, depth_column="a", diameter_column="b")
    np.testing.assert_allclose(by_index.diameter, by_name.diameter)


def test_missing_column_raises_with_a_useful_message(tmp_path):
    (tmp_path / "c.csv").write_text("foo,bar\n1,2\n3,4\n")
    with pytest.raises(ValueError, match="could not find a depth column"):
        read_caliper(tmp_path / "c.csv")


def test_las_reads_curve_units_from_the_c_section(tmp_path):
    (tmp_path / "w.las").write_text(
        "~C\nDEPT .M   : depth\nCALI .IN  : caliper\n~A\n0.0 8.5\n100.0 9.2\n200.0 8.6\n"
    )
    log = read_caliper(tmp_path / "w.las")
    np.testing.assert_allclose(log.depth, [0.0, 100.0, 200.0])
    np.testing.assert_allclose(log.diameter, inch_to_m(np.array([8.5, 9.2, 8.6])), rtol=1e-12)
    assert "given" in log.units


# --- data hygiene ----------------------------------------------------------


def test_null_and_non_finite_rows_are_dropped(tmp_path):
    (tmp_path / "c.csv").write_text(
        "DEPT,CALI\n0,0.20\n10,-999.25\n20,0.22\n30,nan\n40,0.21\n"
    )
    log = read_caliper(tmp_path / "c.csv")
    np.testing.assert_allclose(log.depth, [0.0, 20.0, 40.0])


def test_duplicate_depths_are_averaged(tmp_path):
    (tmp_path / "c.csv").write_text("DEPT,CALI\n0,0.20\n10,0.22\n10,0.24\n20,0.21\n")
    log = read_caliper(tmp_path / "c.csv")
    np.testing.assert_allclose(log.depth, [0.0, 10.0, 20.0])
    assert log.diameter[1] == pytest.approx(0.23)


def test_unsorted_input_is_sorted(tmp_path):
    (tmp_path / "c.csv").write_text("DEPT,CALI\n20,0.21\n0,0.20\n10,0.22\n")
    log = read_caliper(tmp_path / "c.csv")
    assert np.all(np.diff(log.depth) > 0)


def test_too_few_samples_raises(tmp_path):
    (tmp_path / "c.csv").write_text("DEPT,CALI\n0,0.20\n")
    with pytest.raises(ValueError, match="fewer than two"):
        read_caliper(tmp_path / "c.csv")


def test_construction_validates(caplog):
    with pytest.raises(ValueError, match="strictly increasing"):
        CaliperLog([0.0, 10.0, 10.0], [0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match="must be positive"):
        CaliperLog([0.0, 10.0], [0.2, -0.1])
    with pytest.raises(ValueError, match="at least two"):
        CaliperLog([0.0], [0.2])


# --- interpolation and metrics ---------------------------------------------


def test_diameter_at_interpolates_and_holds_outside_the_log(log):
    assert log.diameter_at(0.0) == pytest.approx(log.diameter[0])
    assert log.diameter_at(-50.0) == pytest.approx(log.diameter[0])
    assert log.diameter_at(1e6) == pytest.approx(log.diameter[-1])
    mid = log.diameter_at(0.5 * (log.depth[10] + log.depth[11]))
    assert min(log.diameter[10], log.diameter[11]) <= mid <= max(
        log.diameter[10], log.diameter[11]
    )


def test_synthetic_log_has_the_requested_features(log):
    assert log.depth[0] == 0.0 and log.depth[-1] == pytest.approx(LENGTH)
    assert log.gauge == pytest.approx(GAUGE, rel=0.05)
    # Washouts and a tight zone are present.
    assert log.diameter.max() > 1.4 * GAUGE
    assert log.diameter.min() < 0.98 * GAUGE
    assert 0.05 < log.washout_fraction() < 0.6


def test_excess_volume_is_positive_and_zero_for_a_smooth_hole():
    smooth = synthetic_caliper(LENGTH, GAUGE, washouts=(), tight_zones=(), roughness=0.0)
    assert smooth.excess_volume(0.1397) == pytest.approx(0.0, abs=1e-12)
    assert np.ptp(smooth.diameter) == pytest.approx(0.0, abs=1e-12)
    rough = synthetic_caliper(LENGTH, GAUGE)
    assert rough.excess_volume(0.1397) > 0.5


def test_summary_mentions_units_and_washout(log):
    text = log.summary(casing_od=0.1397)
    for key in ("units", "gauge", "washed out", "excess annular volume"):
        assert key in text
