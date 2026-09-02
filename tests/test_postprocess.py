"""Export and post-processing - the interface to an external CFD comparison."""

import math

import numpy as np
import pytest

from inpipe.config import GeometryConfig, GridConfig, NumericsConfig, SimulationConfig
from inpipe.fluid import Fluid, PumpSchedule, PumpStage
from inpipe.postprocess import (
    centreline_velocity,
    cross_section_average,
    radial_concentration,
    save_results,
    summary_table,
)
from inpipe.solver import InPipeSolver

LENGTH, INNER_D, U_BAR = 4.0, 0.019, 0.05


@pytest.fixture(scope="module")
def result():
    area = math.pi * (0.5 * INNER_D) ** 2
    q = U_BAR * area
    mud = Fluid.newtonian(1000.0, 0.001, "mud")
    cement = Fluid.newtonian(1200.0, 0.002, "cement")
    config = SimulationConfig(
        geometry=GeometryConfig(length=LENGTH, inner_diameter=INNER_D),
        grid=GridConfig(n_axial=50, n_layer=13, n_azimuth=18),
        numerics=NumericsConfig(diagnostics_every=5),
    )
    solver = InPipeSolver(config, PumpSchedule([PumpStage(cement, q * 100, q)]),
                          initial_fluid=mud)
    solver.set_initial_interface(1.0, cement, mud)
    return solver.run(t_end=15.0)


def test_centreline_velocity_spans_the_full_diameter_with_no_slip(result):
    y, u = centreline_velocity(result)
    assert y[0] == pytest.approx(-result.grid.radius, rel=1e-14)
    assert y[-1] == pytest.approx(result.grid.radius, rel=1e-14)
    assert u[0] == 0.0 and u[-1] == 0.0
    assert np.all(np.diff(y) > 0.0), "y must be sorted for a line plot"
    # Symmetric about the axis, and peaked at the centre.
    np.testing.assert_allclose(u, u[::-1], rtol=1e-10, atol=1e-15)
    assert u.argmax() in (len(u) // 2, len(u) // 2 - 1)


def test_centreline_velocity_without_wall_points(result):
    y, u = centreline_velocity(result, include_wall=False)
    assert len(y) == result.grid.n_layer
    assert abs(y[0]) < result.grid.radius


def test_cross_section_average_partitions_to_one(result):
    cols = np.array([cross_section_average(result, i) for i in range(len(result.fluids))])
    np.testing.assert_allclose(cols.sum(axis=0), 1.0, atol=1e-12)
    assert cols.shape == (len(result.fluids), result.grid.n_axial)


def test_cross_section_average_is_monotone_for_a_single_front(result):
    """The displacing fluid decreases with depth: one front, no spurious wiggles."""
    i = result.fluids.index([f for f in result.fluids if f.name == "cement"][0])
    prof = cross_section_average(result, i)
    assert np.all(np.diff(prof) <= 1e-12)


def test_radial_concentration_is_symmetric_and_bounded(result):
    y, f = radial_concentration(result, 1, 2.0)
    np.testing.assert_allclose(f, f[::-1], rtol=1e-9, atol=1e-14)
    assert f.min() >= -1e-14 and f.max() <= 1.0 + 1e-14
    assert np.all(np.diff(y) > 0.0)
    # The front runs ahead on the axis, so the centre is richer than the wall.
    assert f[len(f) // 2] > f[0]


def test_radial_concentration_clamps_depths_outside_the_pipe(result):
    for z in (-5.0, 0.0, LENGTH, LENGTH + 10.0):
        y, f = radial_concentration(result, 0, z)
        assert len(f) == result.grid.n_layer and np.all(np.isfinite(f))


def test_save_results_writes_every_file_and_round_trips(result, tmp_path):
    paths = save_results(result, tmp_path, prefix="case")
    names = {p.name for p in paths}
    assert "case_fields.npz" in names
    assert "case_axial_profile.csv" in names
    assert "case_centreline_velocity.csv" in names
    assert "case_outlet_history.csv" in names
    for fl in result.fluids:
        assert f"case_centreplane_{fl.name}.csv" in names
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)

    data = np.load(tmp_path / "case_fields.npz", allow_pickle=False)
    np.testing.assert_allclose(data["fractions"], result.fractions, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(data["velocity"], result.velocity, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(data["cell_area"], result.grid.cell_area)
    assert list(data["fluid_names"]) == [f.name for f in result.fluids]
    np.testing.assert_allclose(
        data["fluid_rheology"],
        [[f.rho, f.tau0, f.k, f.n] for f in result.fluids],
    )
    assert float(data["radius"]) == pytest.approx(result.grid.radius)


def test_saved_csv_shapes_match_the_grid(result, tmp_path):
    save_results(result, tmp_path, prefix="case")
    g = result.grid

    lines = (tmp_path / "case_axial_profile.csv").read_text().strip().split("\n")
    assert len(lines) == g.n_axial + 1
    assert lines[0] == "z_m," + ",".join(f.name for f in result.fluids)

    lines = (tmp_path / "case_centreplane_mud.csv").read_text().strip().split("\n")
    assert len(lines) == g.n_layer + 1
    assert len(lines[1].split(",")) == g.n_axial + 1

    lines = (tmp_path / "case_outlet_history.csv").read_text().strip().split("\n")
    assert len(lines) == len(result.outlet_time) + 1


def test_summary_table_reports_the_invariants(result):
    text = summary_table(result)
    for key in ("grid", "mass budget err", "sum-to-one err", "wall time", "Dm_num"):
        assert key in text
    for fl in result.fluids:
        assert fl.name in text


def test_plotting_routines_run_headless(result, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from inpipe.postprocess import plot_centre_plane, plot_diagnostics, plot_outlet_history

    plot_centre_plane(result, fluid_index=0)
    plot_outlet_history(result)
    fig = plot_diagnostics(result, path=tmp_path / "diag.png")
    assert (tmp_path / "diag.png").exists()
    plt.close("all")
    assert fig is not None
