"""Test gate 3 - stratified cross-section mesh geometry and velocity mapping."""

import math

import numpy as np
import pytest

from inpipe.config import GridConfig
from inpipe.fluid import Fluid
from inpipe.grid import Grid, cell_moments, layer_boundaries
from inpipe.velocity import solve_profile

R = 0.05
L = 4.0
MESHES = [(3, 4), (13, 18), (26, 36), (52, 72), (7, 5)]


def make_grid(n_layer, n_azimuth, n_axial=10, rule="uniform_y", **kw):
    return Grid(R, L, GridConfig(n_axial=n_axial, n_layer=n_layer, n_azimuth=n_azimuth,
                                 layer_rule=rule), **kw)


# --- exact geometry --------------------------------------------------------


@pytest.mark.parametrize("n_layer,n_azimuth", MESHES)
@pytest.mark.parametrize("rule", ["uniform_y", "equal_area"])
def test_areas_sum_to_disc(n_layer, n_azimuth, rule):
    """Sum of cell cross-sectional areas equals pi R^2 to rel 1e-12."""
    g = make_grid(n_layer, n_azimuth, rule=rule)
    assert g.total_area == pytest.approx(math.pi * R**2, rel=1e-12)


@pytest.mark.parametrize("n_layer,n_azimuth", MESHES)
def test_volumes_sum_to_pipe(n_layer, n_azimuth):
    """Sum of cell volumes equals pi R^2 L to rel 1e-12."""
    g = make_grid(n_layer, n_azimuth)
    assert g.total_volume == pytest.approx(math.pi * R**2 * L, rel=1e-12)


@pytest.mark.parametrize("n_layer,n_azimuth", MESHES)
def test_symmetry_about_vertical_centreline(n_layer, n_azimuth):
    """Areas mirror across x = 0.

    Tolerances are set by round-off in the thin near-wall slivers, where the
    centroid is a ratio of two small closed-form sums; the areas themselves
    mirror to ~1e-11 relative even for the smallest cell on the finest mesh.
    """
    g = make_grid(n_layer, n_azimuth)
    np.testing.assert_allclose(g.cell_area, g.cell_area[:, ::-1], rtol=1e-10, atol=1e-20)
    np.testing.assert_allclose(g.cell_x, -g.cell_x[:, ::-1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(g.cell_y, g.cell_y[:, ::-1], rtol=0.0, atol=1e-12)


def test_refinement_keeps_area_exact_not_merely_convergent():
    """Doubling n_layer and n_azimuth keeps the total area exact, not closer."""
    errs = []
    nl, na = 13, 18
    for _ in range(4):
        g = make_grid(nl, na)
        errs.append(abs(g.total_area - math.pi * R**2) / (math.pi * R**2))
        nl, na = 2 * nl, 2 * na
    assert max(errs) < 1e-14, f"areas are only convergent, not exact: {errs}"


def test_uniform_y_layer_spacing():
    ys = layer_boundaries(8, R, "uniform_y")
    np.testing.assert_allclose(np.diff(ys), 2 * R / 8, rtol=1e-14)
    assert ys[0] == -R and ys[-1] == R


def test_equal_area_layers_have_equal_area():
    g = make_grid(9, 6, rule="equal_area")
    per_layer = g.cell_area.sum(axis=1)
    np.testing.assert_allclose(per_layer, math.pi * R**2 / 9, rtol=1e-10)


def test_uniform_y_layers_are_area_unequal_by_design():
    """Documents the known consequence of assumption A-03."""
    g = make_grid(13, 18, rule="uniform_y")
    per_layer = g.cell_area.sum(axis=1)
    # The near-wall layers carry well under half the area of the central one,
    # so uniform-y spacing under-resolves exactly where the shear is largest.
    assert per_layer[0] / per_layer[6] < 0.4
    assert per_layer[0] == pytest.approx(per_layer[-1], rel=1e-12)


def test_cell_moments_on_a_full_disc():
    a, xc, yc = cell_moments(-R, R, -R, R, R)
    assert a == pytest.approx(math.pi * R**2, rel=1e-14)
    assert xc == pytest.approx(0.0, abs=1e-15)
    assert yc == pytest.approx(0.0, abs=1e-15)


def test_cell_moments_half_disc_centroid():
    """Upper half disc: y centroid is 4R/(3 pi)."""
    a, xc, yc = cell_moments(-R, R, 0.0, R, R)
    assert a == pytest.approx(0.5 * math.pi * R**2, rel=1e-14)
    assert xc == pytest.approx(0.0, abs=1e-15)
    assert yc == pytest.approx(4.0 * R / (3.0 * math.pi), rel=1e-13)


def test_cell_moments_box_outside_disc_is_empty():
    a, _, _ = cell_moments(0.9 * R, R, 0.9 * R, R, R)
    assert a == 0.0


def test_cell_moments_handles_boundary_crossing():
    """A box wider than the layer's own extent must not accrue negative area.

    This is the case that a breakpoint set covering only the "boundary
    switches" (and not the top/bottom crossing) gets wrong.
    """
    a, _, _ = cell_moments(-R, R, 0.4 * R, 0.6 * R, R)
    # Analytical: area between two chords = segment(0.4R) - segment(0.6R).
    def below(y):
        return y * math.sqrt(R * R - y * y) + R * R * math.asin(y / R) + 0.5 * math.pi * R * R
    assert a == pytest.approx(below(0.6 * R) - below(0.4 * R), rel=1e-13)
    assert a > 0.0


def test_centroids_lie_inside_the_disc():
    g = make_grid(26, 36)
    assert np.all(g.cell_r <= R + 1e-15)


# --- velocity mapping ------------------------------------------------------

FLUIDS = [
    Fluid.newtonian(1000.0, 0.001),
    Fluid.power_law(1000.0, k=0.8, n=0.4),
    Fluid("hb", rho=1400.0, tau0=3.0, k=0.6, n=0.55),
]
Q = 1.0e-4


@pytest.mark.parametrize("fluid", FLUIDS)
def test_area_average_mapping_recovers_flow_rate(fluid):
    """Area-weighted sum of cell velocities reproduces Q well inside 1 %."""
    g = make_grid(13, 18)
    prof = solve_profile(Q, fluid, R)
    q = g.flow_rate_from_cells(g.map_velocity(prof, "area_average"))
    err = abs(q - Q) / Q
    assert err < 1e-4, f"area-average mapping Q error {100 * err:.4f} %"


@pytest.mark.parametrize("fluid", FLUIDS)
def test_centroid_mapping_error_is_reported(fluid, capsys):
    """Records the centroid-mapping error at the paper's 13 x 18 resolution.

    Assumption A-04: the spec's gate is ~1 %.  Newtonian and power-law pass;
    the Herschel-Bulkley case does *not*, which is why ``area_average`` is the
    configured default.  This test asserts only the looser bound it actually
    meets, and prints the number.
    """
    g = make_grid(13, 18)
    prof = solve_profile(Q, fluid, R)
    q = g.flow_rate_from_cells(g.map_velocity(prof, "centroid"))
    err = abs(q - Q) / Q
    with capsys.disabled():
        print(f"\n  centroid mapping, {fluid.name}, 13x18: Q error = {100 * err:.4f} %")
    assert err < 0.02


def test_centroid_mapping_converges_second_order():
    g_coarse = make_grid(13, 18)
    g_fine = make_grid(26, 36)
    prof = solve_profile(Q, FLUIDS[0], R)
    e_c = abs(g_coarse.flow_rate_from_cells(g_coarse.map_velocity(prof, "centroid")) - Q)
    e_f = abs(g_fine.flow_rate_from_cells(g_fine.map_velocity(prof, "centroid")) - Q)
    assert 3.0 < e_c / e_f < 5.0


@pytest.mark.parametrize("fluid", FLUIDS)
def test_radial_weight_mapping_matches_direct_quadrature(fluid):
    """The precomputed radial-weight area average matches 2D Gauss-Legendre."""
    g = make_grid(13, 18)
    prof = solve_profile(Q, fluid, R)
    fast = g.map_velocity(prof, "area_average")
    ref = g._area_averaged_velocity_quadrature(prof, n_gauss=40)
    assert np.max(np.abs(fast - ref)) / prof.u_max < 1e-3


def test_radial_weights_tile_each_annulus_exactly():
    g = make_grid(13, 18, n_radial=32)
    W, _ = g._radial_weights()
    edges = np.linspace(0.0, R, 33)
    np.testing.assert_allclose(
        np.cumsum(np.asarray(W.sum(axis=0)).ravel()), math.pi * edges[1:] ** 2, rtol=1e-13
    )


def test_radial_weights_rows_sum_to_the_cell_areas():
    """Every cell's annular weights must add back up to its own area."""
    g = make_grid(13, 18, n_radial=128)
    W, _ = g._radial_weights()
    np.testing.assert_allclose(
        np.asarray(W.sum(axis=1)).ravel(), g.cell_area.ravel(), rtol=1e-12
    )


def test_radial_weight_matrix_is_sparse():
    """The sparsity is the point: each cell spans only a slice of the radius."""
    g = make_grid(26, 36, n_radial=512)
    W, _ = g._radial_weights()
    density = W.nnz / (W.shape[0] * W.shape[1])
    assert density < 0.15, f"weight matrix is {100 * density:.1f} % dense"


def test_radial_weight_mapping_is_second_order_in_n_radial():
    prof = solve_profile(Q, FLUIDS[2], R)
    errs = []
    for nr in (256, 512, 1024):
        g = make_grid(13, 18, n_radial=nr)
        ref = g._area_averaged_velocity_quadrature(prof, n_gauss=40)
        errs.append(np.max(np.abs(g.map_velocity(prof, "area_average") - ref)))
    assert 3.0 < errs[0] / errs[1] < 5.0
    assert 3.0 < errs[1] / errs[2] < 5.0


def test_unknown_mapping_and_rule_raise():
    g = make_grid(5, 4)
    with pytest.raises(ValueError):
        g.map_velocity(solve_profile(Q, FLUIDS[0], R), "nonsense")
    with pytest.raises(ValueError):
        layer_boundaries(4, R, "nonsense")


def test_grid_shapes_and_axial_layout():
    g = make_grid(13, 18, n_axial=100)
    assert g.shape == (100, 13, 18)
    assert g.dz == pytest.approx(L / 100)
    assert g.z_centers[0] == pytest.approx(0.5 * L / 100)
    np.testing.assert_allclose(g.axial_face_area, g.cell_area)
