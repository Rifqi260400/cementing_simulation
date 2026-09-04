"""Test gate 4 - axial VOF transport."""

import numpy as np
import pytest

from inpipe.config import GridConfig
from inpipe.fluid import Fluid
from inpipe.grid import Grid
from inpipe.transport import (
    BoundednessViolation,
    CFLViolation,
    advect,
    advect_multi,
    assert_cfl,
    cfl_timestep,
    check_bounded,
    check_sum_to_one,
    numerical_diffusivity,
    upwind_diffusivity,
)
from inpipe.velocity import solve_profile

NZ = 200
DZ = 0.02  # m, so a 4 m pipe
CFL = 0.4


def column(nz=NZ):
    """Shape helper: a single (layer, azimuth) column."""
    return (nz, 1, 1)


# ---------------------------------------------------------------------------


def test_uniform_field_is_unchanged():
    """Uniform flow, uniform f -> f unchanged to machine precision."""
    f = np.full(column(), 0.37)
    u = np.full(column(), 0.5)
    dt = CFL * DZ / 0.5
    for _ in range(50):
        f = advect(f, u, DZ, dt, inlet_value=0.37)
    np.testing.assert_allclose(f, 0.37, rtol=0.0, atol=1e-15)


def test_mass_budget_is_exact():
    """Total volume of each fluid obeys the flux budget to rel 1e-12, 1000 steps."""
    nz = 400
    f = np.zeros(column(nz))
    f[: nz // 2] = 1.0
    u_val = 0.5
    u = np.full(column(nz), u_val)
    dt = CFL * DZ / u_val
    v0 = f.sum() * DZ
    influx = 0.0
    outflux = 0.0
    for _ in range(1000):
        outflux += u_val * f[-1, 0, 0] * dt  # zero-gradient outflow
        influx += u_val * 1.0 * dt  # Dirichlet inlet on fluid 1
        f = advect(f, u, DZ, dt, inlet_value=1.0)
    v1 = f.sum() * DZ
    expected = v0 + influx - outflux
    assert v1 == pytest.approx(expected, rel=1e-12)


def test_interior_mass_is_exactly_conserved_before_breakthrough():
    """A slug bounded away from both ends conserves its volume exactly.

    Nothing enters or leaves the domain, so the conservative update must hold
    the slug volume to round-off over 1000 steps.
    """
    nz = 1200
    g = np.zeros(column(nz))
    g[100:200] = 1.0  # a slug of fluid 2, clear of both ends
    u_val = 0.5
    u = np.full(column(nz), u_val)
    dt = CFL * DZ / u_val
    v0 = g.sum()
    # 1000 steps advances the slug by 0.4 * 1000 = 400 cells, so its leading
    # edge reaches cell ~600 with an upwind tail of sigma ~ 16 cells - some
    # 38 sigma clear of the outlet at cell 1200.
    for _ in range(1000):
        g = advect(g, u, DZ, dt, inlet_value=0.0)
    assert g[-1, 0, 0] < 1e-12, "slug reached the outlet; test is no longer closed"
    assert g.sum() == pytest.approx(v0, rel=1e-12)


def test_sum_to_one_every_step():
    """sum_i f_i = 1 in every cell at every step, abs 1e-12."""
    nz = 200
    fields = np.zeros((3,) + column(nz))
    fields[0, :60] = 1.0
    fields[1, 60:130] = 1.0
    fields[2, 130:] = 1.0
    u = np.full(column(nz), 0.5)
    area = np.ones((1, 1))
    dt = CFL * DZ / 0.5
    for _ in range(300):
        fields = advect_multi(fields, u, DZ, dt, inlet_values=[1.0, 0.0, 0.0],
                              area=area)
        assert check_sum_to_one(fields, atol=1e-12) <= 1e-12
        check_bounded(fields, atol=1e-12)


def test_square_wave_advects_at_u_and_stays_monotone():
    nz = 400
    f = np.zeros(column(nz))
    f[:100] = 1.0
    u_val = 0.5
    u = np.full(column(nz), u_val)
    dt = CFL * DZ / u_val
    n = 250
    for _ in range(n):
        f = advect(f, u, DZ, dt, inlet_value=1.0)
    prof = f[:, 0, 0]
    # No new extrema: monotone non-increasing in z, and inside [0, 1].
    assert np.all(np.diff(prof) <= 1e-14)
    assert prof.min() >= -1e-14 and prof.max() <= 1.0 + 1e-14
    # The f = 0.5 crossing sits at the exact front position.
    z = (np.arange(nz) + 0.5) * DZ
    front = np.interp(-0.5, -prof, z)
    exact = 100 * DZ + u_val * n * dt
    assert front == pytest.approx(exact, abs=2.5 * DZ)


def test_zero_velocity_is_a_no_op():
    f = np.random.default_rng(0).random(column())
    out = advect(f, np.zeros(column()), DZ, 1.0, inlet_value=1.0)
    np.testing.assert_allclose(out, f, rtol=0.0, atol=0.0)


# --- CFL -------------------------------------------------------------------


def test_cfl_timestep_formula():
    u = np.array([0.1, -0.7, 0.3])
    assert cfl_timestep(u, DZ, CFL) == pytest.approx(CFL * DZ / 0.7)
    assert cfl_timestep(np.zeros(3), DZ, CFL) == np.inf


def test_cfl_violation_raises_and_does_not_proceed_silently():
    u = np.full(column(), 1.0)
    dt_ok = cfl_timestep(u, DZ, CFL)
    assert assert_cfl(u, DZ, dt_ok, CFL) == pytest.approx(CFL)
    with pytest.raises(CFLViolation, match="exceeds the configured limit"):
        assert_cfl(u, DZ, dt_ok * 1.001, CFL)


def test_boundedness_violation_is_reported_not_clipped():
    bad = np.array([[[[1.2]]], [[[-0.2]]]])
    with pytest.raises(BoundednessViolation, match=r"left \[0, 1\]"):
        check_bounded(bad)
    with pytest.raises(BoundednessViolation, match="do not sum to one"):
        check_sum_to_one(np.array([[[[0.6]]], [[[0.6]]]]))


def test_unknown_face_scheme_raises():
    with pytest.raises(ValueError, match="unknown face scheme"):
        advect(np.zeros(column()), np.ones(column()), DZ, 1e-3, face_scheme="thinc")


# --- divergence correction (assumption A-07) -------------------------------


def test_divergence_correction_is_inert_for_a_uniform_profile():
    """The correction must change nothing when du/dz = 0."""
    f = np.zeros(column())
    f[:100] = 1.0
    u = np.full(column(), 0.5)
    dt = CFL * DZ / 0.5
    a = advect(f, u, DZ, dt, inlet_value=1.0, divergence_correction=True)
    b = advect(f, u, DZ, dt, inlet_value=1.0, divergence_correction=False)
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)


def test_divergence_correction_restores_sum_to_one_under_depth_varying_velocity():
    """The failure the correction exists to fix, and the proof that it fixes it.

    A depth-varying column velocity (what a depth-varying effective rheology
    produces) breaks the discrete continuity condition sum_j u_j A_j = 0, so
    the purely conservative update loses sum-to-one.
    """
    nz = 120
    z = np.arange(nz)[:, None, None]
    u = 0.4 + 0.2 * np.sin(2 * np.pi * z / nz)  # du/dz != 0
    fields = np.zeros((2,) + column(nz))
    fields[0, :50] = 1.0
    fields[1] = 1.0 - fields[0]
    dt = CFL * DZ / float(np.max(np.abs(u)))

    raw = fields.copy()
    corrected = fields.copy()
    for _ in range(100):
        raw = advect_multi(raw, u, DZ, dt, inlet_values=[1.0, 0.0], closure="none")
        corrected = advect_multi(corrected, u, DZ, dt, inlet_values=[1.0, 0.0],
                                 closure="local")

    raw_err = float(np.max(np.abs(raw.sum(axis=0) - 1.0)))
    corr_err = float(np.max(np.abs(corrected.sum(axis=0) - 1.0)))
    assert raw_err > 1e-3, f"expected the uncorrected form to drift, got {raw_err:.2e}"
    assert corr_err < 1e-12, f"corrected form drifted to {corr_err:.2e}"


# --- parabolic stretching: the key physical test ---------------------------


def build_stretching_case(n_axial=400, n_layer=13, n_azimuth=18, length=4.0):
    R = 0.0095
    grid = Grid(R, length, GridConfig(n_axial=n_axial, n_layer=n_layer, n_azimuth=n_azimuth))
    fluid = Fluid.newtonian(1000.0, 0.001)
    u_bar = 0.05
    q = u_bar * np.pi * R**2
    prof = solve_profile(q, fluid, R)
    u_cells = grid.map_velocity(prof, "area_average")
    u3 = np.broadcast_to(u_cells, grid.shape).copy()
    return grid, prof, u3


def front_positions(f, z):
    """z of the f = 0.5 crossing in every column, by linear interpolation."""
    nz = f.shape[0]
    flat = f.reshape(nz, -1)
    out = np.empty(flat.shape[1])
    for c in range(flat.shape[1]):
        col = flat[:, c]
        out[c] = np.interp(-0.5, -col, z)
    return out.reshape(f.shape[1:])


def test_parabolic_stretching_matches_the_analytical_interface():
    """z(r, t) = z0 + u(r) t, reproduced column by column."""
    grid, prof, u3 = build_stretching_case()
    z0 = 1.0
    f = (grid.z_centers[:, None, None] < z0).astype(float) * np.ones(grid.shape)
    dt = cfl_timestep(u3, grid.dz, CFL)
    t_end = 10.0
    n = int(t_end / dt)
    t = n * dt
    for _ in range(n):
        f = advect(f, u3, grid.dz, dt, inlet_value=1.0)

    got = front_positions(f, grid.z_centers)
    exact = z0 + u3[0] * t
    # Only columns whose front is still well inside the pipe are comparable.
    inside = exact < grid.length - 10 * grid.dz
    err = np.abs(got - exact)[inside]
    assert err.max() < 3.0 * grid.dz, f"max front error {err.max() / grid.dz:.2f} cells"

    # The parabolic shape itself: front displacement must be proportional to
    # (1 - r^2/R^2) along the centre column of layers.
    disp = (got - z0)[inside]
    predicted = (u3[0] * t)[inside]
    np.testing.assert_allclose(disp, predicted, atol=3.0 * grid.dz)


def test_centreline_front_is_twice_the_mean_front():
    """The Newtonian signature: the tip runs at 2 u_bar."""
    grid, prof, u3 = build_stretching_case(length=8.0)
    z0 = 1.0
    f = (grid.z_centers[:, None, None] < z0).astype(float) * np.ones(grid.shape)
    dt = cfl_timestep(u3, grid.dz, CFL)
    n = int(20.0 / dt)
    for _ in range(n):
        f = advect(f, u3, grid.dz, dt, inlet_value=1.0)
    got = front_positions(f, grid.z_centers)
    tip = got.max() - z0
    assert tip / (prof.mean_velocity * n * dt) == pytest.approx(2.0, rel=0.05)


def _effective_diffusivity(col, z, t):
    """Fit D from the variance of -df/dz, using Var = 2 D t for an erf front."""
    dfdz = -np.gradient(col, z)
    w = np.clip(dfdz, 0.0, None)
    m0 = np.trapezoid(w, z)
    mean = np.trapezoid(w * z, z) / m0
    var = np.trapezoid(w * (z - mean) ** 2, z) / m0
    return var / (2.0 * t), mean


@pytest.mark.parametrize("n_axial", [200, 400, 800])
def test_numerical_diffusion_is_measured_and_falls_with_dx(n_axial, capsys):
    """Quantify the upwind smearing as an effective diffusivity.

    Reported against both the paper's scale estimate ``dx^2/dt`` and the
    modified-equation value ``u dz (1 - C) / 2`` for upwind + explicit Euler.

    What to compare it against is the subtle part, and an earlier version of
    this test got it wrong: it printed a ratio to a "physical Dm ~ 1e-3 m^2/s",
    a number with no justification behind it.  The physical diffusivity here is
    **molecular**, ~1e-9 m^2/s, against which the numerical value is some four
    orders of magnitude larger - and Taylor-Aris dispersion, which would raise
    it, is not established: it needs a radial mixing time ``R^2/Dm`` of about
    22 days and the job lasts minutes.

    The mechanism that actually spreads the front in this regime is **shear** -
    the velocity profile, which the model does represent.  So the honest
    comparison is numerical smearing against that, and on the field case it is
    12.6 m of front width against a measured interface length of 108 m: 12 % in
    length, 1.3 % in variance.  Subdominant, which is the right conclusion, but
    it was reached the wrong way before.
    """
    grid, prof, u3 = build_stretching_case(n_axial=n_axial)
    z0 = 1.0
    f = (grid.z_centers[:, None, None] < z0).astype(float) * np.ones(grid.shape)
    dt = cfl_timestep(u3, grid.dz, CFL)
    n = int(10.0 / dt)
    t = n * dt
    for _ in range(n):
        f = advect(f, u3, grid.dz, dt, inlet_value=1.0)

    # Use a mid-radius column so the front is far from both ends.
    li, ji = grid.n_layer // 2, 0
    col = f[:, li, ji]
    u_col = float(u3[0, li, ji])
    d_fit, _ = _effective_diffusivity(col, grid.z_centers, t)
    d_theory = upwind_diffusivity(u_col, grid.dz, dt)
    d_paper = numerical_diffusivity(grid.dz, dt)

    with capsys.disabled():
        print(
            f"\n  dz = {grid.dz:.4f} m, dt = {dt:.4f} s, C = {u_col * dt / grid.dz:.3f}"
            f"\n    Dm_num (paper, dx^2/dt)     = {d_paper:.4e} m^2/s"
            f"\n    Dm_num (upwind mod. eqn.)   = {d_theory:.4e} m^2/s"
            f"\n    Dm_num (fitted from front)  = {d_fit:.4e} m^2/s"
            f"\n    vs molecular Dm ~ 1e-9      = {d_fit / 1e-9:.3g} x larger"
            f"\n    (the physical spreading here is shear, not diffusion - see "
            f"the docstring)"
        )
    assert d_fit == pytest.approx(d_theory, rel=0.25)


def test_numerical_diffusion_falls_linearly_with_dz():
    """Upwind at fixed Courant number: D_num ~ dz, not dz^2/dt."""
    ds = []
    for n_axial in (200, 400, 800):
        grid, prof, u3 = build_stretching_case(n_axial=n_axial)
        z0 = 1.0
        f = (grid.z_centers[:, None, None] < z0).astype(float) * np.ones(grid.shape)
        dt = cfl_timestep(u3, grid.dz, CFL)
        n = int(10.0 / dt)
        for _ in range(n):
            f = advect(f, u3, grid.dz, dt, inlet_value=1.0)
        li = grid.n_layer // 2
        d, _ = _effective_diffusivity(f[:, li, 0], grid.z_centers, n * dt)
        ds.append(d)
    assert 1.7 < ds[0] / ds[1] < 2.3
    assert 1.7 < ds[1] / ds[2] < 2.3


def test_paper_scale_estimate_reproduces_their_quoted_number():
    """dx = 30 m, dt = 0.1 min -> 150 m^2/s, as Dai et al. state."""
    assert numerical_diffusivity(30.0, 6.0) == pytest.approx(150.0, rel=1e-12)


# --- the redistribute closure (assumption A-07) ----------------------------


def _divergent_case(nz=120, n_layer=3, n_azimuth=4):
    """Depth-varying profile *shape* at depth-constant total Q.

    This is what a depth-varying effective rheology produces: every station
    passes the same Q, but the profile is peakier at some depths than others,
    so an individual column has du/dz != 0.  A merely depth-varying *magnitude*
    would not do - normalising to a common Q would flatten it back out.
    """
    area = np.array([[1.0, 2, 3, 4], [2, 3, 4, 5], [1, 1, 1, 1]])[:n_layer, :n_azimuth]
    # Two profile shapes: near-plug and strongly peaked.
    flat = np.ones((n_layer, n_azimuth))
    peaked = np.zeros((n_layer, n_azimuth))
    peaked[n_layer // 2] = 3.0
    peaked += 0.2
    z = np.arange(nz)
    w = 0.5 * (1.0 - np.cos(2 * np.pi * z / nz))  # blends 0 -> 1 -> 0 with depth
    u = (1.0 - w)[:, None, None] * flat + w[:, None, None] * peaked
    q = 0.5 * float(area.sum())  # so that u is O(0.5) m/s
    u *= (q / np.einsum("klm,lm->k", u, area))[:, None, None]
    assert np.max(np.abs(np.einsum("klm,lm->k", u, area) - q)) < 1e-13 * q
    assert np.max(np.abs(np.diff(u, axis=0))) > 1e-3 * np.max(u), "case is not divergent"
    fields = np.zeros((2, nz, n_layer, n_azimuth))
    fields[0, : nz // 2] = 1.0
    fields[1] = 1.0 - fields[0]
    return fields, u, area, q


@pytest.mark.parametrize("closure", ["none", "local", "redistribute"])
def test_all_closures_are_inert_for_a_uniform_profile(closure):
    """No closure may change anything when div(u) = 0."""
    nz = 100
    fields = np.zeros((2, nz, 2, 2))
    fields[0, :50] = 1.0
    fields[1] = 1.0 - fields[0]
    u = np.full((nz, 2, 2), 0.5)
    area = np.ones((2, 2))
    dt = CFL * DZ / 0.5
    ref = advect_multi(fields, u, DZ, dt, inlet_values=[1.0, 0.0], closure="none")
    got = advect_multi(fields, u, DZ, dt, inlet_values=[1.0, 0.0],
                       closure=closure, area=area)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-16)


def test_redistribute_holds_both_invariants_where_the_others_cannot():
    """The point of the closure: sum-to-one *and* per-fluid volume, together."""
    fields, u, area, q = _divergent_case()
    dt = CFL * DZ / float(np.max(np.abs(u)))
    state = {c: fields.copy() for c in ("none", "local", "redistribute")}
    v0 = np.einsum("iklm,lm->i", fields, area) * DZ
    influx = np.zeros(2)
    outflux = {c: np.zeros(2) for c in state}

    for _ in range(150):
        influx[0] += float((u[0] * area).sum()) * dt
        for c, f in state.items():
            outflux[c] += np.array(
                [float((u[-1] * area * f[i, -1]).sum()) * dt for i in range(2)]
            )
            state[c] = advect_multi(f, u, DZ, dt, inlet_values=[1.0, 0.0],
                                    closure=c, area=area)

    report = {}
    for c, f in state.items():
        v1 = np.einsum("iklm,lm->i", f, area) * DZ
        report[c] = (
            float(np.max(np.abs(f.sum(axis=0) - 1.0))),
            float(np.max(np.abs(v1 - (v0 + influx - outflux[c]))) / v0.sum()),
        )
    s1_red, mass_red = report["redistribute"]
    assert s1_red < 1e-12, f"redistribute lost sum-to-one: {s1_red:.2e}"
    assert mass_red < 1e-12, f"redistribute lost mass: {mass_red:.2e}"
    # And each of the other two fails exactly one of the invariants.
    assert report["none"][0] > 1e-4, "expected 'none' to break sum-to-one"
    assert report["local"][1] > 1e-4, "expected 'local' to break per-fluid mass"


def test_redistribute_stays_bounded():
    fields, u, area, q = _divergent_case()
    dt = CFL * DZ / float(np.max(np.abs(u)))
    f = fields.copy()
    for _ in range(300):
        f = advect_multi(f, u, DZ, dt, inlet_values=[1.0, 0.0],
                         closure="redistribute", area=area)
        assert f.min() > -1e-14 and f.max() < 1.0 + 1e-14


def test_redistribute_requires_area_and_rejects_unknown_closures():
    fields, u, area, q = _divergent_case(nz=20)
    with pytest.raises(ValueError, match="needs the cell area"):
        advect_multi(fields, u, DZ, 1e-3, closure="redistribute")
    with pytest.raises(ValueError, match="unknown transverse closure"):
        advect_multi(fields, u, DZ, 1e-3, closure="magic", area=area)


def test_outlet_is_zero_gradient_in_both_directions():
    """A reversed outlet face must not inject the inlet fluid at the shoe."""
    nz = 40
    f = np.zeros(column(nz))
    f[:20] = 1.0
    u = np.full(column(nz), -0.5)  # everything flowing backwards
    dt = CFL * DZ / 0.5
    out = advect(f, u, DZ, dt, inlet_value=1.0)
    # The last cell holds fluid 2 (f = 0) and draws its own composition back in,
    # so it must stay at 0 rather than picking up the inlet's f = 1.
    assert out[-1, 0, 0] == pytest.approx(0.0, abs=1e-15)


def test_reversed_inlet_lets_fluid_leave_through_the_top():
    nz = 40
    f = np.full(column(nz), 1.0)
    u = np.full(column(nz), -0.5)
    out = advect(f, u, DZ, CFL * DZ / 0.5, inlet_value=0.0)
    # Uniform field, uniform velocity: upwinding at both boundaries keeps it flat.
    np.testing.assert_allclose(out, 1.0, rtol=0.0, atol=1e-15)
