"""Herschel-Bulkley regularisation - Fluent's treatment against the exact law."""

import numpy as np
import pytest

import inpipe.rheology as rheology_module
from inpipe.fluid import Fluid
from inpipe.rheology import (
    apparent_viscosity,
    critical_stress,
    effective_consistency,
    plateau_viscosity,
    shear_rate,
    stress,
    stress_moment,
    velocity_integral,
)
from inpipe.slot import SlotProfile, solve_slot_profile
from inpipe.velocity import solve_profile

CEMENT = Fluid("cement slurry", 1200.0, 1.4, 0.6, 0.4)   # Tao et al. Table 1
GC = 5.5
FLUIDS = [
    CEMENT,
    Fluid("bingham", 1500.0, 5.0, 0.03, 1.0),
    Fluid("power-law", 1000.0, 0.0, 0.8, 0.5),
    Fluid("newtonian", 1000.0, 0.0, 1.0e-3, 1.0),
]


# --- the constitutive law ---------------------------------------------------


@pytest.mark.parametrize("fluid", FLUIDS)
@pytest.mark.parametrize("normalise", [False, True])
def test_viscosity_is_continuous_at_the_cut_off(fluid, normalise):
    """Both normalisations join continuously - continuity does not pick one."""
    lo = apparent_viscosity(GC * (1 - 1e-9), fluid, GC, normalise)
    hi = apparent_viscosity(GC * (1 + 1e-9), fluid, GC, normalise)
    assert float(lo) == pytest.approx(float(hi), rel=1e-7)


def test_literal_consistency_preserves_the_papers_own_law():
    """Eq. (14) and Table 1 say tau = 1.4 + 0.6 g^0.4; only literal k gives that."""
    assert effective_consistency(CEMENT, GC, normalise_consistency=False) == CEMENT.k
    for g in (10.0, 50.0, 200.0):
        assert float(stress(g, CEMENT, GC, normalise_consistency=False)) == pytest.approx(
            CEMENT.tau0 + CEMENT.k * g**CEMENT.n, rel=1e-12
        )
    # The normalised reading inflates the consistency by gc^(1-n) = 2.78.
    assert effective_consistency(CEMENT, GC, normalise_consistency=True) == pytest.approx(
        CEMENT.k * GC ** (1 - CEMENT.n), rel=1e-12
    )
    assert effective_consistency(CEMENT, GC, True) / CEMENT.k == pytest.approx(2.781, rel=1e-3)


def test_critical_stress_and_plateau():
    assert critical_stress(CEMENT, GC) == pytest.approx(
        CEMENT.tau0 + CEMENT.k * GC**CEMENT.n, rel=1e-12
    )
    eta0 = plateau_viscosity(CEMENT, GC)
    assert eta0 == pytest.approx(float(apparent_viscosity(1e-12, CEMENT, GC)), rel=1e-6)
    assert np.isfinite(eta0), "the whole point of the regularisation is a finite cap"


@pytest.mark.parametrize("fluid", FLUIDS)
def test_stress_and_shear_rate_round_trip(fluid):
    g = np.array([1e-6, 1e-3, 0.1, 1.0, GC, 2 * GC, 100.0])
    back = shear_rate(stress(g, fluid, GC), fluid, GC)
    np.testing.assert_allclose(back, g, rtol=1e-8)


def test_regularisation_removes_the_plug():
    """The defining difference: nothing is rigid under regularisation."""
    below_yield = np.array([0.2, 0.7, 1.39])
    np.testing.assert_allclose(shear_rate(below_yield, CEMENT, None), 0.0, atol=0.0)
    assert np.all(shear_rate(below_yield, CEMENT, GC) > 0.0)


def test_shear_rate_is_monotone_and_zero_at_zero_stress():
    tau = np.linspace(0.0, 40.0, 400)
    g = shear_rate(tau, CEMENT, GC)
    assert g[0] == 0.0
    assert np.all(np.diff(g) >= -1e-12)


def test_newtonian_is_untouched_by_regularisation():
    """With tau0 = 0 and n = 1 the regularisation is exactly inert."""
    water = Fluid("water", 998.0, 0.0, 1.0e-3, 1.0)
    g = np.array([0.01, 1.0, 5.5, 50.0])
    np.testing.assert_allclose(stress(g, water, GC), water.k * g, rtol=1e-12)
    np.testing.assert_allclose(
        shear_rate(water.k * g, water, GC), g, rtol=1e-12
    )


# --- limits -----------------------------------------------------------------


def test_shrinking_the_cut_off_recovers_the_exact_law():
    """A regularisation must vanish as its parameter does."""
    exact = solve_profile(0.01, CEMENT, 0.08)
    errs = []
    for gc in (1.0, 0.1, 0.01, 1e-3):
        got = solve_profile(0.01, CEMENT, 0.08, gammadot_c=gc)
        errs.append(abs(got.tau_w - exact.tau_w) / exact.tau_w)
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < 1e-6


def test_the_normalised_reading_does_not_have_that_limit():
    """Documented consequence: normalising ties the fluid to gc.

    ``k gc^(1-n) -> 0`` for ``n < 1``, so shrinking gc thins the fluid away
    instead of sharpening the plug.  Recorded so nobody expects convergence.
    """
    exact = solve_profile(0.01, CEMENT, 0.08)
    far = solve_profile(0.01, CEMENT, 0.08, gammadot_c=1e-3)
    from inpipe.velocity import flow_rate, solve_tau_w

    tau_norm = solve_tau_w(0.01, CEMENT, 0.08, gammadot_c=1e-3)
    # With the literal default it converges...
    assert abs(far.tau_w - exact.tau_w) / exact.tau_w < 1e-6
    # ...but the normalised consistency at that gc is essentially zero.
    assert effective_consistency(CEMENT, 1e-3, True) < 0.02 * CEMENT.k
    assert tau_norm > 0.0 and flow_rate(CEMENT, 0.08, tau_norm, 1e-3) > 0.0


# --- integration paths ------------------------------------------------------


def test_velocity_table_matches_direct_quadrature():
    """The cached table is a speed optimisation, not an approximation to accept."""
    tau_w = 3.0
    lower = np.linspace(0.0, tau_w, 41)
    table = velocity_integral(tau_w, lower, CEMENT, GC)
    quad = stress_moment(tau_w, CEMENT, GC, 0, tau_lower=lower)
    assert np.max(np.abs(table - quad)) < 1e-5
    scale = np.max(np.abs(quad))
    assert np.max(np.abs(table - quad)) / scale < 1e-5


@pytest.mark.parametrize("fluid", FLUIDS)
@pytest.mark.parametrize("power", [0, 1, 2])
@pytest.mark.parametrize("tau_w", [1.0e-4, 1.0e-2, 0.2, 3.0, 200.0])
def test_stress_moment_is_converged_at_the_working_node_count(fluid, power, tau_w):
    """Every moment order the solver uses, over the whole stress range.

    ``power = 0`` is the velocity profile, ``1`` the slot flow rate and ``2``
    the pipe flow rate.  The node count is chosen for speed, so it has to be
    pinned against a much finer rule or it becomes a silent modelling change.
    """
    fine = np.polynomial.legendre.leggauss(160)
    working = rheology_module._GAUSS
    for lower in (0.0, 0.5 * tau_w, 0.9 * tau_w):
        coarse = float(stress_moment(tau_w, fluid, GC, power, tau_lower=lower))
        rheology_module._GAUSS = fine
        try:
            reference = float(stress_moment(tau_w, fluid, GC, power, tau_lower=lower))
        finally:
            rheology_module._GAUSS = working
        assert coarse == pytest.approx(reference, rel=1.0e-6, abs=1.0e-300)


def test_stress_moment_is_positive_below_the_critical_stress():
    """The split at the kink must clamp into the interval.

    An interval lying wholly in the regularised branch has nothing above
    ``tau_c``; a yielded piece running from ``tau_c`` back down to ``tau_w``
    would have negative width and subtract a contribution that is not there.
    """
    tau_c = critical_stress(CEMENT, GC)
    for tau_w in (0.1 * tau_c, 0.5 * tau_c, 0.99 * tau_c):
        for power in (0, 1, 2):
            assert stress_moment(tau_w, CEMENT, GC, power) > 0.0


def test_flow_rate_vanishes_with_the_wall_stress():
    """``Q`` must fall to zero as ``tau_w`` does, monotonically.

    ``Q = pi (R/tau_w)^3 int tau^2 gammadot dtau`` divides by a cube, so a
    moment that is even slightly wrong in shape near the origin makes ``Q``
    *diverge* as ``tau_w`` falls.  The wall-stress bracket search walks down
    exactly this way when there is no yield stress to floor it, so a wrong
    limit here overflows rather than converging.
    """
    from inpipe.velocity import flow_rate

    stresses = [1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 1.0e-2, 0.5, 3.0]
    rates = [flow_rate(CEMENT, 0.08, tw, GC) for tw in stresses]
    assert all(b > a for a, b in zip(rates, rates[1:]))
    # Linear in tau_w at the origin: gammadot is linear in tau there, so the
    # moment goes as tau_w^4 and the cube cancels three of them.
    assert rates[1] / rates[0] == pytest.approx(1.0e3, rel=1.0e-3)


@pytest.mark.parametrize("fluid", FLUIDS)
def test_flow_rate_round_trips_under_regularisation(fluid):
    from inpipe.velocity import flow_rate, solve_tau_w

    for q in (1e-4, 1e-2):
        tau_w = solve_tau_w(q, fluid, 0.08, gammadot_c=GC)
        assert flow_rate(fluid, 0.08, tau_w, gammadot_c=GC) == pytest.approx(q, rel=1e-9)


@pytest.mark.parametrize("fluid", FLUIDS)
def test_slot_flow_rate_round_trips_under_regularisation(fluid):
    from inpipe.slot import slot_flow_rate, solve_slot_tau_w, slot_geometry

    b, w = slot_geometry(0.10, 0.20)
    for q in (1e-3, 1e-2):
        tau_w = solve_slot_tau_w(q, fluid, b, w, gammadot_c=GC)
        assert slot_flow_rate(fluid, b, w, tau_w, gammadot_c=GC) == pytest.approx(
            q, rel=1e-9
        )


def test_no_slip_still_holds_under_regularisation():
    for solve, args in ((solve_profile, (0.01, CEMENT, 0.08)),
                        (solve_slot_profile, (0.01, CEMENT, 0.10, 0.20))):
        p = solve(*args, gammadot_c=GC)
        edge = p.radius if hasattr(p, "radius") else p.half_gap
        assert abs(float(p(edge))) < 1e-12


def test_regularised_profile_is_more_peaked_than_the_exact_one():
    """Losing the plug costs displacement: the profile stops being flat.

    On the Tao et al. annulus the exact law puts a plug across 44 % of the gap
    and gives u_max/u_bar = 1.14; Fluent's treatment has no plug and gives 1.36.
    """
    exact = solve_slot_profile(0.01, CEMENT, 0.10, 0.20)
    reg = solve_slot_profile(0.01, CEMENT, 0.10, 0.20, gammadot_c=GC)
    assert exact.plug_half_width > 0.3 * exact.half_gap
    assert reg.plug_half_width == 0.0
    assert reg.u_max / reg.mean_velocity > exact.u_max / exact.mean_velocity
    assert exact.u_max / exact.mean_velocity == pytest.approx(1.14, abs=0.03)
    assert reg.u_max / reg.mean_velocity == pytest.approx(1.36, abs=0.03)


# --- the regularisation must reach the field the solver advects --------------


def _mixture_whose_yield_stress_exceeds_its_wall_stress():
    """A mostly-water mixture carrying a trace of slurry.

    This is what an annulus station holds on the step the cement front reaches
    it, and it is the awkward case: the yield stress inherited from the trace
    of slurry sits *above* the wall stress the flow rate needs, so the exact
    law calls the whole section rigid and the regularised law does not.
    """
    return Fluid("eff", 998.2, 1.4419e-3, 1.6169e-3, 0.99938), 1.0053e-3


def test_annulus_mapping_carries_the_regularisation():
    """The mapped field, not just the reported diagnostics, must be regularised.

    Solving ``tau_w`` under one law and then evaluating the profile under
    another is silent: the wall stress, the plug fraction and ``u_max/u_bar``
    all still read correctly, while the velocity field the solver actually
    advects is a different fluid's.  Here it is not even subtly different - the
    exact law returns zero across the whole station, which breaks discrete
    continuity and takes ``sum_i f_i`` with it.
    """
    from inpipe.annulus_grid import AnnulusGrid
    from inpipe.caliper import CaliperLog
    from inpipe.slot import solve_slot_tau_w

    fluid, q = _mixture_whose_yield_stress_exceeds_its_wall_stress()
    caliper = CaliperLog(np.array([0.0, 1.0]), np.array([0.4, 0.4]))
    grid = AnnulusGrid(1.0, 0.2, caliper, 4, 13, 8)
    b, width = float(grid.half_gap[0]), float(grid.slot_width[0])
    tau_w = solve_slot_tau_w(q, fluid, b, width, gammadot_c=GC)
    assert tau_w < fluid.tau0, "this test is pointless unless the laws disagree"

    profiles = [SlotProfile(fluid, b, width, tau_w, gammadot_c=GC)] * grid.n_axial
    u = grid.map_velocity(profiles, "area_average")
    assert np.all(u[0] > 0.0), "the regularised field must flow everywhere"

    # And it must be the profile's own values, not merely non-zero.
    assert float(np.einsum("lm,lm->", u[0], grid.cell_area[0])) == pytest.approx(
        q, rel=1.0e-3
    )


def test_the_in_pipe_solver_honours_the_regularisation():
    """``NumericsConfig.regularisation_shear_rate`` must not be silently ignored."""
    from inpipe.config import (
        GeometryConfig,
        GridConfig,
        NumericsConfig,
        SimulationConfig,
    )
    from inpipe.fluid import PumpSchedule, PumpStage
    from inpipe.solver import InPipeSolver

    q = 1.0e-3

    def field(gammadot_c):
        config = SimulationConfig(
            geometry=GeometryConfig(length=1.0, inner_diameter=0.16),
            grid=GridConfig(n_axial=8, n_layer=9, n_azimuth=8),
            numerics=NumericsConfig(regularisation_shear_rate=gammadot_c),
        )
        schedule = PumpSchedule([PumpStage(CEMENT, q * 10.0, q)])
        solver = InPipeSolver(config, schedule, initial_fluid=CEMENT)
        return solver.velocity_field(q)

    exact, regularised = field(None), field(GC)
    # No plug under regularisation, so the profile is more peaked.
    assert np.max(regularised) / np.mean(regularised) > np.max(exact) / np.mean(exact)
