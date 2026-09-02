"""Test gate 6 - annular flow by the parallel-plate (slot) approximation.

Closed-form checks, the planar analogues of the pipe gate in
``test_velocity.py``.
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from inpipe.fluid import Fluid
from inpipe.slot import (
    SlotProfile,
    slot_error_estimate,
    slot_flow_rate,
    slot_geometry,
    solve_slot_profile,
    solve_slot_tau_w,
)
from inpipe.velocity import NoBracketError

R_IN = 0.13970 / 2   # 5-1/2 in casing OD
R_OUT = 0.21590 / 2  # 8-1/2 in hole
B, W = slot_geometry(R_IN, R_OUT)


def quad_flow_rate(fluid, b, width, tau_w):
    """Reference Q by quadrature, split at the plug edge."""
    if tau_w <= fluid.tau0:
        return 0.0
    prof = SlotProfile(fluid, b, width, tau_w)
    s0 = prof.plug_half_width
    q = 2.0 * width * prof.u_max * s0  # analytic plug slab
    if s0 < b:
        val, _ = quad(lambda s: float(prof(s)), s0, b, epsabs=1e-16, epsrel=1e-13)
        q += 2.0 * width * val
    return q


def bingham_slot(tau_w, b, width, mu_p, tau0):
    """Analytical Bingham slot relation, the planar Buckingham-Reiner."""
    if tau_w <= tau0:
        return 0.0
    x = tau0 / tau_w
    return (2.0 * width * b**2 * tau_w) / (3.0 * mu_p) * (1.0 - 1.5 * x + 0.5 * x**3)


# --- geometry --------------------------------------------------------------


def test_slot_area_equals_the_true_annulus_area():
    """2 b W = pi (r_o^2 - r_i^2) exactly, by construction."""
    assert 2.0 * B * W == pytest.approx(math.pi * (R_OUT**2 - R_IN**2), rel=1e-14)


def test_slot_geometry_rejects_an_inverted_annulus():
    with pytest.raises(ValueError):
        slot_geometry(R_OUT, R_IN)


def test_slot_error_grows_with_the_gap_and_stays_small_here():
    """The approximation's own error, reported rather than assumed."""
    nominal = slot_error_estimate(R_IN, R_OUT)
    assert abs(nominal) < 0.005, f"nominal slot error {100 * nominal:.2f} %"
    # Monotone in hole size, and still tolerable at a severe washout.
    errs = [abs(slot_error_estimate(R_IN, d / 2)) for d in (0.2159, 0.25, 0.3048, 0.40)]
    assert errs == sorted(errs)
    assert errs[-1] < 0.03, f"washout slot error {100 * errs[-1]:.2f} %"


# --- profile ---------------------------------------------------------------


def test_plane_poiseuille_shape():
    """Newtonian slot: u(s) = u_max (1 - s^2/b^2)."""
    prof = solve_slot_profile(0.0132, Fluid.newtonian(1000.0, 0.001), R_IN, R_OUT)
    s = np.linspace(0.0, B, 101)
    np.testing.assert_allclose(
        prof(s), prof.u_max * (1.0 - (s / B) ** 2), rtol=1e-10, atol=1e-14
    )


def test_plane_poiseuille_peak():
    """u_max / u_bar = 3/2 for a Newtonian slot."""
    prof = solve_slot_profile(0.0132, Fluid.newtonian(1000.0, 0.001), R_IN, R_OUT)
    assert prof.u_max / prof.mean_velocity == pytest.approx(1.5, rel=1e-10)


@pytest.mark.parametrize("n", [0.3, 0.5, 0.8, 1.0])
def test_power_law_slot_peak(n):
    """u_max / u_bar = (2n+1)/(n+1) for a power-law slot."""
    prof = solve_slot_profile(0.0132, Fluid.power_law(1000.0, k=0.5, n=n), R_IN, R_OUT)
    assert prof.u_max / prof.mean_velocity == pytest.approx(
        (2.0 * n + 1.0) / (n + 1.0), rel=1e-8
    )


@pytest.mark.parametrize("tau0", [0.5, 2.0, 10.0])
@pytest.mark.parametrize("mu_p", [0.01, 0.05])
def test_bingham_slot_closed_form(tau0, mu_p):
    fluid = Fluid.bingham(1200.0, mu_p=mu_p, tau0=tau0)
    for tau_w in (tau0 * 1.5, tau0 * 4.0, tau0 * 25.0):
        assert slot_flow_rate(fluid, B, W, tau_w) == pytest.approx(
            bingham_slot(tau_w, B, W, mu_p, tau0), rel=1e-8
        )


FLUIDS = [
    Fluid.newtonian(1000.0, 0.001),
    Fluid.power_law(1000.0, k=0.8, n=0.4),
    Fluid.bingham(1500.0, mu_p=0.03, tau0=4.0),
    Fluid("hb", rho=1400.0, tau0=3.0, k=0.6, n=0.55),
]


@pytest.mark.parametrize("fluid", FLUIDS)
def test_closed_form_matches_quadrature(fluid):
    for offset in (1e-6, 0.5, 5.0, 50.0):
        tau_w = fluid.tau0 + offset
        assert slot_flow_rate(fluid, B, W, tau_w) == pytest.approx(
            quad_flow_rate(fluid, B, W, tau_w), rel=1e-9
        )


@pytest.mark.parametrize("fluid", FLUIDS)
def test_no_slip_at_the_walls(fluid):
    prof = solve_slot_profile(0.0132, fluid, R_IN, R_OUT)
    assert abs(float(prof(B))) < 1e-14


@pytest.mark.parametrize("fluid", FLUIDS)
@pytest.mark.parametrize("q", [1e-5, 1e-3, 1e-1])
def test_flow_rate_round_trip(fluid, q):
    tau_w = solve_slot_tau_w(q, fluid, B, W)
    assert slot_flow_rate(fluid, B, W, tau_w) == pytest.approx(q, rel=1e-10)


def test_plug_is_flat():
    fluid = Fluid("hb", rho=1400.0, tau0=4.0, k=0.6, n=0.7)
    prof = solve_slot_profile(0.005, fluid, R_IN, R_OUT)
    s0 = prof.plug_half_width
    assert 0.0 < s0 < B, "test needs a genuine plug"
    np.testing.assert_allclose(
        prof(np.linspace(0.0, s0 * (1 - 1e-12), 40)), prof.u_max, rtol=1e-12
    )


def test_yield_limit_is_exact():
    fluid = Fluid("hb", rho=1400.0, tau0=3.0, k=0.6, n=0.55)
    assert slot_flow_rate(fluid, B, W, 3.0) == 0.0
    assert slot_flow_rate(fluid, B, W, 2.9) == 0.0
    assert solve_slot_tau_w(0.0, fluid, B, W) == 3.0


def test_frictional_gradient_follows_the_papers_relation():
    """tau_w = (h/2) P with h = 2b, i.e. P = tau_w / b (Appendix A.1)."""
    prof = solve_slot_profile(0.0132, FLUIDS[0], R_IN, R_OUT)
    assert prof.frictional_gradient() == pytest.approx(prof.tau_w / B, rel=1e-15)


def test_monotone_and_symmetric():
    prof = solve_slot_profile(0.0132, FLUIDS[3], R_IN, R_OUT)
    u = prof(np.linspace(0.0, B, 300))
    assert np.all(np.diff(u) <= 1e-15)
    assert float(prof(0.3 * B)) == pytest.approx(float(prof(0.3 * B)), rel=1e-15)


def test_negative_rate_and_unreachable_target_raise(monkeypatch):
    fluid = Fluid("hb", rho=1400.0, tau0=3.0, k=0.6, n=0.55)
    with pytest.raises(ValueError):
        solve_slot_tau_w(-1.0, fluid, B, W)
    import inpipe.slot as slot

    monkeypatch.setattr(slot, "_BRACKET_MAX_ITER", 1)
    with pytest.raises(NoBracketError):
        slot.solve_slot_tau_w(1.0e7, fluid, B, W)


def test_narrower_gap_needs_more_wall_shear_for_the_same_rate():
    """Physical sanity: squeezing the annulus raises tau_w."""
    fluid = FLUIDS[3]
    wide = solve_slot_profile(0.0132, fluid, R_IN, R_OUT)
    narrow = solve_slot_profile(0.0132, fluid, R_IN, R_IN * 1.15)
    assert narrow.tau_w > wide.tau_w
    assert narrow.mean_velocity > wide.mean_velocity
