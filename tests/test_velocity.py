"""Test gate 2 - closed-form checks on the concentric velocity profile.

Every tolerance below is the one stated in the build spec, Section 3.5.  If any
of these fail the error is in the profile (velocity.py Section "Physics") or in
the flow-rate integral - it must be fixed there, never by a correction factor.
"""

import math

import numpy as np
import pytest

from inpipe.fluid import Fluid
from inpipe.velocity import (
    NoBracketError,
    flow_rate,
    flow_rate_quad,
    plug_radius,
    pressure_gradient,
    solve_profile,
    solve_tau_w,
    velocity_profile,
)

R = 0.05  # m


def buckingham_reiner(tau_w, R, mu_p, tau0):
    """Analytical Bingham Q-tau_w relation."""
    if tau_w <= tau0:
        return 0.0
    x = tau0 / tau_w
    return (math.pi * R**3 * tau_w) / (4.0 * mu_p) * (1.0 - 4.0 / 3.0 * x + x**4 / 3.0)


# ---------------------------------------------------------------------------


def test_poiseuille_shape():
    """Newtonian, tau0 = 0, n = 1  ->  u(r) = u_max (1 - r^2/R^2), rel 1e-10."""
    fluid = Fluid.newtonian(1000.0, 0.001)
    q = 1.0e-4
    prof = solve_profile(q, fluid, R)
    r = np.linspace(0.0, R, 101)
    u = prof(r)
    expected = prof.u_max * (1.0 - (r / R) ** 2)
    np.testing.assert_allclose(u, expected, rtol=1e-10, atol=1e-14)


def test_poiseuille_peak():
    """u_max / u_bar = 2 for a Newtonian pipe flow, rel 1e-10."""
    fluid = Fluid.newtonian(1000.0, 0.001)
    prof = solve_profile(1.0e-4, fluid, R)
    assert prof.u_max / prof.mean_velocity == pytest.approx(2.0, rel=1e-10)


@pytest.mark.parametrize("n", [0.3, 0.5, 0.8, 1.0])
def test_power_law_peak(n):
    """u_max / u_bar = (3n+1)/(n+1) for a power-law fluid, rel 1e-8."""
    fluid = Fluid.power_law(1000.0, k=0.5, n=n)
    prof = solve_profile(1.0e-4, fluid, R)
    expected = (3.0 * n + 1.0) / (n + 1.0)
    assert prof.u_max / prof.mean_velocity == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("tau0", [0.5, 2.0, 10.0])
@pytest.mark.parametrize("mu_p", [0.01, 0.05])
def test_buckingham_reiner(tau0, mu_p):
    """Q(tau_w) matches the Buckingham-Reiner closed form, rel 1e-8."""
    fluid = Fluid.bingham(1200.0, mu_p=mu_p, tau0=tau0)
    for tau_w in [tau0 * 1.5, tau0 * 4.0, tau0 * 25.0]:
        got = flow_rate(fluid, R, tau_w)
        want = buckingham_reiner(tau_w, R, mu_p, tau0)
        assert got == pytest.approx(want, rel=1e-8)


@pytest.mark.parametrize(
    "fluid",
    [
        Fluid.newtonian(1000.0, 0.001),
        Fluid.power_law(1000.0, k=0.8, n=0.4),
        Fluid.bingham(1500.0, mu_p=0.03, tau0=4.0),
        Fluid("hb", rho=1400.0, tau0=3.0, k=0.6, n=0.55),
    ],
)
def test_no_slip(fluid):
    """u(R) = 0 to abs 1e-14 for any fluid."""
    prof = solve_profile(1.0e-4, fluid, R)
    assert abs(prof(R)) < 1e-14


@pytest.mark.parametrize("tau0", [1.0, 5.0])
def test_plug_flatness(tau0):
    """u(r) is constant for r < r0 when tau0 > 0, rel 1e-12."""
    fluid = Fluid("hb", rho=1400.0, tau0=tau0, k=0.6, n=0.7)
    prof = solve_profile(1.0e-4, fluid, R)
    r0 = prof.plug_radius
    assert 0.0 < r0 < R, "test needs a genuine plug"
    r = np.linspace(0.0, r0 * (1.0 - 1e-12), 50)
    u = prof(r)
    np.testing.assert_allclose(u, prof.plug_velocity, rtol=1e-12)
    assert plug_radius(tau0, R, prof.tau_w) == pytest.approx(r0, rel=1e-14)


@pytest.mark.parametrize(
    "fluid",
    [
        Fluid.newtonian(1000.0, 0.001),
        Fluid.power_law(1000.0, k=0.8, n=0.4),
        Fluid.bingham(1500.0, mu_p=0.03, tau0=4.0),
        Fluid("hb", rho=1400.0, tau0=3.0, k=0.6, n=0.55),
    ],
)
@pytest.mark.parametrize("q", [1.0e-6, 1.0e-4, 1.0e-2])
def test_round_trip(fluid, q):
    """Q(solve_tau_w(Q)) == Q, rel 1e-10."""
    tau_w = solve_tau_w(q, fluid, R)
    assert flow_rate(fluid, R, tau_w) == pytest.approx(q, rel=1e-10)


def test_yield_limit_exact():
    """tau_w <= tau0  ->  Q = 0 exactly."""
    fluid = Fluid("hb", rho=1400.0, tau0=3.0, k=0.6, n=0.55)
    assert flow_rate(fluid, R, 3.0) == 0.0
    assert flow_rate(fluid, R, 2.999) == 0.0
    assert np.all(velocity_profile(np.linspace(0, R, 11), fluid, R, 2.5) == 0.0)
    # Zero target rate returns exactly the yield stress.
    assert solve_tau_w(0.0, fluid, R) == 3.0


def test_centreline_equals_plug_velocity():
    fluid = Fluid("hb", rho=1400.0, tau0=2.0, k=0.6, n=0.7)
    prof = solve_profile(1e-4, fluid, R)
    assert prof(0.0) == pytest.approx(prof.plug_velocity, rel=1e-15)


def test_monotone_decreasing():
    """u must decrease monotonically from the centreline to the wall."""
    fluid = Fluid("hb", rho=1400.0, tau0=2.0, k=0.6, n=0.7)
    prof = solve_profile(1e-4, fluid, R)
    u = prof(np.linspace(0.0, R, 500))
    assert np.all(np.diff(u) <= 1e-15)


def test_pressure_gradient_vertical():
    """Vertical pipe: dp/dz = rho g + 2 tau_w / R."""
    rho, g, tau_w = 1500.0, 9.80665, 12.0
    got = pressure_gradient(tau_w, R, rho, beta=0.0, g=g)
    assert got == pytest.approx(rho * g + 2.0 * tau_w / R, rel=1e-15)


def test_pressure_gradient_horizontal_has_no_hydrostatic_term():
    got = pressure_gradient(12.0, R, 1500.0, beta=math.pi / 2.0, g=9.80665)
    assert got == pytest.approx(2.0 * 12.0 / R, rel=1e-12)


def test_solve_tau_w_rejects_negative_rate():
    with pytest.raises(ValueError):
        solve_tau_w(-1.0, Fluid.newtonian(1000.0, 1e-3), R)


def test_flow_rate_monotone_in_tau_w():
    fluid = Fluid("hb", rho=1400.0, tau0=3.0, k=0.6, n=0.55)
    taus = np.linspace(3.001, 60.0, 40)
    qs = [flow_rate(fluid, R, t) for t in taus]
    assert np.all(np.diff(qs) > 0.0)


@pytest.mark.parametrize(
    "fluid",
    [
        Fluid.newtonian(1000.0, 0.001),
        Fluid.power_law(1000.0, k=0.8, n=0.4),
        Fluid.power_law(1000.0, k=0.2, n=1.0),
        Fluid.bingham(1500.0, mu_p=0.03, tau0=4.0),
        Fluid("hb", rho=1400.0, tau0=3.0, k=0.6, n=0.55),
        Fluid("cement", rho=1870.0, tau0=6.0, k=0.55, n=0.65),
    ],
)
def test_closed_form_flow_rate_matches_quadrature(fluid):
    """The closed-form Q(tau_w) equals the spec's quadrature to rel 1e-10.

    This is the check that licenses the deviation logged as A-01: the closed
    form is used in the hot path, the quadrature stands as the independent
    reference.
    """
    for offset in (1e-6, 0.5, 5.0, 50.0, 500.0):
        tau_w = fluid.tau0 + offset
        assert flow_rate(fluid, R, tau_w) == pytest.approx(
            flow_rate_quad(fluid, R, tau_w), rel=1e-10
        )


def test_closed_form_reduces_to_hagen_poiseuille():
    mu = 0.002
    fluid = Fluid.newtonian(1000.0, mu)
    for tau_w in (0.1, 1.0, 25.0):
        assert flow_rate(fluid, R, tau_w) == pytest.approx(
            math.pi * R**3 * tau_w / (4.0 * mu), rel=1e-14
        )


@pytest.mark.parametrize("n", [0.3, 0.5, 0.8, 1.0, 1.2])
def test_closed_form_reduces_to_the_power_law_result(n):
    """Q = pi R^3 n/(3n+1) (tau_w/k)^(1/n) for tau0 = 0."""
    k = 0.7
    fluid = Fluid.power_law(1000.0, k=k, n=n)
    for tau_w in (0.5, 10.0, 200.0):
        want = math.pi * R**3 * n / (3.0 * n + 1.0) * (tau_w / k) ** (1.0 / n)
        assert flow_rate(fluid, R, tau_w) == pytest.approx(want, rel=1e-13)
