"""Gravity in the cross-section - Dai et al. (2024) Eqs. 4, A.10-A.18."""

import math

import numpy as np
import pytest

from inpipe.fluid import Fluid
from inpipe.segregation import (
    atwood,
    cross_section_regime,
    effective_viscosity,
    inertial_velocity,
    mix_uniformly,
    segregate,
)

MUD = Fluid("mud", 1198.0, 2.0, 0.30, 0.72)
CEMENT = Fluid("cement", 1870.0, 6.0, 0.55, 0.65)


# --- the inertial velocity, and why a vertical well is a special case -------


def test_atwood_number_matches_the_paper_definition():
    assert atwood(1870.0, 1198.0) == pytest.approx(672.0 / 3068.0)
    assert atwood(1500.0, 1500.0) == 0.0


def test_a_vertical_well_has_no_transverse_buoyancy_at_all():
    """``sin(beta)`` is the whole story, and it is zero.

    Buoyancy can only stratify a cross-section if gravity has a component *in*
    it.  A vertical well has none, so the paper's inertial velocity vanishes
    and both its gravity mechanisms switch themselves off.  This is not an
    approximation - it is the geometry.
    """
    at = atwood(CEMENT.rho, MUD.rho)
    assert inertial_velocity(at, 0.0872, inclination=0.0) == 0.0
    assert inertial_velocity(at, 0.0872, math.radians(90.0)) > 0.4


def test_the_inertial_velocity_grows_with_inclination():
    at = atwood(CEMENT.rho, MUD.rho)
    v = [inertial_velocity(at, 0.0872, math.radians(b)) for b in (0, 30, 60, 90)]
    assert v == sorted(v)
    assert v[3] == pytest.approx(math.sqrt(at * 9.80665 * 0.0872))


def test_effective_viscosity_is_the_papers_nominal_form():
    """Eq. A.11 - built from bulk velocity and diameter, not a shear solve."""
    mu = effective_viscosity(MUD, velocity=0.3, diameter=0.09)
    expected = 2.0 * 0.09 / 0.3 + 0.30 * (0.3 / 0.09) ** (0.72 - 1.0)
    assert mu == pytest.approx(expected)


# --- the regime switches ----------------------------------------------------


def test_a_vertical_section_neither_segregates_nor_mixes():
    """The K-GEP-1 case: the paper's machinery is inert here, by geometry."""
    reg = cross_section_regime(CEMENT, MUD, velocity=0.277, diameter=0.0872,
                               inclination=0.0, density_stable=True)
    assert reg.inertial_velocity == 0.0
    assert math.isinf(reg.froude)
    assert reg.reynolds_inertial == 0.0
    assert not reg.segregates
    assert not reg.mixes


def test_a_horizontal_section_segregates():
    reg = cross_section_regime(CEMENT, MUD, velocity=0.277, diameter=0.0872,
                               inclination=math.radians(90.0), density_stable=False)
    assert reg.segregates
    assert reg.reynolds_inertial > 1.0


def test_density_stable_flow_is_spared_the_buoyancy_instabilities():
    """The paper says so outright, and it matters for a vertical cement job.

    Cement is denser than the mud it lifts, so the annulus is density-stable
    and the buoyancy-driven criteria do not apply - only turbulence can mix it.
    """
    # Fast enough that a buoyancy criterion actually fires (Eq. A.15 here);
    # at the job's own 0.277 m/s nothing fires either way, because a slow
    # viscous flow is stable whichever fluid is on top.
    kw = dict(velocity=2.0, diameter=0.0872, inclination=math.radians(45.0))
    stable = cross_section_regime(CEMENT, MUD, density_stable=True, **kw)
    unstable = cross_section_regime(CEMENT, MUD, density_stable=False, **kw)
    assert unstable.exchange_instability, "this test needs a criterion to fire"
    assert not stable.turbulent
    assert not stable.mixes         # spared, because it is density-stable
    assert unstable.mixes           # same numbers, only the stability differs


def test_a_slow_viscous_section_is_stable_whichever_fluid_is_on_top():
    """Density-unstable is not the same as unstable.

    At the job's own velocity nothing fires: Re is 27 and every criterion wants
    more.  Treating "heavy over light" as automatically mixing would smear the
    interface for no reason.
    """
    kw = dict(velocity=0.277, diameter=0.0872, inclination=math.radians(45.0))
    unstable = cross_section_regime(CEMENT, MUD, density_stable=False, **kw)
    assert unstable.reynolds < 50.0
    assert not unstable.mixes


def test_turbulence_mixes_regardless_of_stability():
    """Eq. A.16 is not a buoyancy criterion, so stability does not exempt it."""
    thin = Fluid("water", 998.0, 0.0, 1.0e-3, 1.0)
    reg = cross_section_regime(thin, thin, velocity=2.0, diameter=0.15,
                               inclination=0.0, density_stable=True)
    assert reg.turbulent and reg.mixes


# --- the two rearrangements -------------------------------------------------


def _section(n_layer=4, n_azimuth=3, volume=1.0):
    return np.full((n_layer, n_azimuth), volume)


def test_segregation_puts_the_heavy_fluid_at_the_bottom():
    """Layer 0 is the bottom of the section, so it fills with cement first."""
    vol = _section()
    f = np.zeros((2, 4, 3))
    f[0] = 0.5          # mud, everywhere
    f[1] = 0.5          # cement, everywhere
    out = segregate(f, vol, densities=[MUD.rho, CEMENT.rho])
    assert out[1, 0].mean() == pytest.approx(1.0)     # cement fills the bottom
    assert out[0, -1].mean() == pytest.approx(1.0)    # mud left at the top


def test_segregation_conserves_every_fluid_exactly():
    """It runs inside a solver whose correctness rests on that."""
    rng = np.random.default_rng(0)
    vol = _section(volume=2.0) * rng.uniform(0.5, 1.5, (4, 3))
    raw = rng.uniform(size=(3, 4, 3))
    f = raw / raw.sum(axis=0)
    before = np.einsum("ilm,lm->i", f, vol)
    out = segregate(f, vol, densities=[1000.0, 1500.0, 1870.0])
    after = np.einsum("ilm,lm->i", out, vol)
    assert after == pytest.approx(before, rel=1e-12)
    assert out.sum(axis=0) == pytest.approx(np.ones((4, 3)), rel=1e-12)


def test_mixing_conserves_every_fluid_and_flattens_the_section():
    vol = _section()
    f = np.zeros((2, 4, 3))
    f[1, :2] = 1.0
    f[0, 2:] = 1.0
    before = np.einsum("ilm,lm->i", f, vol)
    out = mix_uniformly(f, vol)
    assert np.einsum("ilm,lm->i", out, vol) == pytest.approx(before, rel=1e-12)
    assert np.allclose(out[0], out[0, 0, 0])
    assert out.sum(axis=0) == pytest.approx(np.ones((4, 3)), rel=1e-12)


def test_segregating_an_already_uniform_single_fluid_changes_nothing():
    vol = _section()
    f = np.zeros((2, 4, 3))
    f[0] = 1.0
    out = segregate(f, vol, densities=[MUD.rho, CEMENT.rho])
    assert out == pytest.approx(f)
