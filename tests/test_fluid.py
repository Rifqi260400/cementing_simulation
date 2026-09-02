"""Test gate 1 - fluids, unit conversions and the pump schedule.

The build spec's Section 2 gate: unit helpers must round-trip, and
``PumpSchedule.rate_at`` must return the right stage at the edges - t = 0, t
exactly on a boundary, and t past the end.
"""


import numpy as np
import pytest

from inpipe.config import (
    bpm_to_m3s,
    cp_to_pas,
    ft_to_m,
    inch_to_m,
    kgm3_to_ppg,
    lbf100ft2_to_pa,
    m3s_to_bpm,
    m_to_ft,
    m_to_inch,
    pa_to_lbf100ft2,
    pa_to_psi,
    pas_to_cp,
    ppg_to_kgm3,
    psi_to_pa,
)
from inpipe.fluid import Fluid, PumpSchedule, PumpStage, mix_fluids

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

ROUND_TRIPS = [
    (bpm_to_m3s, m3s_to_bpm, 5.0),
    (ft_to_m, m_to_ft, 5000.0),
    (inch_to_m, m_to_inch, 5.0),
    (cp_to_pas, pas_to_cp, 1.0),
    (ppg_to_kgm3, kgm3_to_ppg, 8.33),
    (psi_to_pa, pa_to_psi, 5000.0),
    (lbf100ft2_to_pa, pa_to_lbf100ft2, 12.0),
]


@pytest.mark.parametrize("fwd,back,value", ROUND_TRIPS)
def test_unit_helpers_round_trip(fwd, back, value):
    assert back(fwd(value)) == pytest.approx(value, rel=1e-14)


def test_newtonian_fluid_round_trips_through_unit_helpers():
    """The spec's gate-1 example: Fluid.newtonian(1000, 0.001) survives a
    round trip through the unit helpers."""
    fluid = Fluid.newtonian(1000.0, 0.001, "water")
    rho_field = kgm3_to_ppg(fluid.rho)
    mu_field = pas_to_cp(fluid.k)
    recovered = Fluid.newtonian(ppg_to_kgm3(rho_field), cp_to_pas(mu_field), "water")
    assert recovered.rho == pytest.approx(1000.0, rel=1e-14)
    assert recovered.k == pytest.approx(0.001, rel=1e-14)
    assert recovered == fluid


def test_conversions_match_the_paper_and_api_reference_values():
    """Anchor the conversions to values quoted in the source paper."""
    # Dai et al. quote "2 bpm (0.318 m3/min)" and "5 bpm (0.795 m3/min)".
    assert bpm_to_m3s(2.0) * 60.0 == pytest.approx(0.318, abs=5e-4)
    assert bpm_to_m3s(5.0) * 60.0 == pytest.approx(0.795, abs=5e-4)
    # "5 in., or 127 mm" and "5000 ft (1524 m)".
    assert inch_to_m(5.0) == pytest.approx(0.127, rel=1e-12)
    assert ft_to_m(5000.0) == pytest.approx(1524.0, rel=1e-12)
    assert ft_to_m(3.0) == pytest.approx(0.9144, rel=1e-12)
    # Fresh water is 8.33 ppg.
    assert ppg_to_kgm3(8.33) == pytest.approx(998.2, abs=1.0)
    assert cp_to_pas(1.0) == pytest.approx(1e-3, rel=1e-15)


# ---------------------------------------------------------------------------
# Fluid model
# ---------------------------------------------------------------------------


def test_convenience_constructors():
    n = Fluid.newtonian(1000.0, 0.002)
    assert (n.tau0, n.k, n.n) == (0.0, 0.002, 1.0) and n.is_newtonian

    b = Fluid.bingham(1500.0, mu_p=0.03, tau0=4.0)
    assert (b.tau0, b.k, b.n) == (4.0, 0.03, 1.0) and not b.is_newtonian

    p = Fluid.power_law(1000.0, k=0.8, n=0.4)
    assert (p.tau0, p.k, p.n) == (0.0, 0.8, 0.4) and not p.is_newtonian


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rho=-1.0, tau0=0.0, k=1e-3, n=1.0),
        dict(rho=1000.0, tau0=-1.0, k=1e-3, n=1.0),
        dict(rho=1000.0, tau0=0.0, k=0.0, n=1.0),
        dict(rho=1000.0, tau0=0.0, k=1e-3, n=0.0),
        dict(rho=1000.0, tau0=0.0, k=1e-3, n=5.0),
    ],
)
def test_fluid_rejects_unphysical_parameters(kwargs):
    with pytest.raises(ValueError):
        Fluid(name="bad", **kwargs)


def test_apparent_viscosity_matches_the_constitutive_law():
    """mu = tau0/gammadot + k gammadot^(n-1), i.e. tau = tau0 + k gammadot^n."""
    fluid = Fluid("hb", rho=1400.0, tau0=3.0, k=0.6, n=0.55)
    for gd in (0.1, 1.0, 100.0):
        tau = fluid.tau0 + fluid.k * gd**fluid.n
        assert fluid.apparent_viscosity(gd) == pytest.approx(tau / gd, rel=1e-14)
    with pytest.raises(ValueError):
        fluid.apparent_viscosity(0.0)


def test_newtonian_apparent_viscosity_is_shear_independent():
    fluid = Fluid.newtonian(1000.0, 0.002)
    assert fluid.apparent_viscosity(0.1) == pytest.approx(0.002, rel=1e-14)
    assert fluid.apparent_viscosity(1000.0) == pytest.approx(0.002, rel=1e-14)


# ---------------------------------------------------------------------------
# Volume-weighted mixing (assumption A-06)
# ---------------------------------------------------------------------------


def test_mix_fluids_is_volume_weighted_and_normalises():
    a = Fluid("a", rho=1000.0, tau0=1.0, k=0.2, n=0.6)
    b = Fluid("b", rho=2000.0, tau0=3.0, k=0.6, n=1.0)
    m = mix_fluids([a, b], [1.0, 3.0])
    assert m.rho == pytest.approx(1750.0, rel=1e-14)
    assert m.tau0 == pytest.approx(2.5, rel=1e-14)
    assert m.k == pytest.approx(0.5, rel=1e-14)
    assert m.n == pytest.approx(0.9, rel=1e-14)
    # Unnormalised weights give the same answer as normalised ones.
    assert mix_fluids([a, b], [0.25, 0.75]) == m


def test_mix_fluids_of_one_fluid_is_that_fluid():
    a = Fluid("a", rho=1000.0, tau0=1.0, k=0.2, n=0.6)
    m = mix_fluids([a, a], [0.3, 0.7])
    for attr in ("rho", "tau0", "k", "n"):
        assert getattr(m, attr) == pytest.approx(getattr(a, attr), rel=1e-14)


def test_mix_fluids_rejects_bad_input():
    a = Fluid.newtonian(1000.0, 1e-3)
    with pytest.raises(ValueError):
        mix_fluids([a, a], [1.0])
    with pytest.raises(ValueError):
        mix_fluids([a, a], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Pump schedule - the edges, explicitly
# ---------------------------------------------------------------------------


@pytest.fixture
def schedule():
    mud = Fluid.newtonian(1000.0, 0.001, "mud")
    spacer = Fluid.newtonian(1100.0, 0.002, "spacer")
    cement = Fluid.newtonian(1800.0, 0.005, "cement")
    return PumpSchedule(
        [
            PumpStage(mud, volume=2.0, flow_rate=0.5),      # 0 s  -> 4 s
            PumpStage(spacer, volume=3.0, flow_rate=1.5),   # 4 s  -> 6 s
            PumpStage(cement, volume=4.0, flow_rate=0.8),   # 6 s  -> 11 s
        ]
    )


def test_stage_durations_and_totals(schedule):
    assert [s.duration for s in schedule.stages] == pytest.approx([4.0, 2.0, 5.0])
    assert schedule.total_time == pytest.approx(11.0)
    assert schedule.total_volume == pytest.approx(9.0)
    assert len(schedule) == 3


def test_rate_at_t_zero(schedule):
    assert schedule.rate_at(0.0) == 0.5
    assert schedule.fluid_at_inlet(0.0).name == "mud"


@pytest.mark.parametrize(
    "t,rate,name",
    [
        (3.999999, 0.5, "mud"),
        (4.0, 1.5, "spacer"),   # exactly on a boundary -> the later stage
        (4.000001, 1.5, "spacer"),
        (5.999999, 1.5, "spacer"),
        (6.0, 0.8, "cement"),
        (6.000001, 0.8, "cement"),
    ],
)
def test_rate_at_stage_boundaries(schedule, t, rate, name):
    assert schedule.rate_at(t) == rate
    assert schedule.fluid_at_inlet(t).name == name


def test_rate_at_past_the_end_holds_the_last_stage(schedule):
    assert schedule.rate_at(11.0) == 0.8
    assert schedule.rate_at(1e6) == 0.8
    assert schedule.fluid_at_inlet(1e6).name == "cement"
    assert schedule.stage_index_at(1e6) == 2


def test_rate_at_rejects_negative_time(schedule):
    with pytest.raises(ValueError):
        schedule.rate_at(-1e-9)


def test_stage_index_is_monotone_over_the_job(schedule):
    ts = np.linspace(0.0, schedule.total_time, 500)
    idx = [schedule.stage_index_at(t) for t in ts]
    assert idx == sorted(idx)
    assert idx[0] == 0 and idx[-1] == 2


def test_pumped_volume_integrates_back_to_the_stage_volumes(schedule):
    """Integrating rate_at over the job must recover each stage's volume.

    rate_at is piecewise constant, so the midpoint rectangle rule is exact -
    a trapezoid rule would clip the sliver at the closed end of each stage.
    """
    for i, stage in enumerate(schedule.stages):
        t0, t1 = schedule.stage_start(i), schedule.stage_start(i + 1)
        edges = np.linspace(t0, t1, 2001)
        mids = 0.5 * (edges[:-1] + edges[1:])
        rates = np.array([schedule.rate_at(x) for x in mids])
        assert float((rates * np.diff(edges)).sum()) == pytest.approx(
            stage.volume, rel=1e-12
        )


def test_empty_schedule_and_bad_stages_are_rejected():
    mud = Fluid.newtonian(1000.0, 0.001, "mud")
    with pytest.raises(ValueError):
        PumpSchedule([])
    with pytest.raises(ValueError):
        PumpStage(mud, volume=0.0, flow_rate=1.0)
    with pytest.raises(ValueError):
        PumpStage(mud, volume=1.0, flow_rate=0.0)
