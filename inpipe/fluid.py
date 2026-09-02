"""Module 1 - Herschel-Bulkley fluids and the pump schedule.

Rheology (paper Eq. 1)::

    tau = tau0 + k * gammadot**n   if tau >  tau0
    gammadot = 0                   if tau <= tau0

All quantities are SI.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Fluid:
    """A Herschel-Bulkley fluid.

    Parameters
    ----------
    name : label used in diagnostics and plots.
    rho : density [kg/m^3].
    tau0 : yield stress [Pa].
    k : consistency index [Pa.s^n].
    n : flow behaviour index [-].
    """

    name: str
    rho: float
    tau0: float
    k: float
    n: float

    def __post_init__(self) -> None:
        if self.rho <= 0.0:
            raise ValueError(f"density must be positive, got {self.rho}")
        if self.tau0 < 0.0:
            raise ValueError(f"yield stress must be non-negative, got {self.tau0}")
        if self.k <= 0.0:
            raise ValueError(f"consistency index must be positive, got {self.k}")
        if not 0.0 < self.n <= 2.0:
            raise ValueError(f"flow index out of plausible range, got {self.n}")

    # -- convenience constructors ------------------------------------------

    @classmethod
    def newtonian(cls, rho: float, mu: float, name: str = "newtonian") -> "Fluid":
        return cls(name=name, rho=rho, tau0=0.0, k=mu, n=1.0)

    @classmethod
    def bingham(
        cls, rho: float, mu_p: float, tau0: float, name: str = "bingham"
    ) -> "Fluid":
        return cls(name=name, rho=rho, tau0=tau0, k=mu_p, n=1.0)

    @classmethod
    def power_law(cls, rho: float, k: float, n: float, name: str = "power_law") -> "Fluid":
        return cls(name=name, rho=rho, tau0=0.0, k=k, n=n)

    # -- derived -----------------------------------------------------------

    @property
    def is_newtonian(self) -> bool:
        return self.tau0 == 0.0 and self.n == 1.0

    def apparent_viscosity(self, shear_rate: float) -> float:
        """mu = tau / gammadot for a given shear rate [Pa.s].

        Diverges as shear_rate -> 0 for a yield-stress fluid; that is physical,
        not a bug.
        """
        if shear_rate <= 0.0:
            raise ValueError("shear rate must be positive")
        return self.tau0 / shear_rate + self.k * shear_rate ** (self.n - 1.0)

    def with_name(self, name: str) -> "Fluid":
        return Fluid(name=name, rho=self.rho, tau0=self.tau0, k=self.k, n=self.n)


def mix_fluids(fluids: Sequence[Fluid], fractions: Sequence[float], name: str = "mixture") -> Fluid:
    """Volume-fraction-weighted average of rho, tau0, k and n.

    The paper (Appendix A.1) states only that "averaged rheological parameters
    and density of fluids are used"; the averaging rule is our choice.

    WARNING: averaging ``n`` is not physically rigorous - it is an exponent,
    not an extensive property.  Logged as assumption A-06 and flagged as a
    candidate for a sensitivity study.
    """
    if len(fluids) != len(fractions):
        raise ValueError("fluids and fractions must have equal length")
    total = float(sum(fractions))
    if total <= 0.0:
        raise ValueError("fractions must sum to a positive value")
    w = [f / total for f in fractions]
    return Fluid(
        name=name,
        rho=sum(wi * fl.rho for wi, fl in zip(w, fluids)),
        tau0=sum(wi * fl.tau0 for wi, fl in zip(w, fluids)),
        k=sum(wi * fl.k for wi, fl in zip(w, fluids)),
        n=sum(wi * fl.n for wi, fl in zip(w, fluids)),
    )


@dataclass(frozen=True)
class PumpStage:
    """One stage of the pump schedule."""

    fluid: Fluid
    volume: float  # m^3
    flow_rate: float  # m^3/s

    def __post_init__(self) -> None:
        if self.volume <= 0.0:
            raise ValueError("stage volume must be positive")
        if self.flow_rate <= 0.0:
            raise ValueError("stage flow rate must be positive")

    @property
    def duration(self) -> float:
        """Stage duration [s]."""
        return self.volume / self.flow_rate


class PumpSchedule:
    """An ordered list of pump stages.

    Time ``t`` is measured from the start of pumping.  Stage ``i`` occupies the
    half-open interval ``[t_i, t_{i+1})``; at a stage boundary the *later*
    stage is returned.  Past the end of the schedule the last stage is held
    (the pump keeps running on the tail fluid) - callers that want the job to
    stop should compare ``t`` against :attr:`total_time`.
    """

    def __init__(self, stages: Sequence[PumpStage]):
        if not stages:
            raise ValueError("pump schedule needs at least one stage")
        self.stages = list(stages)
        # Cumulative stage start times.
        self._starts = [0.0]
        for st in self.stages:
            self._starts.append(self._starts[-1] + st.duration)

    @property
    def total_time(self) -> float:
        return self._starts[-1]

    @property
    def total_volume(self) -> float:
        return sum(st.volume for st in self.stages)

    def stage_index_at(self, t: float) -> int:
        """Index of the stage active at time ``t``."""
        if t < 0.0:
            raise ValueError("time must be non-negative")
        if t >= self._starts[-1]:
            return len(self.stages) - 1
        # bisect_right on interior boundaries: t exactly on a boundary selects
        # the later stage.
        return bisect_right(self._starts, t) - 1

    def rate_at(self, t: float) -> float:
        """Imposed volumetric flow rate at time ``t`` [m^3/s]."""
        return self.stages[self.stage_index_at(t)].flow_rate

    def fluid_at_inlet(self, t: float) -> Fluid:
        """Fluid entering the pipe at time ``t``."""
        return self.stages[self.stage_index_at(t)].fluid

    def stage_start(self, i: int) -> float:
        return self._starts[i]

    def __len__(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        names = ", ".join(f"{s.fluid.name}:{s.volume:.3g}m3" for s in self.stages)
        return f"PumpSchedule([{names}], total={self.total_time:.4g}s)"
