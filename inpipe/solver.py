"""Module 6 - the coupled time loop.

Per timestep (spec Section 6.1)::

    Q = schedule.rate_at(t)
    for each depth station z:
        tau_w = solve_tau_w(Q, effective_fluid_at(z))
        u(r)  = velocity_profile(tau_w, fluid)
    map u onto grid cells
    dt = cfl_limit(u, dz)
    advect concentrations f_i
    advect mixing status s
    apply inlet BC from the pump schedule
    record diagnostics

Sign convention: ``+z`` is the flow direction, so ``z = 0`` is the inlet (top of
the casing) and ``z = L`` the shoe.  See :mod:`inpipe.velocity`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import SimulationConfig
from .fluid import Fluid, PumpSchedule
from .grid import Grid
from .transport import (
    advect,
    advect_multi,
    assert_cfl,
    cfl_timestep,
    check_bounded,
    check_sum_to_one,
    numerical_diffusivity,
    upwind_diffusivity,
)
from .velocity import VelocityProfile, solve_tau_w

__all__ = ["Diagnostics", "SimulationResult", "InPipeSolver"]


@dataclass
class Diagnostics:
    """Time series recorded every ``NumericsConfig.diagnostics_every`` steps."""

    time: list[float] = field(default_factory=list)
    step: list[int] = field(default_factory=list)
    #: Volume of each fluid in the pipe [m^3], shape (n_records, n_fluids).
    fluid_volumes: list[np.ndarray] = field(default_factory=list)
    #: Cumulative volume of each fluid pumped in / out through the boundaries.
    influx: list[np.ndarray] = field(default_factory=list)
    outflux: list[np.ndarray] = field(default_factory=list)
    #: Front position (z of the f = 0.5 crossing) per column, for the fluid
    #: currently heading for the shoe.  Shape (n_records, n_layer, n_azimuth).
    front_position: list[np.ndarray] = field(default_factory=list)
    f_min: list[float] = field(default_factory=list)
    f_max: list[float] = field(default_factory=list)
    sum_to_one_error: list[float] = field(default_factory=list)
    #: Numerical diffusivity, both the paper's scale estimate and the
    #: modified-equation value for the active scheme [m^2/s].
    dm_num_paper: list[float] = field(default_factory=list)
    dm_num_scheme: list[float] = field(default_factory=list)
    courant: list[float] = field(default_factory=list)
    dt: list[float] = field(default_factory=list)
    wall_time_per_step: list[float] = field(default_factory=list)

    def as_arrays(self) -> dict:
        return {
            k: np.asarray(v)
            for k, v in self.__dict__.items()
            if isinstance(v, list) and v
        }


@dataclass
class SimulationResult:
    """Everything a run produces."""

    grid: Grid
    fluids: list[Fluid]
    #: Volume fractions, (n_fluids, n_axial, n_layer, n_azimuth).
    fractions: np.ndarray
    #: Mixing status s, (n_axial, n_layer, n_azimuth).
    mixing_status: np.ndarray
    #: Axial cell velocities at the final step, (n_axial, n_layer, n_azimuth).
    velocity: np.ndarray
    diagnostics: Diagnostics
    #: Outlet fluid-fraction time series: (n_samples,) times and
    #: (n_samples, n_fluids) area-weighted mean fractions leaving the shoe.
    outlet_time: np.ndarray
    outlet_fractions: np.ndarray
    time: float
    n_steps: int
    wall_time: float

    def volume_per_fluid(self) -> np.ndarray:
        cv = self.grid.cell_volume
        return np.array([float((f * cv).sum()) for f in self.fractions])

    def centre_plane(self, fluid_index: int = 0) -> np.ndarray:
        """Concentration on the vertical centre plane, shape (n_layer, n_axial).

        The centre plane is the column of cells straddling ``x = 0``, i.e. the
        plane the paper plots in Fig. 5.
        """
        j = self.grid.n_azimuth // 2
        if self.grid.n_azimuth % 2 == 0:
            sl = self.fractions[fluid_index][:, :, j - 1 : j + 1].mean(axis=2)
        else:
            sl = self.fractions[fluid_index][:, :, j]
        return sl.T


class InPipeSolver:
    """Reduced-order in-pipe displacement solver."""

    def __init__(
        self,
        config: SimulationConfig,
        schedule: PumpSchedule,
        initial_fluid: Fluid,
        extra_fluids: Sequence[Fluid] = (),
    ):
        self.config = config
        self.schedule = schedule
        self.grid = Grid(config.geometry.radius, config.geometry.length, config.grid)

        # Fluid registry: the initial in-situ fluid first, then every distinct
        # fluid appearing in the pump schedule, then any extras.
        fluids: list[Fluid] = [initial_fluid]
        for st in schedule.stages:
            if st.fluid not in fluids:
                fluids.append(st.fluid)
        for fl in extra_fluids:
            if fl not in fluids:
                fluids.append(fl)
        self.fluids = fluids
        self.n_fluids = len(fluids)
        self._index = {fl: i for i, fl in enumerate(fluids)}

        shape = self.grid.shape
        self.f = np.zeros((self.n_fluids,) + shape)
        self.f[0] = 1.0  # pipe initially full of the in-situ fluid
        self.s = np.zeros(shape)  # mixing status; nothing sets it in Phase 1

        self.t = 0.0
        self.n_steps = 0
        self.diagnostics = Diagnostics()
        self._influx = np.zeros(self.n_fluids)
        self._outflux = np.zeros(self.n_fluids)
        self._outlet_time: list[float] = []
        self._outlet_fractions: list[np.ndarray] = []
        self._profile_cache: dict = {}
        self._cache_limit = 4096
        #: (n_fluids, 4) matrix of [rho, tau0, k, n] for fast weighted averaging.
        self._fluid_params = np.array(
            [[fl.rho, fl.tau0, fl.k, fl.n] for fl in fluids], dtype=float
        )

    # -- initial condition --------------------------------------------------

    def set_initial_interface(self, z_interface: float, upstream: Fluid, downstream: Fluid):
        """Place a flat interface at ``z_interface`` [m].

        Cells straddling the interface get the exact volume-weighted split, so
        the initial condition is second-order rather than staircased.
        """
        iu, idn = self._index[upstream], self._index[downstream]
        zf = self.grid.z_faces
        frac_up = np.clip((z_interface - zf[:-1]) / self.grid.dz, 0.0, 1.0)
        self.f[...] = 0.0
        self.f[iu] = frac_up[:, None, None]
        self.f[idn] = 1.0 - frac_up[:, None, None]
        return self

    # -- per-station effective fluid ---------------------------------------

    def effective_parameters(self) -> np.ndarray:
        """Volume-fraction-weighted ``(rho, tau0, k, n)`` per axial station.

        Shape ``(n_axial, 4)``.  The paper says only that "averaged rheological
        parameters and density of fluids are used" (Appendix A.1); averaging
        ``n`` is not physically rigorous - see assumption A-06.
        """
        # weights[k, i] = volume of fluid i at station k
        weights = np.einsum("iklm,lm->ki", self.f, self.grid.cell_volume)
        total = weights.sum(axis=1, keepdims=True)
        np.divide(weights, total, out=weights, where=total > 0.0)
        return weights @ self._fluid_params

    def effective_fluid_at(self, k: int) -> Fluid:
        """Effective fluid at axial station ``k``."""
        rho, tau0, kk, n = self.effective_parameters()[k]
        return Fluid(name=f"eff@{k}", rho=float(rho), tau0=float(tau0),
                     k=float(kk), n=float(n))

    def _station_velocity(self, q: float, params) -> np.ndarray:
        """Cross-sectional velocity field for one effective rheology, cached.

        Stations holding a single fluid share an effective rheology exactly, so
        a whole pipe of undisturbed mud collapses to one Brent solve.  The cache
        key rounds to ~1e-9 relative (assumption A-21), which keeps stations
        that differ only by round-off from re-solving.
        """
        rho, tau0, kk, n = params
        key = (round(q, 15), round(float(tau0), 9), round(float(kk), 12), round(float(n), 9))
        cached = self._profile_cache.get(key)
        if cached is None:
            fluid = Fluid(name="eff", rho=float(rho), tau0=float(tau0),
                          k=float(kk), n=float(n))
            tau_w = solve_tau_w(
                q, fluid, self.grid.radius,
                xtol=self.config.numerics.brentq_xtol,
                rtol=self.config.numerics.brentq_rtol,
            )
            prof = VelocityProfile(fluid=fluid, radius=self.grid.radius, tau_w=tau_w)
            cached = self.grid.map_velocity(prof, self.config.numerics.velocity_mapping)
            if len(self._profile_cache) >= self._cache_limit:
                self._profile_cache.clear()
            self._profile_cache[key] = cached
        return cached

    def velocity_field(self, q: float) -> np.ndarray:
        """Axial cell velocities for the whole pipe, shape ``grid.shape``.

        Stations are grouped by rounded effective rheology, so the expensive
        ``tau_w`` root find runs once per *distinct* rheology rather than once
        per station.  In a typical displacement only the few stations straddling
        an interface are mixed, so this collapses ~100 solves to a handful.
        """
        params = self.effective_parameters()
        rounded = np.stack([
            np.round(params[:, 1], 9),
            np.round(params[:, 2], 12),
            np.round(params[:, 3], 9),
        ], axis=1)
        _, first_idx, inverse = np.unique(rounded, axis=0, return_index=True,
                                          return_inverse=True)
        inverse = inverse.reshape(-1)
        unique_u = [self._station_velocity(q, params[i]) for i in first_idx]
        u = np.empty(self.grid.shape)
        for g, uu in enumerate(unique_u):
            u[inverse == g] = uu
        self._n_unique_stations = len(unique_u)

        if self.config.numerics.enforce_discrete_continuity and q > 0.0:
            # Rescale each station so that sum_cells u*A == Q exactly.  The
            # area-averaged mapping is already accurate to ~4e-6 relative, so
            # this is a tiny correction; it makes sum_c div(u)_c A_c vanish to
            # round-off, which is what the "redistribute" closure needs to
            # conserve exactly (assumption A-22).
            station_q = np.einsum("klm,lm->k", u, self.grid.cell_area)
            np.divide(q, station_q, out=station_q, where=station_q != 0.0)
            u *= station_q[:, None, None]
        return u

    # -- time loop ----------------------------------------------------------

    def step(self, dt: float | None = None) -> float:
        """Advance one timestep.  Returns the timestep actually taken [s]."""
        t0 = time.perf_counter()
        num = self.config.numerics

        q = self.schedule.rate_at(self.t)
        u = self.velocity_field(q)

        dt_max = cfl_timestep(u, self.grid.dz, num.cfl)
        dt = dt_max if dt is None else dt
        courant = assert_cfl(u, self.grid.dz, dt, num.cfl)

        inlet = self.schedule.fluid_at_inlet(self.t)
        inlet_values = np.zeros(self.n_fluids)
        inlet_values[self._index[inlet]] = 1.0

        # Boundary bookkeeping before the update (explicit Euler uses f^n).
        area = self.grid.cell_area
        inflow_vol = float((np.maximum(u[0], 0.0) * area).sum()) * dt
        self._influx += inlet_values * inflow_vol
        self._outflux += np.array(
            [float((np.maximum(u[-1], 0.0) * area * self.f[i, -1]).sum()) * dt
             for i in range(self.n_fluids)]
        )
        outlet_mean = np.array(
            [float((self.f[i, -1] * area).sum() / area.sum()) for i in range(self.n_fluids)]
        )
        self._outlet_time.append(self.t)
        self._outlet_fractions.append(outlet_mean)

        self.f = advect_multi(
            self.f,
            u,
            self.grid.dz,
            dt,
            face_scheme=num.face_scheme,
            inlet_values=inlet_values,
            closure=num.transverse_closure,
            area=self.grid.cell_area,
        )
        # The mixing status advects with the same kernel and scheme.  Nothing
        # sets s = 1 in Phase 1 (assumption A-17); the inlet carries s = 0.
        self.s = advect(
            self.s,
            u,
            self.grid.dz,
            dt,
            face_scheme=num.face_scheme,
            inlet_value=0.0,
            # Eq. A.20 is stated in advective form, so s always uses it.
            divergence_correction=True,
        )

        self.t += dt
        self.n_steps += 1
        self._last_u = u
        self._last_courant = courant
        self._last_wall = time.perf_counter() - t0
        return dt

    def record(self) -> None:
        d = self.diagnostics
        cv = self.grid.cell_volume
        d.time.append(self.t)
        d.step.append(self.n_steps)
        d.fluid_volumes.append(np.array([float((f * cv).sum()) for f in self.f]))
        d.influx.append(self._influx.copy())
        d.outflux.append(self._outflux.copy())
        d.front_position.append(self.front_position())
        d.f_min.append(float(self.f.min()))
        d.f_max.append(float(self.f.max()))
        d.sum_to_one_error.append(float(np.max(np.abs(self.f.sum(axis=0) - 1.0))))
        dt = self.diagnostics.dt[-1] if self.diagnostics.dt else float("nan")
        d.courant.append(getattr(self, "_last_courant", float("nan")))
        d.wall_time_per_step.append(getattr(self, "_last_wall", float("nan")))

    def front_position(self, fluid_index: int | None = None) -> np.ndarray:
        """z of the ``f = 0.5`` crossing per column [m], NaN where absent.

        By default tracks the fluid currently entering at the inlet, which is
        the one heading for the shoe.
        """
        if fluid_index is None:
            fluid_index = self._index[self.schedule.fluid_at_inlet(min(self.t, self.schedule.total_time))]
        f = self.f[fluid_index]
        z = self.grid.z_centers
        nz = f.shape[0]
        flat = f.reshape(nz, -1)
        out = np.full(flat.shape[1], np.nan)
        for c in range(flat.shape[1]):
            col = flat[:, c]
            if col[0] < 0.5 or col[-1] > 0.5:
                # No downward 0.5 crossing in this column.
                if col.max() < 0.5 or col.min() > 0.5:
                    continue
            idx = np.nonzero((col[:-1] >= 0.5) & (col[1:] < 0.5))[0]
            if idx.size == 0:
                continue
            i = idx[-1]
            w = (col[i] - 0.5) / (col[i] - col[i + 1])
            out[c] = z[i] + w * (z[i + 1] - z[i])
        return out.reshape(f.shape[1:])

    def run(
        self,
        t_end: float | None = None,
        max_steps: int = 2_000_000,
        progress: bool = False,
    ) -> SimulationResult:
        """Run to ``t_end`` (default: the end of the pump schedule)."""
        num = self.config.numerics
        if t_end is None:
            t_end = self.schedule.total_time
        wall0 = time.perf_counter()
        self.record()

        while self.t < t_end and self.n_steps < max_steps:
            q = self.schedule.rate_at(self.t)
            u = self.velocity_field(q)
            dt = min(cfl_timestep(u, self.grid.dz, num.cfl), t_end - self.t)
            self.diagnostics.dt.append(dt)
            self.diagnostics.dm_num_paper.append(numerical_diffusivity(self.grid.dz, dt))
            self.diagnostics.dm_num_scheme.append(
                upwind_diffusivity(float(np.max(np.abs(u))), self.grid.dz, dt)
            )
            self.step(dt)
            check_sum_to_one(self.f, atol=num.sum_to_one_atol)
            check_bounded(self.f, atol=num.boundedness_atol)
            if self.n_steps % num.diagnostics_every == 0:
                self.record()
                if progress:
                    print(
                        f"  step {self.n_steps:6d}  t = {self.t:9.3f} s  "
                        f"dt = {dt:.4g} s  C = {self._last_courant:.3f}"
                    )

        self.record()
        return SimulationResult(
            grid=self.grid,
            fluids=self.fluids,
            fractions=self.f,
            mixing_status=self.s,
            velocity=getattr(self, "_last_u", np.zeros(self.grid.shape)),
            diagnostics=self.diagnostics,
            outlet_time=np.asarray(self._outlet_time),
            outlet_fractions=np.asarray(self._outlet_fractions),
            time=self.t,
            n_steps=self.n_steps,
            wall_time=time.perf_counter() - wall0,
        )
