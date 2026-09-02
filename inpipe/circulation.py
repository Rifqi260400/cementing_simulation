"""Coupled casing-and-annulus circulation - a cementing job end to end.

Cement is pumped down the casing, turns at the shoe, and returns up the
annulus displacing mud ahead of it.  The two legs share one depth axis:

    z = measured depth, increasing downward, 0 at surface, L at the shoe.

The casing leg flows along ``+z``; the annulus leg flows along ``-z``.  The
annulus grid stores its cells in *flow order* (index 0 at the shoe), so the
same 1D advection kernel serves both legs unchanged.

Coupling
--------
At the shoe, the casing outlet feeds the annulus inlet at the same volumetric
rate.  The composition handed over is the casing outlet's **mixing cup** - the
flux-weighted mean over the outlet face - because the flow reverses through the
shoe and float equipment, which no reduced-order model resolves.  Any radial
structure in the casing is therefore lost at the turn; the annulus develops its
own profile from a uniform inlet.  Logged as A-26.

Hydraulics
----------
Gravity enters as hydrostatic head and friction only: the flow rate is the
pump rate, as chosen for this phase.  Pressure is integrated down the casing
and back up the annulus (see :mod:`inpipe.hydraulics`), giving pump pressure,
shoe pressure and ECD.  The U-tube imbalance is *reported* so it is visible
when the well would tend to free-fall, but it does not drive the flow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .annulus_grid import AnnulusGrid
from .caliper import CaliperLog
from .config import G_ACCEL, GridConfig, NumericsConfig
from .fluid import Fluid, PumpSchedule, mix_fluids
from .grid import Grid
from .hydraulics import HydraulicsReport, circulation_pressure
from .slot import SlotProfile, solve_slot_tau_w
from .transport import (
    advect_multi,
    assert_cfl,
    cfl_timestep,
    check_bounded,
    check_sum_to_one,
    face_velocity_stack,
)
from .velocity import VelocityProfile, solve_tau_w

__all__ = ["WellConfig", "CirculationResult", "CirculationSolver"]


@dataclass(frozen=True)
class WellConfig:
    """Geometry of a cased vertical well with an open-hole annulus."""

    length: float               # measured depth to the shoe [m]
    casing_id: float            # casing inner diameter [m]
    casing_od: float            # casing outer diameter [m]
    caliper: CaliperLog         # hole diameter against depth
    inclination: float = 0.0    # from vertical [rad]; 0 is vertical

    def __post_init__(self) -> None:
        if not 0.0 < self.casing_id < self.casing_od:
            raise ValueError(
                f"need 0 < casing ID ({self.casing_id}) < casing OD ({self.casing_od})"
            )

    @property
    def casing_radius(self) -> float:
        return 0.5 * self.casing_id


@dataclass
class CirculationResult:
    """State and history of a coupled circulation run."""

    casing_grid: Grid
    annulus_grid: AnnulusGrid
    fluids: list
    casing_fractions: np.ndarray    # (n_fluids, n_axial, n_layer, n_azimuth)
    annulus_fractions: np.ndarray   # same, in annulus flow order
    casing_velocity: np.ndarray
    annulus_velocity: np.ndarray    # magnitude along the flow (upward)
    time: float
    n_steps: int
    wall_time: float
    history: dict = field(default_factory=dict)
    snapshots: list = field(default_factory=list)
    hydraulics: HydraulicsReport | None = None
    #: Wall shear stress per annular station [Pa], in annulus flow order.
    annulus_tau_w: np.ndarray | None = None

    def yield_diagnostic(self, fluid) -> dict:
        """Where the annular wall shear falls below ``fluid``'s yield stress.

        A displaced fluid only moves where the flow can yield it.  Below its
        yield stress it is immobile no matter how long the job runs - the
        classic unyielded-mud channel.  This model does *not* enforce that: it
        averages rheology per station and solves one profile, so it will
        happily "displace" fluid the real well would leave behind.  The check
        is reported so the omission is visible.

        Returns the per-station wall shear, the fraction of the annulus below
        the yield stress, and the depth interval affected.
        """
        if self.annulus_tau_w is None:  # pragma: no cover - defensive
            raise ValueError("no wall shear recorded")
        g = self.annulus_grid
        order = np.argsort(g.z_centers)
        tau = self.annulus_tau_w[order]
        z = g.z_centers[order]
        below = tau < fluid.tau0
        vol = g.cell_volume.sum(axis=(1, 2))[order]
        return {
            "depth": z,
            "tau_w": tau,
            "below_yield": below,
            "volume_fraction_below": float(vol[below].sum() / vol.sum()),
            "min_tau_w": float(tau.min()),
            "yield_stress": fluid.tau0,
        }

    def annulus_profile(self, fluid_index: int) -> tuple:
        """Area-averaged annular concentration against *ascending* depth."""
        a = self.annulus_grid.cell_area
        prof = (self.annulus_fractions[fluid_index] * a).sum(axis=(1, 2)) / a.sum(axis=(1, 2))
        z = self.annulus_grid.z_centers
        order = np.argsort(z)
        return z[order], prof[order]

    def casing_profile(self, fluid_index: int) -> tuple:
        a = self.casing_grid.cell_area
        prof = (self.casing_fractions[fluid_index] * a).sum(axis=(1, 2)) / a.sum()
        return self.casing_grid.z_centers, prof

    def annular_displacement_efficiency(self, cement_index: int) -> float:
        """Fraction of the annular volume occupied by cement."""
        v = self.annulus_grid.cell_volume
        return float((self.annulus_fractions[cement_index] * v).sum() / v.sum())


class CirculationSolver:
    """Couples the in-pipe model to an annular leg through the shoe."""

    def __init__(
        self,
        well: WellConfig,
        schedule: PumpSchedule,
        initial_fluid: Fluid,
        grid: GridConfig,
        annulus_grid: GridConfig | None = None,
        numerics: NumericsConfig | None = None,
        gravity: float = G_ACCEL,
        surface_pressure: float = 0.0,
    ):
        self.well = well
        self.schedule = schedule
        self.numerics = numerics or NumericsConfig()
        self.gravity = gravity
        self.surface_pressure = surface_pressure

        ann = annulus_grid or grid
        self.casing_grid = Grid(well.casing_radius, well.length, grid)
        self.annulus_grid = AnnulusGrid(
            length=well.length,
            casing_od=well.casing_od,
            caliper=well.caliper,
            n_axial=ann.n_axial,
            n_layer=ann.n_layer,
            n_azimuth=ann.n_azimuth,
        )

        fluids = [initial_fluid]
        for st in schedule.stages:
            if st.fluid not in fluids:
                fluids.append(st.fluid)
        self.fluids = fluids
        self.n_fluids = len(fluids)
        self._index = {f: i for i, f in enumerate(fluids)}
        self._params = np.array([[f.rho, f.tau0, f.k, f.n] for f in fluids])

        self.f_casing = np.zeros((self.n_fluids,) + self.casing_grid.shape)
        self.f_casing[0] = 1.0
        self.f_annulus = np.zeros((self.n_fluids,) + self.annulus_grid.shape)
        self.f_annulus[0] = 1.0

        self.t = 0.0
        self.n_steps = 0
        self._casing_cache: dict = {}
        self._slot_cache: dict = {}
        self._history = {k: [] for k in (
            "time", "shoe_fraction", "returns_fraction", "annular_efficiency",
            "pump_pressure", "shoe_pressure", "ecd_at_shoe", "utube_imbalance",
            "mass_error", "sum_to_one_error", "dt",
        )}
        self._snapshots: list = []

    # -- effective fluids ---------------------------------------------------

    @staticmethod
    def _effective(fields, volume, params):
        """Volume-weighted (rho, tau0, k, n) per station (assumption A-06).

        ``volume`` is 2D for the casing, whose one cross-section is extruded,
        and 3D for the annulus, whose section varies with depth.
        """
        subscripts = "iklm,lm->ki" if np.ndim(volume) == 2 else "iklm,klm->ki"
        w = np.einsum(subscripts, fields, volume)
        total = w.sum(axis=1, keepdims=True)
        np.divide(w, total, out=w, where=total > 0.0)
        return w @ params

    def _casing_velocity(self, q):
        grid = self.casing_grid
        params = self._effective(self.f_casing, grid.cell_volume, self._params)
        d0, d1, d2 = self.numerics.cache_key_decimals
        rounded = np.stack([np.round(params[:, 1], d0), np.round(params[:, 2], d1),
                            np.round(params[:, 3], d2)], axis=1)
        _, first, inverse = np.unique(rounded, axis=0, return_index=True,
                                      return_inverse=True)
        u = np.empty(grid.shape)
        for g_i, k in enumerate(first):
            rho, tau0, kk, n = params[k]
            key = (round(q, 15), round(tau0, d0), round(kk, d1), round(n, d2))
            cached = self._casing_cache.get(key)
            if cached is None:
                fl = Fluid("eff", float(rho), float(tau0), float(kk), float(n))
                tau_w = solve_tau_w(q, fl, grid.radius,
                                    xtol=self.numerics.brentq_xtol,
                                    rtol=self.numerics.brentq_rtol)
                cached = grid.map_velocity(
                    VelocityProfile(fl, grid.radius, tau_w),
                    self.numerics.velocity_mapping,
                )
                if len(self._casing_cache) >= self.numerics.cache_limit:
                    self._casing_cache.clear()
                self._casing_cache[key] = cached
            u[inverse.reshape(-1) == g_i] = cached
        station_q = np.einsum("klm,lm->k", u, grid.cell_area)
        np.divide(q, station_q, out=station_q, where=station_q != 0.0)
        return u * station_q[:, None, None]

    def _annulus_profiles(self, q):
        """Slot profile per annular station, with its local gap and rheology."""
        g = self.annulus_grid
        params = self._effective(self.f_annulus, g.cell_volume, self._params)
        profiles = []
        d0, d1, d2 = self.numerics.cache_key_decimals
        for k in range(g.n_axial):
            rho, tau0, kk, n = params[k]
            b, width = float(g.half_gap[k]), float(g.slot_width[k])
            key = (round(q, 15), round(b, 9), round(width, 9),
                   round(tau0, d0), round(kk, d1), round(n, d2))
            prof = self._slot_cache.get(key)
            if prof is None:
                fl = Fluid("eff", float(rho), float(tau0), float(kk), float(n))
                prof = SlotProfile(fl, b, width,
                                   solve_slot_tau_w(q, fl, b, width,
                                                    xtol=self.numerics.brentq_xtol,
                                                    rtol=self.numerics.brentq_rtol))
                if len(self._slot_cache) >= self.numerics.cache_limit:
                    self._slot_cache.clear()
                self._slot_cache[key] = prof
            profiles.append(prof)
        return profiles

    # -- one step -----------------------------------------------------------

    def step(self, dt=None):
        num = self.numerics
        q = self.schedule.rate_at(self.t)

        u_c = self._casing_velocity(q)
        profiles = self._annulus_profiles(q)
        u_a = self.annulus_grid.normalise_to_flow_rate(
            self.annulus_grid.map_velocity(profiles, "area_average"), q
        )

        dt_limit = min(
            cfl_timestep(u_c, self.casing_grid.dz, num.cfl),
            cfl_timestep(u_a, self.annulus_grid.dz, num.cfl),
        )
        if not np.isfinite(dt_limit):
            dt_limit = np.inf
        dt = dt_limit if dt is None else min(dt, dt_limit)
        assert_cfl(u_c, self.casing_grid.dz, dt, num.cfl)
        assert_cfl(u_a, self.annulus_grid.dz, dt, num.cfl)

        # Inlet of the casing: whatever the pump is delivering.
        inlet = self.schedule.fluid_at_inlet(self.t)
        casing_inlet = np.zeros(self.n_fluids)
        casing_inlet[self._index[inlet]] = 1.0

        # Shoe hand-over: the casing outlet mixing cup feeds the annulus inlet
        # (assumption A-26).  Taken from the state before the update, matching
        # the explicit Euler the rest of the step uses.
        shoe = self._shoe_mixing_cup(u_c)

        cg, ag = self.casing_grid, self.annulus_grid
        self.f_casing = advect_multi(
            self.f_casing, u_c, cg.dz, dt,
            face_scheme=num.face_scheme, inlet_values=casing_inlet,
            closure=num.transverse_closure, area=cg.cell_area,
        )
        # Every axial face must pass the imposed rate; see
        # AnnulusGrid.normalise_face_flux.
        uf_a = ag.normalise_face_flux(face_velocity_stack(u_a), q)
        self.f_annulus = advect_multi(
            self.f_annulus, u_a, ag.dz, dt,
            face_scheme=num.face_scheme, inlet_values=shoe,
            closure=num.transverse_closure, area=ag.cell_area,
            u_faces=uf_a,
            face_area=ag.face_area, cell_volume=ag.cell_volume,
        )

        self.t += dt
        self.n_steps += 1
        self._last = dict(q=q, u_c=u_c, u_a=u_a, profiles=profiles, shoe=shoe, dt=dt)
        return dt

    def _shoe_mixing_cup(self, u_c):
        """Flux-weighted composition leaving the casing at the shoe."""
        w = np.maximum(u_c[-1], 0.0) * self.casing_grid.cell_area
        total = w.sum()
        if total <= 0.0:
            return np.zeros(self.n_fluids)
        return np.einsum("ilm,lm->i", self.f_casing[:, -1], w) / total

    # -- reporting ----------------------------------------------------------

    def hydraulics(self) -> HydraulicsReport:
        """Pressure profile, pump pressure, ECD and the U-tube imbalance."""
        last = getattr(self, "_last", None)
        q = last["q"] if last else self.schedule.rate_at(self.t)
        if last is None:
            self._last = dict(q=q, u_c=self._casing_velocity(q),
                              profiles=self._annulus_profiles(q))
        casing_params = self._effective(self.f_casing, self.casing_grid.cell_volume,
                                        self._params)
        annulus_params = self._effective(self.f_annulus, self.annulus_grid.cell_volume,
                                         self._params)
        casing_tau_w = []
        d0, d1, d2 = self.numerics.cache_key_decimals
        for rho, tau0, kk, n in casing_params:
            fl = Fluid("eff", float(rho), float(tau0), float(kk), float(n))
            casing_tau_w.append(
                solve_tau_w(q, fl, self.casing_grid.radius) if q > 0.0 else 0.0
            )
        return circulation_pressure(
            casing_z=self.casing_grid.z_centers,
            casing_dz=self.casing_grid.dz,
            casing_rho=casing_params[:, 0],
            casing_tau_w=np.array(casing_tau_w),
            casing_radius=self.casing_grid.radius,
            annulus_z=self.annulus_grid.z_centers,
            annulus_dz=self.annulus_grid.dz,
            annulus_rho=annulus_params[:, 0],
            annulus_tau_w=np.array([p.tau_w for p in self._last["profiles"]]),
            annulus_half_gap=self.annulus_grid.half_gap,
            inclination=self.well.inclination,
            gravity=self.gravity,
            surface_pressure=self.surface_pressure,
        )

    def record(self, snapshot=False):
        h = self._history
        cem = self.n_fluids - 1
        rep = self.hydraulics()
        h["time"].append(self.t)
        h["shoe_fraction"].append(getattr(self, "_last", {}).get(
            "shoe", np.zeros(self.n_fluids)).copy())
        av = self.annulus_grid.cell_area[-1]
        h["returns_fraction"].append(
            np.einsum("ilm,lm->i", self.f_annulus[:, -1], av) / av.sum())
        h["annular_efficiency"].append(self._annular_efficiency(cem))
        h["pump_pressure"].append(rep.pump_pressure)
        h["shoe_pressure"].append(rep.shoe_pressure)
        h["ecd_at_shoe"].append(rep.ecd_at_shoe)
        h["utube_imbalance"].append(rep.utube_imbalance)
        h["sum_to_one_error"].append(max(
            float(np.max(np.abs(self.f_casing.sum(axis=0) - 1.0))),
            float(np.max(np.abs(self.f_annulus.sum(axis=0) - 1.0))),
        ))
        h["mass_error"].append(self._total_volume_error())
        h["dt"].append(getattr(self, "_last", {}).get("dt", float("nan")))
        if snapshot:
            self._snapshots.append(dict(
                time=self.t,
                casing=self.f_casing.copy(),
                annulus=self.f_annulus.copy(),
            ))

    def _annular_efficiency(self, cement_index):
        v = self.annulus_grid.cell_volume
        return float((self.f_annulus[cement_index] * v).sum() / v.sum())

    def _total_volume_error(self):
        """Relative departure of the total fluid volume from the pore volume.

        The volume fractions must fill the well: summed over fluids and
        weighted by cell volume they should reproduce the geometric volume of
        both legs exactly.
        """
        vc = float((self.f_casing.sum(axis=0) * self.casing_grid.cell_volume).sum())
        va = float((self.f_annulus.sum(axis=0) * self.annulus_grid.cell_volume).sum())
        exact = self.casing_grid.total_volume + self.annulus_grid.total_volume
        return abs(vc + va - exact) / exact

    # -- run ----------------------------------------------------------------

    def run(self, t_end=None, n_snapshots=0, progress=False) -> CirculationResult:
        num = self.numerics
        if t_end is None:
            t_end = self.schedule.total_time
        snap_times = (
            list(np.linspace(0.0, t_end, n_snapshots)) if n_snapshots > 0 else []
        )
        wall0 = time.perf_counter()
        self.record(snapshot=bool(snap_times))
        if snap_times:
            snap_times.pop(0)

        while self.t < t_end:
            self.step(min(np.inf, t_end - self.t))
            check_sum_to_one(self.f_casing, atol=num.sum_to_one_atol)
            check_sum_to_one(self.f_annulus, atol=num.sum_to_one_atol)
            check_bounded(self.f_casing, atol=num.boundedness_atol)
            check_bounded(self.f_annulus, atol=num.boundedness_atol)
            take = bool(snap_times) and self.t >= snap_times[0]
            if take:
                snap_times.pop(0)
            if self.n_steps % num.diagnostics_every == 0 or take:
                self.record(snapshot=take)
                if progress:
                    print(f"  t = {self.t:8.1f} s  eff = "
                          f"{self._annular_efficiency(self.n_fluids - 1):.4f}")

        self.record(snapshot=bool(snap_times))
        return CirculationResult(
            casing_grid=self.casing_grid,
            annulus_grid=self.annulus_grid,
            fluids=self.fluids,
            casing_fractions=self.f_casing,
            annulus_fractions=self.f_annulus,
            casing_velocity=self._last["u_c"],
            annulus_velocity=self._last["u_a"],
            time=self.t,
            n_steps=self.n_steps,
            wall_time=time.perf_counter() - wall0,
            history={k: np.asarray(v) for k, v in self._history.items()},
            snapshots=self._snapshots,
            hydraulics=self.hydraulics(),
            annulus_tau_w=np.array([p.tau_w for p in self._last["profiles"]]),
        )
