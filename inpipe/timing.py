"""Cement arrival time against depth - the quantity a fiber optic log sees.

A DTS or DAS waterfall shows *when* something reaches each depth, not how much
of it is there.  The comparable quantity from this model is the first time the
cement fraction at a station crosses a threshold, recorded against measured
depth.  That is what :class:`ArrivalTracker` accumulates, and it is recorded on
every timestep rather than on snapshots, so its resolution is the timestep
(~0.03 s here) instead of the snapshot interval (~10 s).

What this number is, and is not
-------------------------------
The flow rate is imposed: it is the volume entering the inlet per unit time
(assumption A-29).  So arrival time here is a **volumetric** result - the time
to fill the annulus below a depth, at the pumped rate - with the caliper
supplying the volume and the velocity profile supplying only the smearing of
the front.  Two consequences worth carrying into any validation:

1. The caliper is what makes it non-trivial.  On K-GEP-1 the annulus holds
   59 % more than an in-gauge hole, so arrival times computed from bit size are
   too early by roughly that fraction.
2. **It is an upper bound on time.**  This model reports the U-tube imbalance
   but does not let it drive the flow, and on this well that imbalance reaches
   21.8 bar for 99 % of the job.  A well that free-falls returns faster than
   the pump imposes, so cement arrives *earlier* than computed here.  If a
   fiber optic log shows early arrival, suspect that before suspecting the
   rheology.

Comparing against a fiber optic log
-----------------------------------
Hart et al. (2025), *Sci Rep* 15:11365, track the rising interface behind a
surface casing with DAS and compare it with an analytical displacement model.
Their Eq. 2-4 are exactly the :attr:`ArrivalReport.volumetric` curve here:
cumulative pumped volume against cumulative annular volume from the caliper,
with the stinger and rat hole filled first, and a rise velocity of ``Q / A``.
So this model reproduces theirs and adds the front on top of it.

Three points carry over from their field data:

* **DAS, not DTS.**  Their cold front in the DTS lagged every modelled rise -
  the cable's thermal response is on a minute scale and heat transfer to the
  formation blurs the step.  The interface is tracked in the low-frequency DAS
  response.  So the curve to overlay is :attr:`front`, against a DAS waterfall.
* **Which curve a fluid follows depends on its density.**  Their light
  freshwater spacer took the path of least resistance and tracked the in-gauge
  "fast rise"; their cement, denser than the mud it displaced, filled the whole
  annulus including washouts and tracked the caliper curve.  Cement displacing
  mud with no spacer - the case here - should therefore track the caliper
  curve, and the model puts the front a little ahead of it.
* **Deviations are the diagnostic.**  They read a rise *faster* than the
  caliper curve as annular volume that never got filled - gelled mud sitting in
  the washouts - and a rise *slower* as losses or fresh breakouts.  This model
  predicts the first of those directly, because it computes how much mud is
  left behind at each depth rather than assuming perfect displacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ArrivalReport", "ArrivalTracker"]

#: Fractions whose crossing is recorded.  0.5 is the front proper; 0.1 and 0.9
#: bracket it, and their separation is the width of the mixing zone in time.
DEFAULT_THRESHOLDS = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class ArrivalReport:
    """When cement first reached each depth."""

    depth: np.ndarray        # ascending measured depth [m]
    arrival: np.ndarray      # (n_threshold, n_depth) [s]; nan where never reached
    thresholds: tuple
    job_time: float          # length of the run [s]
    fluid_name: str = "cement"
    #: Volumetric arrival [s]: the hand calculation - cumulative displacing
    #: fluid pumped, against casing volume plus annulus volume below that depth,
    #: i.e. plug displacement with no mixing.  See :meth:`front_lead`.
    volumetric: np.ndarray | None = None
    #: The same on an in-gauge hole: the "fast rise" curve of Hart et al.
    #: (2025) - what the interface would do if washouts held no cement at all.
    #: It is a *reference*, not a strict bound, unless the diameter it was built
    #: on is the bit size: a caliper reads under gauge in places, and there the
    #: real hole holds less than the reference and the interface runs ahead of
    #: it.  Give ``bit_diameter`` in the case file to make it a true bound.
    in_gauge: np.ndarray | None = None
    #: Pumping time of the rat hole volume [s], reported only.  It is *not*
    #: added to :attr:`arrival`: the solver holds the rat hole as a real
    #: volume, so the delay it causes is already in the simulated front, and
    #: adding it again would count it twice.  It is still added to the
    #: volumetric curves, which are a plug-displacement hand calculation and
    #: have no rat hole of their own.
    rat_hole_delay: float = 0.0

    def at(self, threshold: float) -> np.ndarray:
        """Arrival times for one threshold, ascending in depth."""
        i = self.thresholds.index(threshold)
        return self.arrival[i]

    @property
    def front(self) -> np.ndarray:
        """Arrival of the 0.5 contour - the front proper [s]."""
        return self.at(0.5) if 0.5 in self.thresholds else self.arrival[len(self.arrival) // 2]

    @property
    def front_envelope(self) -> np.ndarray:
        """Leading edge of the front: first time cement reached this depth or above.

        The raw :attr:`front` is not monotonic in depth.  Cement channels
        through the narrow side of a washout and reaches a shallower station
        before the wide one is half full, so a station can record an *earlier*
        arrival than the one below it.  Differencing that directly gives
        nonsense - rise velocities of hundreds of m/min, and negative ones.

        This is the running minimum from the top down, which is the same curve
        an operator traces as the leading feature on a DAS waterfall: the
        shallowest depth cement has reached, against time.  It is monotonic by
        construction, so its rise velocity is well posed.
        """
        return np.minimum.accumulate(np.where(np.isnan(self.front), np.inf,
                                              self.front))

    @property
    def overtaking_depths(self) -> int:
        """How many depths the front reached out of order.

        Each one is cement arriving above a station before that station is half
        displaced - channelling past a wide spot rather than sweeping it.
        """
        front = self.front
        got = np.isfinite(front)
        return int(np.sum(front[got] > self.front_envelope[got] + 1e-9))

    @property
    def shoe_depth(self) -> float:
        return float(self.depth[-1])

    @property
    def top_depth(self) -> float:
        return float(self.depth[0])

    @property
    def shoe_arrival(self) -> float:
        """When cement entered the annulus [s]; nan if it never did."""
        return float(self.front[-1])

    @property
    def top_arrival(self) -> float:
        """When cement reached the top of the modelled interval [s]."""
        return float(self.front[0])

    @property
    def rising_time(self) -> float:
        """Time to travel the modelled interval, shoe to top [s].

        ``nan`` if cement never reached the top within the run - which is a
        result, not a failure: it means the job as scheduled does not cover the
        interval.
        """
        return self.top_arrival - self.shoe_arrival

    @property
    def reached(self) -> np.ndarray:
        """Mask of depths cement actually reached."""
        return np.isfinite(self.front)

    @property
    def top_of_cement(self) -> float:
        """Shallowest depth cement reached [m]; the shoe depth if none did."""
        got = self.reached
        return float(self.depth[got][0]) if np.any(got) else self.shoe_depth

    def front_lead(self) -> np.ndarray:
        """How far the front runs ahead of the volume balance, as a fraction.

        Two different questions, and they do not have the same answer:

        ``front``
            when the *interface* passes a depth - the 0.5 contour of the local
            cement fraction.  This is what a fiber optic log sees pass.
        ``volumetric``
            when enough cement has been pumped to fill the casing and the
            annulus below that depth, as plug displacement with no mixing.
            This is the hand calculation an engineer does from the pump rate
            and the caliper, and it needs no simulator.

        The front leads, because the annulus does not move as a plug: roughly
        half its area travels faster than the mean, so the interface reaches a
        depth before the volume balance says it should.  On the synthetic 200 m
        case that lead is **12 %**, and it does *not* shrink under mesh
        refinement (-10.2 % at 60 stations, -11.7 % at 240), so it is the
        velocity profile and not numerical diffusion.  That 12 % is what this
        model adds over the hand calculation; which of the two to compare a
        fiber optic log against depends on what the log picks up, so both are
        exported.
        """
        if self.volumetric is None:
            raise ValueError("no volume-balance arrival was recorded")
        return (self.front - self.volumetric) / self.volumetric

    def rise_velocity(self, arrival: np.ndarray, per_minute: bool = True) -> np.ndarray:
        """Rate of climb of an arrival curve [m/min], against depth.

        Hart et al. (2025) plot this against depth (their Fig. 2b) and read the
        borehole geometry straight off it: with an in-gauge hole it is flat at
        ``Q / A``, and every washout shows as a dip, because the same pumped
        rate has more area to fill.  Differencing the arrival curve rather than
        forming ``Q / A`` directly means the same routine works for the front,
        which has no closed form.
        """
        good = np.isfinite(arrival)
        out = np.full(arrival.shape, np.nan)
        if good.sum() < 2:
            return out
        depth, time = self.depth[good], arrival[good]

        # Depths sharing an arrival time have no separable speed: the interface
        # crossed them inside one recording interval.  Differencing across them
        # divides by zero, so they are merged and the speed interpolated back.
        keep = np.ones(time.size, dtype=bool)
        keep[1:] = np.diff(time) != 0.0
        if keep.sum() < 2:
            return out

        # Depth decreases as the front climbs, so negate to get a positive speed.
        speed = -np.gradient(depth[keep], time[keep])
        out[good] = np.interp(depth, depth[keep], speed)
        return out * 60.0 if per_minute else out

    def mixing_zone_duration(self) -> np.ndarray:
        """Time between the 0.1 and 0.9 crossings [s] - the front's width."""
        if 0.1 not in self.thresholds or 0.9 not in self.thresholds:
            raise ValueError("mixing zone needs the 0.1 and 0.9 thresholds")
        return self.at(0.9) - self.at(0.1)

    def to_csv(self, path) -> None:
        cols = [self.depth] + [self.arrival[i] for i in range(len(self.thresholds))]
        header = "depth_m," + ",".join(f"arrival_f{t:g}_s" for t in self.thresholds)
        if self.volumetric is not None:
            cols.append(self.volumetric)
            header += ",arrival_volumetric_s"
        if self.in_gauge is not None:
            cols.append(self.in_gauge)
            header += ",arrival_in_gauge_s"
        cols.append(self.front_envelope)
        cols.append(self.rise_velocity(self.front_envelope))
        header += ",front_envelope_s,front_rise_velocity_m_per_min"
        np.savetxt(path, np.column_stack(cols), delimiter=",", header=header,
                   comments="", fmt="%.6g")

    def summary(self) -> str:
        got = self.reached
        lines = [
            f"{self.fluid_name} arrival (imposed pump rate; an upper bound on time "
            "- see inpipe/timing.py)",
            f"  interval          : {self.top_depth:.2f} - {self.shoe_depth:.2f} m",
        ]
        if not np.any(got):
            lines.append(f"  cement never entered the annulus in {self.job_time:.1f} s")
            return "\n".join(lines)
        lines.append(f"  reached the shoe  : {self.shoe_arrival / 60:.2f} min")
        if np.all(got):
            lines.append(f"  reached the top   : {self.top_arrival / 60:.2f} min")
            lines.append(f"  rising time       : {self.rising_time / 60:.2f} min "
                         f"over {self.shoe_depth - self.top_depth:.1f} m "
                         f"({(self.shoe_depth - self.top_depth) / self.rising_time:.4f} m/s)")
        else:
            lines.append(
                f"  top of cement     : {self.top_of_cement:.2f} m at the end of the "
                f"run ({self.job_time / 60:.2f} min) - it did not cover the interval"
            )
        if self.volumetric is not None:
            lead = self.front_lead()[got]
            if np.any(np.isfinite(lead)):
                lines.append(
                    f"  volumetric (hand) : top at "
                    f"{self.volumetric[got][0] / 60:.2f} min; the front runs "
                    f"{-100 * np.nanmean(lead):.1f} % ahead of it, because half the "
                    "annular area moves faster than the mean"
                )
        if self.in_gauge is not None:
            lines.append(
                f"  in-gauge (fast)   : top at {self.in_gauge[got][0] / 60:.2f} min "
                "- the bound if washouts held no cement at all"
            )
        if self.volumetric is not None:
            geo = self.rise_velocity(self.volumetric)
            geo = geo[np.isfinite(geo)]
            if geo.size:
                lines.append(
                    f"  rise velocity     : {geo.min():.2f} - {geo.max():.2f} m/min "
                    f"from the geometry (median {np.median(geo):.2f}); the front "
                    f"averages {60 * (self.shoe_depth - self.top_depth) / self.rising_time:.2f}"
                    if np.isfinite(self.rising_time) else
                    f"  rise velocity     : {geo.min():.2f} - {geo.max():.2f} m/min"
                )
        if self.overtaking_depths:
            lines.append(
                f"  channelling       : the front reached {self.overtaking_depths} of "
                f"{self.depth.size} depths out of order - cement passing a wide spot "
                "before sweeping it"
            )
        if self.rat_hole_delay > 0.0:
            lines.append(
                f"  rat hole          : {self.rat_hole_delay / 60:.2f} min to pump its "
                "volume; already in the front, and added to the volumetric curves"
            )
        if 0.1 in self.thresholds and 0.9 in self.thresholds:
            width = self.mixing_zone_duration()[got]
            if np.any(np.isfinite(width)):
                lines.append(f"  mixing zone       : {np.nanmedian(width):.1f} s "
                             "between the 0.1 and 0.9 contours (median)")
        return "\n".join(lines)


class ArrivalTracker:
    """Records the first threshold crossing per annular station, as it happens.

    Updated every timestep from inside the run loop.  The crossing time is
    linearly interpolated between the two steps that straddle it, so the answer
    is not quantised to the timestep.
    """

    def __init__(self, grid, fluid_index: int, thresholds=DEFAULT_THRESHOLDS,
                 fluid_name: str = "cement", casing_volume: float = 0.0,
                 rat_hole_volume: float = 0.0, gauge_diameter: float | None = None):
        self.grid = grid
        self.fluid_index = int(fluid_index)
        self.thresholds = tuple(thresholds)
        self.fluid_name = fluid_name
        self._station_volume = grid.cell_volume.sum(axis=(1, 2))
        self._arrival = np.full((len(self.thresholds), grid.n_axial), np.nan)
        self._previous = None       # station fractions at the previous update
        self._previous_time = None
        # Volume that must be pumped to fill plug-wise up to each station: the
        # casing, the rat hole below the shoe, then the annulus from the shoe
        # upward.  Cells are in flow order, so the annular part is a plain
        # cumulative sum.  This is Eq. 2 of Hart et al. (2025).
        self._offset = float(casing_volume) + float(rat_hole_volume)
        self._rat_hole_volume = float(rat_hole_volume)
        self._volume_below = self._offset + np.cumsum(self._station_volume)
        self._volumetric = np.full(grid.n_axial, np.nan)

        # The same on an in-gauge hole - their "fast rise" bound.  A washout
        # holding no cement at all would put the interface on this curve.
        self._in_gauge = None
        if gauge_diameter is not None:
            in_gauge_area = 0.25 * np.pi * (gauge_diameter**2 - grid.casing_od**2)
            self._in_gauge_below = self._offset + np.cumsum(
                np.full(grid.n_axial, in_gauge_area * grid.dz))
            self._in_gauge = np.full(grid.n_axial, np.nan)

        # Times the pumped volume passed the casing alone and the casing plus
        # the rat hole; their difference delays everything in the annulus, and
        # the mesh has no rat hole in it to produce that delay by itself.
        self._t_casing_full = np.nan
        self._t_offset_full = np.nan
        self._casing_volume = float(casing_volume)
        self._delivered = 0.0       # tracked fluid pumped at the inlet [m^3]

    def station_fraction(self, f_annulus: np.ndarray) -> np.ndarray:
        """Volume-averaged fraction of the tracked fluid per station."""
        return (np.einsum("klm,klm->k", f_annulus[self.fluid_index], self.grid.cell_volume)
                / self._station_volume)

    def update(self, t: float, f_annulus: np.ndarray, delivered=None) -> None:
        """Record crossings at time ``t``.

        ``delivered`` is the cumulative volume of the tracked fluid pumped in at
        the inlet [m^3]; supplying it adds the volumetric arrival.  Taking it
        from the solver rather than assuming ``rate x time`` keeps it right when
        the rate changes or another fluid is pumped for part of the job.
        """
        if delivered is not None:
            self._cross(t, delivered, self._volume_below, self._volumetric)
            if self._in_gauge is not None:
                self._cross(t, delivered, self._in_gauge_below, self._in_gauge)
            if np.isnan(self._t_casing_full) and delivered >= self._casing_volume:
                self._t_casing_full = self._interpolate(t, delivered,
                                                        self._casing_volume)
            if np.isnan(self._t_offset_full) and delivered >= self._offset:
                self._t_offset_full = self._interpolate(t, delivered, self._offset)
            self._delivered = float(delivered)

        current = self.station_fraction(f_annulus)
        previous = self._previous
        for i, threshold in enumerate(self.thresholds):
            crossed = (current >= threshold) & np.isnan(self._arrival[i])
            if previous is None:
                # Already above at the first sample: it arrived at t, not before.
                self._arrival[i, crossed] = t
                continue
            crossed &= previous < threshold
            if not np.any(crossed):
                continue
            span = current[crossed] - previous[crossed]
            # span > 0 wherever the mask holds, since previous < threshold <= current.
            frac = (threshold - previous[crossed]) / span
            self._arrival[i, crossed] = (
                self._previous_time + frac * (t - self._previous_time)
            )
        self._previous = current
        self._previous_time = t

    def _interpolate(self, t: float, delivered: float, target: float) -> float:
        """Time the pumped volume passed ``target``, between the last two calls."""
        if self._previous_time is None or delivered <= self._delivered:
            return t
        frac = (target - self._delivered) / (delivered - self._delivered)
        return self._previous_time + frac * (t - self._previous_time)

    def _cross(self, t: float, delivered: float, targets, out) -> None:
        """Record the first time ``delivered`` passed each entry of ``targets``."""
        crossed = (delivered >= targets) & np.isnan(out)
        if not np.any(crossed):
            return
        if self._previous_time is None or delivered <= self._delivered:
            out[crossed] = t
            return
        frac = (targets[crossed] - self._delivered) / (delivered - self._delivered)
        out[crossed] = self._previous_time + frac * (t - self._previous_time)

    def report(self, job_time: float) -> ArrivalReport:
        order = np.argsort(self.grid.z_centers)   # ascending depth
        # The rat hole is in the solver as a real volume, so the front already
        # waits for it.  This is the plug-displacement pumping time of that
        # volume, reported for comparison with the volumetric curves - which do
        # carry it, being a hand calculation with no rat hole in them.
        delay = 0.0
        if self._rat_hole_volume > 0.0 and np.isfinite(self._t_offset_full):
            delay = float(self._t_offset_full - self._t_casing_full)
        return ArrivalReport(
            depth=self.grid.z_centers[order],
            arrival=self._arrival[:, order],
            thresholds=self.thresholds,
            job_time=float(job_time),
            fluid_name=self.fluid_name,
            volumetric=self._volumetric[order],
            in_gauge=None if self._in_gauge is None else self._in_gauge[order],
            rat_hole_delay=delay,
        )
