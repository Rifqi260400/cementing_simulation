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

    def at(self, threshold: float) -> np.ndarray:
        """Arrival times for one threshold, ascending in depth."""
        i = self.thresholds.index(threshold)
        return self.arrival[i]

    @property
    def front(self) -> np.ndarray:
        """Arrival of the 0.5 contour - the front proper [s]."""
        return self.at(0.5) if 0.5 in self.thresholds else self.arrival[len(self.arrival) // 2]

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
                 fluid_name: str = "cement", casing_volume: float = 0.0):
        self.grid = grid
        self.fluid_index = int(fluid_index)
        self.thresholds = tuple(thresholds)
        self.fluid_name = fluid_name
        self._station_volume = grid.cell_volume.sum(axis=(1, 2))
        self._arrival = np.full((len(self.thresholds), grid.n_axial), np.nan)
        self._previous = None       # station fractions at the previous update
        self._previous_time = None
        # Volume that must be pumped to fill plug-wise up to each station: the
        # casing, then the annulus from the shoe upward.  Cells are in flow
        # order, so the annular part is a plain cumulative sum.
        self._volume_below = casing_volume + np.cumsum(self._station_volume)
        self._volumetric = np.full(grid.n_axial, np.nan)
        self._delivered = 0.0       # tracked fluid past the shoe [m^3]

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
            crossed = ((delivered >= self._volume_below)
                       & np.isnan(self._volumetric))
            if np.any(crossed):
                if self._previous_time is None or self._delivered >= delivered:
                    self._volumetric[crossed] = t
                else:
                    frac = ((self._volume_below[crossed] - self._delivered)
                            / (delivered - self._delivered))
                    self._volumetric[crossed] = (
                        self._previous_time + frac * (t - self._previous_time)
                    )
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

    def report(self, job_time: float) -> ArrivalReport:
        order = np.argsort(self.grid.z_centers)   # ascending depth
        return ArrivalReport(
            depth=self.grid.z_centers[order],
            arrival=self._arrival[:, order],
            thresholds=self.thresholds,
            job_time=float(job_time),
            fluid_name=self.fluid_name,
            volumetric=self._volumetric[order],
        )
