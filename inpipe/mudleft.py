"""Where the displaced fluid is left behind, and how much of that is the hole.

The question this answers: with an irregular open hole, how much mud is still
in the annulus at the end of the job, and how much of it is attributable to the
enlargements rather than to the displacement being imperfect anyway.

What this model can and cannot say
----------------------------------
It has **no mechanism that traps fluid permanently**.  Transport is axial only,
every column moves at a finite speed, so circulating for long enough sweeps
everything: pumping 2.5 annular volumes on the K-GEP-1 open hole leaves 0.002
of 10.2 m3.  Residual mud here is therefore always *"not yet swept at the
volume actually pumped"*, never *"stuck"*.

That is not the whole physical story.  In a real concentric washout the flow
separates at the expansion and recirculates in the cavity, which does strand
fluid however long you circulate.  That is a two-dimensional effect - radial and
axial - and a reduced-order model with a fully-developed profile at every
station cannot represent it.  So the numbers below are a *lower bound* on what a
washout costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MudLeftReport", "mud_left_behind"]


@dataclass
class MudLeftReport:
    """Residual displaced fluid against depth, split by hole condition."""

    depth: np.ndarray            # ascending true depth [m]
    hole_diameter: np.ndarray    # [m]
    cell_volume: np.ndarray      # annular volume per station [m^3]
    mud_volume: np.ndarray       # residual displaced fluid per station [m^3]
    gauge: float                 # in-gauge hole diameter [m]
    washout_threshold: float     # multiple of gauge counted as washed out

    @property
    def is_washout(self) -> np.ndarray:
        return self.hole_diameter > self.washout_threshold * self.gauge

    @property
    def local_fraction(self) -> np.ndarray:
        """Residual mud as a fraction of the local annular volume."""
        return self.mud_volume / np.maximum(self.cell_volume, 1e-300)

    @property
    def total_mud(self) -> float:
        return float(self.mud_volume.sum())

    @property
    def total_volume(self) -> float:
        return float(self.cell_volume.sum())

    @property
    def efficiency(self) -> float:
        return 1.0 - self.total_mud / self.total_volume

    @property
    def washout_volume_share(self) -> float:
        """Share of the annulus that is washed out, by volume."""
        w = self.is_washout
        return float(self.cell_volume[w].sum() / self.total_volume)

    @property
    def washout_mud_share(self) -> float:
        """Share of the residual mud sitting in washed-out hole."""
        w = self.is_washout
        if self.total_mud <= 0.0:
            return float("nan")
        return float(self.mud_volume[w].sum() / self.total_mud)

    @property
    def concentration_ratio(self) -> float:
        """How over-represented washouts are in the residual.

        ``1`` means the leftover mud is spread in proportion to volume;
        ``> 1`` means the enlargements hold more than their share.
        """
        share = self.washout_volume_share
        if share <= 0.0:
            return float("nan")
        return self.washout_mud_share / share

    def excess_over_gauge(self, gauge_report: "MudLeftReport") -> float:
        """Extra residual mud against the same job in an in-gauge hole [m^3]."""
        return self.total_mud - gauge_report.total_mud

    def worst_intervals(self, n=5, min_length=1.0):
        """The ``n`` depth intervals holding the most residual mud per metre.

        Returns a list of ``(top, bottom, mud_volume, mean_hole_diameter)``,
        merging adjacent stations whose local fraction is above the median.
        """
        frac = self.local_fraction
        flagged = frac > max(np.median(frac), 1e-12)
        out = []
        i = 0
        dz = float(np.mean(np.diff(self.depth))) if self.depth.size > 1 else 0.0
        while i < len(flagged):
            if not flagged[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(flagged) and flagged[j + 1]:
                j += 1
            top, bottom = self.depth[i] - 0.5 * dz, self.depth[j] + 0.5 * dz
            if bottom - top >= min_length:
                out.append((
                    float(top), float(bottom),
                    float(self.mud_volume[i:j + 1].sum()),
                    float(np.mean(self.hole_diameter[i:j + 1])),
                ))
            i = j + 1
        out.sort(key=lambda r: r[2], reverse=True)
        return out[:n]

    def summary(self) -> str:
        lines = [
            f"residual mud        : {self.total_mud:.4f} m^3 of "
            f"{self.total_volume:.4f} m^3 annulus "
            f"(displacement efficiency {self.efficiency:.4f})",
            f"washed-out hole     : {100 * self.washout_volume_share:.1f} % of the "
            f"annular volume (> {self.washout_threshold:g} x gauge)",
            f"  holds             : {100 * self.washout_mud_share:.1f} % of the "
            f"residual mud",
            f"  concentration     : {self.concentration_ratio:.2f} x its volume share",
        ]
        worst = self.worst_intervals()
        if worst:
            lines.append("worst intervals     :")
            for top, bottom, mud, dia in worst:
                lines.append(
                    f"    {top:7.1f} - {bottom:7.1f} m   {mud:7.4f} m^3   "
                    f"mean hole {dia * 1e3:6.1f} mm"
                )
        return "\n".join(lines)


def mud_left_behind(result, fluid, gauge=None, washout_threshold=1.3) -> MudLeftReport:
    """Residual ``fluid`` in the annulus at the end of a circulation run."""
    g = result.annulus_grid
    i = result.fluids.index(fluid)
    order = np.argsort(g.z_centers)
    return MudLeftReport(
        depth=g.z_centers[order],
        hole_diameter=g.hole_diameter[order],
        cell_volume=g.cell_volume.sum(axis=(1, 2))[order],
        mud_volume=(result.annulus_fractions[i] * g.cell_volume).sum(axis=(1, 2))[order],
        gauge=float(gauge if gauge is not None else np.median(g.hole_diameter)),
        washout_threshold=washout_threshold,
    )
