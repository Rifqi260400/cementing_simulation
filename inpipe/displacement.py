"""Interface length and swept-region efficiency - Xue et al. (2022).

Xue, Han, Zhang & Fan (2022), *J. Pet. Sci. Eng.* 208:109393, make a criticism
of the usual displacement efficiency that applies squarely to this model.
Volumetric efficiency is ``eta = V_cement / V_annulus``.  Before the cement
front reaches the outlet, the cement in the annulus is just what has been
pumped into it, so ``eta`` rises along a straight line whose slope is set by
the pump rate and the annulus volume - **and nothing else**.  Every rheology,
every rate, every geometry gives the same line.  All the physics shows up only
as a difference in the time the front arrives.

So the efficiency history this model plots carries almost no information while
the job is running, and its final value is meaningful only because the job
pumps past breakthrough.  Their fix is two quantities that are informative from
the first minute:

``interface length``
    the distance between the 20 % and 80 % contours of the displacing fluid.
    It measures how far the front has been stretched - the mixed volume - and
    it separates a sharp front from a smeared one immediately.

``swept efficiency``
    efficiency of the region the front has already passed, rather than of the
    whole annulus.  It asks "of what the cement has reached, how much did it
    actually clean", and it settles to a value early instead of ramping.

Their own definition of the second is the ratio of the volume behind the 80 %
contour to the volume behind the 20 % contour - two contour positions, which is
what a CFD post-processor can extract.  This model holds the whole field, so
the actual cement volume in the swept region is used instead; the two agree in
the sharp-interface limit and this one does not assume the region behind the
back edge is clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["InterfaceMetrics", "interface_metrics", "FRONT_EDGE", "BACK_EDGE"]

#: Contours bounding the interface, following Xue et al. (2022) Section 3.1.1.
FRONT_EDGE = 0.2
BACK_EDGE = 0.8


@dataclass(frozen=True)
class InterfaceMetrics:
    """Where the front is, how long it is, and how clean it left things."""

    front_depth: float       # 20 % contour [m]; nan before the front exists
    back_depth: float        # 80 % contour [m]
    length: float            # back - front [m]; the stretched, mixed zone
    swept_efficiency: float  # cement fraction of the region behind the front


def _crossing(fraction, depth, threshold):
    """Depth where ``fraction`` falls through ``threshold`` on the way up.

    ``fraction`` and ``depth`` are in flow order - index 0 at the shoe - so the
    furthest-travelled crossing is the *last* index still above the threshold.
    Taking the last rather than the first matters when the front is not
    monotonic, which it is not: cement channels through the narrow side of a
    washout and gets ahead of itself.
    """
    above = np.flatnonzero(fraction >= threshold)
    if above.size == 0:
        return np.nan
    k = int(above[-1])
    if k == fraction.size - 1:
        return float(depth[k])
    c0, c1 = float(fraction[k]), float(fraction[k + 1])
    if c0 == c1:
        return float(depth[k])
    step = (c0 - threshold) / (c0 - c1)
    return float(depth[k] + step * (depth[k + 1] - depth[k]))


def interface_metrics(fraction, station_volume, depth,
                      front_edge=FRONT_EDGE, back_edge=BACK_EDGE):
    """Interface length and swept efficiency from one instant of the annulus.

    ``fraction`` is the volume-averaged displacing-fluid fraction per station,
    ``station_volume`` the annular volume of each, both in flow order.
    """
    fraction = np.asarray(fraction, dtype=float)
    depth = np.asarray(depth, dtype=float)

    z_front = _crossing(fraction, depth, front_edge)
    z_back = _crossing(fraction, depth, back_edge)
    length = np.nan if (np.isnan(z_front) or np.isnan(z_back)) else z_back - z_front

    swept = np.nan
    above = np.flatnonzero(fraction >= front_edge)
    if above.size:
        # Whole stations up to and including the front's own station.  Cells are
        # in flow order, so that is a prefix.
        k = int(above[-1])
        volume = np.asarray(station_volume, dtype=float)[: k + 1]
        if volume.sum() > 0.0:
            swept = float((fraction[: k + 1] * volume).sum() / volume.sum())
    return InterfaceMetrics(front_depth=z_front, back_depth=z_back,
                            length=length, swept_efficiency=swept)
