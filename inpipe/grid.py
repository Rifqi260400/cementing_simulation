"""Module 3 - the structured 3D stratified mesh (paper Section 2.2, Figs. 2b/3a).

The circular cross-section is cut by straight *horizontal* lines into layers,
and each layer is cut by *longitudinal* (vertical) lines into cells.  The mesh
is ``n_axial x n_layer x n_azimuth``; a 2D cross-section plane is built once and
extruded along measured depth, because the pipe has uniform diameter.

Geometry is exact
-----------------
Cell areas and centroids are obtained in closed form by integrating the
circular-segment height

    h(x) = [ min(y_hi, sqrt(R^2 - x^2)) - max(y_lo, -sqrt(R^2 - x^2)) ]_+

analytically over sub-intervals split at the circle-intersection breakpoints
(assumption A-12).  No Monte Carlo, and no quadrature error: the cell areas sum
to ``pi R^2`` to round-off at *any* resolution.

Orientation
-----------
In a vertical pipe the direction of the central longitudinal line is not
uniquely defined (paper Section 2.2).  A fixed reference azimuth is used
(``GridConfig.reference_azimuth``, default 0); at ``beta = 0`` nothing in the
model depends on it (assumption A-09).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .config import GridConfig
from .velocity import VelocityProfile

__all__ = ["Grid", "layer_boundaries", "cell_moments"]

#: Cells whose area falls below this fraction of pi R^2 fall back to the
#: bounding-box midpoint for their centroid, to avoid 0/0 cancellation.
_DEGENERATE_AREA_FRAC = 1.0e-14


# ---------------------------------------------------------------------------
# Closed-form segment integrals
# ---------------------------------------------------------------------------


def _int_s(a: float, b: float, R: float) -> float:
    """int_a^b sqrt(R^2 - x^2) dx."""

    def F(x):
        x = min(max(x, -R), R)
        return 0.5 * (x * math.sqrt(max(R * R - x * x, 0.0)) + R * R * math.asin(x / R))

    return F(b) - F(a)


def _int_x_s(a: float, b: float, R: float) -> float:
    """int_a^b x sqrt(R^2 - x^2) dx."""

    def F(x):
        x = min(max(x, -R), R)
        return -((max(R * R - x * x, 0.0)) ** 1.5) / 3.0

    return F(b) - F(a)


def _int_s2(a: float, b: float, R: float) -> float:
    """int_a^b (R^2 - x^2) dx."""

    def F(x):
        return R * R * x - x**3 / 3.0

    return F(b) - F(a)


def _boundary_integrals(a: float, b: float, R: float, value, is_circle: bool, sign: float):
    """Return ``(int g, int x g, int g^2)`` on ``[a, b]``.

    ``g`` is either the constant ``value`` (``is_circle=False``) or
    ``sign * sqrt(R^2 - x^2)`` (``is_circle=True``).
    """
    if is_circle:
        return (
            sign * _int_s(a, b, R),
            sign * _int_x_s(a, b, R),
            _int_s2(a, b, R),  # (sign * s)^2 = s^2
        )
    c = value
    return (c * (b - a), c * (b * b - a * a) / 2.0, c * c * (b - a))


def cell_moments(xa: float, xb: float, y_lo: float, y_hi: float, R: float):
    """Exact area and centroid of the region ``[xa, xb] x [y_lo, y_hi]`` clipped
    to the disc of radius ``R``.

    Returns ``(area, x_centroid, y_centroid)``.  For a zero-area region the
    centroid falls back to the box midpoint.
    """
    xa = max(xa, -R)
    xb = min(xb, R)
    if xb <= xa or y_hi <= y_lo:
        return 0.0, 0.5 * (xa + xb), 0.5 * (y_lo + y_hi)

    # Breakpoints at every |x| where a horizontal cut line meets the circle.
    # These are the points where (a) the top or bottom boundary switches
    # between the cut line and the circle, and (b) the top and bottom
    # boundaries cross, i.e. where the clipped region becomes empty.  Both
    # kinds must be included, otherwise a sub-interval can straddle a crossing
    # and contribute negative height.
    breaks = {xa, xb}
    for y_b in (y_hi, y_lo):
        if abs(y_b) <= R:
            xc = math.sqrt(max(R * R - y_b * y_b, 0.0))
            breaks.update({-xc, xc})
    xs = sorted(x for x in breaks if xa <= x <= xb)

    area = mx = my = 0.0
    for a, b in zip(xs[:-1], xs[1:]):
        if b <= a:
            continue
        xm = 0.5 * (a + b)
        s_mid = math.sqrt(max(R * R - xm * xm, 0.0))

        # top(x) = min(y_hi, s(x));  bot(x) = max(y_lo, -s(x))
        top_is_circle = s_mid < y_hi
        bot_is_circle = -s_mid > y_lo
        top_val = y_hi if not top_is_circle else 0.0
        bot_val = y_lo if not bot_is_circle else 0.0

        if (top_val if not top_is_circle else s_mid) <= (
            bot_val if not bot_is_circle else -s_mid
        ):
            continue  # empty sliver

        t0, t1, t2 = _boundary_integrals(a, b, R, top_val, top_is_circle, +1.0)
        b0, b1, b2 = _boundary_integrals(a, b, R, bot_val, bot_is_circle, -1.0)

        area += t0 - b0
        mx += t1 - b1
        my += 0.5 * (t2 - b2)

    if area <= _DEGENERATE_AREA_FRAC * math.pi * R * R:
        return max(area, 0.0), 0.5 * (xa + xb), 0.5 * (y_lo + y_hi)
    return area, mx / area, my / area


# ---------------------------------------------------------------------------
# Layer boundaries
# ---------------------------------------------------------------------------


def _disc_area_below(y: float, R: float) -> float:
    """Area of the disc of radius ``R`` lying below the chord at height ``y``."""
    y = min(max(y, -R), R)
    return y * math.sqrt(max(R * R - y * y, 0.0)) + R * R * math.asin(y / R) + 0.5 * math.pi * R * R


def layer_boundaries(n_layer: int, R: float, rule: str = "uniform_y") -> np.ndarray:
    """Horizontal cut positions ``y`` in ``[-R, R]``, length ``n_layer + 1``.

    ``uniform_y`` (default, assumption A-03)
        Equal spacing in the chord coordinate.  Simple; near-wall layers are
        thin in *area*.
    ``equal_area``
        Each layer holds ``pi R^2 / n_layer`` of cross-sectional area.  Offered
        as the swap-in alternative required by A-03.
    """
    if n_layer < 1:
        raise ValueError("n_layer must be >= 1")
    if rule == "uniform_y":
        return np.linspace(-R, R, n_layer + 1)
    if rule == "equal_area":
        total = math.pi * R * R
        ys = [-R]
        for i in range(1, n_layer):
            target = total * i / n_layer
            ys.append(brentq(lambda y: _disc_area_below(y, R) - target, -R, R, xtol=1e-15))
        ys.append(R)
        return np.array(ys)
    raise ValueError(f"unknown layer rule {rule!r}")


def _box_radial_extent(xa: float, xb: float, y_lo: float, y_hi: float):
    """Min and max distance from the origin to the box ``[xa,xb] x [y_lo,y_hi]``."""
    dx = 0.0 if xa <= 0.0 <= xb else min(abs(xa), abs(xb))
    dy = 0.0 if y_lo <= 0.0 <= y_hi else min(abs(y_lo), abs(y_hi))
    r_min = math.hypot(dx, dy)
    r_max = math.hypot(max(abs(xa), abs(xb)), max(abs(y_lo), abs(y_hi)))
    return r_min, r_max


def _layer_half_width(y_lo: float, y_hi: float, R: float) -> float:
    """Maximum ``|x|`` reached by the layer inside the disc."""
    if y_lo <= 0.0 <= y_hi:
        return R
    d = min(abs(y_lo), abs(y_hi))
    return math.sqrt(max(R * R - d * d, 0.0))


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


@dataclass
class Grid:
    """Structured ``n_axial x n_layer x n_azimuth`` mesh for a uniform-ID pipe.

    Cross-section arrays are shaped ``(n_layer, n_azimuth)``; 3D fields are
    shaped ``(n_axial, n_layer, n_azimuth)``.
    """

    radius: float
    length: float
    config: GridConfig
    #: Number of annular bins used to build the exact area-averaging weights.
    n_radial: int = 1024
    #: Measured depth of the top of the modelled interval [m].  Non-zero when
    #: only a sub-interval of the well is modelled, so that ``z_centers`` stay
    #: true depths.
    z_offset: float = 0.0

    def __post_init__(self) -> None:
        R, cfg = self.radius, self.config
        self._radial_cache = None
        self.n_axial = cfg.n_axial
        self.n_layer = cfg.n_layer
        self.n_azimuth = cfg.n_azimuth

        # --- axial ---------------------------------------------------------
        self.dz = self.length / self.n_axial
        self.z_faces = np.linspace(
            self.z_offset, self.z_offset + self.length, self.n_axial + 1
        )
        self.z_centers = 0.5 * (self.z_faces[:-1] + self.z_faces[1:])

        # --- cross-section -------------------------------------------------
        self.y_layer_faces = layer_boundaries(self.n_layer, R, cfg.layer_rule)
        shape = (self.n_layer, self.n_azimuth)
        self.cell_area = np.zeros(shape)
        self.cell_x = np.zeros(shape)
        self.cell_y = np.zeros(shape)
        # x-edges of each layer's longitudinal cuts, (n_layer, n_azimuth + 1)
        self.x_layer_faces = np.zeros((self.n_layer, self.n_azimuth + 1))

        for i in range(self.n_layer):
            y_lo, y_hi = self.y_layer_faces[i], self.y_layer_faces[i + 1]
            w = _layer_half_width(y_lo, y_hi, R)
            edges = np.linspace(-w, w, self.n_azimuth + 1)
            self.x_layer_faces[i] = edges
            for j in range(self.n_azimuth):
                a, xc, yc = cell_moments(edges[j], edges[j + 1], y_lo, y_hi, R)
                self.cell_area[i, j] = a
                self.cell_x[i, j] = xc
                self.cell_y[i, j] = yc

        self.cell_r = np.minimum(np.hypot(self.cell_x, self.cell_y), R)
        self.cell_theta = np.mod(
            np.arctan2(self.cell_y, self.cell_x) + cfg.reference_azimuth, 2.0 * math.pi
        )
        self.cell_volume = self.cell_area * self.dz
        #: Axial face area equals the cell cross-sectional area (uniform ID).
        self.axial_face_area = self.cell_area

    # -- derived quantities -------------------------------------------------

    @property
    def total_area(self) -> float:
        return float(self.cell_area.sum())

    @property
    def total_volume(self) -> float:
        """Total pipe volume represented by the mesh [m^3]."""
        return float(self.cell_volume.sum()) * self.n_axial

    @property
    def cross_section_shape(self) -> tuple:
        return (self.n_layer, self.n_azimuth)

    @property
    def shape(self) -> tuple:
        return (self.n_axial, self.n_layer, self.n_azimuth)

    # -- velocity mapping ---------------------------------------------------

    def map_velocity(self, profile: VelocityProfile, method: str = "centroid") -> np.ndarray:
        """Map a 1D ``u(r)`` onto the cross-section, shape ``(n_layer, n_azimuth)``.

        ``centroid`` (default, assumption A-04)
            Evaluate ``u`` at the cell centroid radius.
        ``area_average``
            Exact area average of ``u`` over the cell, by tensor
            Gauss-Legendre quadrature respecting the circular boundary.
        """
        if method == "centroid":
            return profile(self.cell_r)
        if method == "area_average":
            return self._area_averaged_velocity(profile)
        raise ValueError(f"unknown velocity mapping {method!r}")

    def _area_averaged_velocity(self, profile: VelocityProfile) -> np.ndarray:
        """Area average of ``u(r)`` over each cell via precomputed radial weights.

        Because ``u`` depends on radius alone, the exact cell integral is

            int_cell u dA = int_0^R u(r) * (dA_cell/dr) dr

        so the cell integral is a *linear functional* of the 1D profile.  The
        weights ``W[cell, k] = area(cell & disc(r_{k+1})) - area(cell & disc(r_k))``
        depend only on geometry and are built once (exactly, by the same
        closed-form segment integrals used for the cell areas).  At run time the
        mapping is one profile evaluation at ``n_radial`` points plus a matvec,
        which is what makes area-averaging affordable inside the time loop.
        """
        W, r_nodes = self._radial_weights()
        u_nodes = np.asarray(profile(r_nodes), dtype=float)
        integrals = np.asarray(W @ u_nodes).ravel()
        out = np.zeros(self.cross_section_shape)
        area = self.cell_area
        good = area > 0.0
        out[good] = integrals.reshape(self.cross_section_shape)[good] / area[good]
        return out

    def _radial_weights(self):
        """Build (and cache) the exact cell-by-annulus area matrix.

        Returns ``(W, r_nodes)`` with ``W`` of shape ``(n_layer*n_azimuth,
        n_radial)`` and ``r_nodes`` the area-weighted mean radius of each
        annular bin within each cell's contribution -- here the bin midpoint,
        which makes the radial quadrature second-order accurate in ``dr``.
        """
        if getattr(self, "_radial_cache", None) is not None:
            return self._radial_cache

        R, m = self.radius, self.n_radial
        edges = np.linspace(0.0, R, m + 1)
        r_nodes = 0.5 * (edges[:-1] + edges[1:])
        n_cells = self.n_layer * self.n_azimuth
        W = np.zeros((n_cells, m))

        for i in range(self.n_layer):
            y_lo, y_hi = self.y_layer_faces[i], self.y_layer_faces[i + 1]
            for j in range(self.n_azimuth):
                xa = self.x_layer_faces[i, j]
                xb = self.x_layer_faces[i, j + 1]
                c = i * self.n_azimuth + j
                full = self.cell_area[i, j]
                if full <= 0.0:
                    continue
                # Radial extent of the box: bins entirely inside contribute
                # nothing beyond the cell's full area, bins entirely outside
                # contribute nothing at all, so only the overlap is computed.
                r_min, r_max = _box_radial_extent(xa, xb, y_lo, y_hi)
                k_lo = max(int(np.searchsorted(edges, r_min, side="right")) - 1, 0)
                k_hi = min(int(np.searchsorted(edges, r_max, side="left")), m)
                # Cumulative area of the cell inside the disc of radius r.
                prev = 0.0
                for k in range(k_lo, k_hi):
                    r_out = edges[k + 1]
                    # Cell clipped to the disc of radius r_out.  Layer cut
                    # lines are clipped to that disc as well, which is exactly
                    # what "cell & disc(r_out)" means.
                    cur, _, _ = cell_moments(
                        max(xa, -r_out), min(xb, r_out),
                        max(y_lo, -r_out), min(y_hi, r_out),
                        r_out,
                    )
                    W[c, k] = cur - prev
                    prev = cur
                # Everything beyond r_max is the remainder of the cell.
                if k_hi < m:
                    W[c, k_hi] += full - prev
                elif abs(prev - full) > 1e-12 * full:  # pragma: no cover
                    W[c, m - 1] += full - prev

        # W is mostly zeros - each cell spans only a slice of the radius - so a
        # sparse matrix cuts both the memory and the matvec cost, which is what
        # dominates the velocity mapping at fine cross-sections.
        from scipy.sparse import csr_matrix

        self._radial_cache = (csr_matrix(W), r_nodes)
        return self._radial_cache

    def _area_averaged_velocity_quadrature(
        self, profile: VelocityProfile, n_gauss: int = 24
    ) -> np.ndarray:
        """Reference implementation of the area average by 2D Gauss-Legendre.

        Slow; used only to verify :meth:`_area_averaged_velocity` in the tests.
        """
        R = self.radius
        gx, gw = np.polynomial.legendre.leggauss(n_gauss)
        out = np.zeros(self.cross_section_shape)
        for i in range(self.n_layer):
            y_lo, y_hi = self.y_layer_faces[i], self.y_layer_faces[i + 1]
            for j in range(self.n_azimuth):
                xa = max(self.x_layer_faces[i, j], -R)
                xb = min(self.x_layer_faces[i, j + 1], R)
                if xb <= xa or self.cell_area[i, j] <= 0.0:
                    continue
                brk = {xa, xb}
                for yb in (y_hi, y_lo):
                    if abs(yb) <= R:
                        xc = math.sqrt(max(R * R - yb * yb, 0.0))
                        brk.update({-xc, xc})
                xs = sorted(x for x in brk if xa <= x <= xb)
                acc = 0.0
                for a, b in zip(xs[:-1], xs[1:]):
                    if b <= a:
                        continue
                    xn = 0.5 * (b - a) * gx + 0.5 * (a + b)
                    wx = 0.5 * (b - a) * gw
                    s = np.sqrt(np.maximum(R * R - xn * xn, 0.0))
                    top = np.minimum(y_hi, s)
                    bot = np.maximum(y_lo, -s)
                    hgt = np.maximum(top - bot, 0.0)
                    yn = 0.5 * hgt[:, None] * gx[None, :] + 0.5 * (top + bot)[:, None]
                    wy = 0.5 * hgt[:, None] * gw[None, :]
                    rr = np.minimum(np.hypot(xn[:, None], yn), R)
                    acc += float((profile(rr) * wy).sum(axis=1) @ wx)
                out[i, j] = acc / self.cell_area[i, j]
        return out

    def flow_rate_from_cells(self, u_cells: np.ndarray) -> float:
        """Area-weighted flow rate implied by a cross-sectional velocity field."""
        return float((u_cells * self.cell_area).sum())
