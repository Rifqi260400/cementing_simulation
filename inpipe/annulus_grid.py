"""Annular mesh with a depth-varying cross-section.

Unlike the casing, the open hole is not a smooth cylinder: the caliper gives a
different hole diameter at every depth, so the annular area, the cell volumes
and the axial face areas all vary with ``z``.  That breaks the assumption the
in-pipe grid relies on - there, one cross-section is extruded and the area
cancels out of the transport update.  Here it does not, and the transport
kernel must carry face areas explicitly.

Discretisation
--------------
For a vertical well with concentric casing the flow is axisymmetric, so the
cross-section is cut into ``n_layer`` rings across the gap and ``n_azimuth``
equal sectors.  A ring between radii ``r1`` and ``r2`` has area
``pi (r2^2 - r1^2)``, split equally between sectors - exact, no quadrature.

The azimuthal dimension is uniform in a concentric vertical well, so it carries
no information yet; it is kept because eccentricity (which the paper's stratified
grid exists to represent) is where it would start to matter.

Geometry is evaluated at both cell centres and axial faces, because the
conservative update needs face areas that are consistent with the cell volumes
either side of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .caliper import CaliperLog
from .velocity import velocity_profile

__all__ = ["AnnulusGrid"]


@dataclass
class AnnulusGrid:
    """Structured ``n_axial x n_layer x n_azimuth`` annular mesh.

    Parameters
    ----------
    length : total measured depth spanned by the annulus [m].
    casing_od : casing outer diameter [m] - the inner wall of the annulus.
    caliper : hole diameter against depth; the outer wall.
    n_axial, n_layer, n_azimuth : mesh resolution.

    Depth convention
    ----------------
    ``z`` is measured depth, increasing downward, the same coordinate as the
    casing grid.  Annular flow is *upward*, so its velocities are negative in
    ``+z``.  Keeping one depth axis for both legs is what lets the shoe couple
    them without a coordinate flip.
    """

    length: float
    casing_od: float
    caliper: CaliperLog
    n_axial: int
    n_layer: int
    n_azimuth: int
    #: Store cells in *flow order*: index 0 at the shoe, the last index at
    #: surface, so annular flow runs along increasing index and the shared 1D
    #: advection kernel applies unchanged.  Depths therefore descend.
    flow_order: bool = True
    #: Measured depth of the top of the modelled interval [m].  The caliper is
    #: sampled at true depth, so this must be right when only part of the well
    #: is modelled.
    z_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.casing_od <= 0.0:
            raise ValueError("casing OD must be positive")
        self.dz = self.length / self.n_axial
        self.z_faces = np.linspace(
            self.z_offset, self.z_offset + self.length, self.n_axial + 1
        )
        if self.flow_order:
            self.z_faces = self.z_faces[::-1].copy()
        self.z_centers = 0.5 * (self.z_faces[:-1] + self.z_faces[1:])

        self.r_inner = 0.5 * self.casing_od
        self.hole_diameter = self.caliper.diameter_at(self.z_centers)
        self.hole_diameter_faces = self.caliper.diameter_at(self.z_faces)
        self.r_outer = 0.5 * self.hole_diameter
        self.r_outer_faces = 0.5 * self.hole_diameter_faces

        if np.any(self.r_outer <= self.r_inner) or np.any(
            self.r_outer_faces <= self.r_inner
        ):
            worst = float(np.min(np.concatenate([self.hole_diameter,
                                                 self.hole_diameter_faces])))
            raise ValueError(
                f"hole diameter falls to {worst:.4g} m, at or below the casing OD "
                f"{self.casing_od:.4g} m - the annulus closes"
            )

        # Cell geometry at centres and at faces.
        self.cell_area, self.cell_r, self.ring_faces = self._section(self.r_outer)
        self.face_area, _, _ = self._section(self.r_outer_faces)
        self.cell_volume = self.cell_area * self.dz

        # Slot parameters per station, for the velocity profile.
        self.half_gap = 0.5 * (self.r_outer - self.r_inner)
        self.slot_width = math.pi * (self.r_outer + self.r_inner)
        #: Distance of each cell centroid from the mid-gap, in [0, half_gap].
        self.cell_s = np.abs(self.cell_r - 0.5 * (self.r_outer + self.r_inner)[:, None, None])

    def _section(self, r_outer):
        """Areas, centroid radii and ring boundaries for each station.

        ``r_outer`` is ``(n,)``; returns ``(n, n_layer, n_azimuth)`` areas and
        centroid radii, and ``(n, n_layer + 1)`` ring boundaries.
        """
        r_outer = np.asarray(r_outer, dtype=float)
        frac = np.linspace(0.0, 1.0, self.n_layer + 1)
        # Rings equally spaced across the gap, the annular analogue of the
        # in-pipe uniform-y layering (assumption A-03).
        rings = self.r_inner + (r_outer[:, None] - self.r_inner) * frac[None, :]
        ring_area = math.pi * (rings[:, 1:] ** 2 - rings[:, :-1] ** 2)
        area = np.repeat(ring_area[:, :, None], self.n_azimuth, axis=2) / self.n_azimuth
        # Area-weighted centroid radius of a ring: 2/3 (r2^3-r1^3)/(r2^2-r1^2).
        r1, r2 = rings[:, :-1], rings[:, 1:]
        centroid = (2.0 / 3.0) * (r2**3 - r1**3) / np.maximum(r2**2 - r1**2, 1e-300)
        centroid = np.repeat(centroid[:, :, None], self.n_azimuth, axis=2)
        return area, centroid, rings

    # -- properties ---------------------------------------------------------

    @property
    def shape(self) -> tuple:
        return (self.n_axial, self.n_layer, self.n_azimuth)

    @property
    def station_area(self) -> np.ndarray:
        """Total annular area at each station [m^2], shape ``(n_axial,)``."""
        return self.cell_area.sum(axis=(1, 2))

    @property
    def total_volume(self) -> float:
        return float(self.cell_volume.sum())

    def exact_station_area(self) -> np.ndarray:
        """``pi (r_o^2 - r_i^2)`` per station, for checking the discretisation."""
        return math.pi * (self.r_outer**2 - self.r_inner**2)

    # -- velocity -----------------------------------------------------------

    def map_velocity(self, profiles, method: str = "area_average") -> np.ndarray:
        """Map a per-station :class:`~inpipe.slot.SlotProfile` list onto cells.

        ``area_average`` (default)
            Exact area average of ``u`` over each ring, by Gauss-Legendre
            quadrature in ``r`` with weights precomputed once (the annulus
            geometry is fixed for the whole run, so they never change).
        ``centroid``
            Evaluate at the ring's area-weighted centroid radius.  Cheaper, but
            it mis-states the velocity distribution by ~1.6 % at nine rings -
            the same failure the in-pipe grid showed (assumption A-04).

        Returns velocities in the magnitude sense; the solver applies the
        upward sign for the annulus.
        """
        if method == "centroid":
            u = np.empty(self.shape)
            for k, prof in enumerate(profiles):
                u[k] = velocity_profile(
                    np.minimum(self.cell_s[k], prof.half_gap),
                    prof.fluid, prof.half_gap, prof.tau_w,
                )
            return u
        if method != "area_average":
            raise ValueError(f"unknown velocity mapping {method!r}")

        S, W = self._quadrature()
        u = np.empty(self.shape)
        for k, prof in enumerate(profiles):
            vals = velocity_profile(
                np.minimum(S[k], prof.half_gap), prof.fluid, prof.half_gap, prof.tau_w
            )
            u[k] = (W[k] * vals).sum(axis=1)[:, None]
        return u

    def _quadrature(self, n_gauss: int = 12):
        """Precompute per-ring quadrature nodes and weights.

        Returns ``(S, W)`` of shape ``(n_axial, n_layer, n_gauss)``: ``S`` the
        distance from mid-gap at each node, ``W`` weights that sum to one over
        each ring, so ``sum_j W u(S)`` is the ring's area average of ``u``.
        """
        cached = getattr(self, "_quad_cache", None)
        if cached is not None:
            return cached

        gx, gw = np.polynomial.legendre.leggauss(n_gauss)
        r1 = self.ring_faces[:, :-1]           # (n_axial, n_layer)
        r2 = self.ring_faces[:, 1:]
        mid = 0.5 * (r1 + r2)[:, :, None]
        half = 0.5 * (r2 - r1)[:, :, None]
        r_nodes = mid + half * gx[None, None, :]
        # dA = 2 pi r dr, normalised by the ring area so the weights sum to 1.
        w = gw[None, None, :] * half * 2.0 * math.pi * r_nodes
        ring_area = math.pi * (r2**2 - r1**2)
        w = w / np.maximum(ring_area, 1e-300)[:, :, None]

        r_mid_gap = 0.5 * (self.r_outer + self.r_inner)[:, None, None]
        S = np.abs(r_nodes - r_mid_gap)
        self._quad_cache = (S, w)
        return self._quad_cache

    def normalise_to_flow_rate(self, u: np.ndarray, q: float) -> np.ndarray:
        """Rescale each station so ``sum_cells u*A`` equals ``q`` exactly.

        The annular analogue of assumption A-22, applied at cell centres.
        """
        station_q = np.einsum("klm,klm->k", u, self.cell_area)
        scale = np.divide(q, station_q, out=np.ones_like(station_q),
                          where=station_q != 0.0)
        return u * scale[:, None, None]

    def normalise_face_flux(self, u_faces: np.ndarray, q: float) -> np.ndarray:
        """Rescale each *face* so ``sum_cells u*A_face`` equals ``q`` exactly.

        This, not the cell-centre version, is what the conservative update
        needs.  The flux crosses **face** areas, and where the hole diameter
        varies those differ from the cell areas either side, so a velocity
        field normalised at cell centres does not carry ``q`` through the faces
        between them.  Left uncorrected it breaks the discrete continuity the
        transverse closure depends on, and ``sum_i f_i`` drifts by percent.

        Enforcing it is physics, not a fudge: the well is incompressible, so
        every axial face passes the same volumetric rate.
        """
        face_q = np.einsum("klm,klm->k", u_faces, self.face_area)
        scale = np.divide(q, face_q, out=np.ones_like(face_q), where=face_q != 0.0)
        return u_faces * scale[:, None, None]
