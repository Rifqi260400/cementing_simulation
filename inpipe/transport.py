"""Module 4 - VOF transport of fluid volume fractions (paper Section 2.4, A.2).

Governing equation (paper Eq. 2), with the diffusive right-hand side dropped as
the paper does::

    df_i/dt + u . grad(f_i) = 0

Because the reduced-order model never solves a transverse velocity, transport
is purely axial: each ``(layer, azimuth)`` column is an independent 1D
advection problem carrying its own velocity ``u(r, theta)``.  That is exactly
the mechanism that stretches a flat interface into a parabola, so the kernel
below is written once for a column and mapped over all columns at once.

Discretisation
--------------
The paper's Eq. A.9 as printed is dimensionally inconsistent (no face area) and
carries the wrong sign for outward normals.  The correct conservative
finite-volume update (assumption A-14) is::

    f_i^{n+1} = f_i^n - (dt / dV) * sum_j ( u_{n,j} * A_j * f_{s,j} )

with outward-positive face normals, integrated in time with explicit Euler.

Divergence correction (assumption A-07)
---------------------------------------
The conservative update preserves ``sum_i f_i = 1`` only when the discrete
continuity condition ``sum_j u_j A_j = 0`` holds per cell.  In this model it
does *not* in general: when neighbouring depth stations hold different
effective fluids, their velocity profiles differ, so a column has
``du/dz != 0`` even though the total ``Q`` is constant.  Subtracting
``f_i^n * sum_j u_j A_j`` restores the advective form of Eq. (2) and with it
sum-to-one.  The correction vanishes identically for a depth-uniform profile,
so it changes nothing in the single-rheology cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "CFLViolation",
    "BoundednessViolation",
    "upwind_faces",
    "cfl_timestep",
    "advect",
    "advect_multi",
    "check_sum_to_one",
    "check_bounded",
    "numerical_diffusivity",
    "FACE_SCHEMES",
]


class CFLViolation(RuntimeError):
    """Raised when the requested timestep violates the CFL condition."""


class BoundednessViolation(RuntimeError):
    """Raised when volume fractions leave ``[0, 1]`` or stop summing to one."""


# ---------------------------------------------------------------------------
# Face-value schemes.  Injected into :func:`advect` so Phase 2 can add
# donor-acceptor and THINC without touching the solver (assumption A-08).
# ---------------------------------------------------------------------------


def upwind_faces(f: np.ndarray, u_faces: np.ndarray, dz: float, dt: float) -> np.ndarray:
    """First-order upwind interior face values.

    Parameters
    ----------
    f : ``(n_axial, ...)`` cell values.
    u_faces : ``(n_axial - 1, ...)`` velocities on the *interior* axial faces.

    Returns the ``(n_axial - 1, ...)`` interior face values.  Deliberately
    diffusive - this is the Phase 1 baseline, and quantifying its smearing is
    the point.
    """
    return np.where(u_faces >= 0.0, f[:-1], f[1:])


#: Registry of available face-value schemes.
FACE_SCHEMES: dict[str, Callable] = {"upwind": upwind_faces}


def resolve_face_scheme(scheme) -> Callable:
    if callable(scheme):
        return scheme
    try:
        return FACE_SCHEMES[scheme]
    except KeyError:
        raise ValueError(
            f"unknown face scheme {scheme!r}; available: {sorted(FACE_SCHEMES)}"
        ) from None


# ---------------------------------------------------------------------------
# Timestep
# ---------------------------------------------------------------------------


def cfl_timestep(u: np.ndarray, dz: float, cfl: float) -> float:
    """Largest timestep satisfying ``dt < cfl * dz / max|u|`` [s]."""
    u_max = float(np.max(np.abs(u)))
    if u_max <= 0.0:
        return np.inf
    return cfl * dz / u_max


def assert_cfl(u: np.ndarray, dz: float, dt: float, cfl: float) -> float:
    """Assert the CFL condition and return the realised Courant number.

    Raises :class:`CFLViolation` rather than clipping ``dt`` silently: a
    violation is a modelling error, not something to hide (assumption A-05).
    """
    u_max = float(np.max(np.abs(u)))
    courant = u_max * dt / dz
    if courant > cfl * (1.0 + 1e-12):
        raise CFLViolation(
            f"Courant number {courant:.6g} exceeds the configured limit {cfl:.6g} "
            f"(max|u| = {u_max:.6g} m/s, dz = {dz:.6g} m, dt = {dt:.6g} s)"
        )
    return courant


# ---------------------------------------------------------------------------
# Advection kernel
# ---------------------------------------------------------------------------


def _face_velocities(u_cells: np.ndarray) -> np.ndarray:
    """Interior axial face velocities by arithmetic mean of the two neighbours.

    For a depth-uniform profile this is exact; where the profile changes with
    depth it is the natural second-order face reconstruction, and any residual
    ``sum_j u_j A_j != 0`` is handled by the divergence correction.
    """
    return 0.5 * (u_cells[:-1] + u_cells[1:])


def advect(
    f: np.ndarray,
    u_cells: np.ndarray,
    dz: float,
    dt: float,
    face_scheme="upwind",
    inlet_value=0.0,
    divergence_correction: bool = True,
    area: np.ndarray | None = None,
) -> np.ndarray:
    """One explicit Euler step of axial advection.

    Parameters
    ----------
    f : ``(n_axial, ...)`` volume fraction (or mixing status) field.
    u_cells : ``(n_axial, ...)`` axial cell velocities, broadcastable to ``f``.
        Flow in ``+z`` (the flow direction, see :mod:`inpipe.velocity`) is
        positive.
    dz : uniform axial cell height [m].
    dt : timestep [s].
    face_scheme : name or callable returning interior face values.
    inlet_value : Dirichlet value carried in through the ``z = 0`` face.
        Scalar or an array broadcastable to a cross-section.
    divergence_correction : subtract ``f * div(u)`` so that a set of fields
        summing to one keeps summing to one (assumption A-07).
    area : cell cross-sectional area, ``(n_layer, n_azimuth)``.  Only needed
        for the record; the update is area-independent for a uniform-ID pipe
        because the same area divides out of flux and volume.

    Returns the updated field; ``f`` is not modified.
    """
    scheme = resolve_face_scheme(face_scheme)
    u_cells = np.broadcast_to(np.asarray(u_cells, dtype=float), f.shape)

    u_int = _face_velocities(u_cells)
    f_int = scheme(f, u_int, dz, dt)

    # Boundary faces.  Inlet at z = 0: Dirichlet on inflow, upwind on outflow.
    # Outlet at z = L: zero-gradient outflow (assumption A-18).
    u_in = u_cells[0]
    u_out = u_cells[-1]
    f_in = np.where(u_in >= 0.0, np.broadcast_to(inlet_value, u_in.shape), f[0])
    f_out = np.where(u_out >= 0.0, f[-1], np.broadcast_to(inlet_value, u_out.shape))

    # Assemble face velocities and face values including boundaries.
    uf = np.concatenate([u_in[None, ...], u_int, u_out[None, ...]], axis=0)
    ff = np.concatenate([f_in[None, ...], f_int, f_out[None, ...]], axis=0)

    flux = uf * ff  # per unit area; the area cancels for a uniform-ID pipe
    div_flux = (flux[1:] - flux[:-1]) / dz

    if divergence_correction:
        div_u = (uf[1:] - uf[:-1]) / dz
        div_flux = div_flux - f * div_u

    return f - dt * div_flux


def advect_multi(
    fields: np.ndarray,
    u_cells: np.ndarray,
    dz: float,
    dt: float,
    face_scheme="upwind",
    inlet_values=None,
    divergence_correction: bool = True,
) -> np.ndarray:
    """Advect a stack of fields, ``(n_fluids, n_axial, n_layer, n_azimuth)``.

    ``inlet_values`` is a sequence of length ``n_fluids``.
    """
    fields = np.asarray(fields, dtype=float)
    if inlet_values is None:
        inlet_values = np.zeros(fields.shape[0])
    out = np.empty_like(fields)
    for i in range(fields.shape[0]):
        out[i] = advect(
            fields[i],
            u_cells,
            dz,
            dt,
            face_scheme=face_scheme,
            inlet_value=inlet_values[i],
            divergence_correction=divergence_correction,
        )
    return out


# ---------------------------------------------------------------------------
# Multi-fluid closure checks.  Report, never clip: a boundedness failure is a
# Phase 2 diagnostic, not something to hide (spec Section 5.5).
# ---------------------------------------------------------------------------


def check_sum_to_one(fields: np.ndarray, atol: float = 1e-12) -> float:
    """Return the worst ``|sum_i f_i - 1|``; raise if it exceeds ``atol``."""
    err = float(np.max(np.abs(fields.sum(axis=0) - 1.0)))
    if err > atol:
        raise BoundednessViolation(
            f"volume fractions do not sum to one: max |sum f_i - 1| = {err:.3e} "
            f"(tolerance {atol:.1e})"
        )
    return err


def check_bounded(fields: np.ndarray, atol: float = 1e-12) -> tuple[float, float]:
    """Return ``(min f, max f)``; raise if outside ``[0, 1]`` beyond ``atol``."""
    fmin = float(fields.min())
    fmax = float(fields.max())
    if fmin < -atol or fmax > 1.0 + atol:
        raise BoundednessViolation(
            f"volume fractions left [0, 1]: min = {fmin:.6e}, max = {fmax:.6e} "
            f"(tolerance {atol:.1e})"
        )
    return fmin, fmax


# ---------------------------------------------------------------------------
# Numerical diffusion
# ---------------------------------------------------------------------------


def numerical_diffusivity(dz: float, dt: float) -> float:
    """The paper's scale estimate ``Dm_num = dx^2 / dt`` [m^2/s] (Appendix A.2).

    Dai et al. quote ``dx = 30 m``, ``dt = 0.1 min`` giving ~150 m^2/s, against
    a physical ``Dm`` of 1e-3 to 1e-4 m^2/s - a ratio of ~1e5, which is their
    justification for discarding the diffusive term of Eq. (2) entirely.  That
    justification weakens quadratically as the mesh is refined, which is why
    this number is reported at every resolution.
    """
    return dz * dz / dt


def upwind_diffusivity(u: float, dz: float, dt: float) -> float:
    """Modified-equation diffusivity of first-order upwind + explicit Euler.

    ``D = (u * dz / 2) * (1 - C)`` with Courant number ``C = u dt / dz``.  This
    is the *actual* leading-order numerical diffusion of the scheme, and it is a
    sharper statement than the paper's ``dx^2/dt`` scale estimate: it vanishes
    at ``C = 1`` and is proportional to ``dz`` rather than ``dz^2/dt``.
    """
    courant = abs(u) * dt / dz
    return 0.5 * abs(u) * dz * (1.0 - courant)


@dataclass
class InterfaceMetrics:
    """Measured spreading of an initially sharp interface in one column."""

    front_position: float
    thickness: float
    effective_diffusivity: float
