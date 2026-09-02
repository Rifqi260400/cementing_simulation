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

Transverse closure (assumption A-07)
------------------------------------
The conservative update preserves ``sum_i f_i = 1`` only when the discrete
continuity condition ``sum_j u_j A_j = 0`` holds per *cell*.  In this model it
does not: when neighbouring depth stations hold different effective fluids,
their velocity profiles differ, so a column has ``du/dz != 0`` even though the
total ``Q`` is constant.  Three closures are available, and the field-scale
numbers below are why the third is the default:

``"none"``
    Eq. A.9 exactly as printed.  Conserves each fluid's volume to round-off but
    loses sum-to-one badly: on the 200 m case ``max |sum_i f_i - 1| = 0.39``,
    with ``f`` reaching 1.10.  The fractions stop being a partition of the cell.
``"local"``
    Subtract ``f_i * div(u)``, i.e. use the advective form of Eq. (2).
    Sum-to-one holds to 3e-14, but volume is created and destroyed locally:
    1.8 % per-fluid error on the same case.
``"redistribute"`` (default)
    The column imbalance ``D_c = div(u)_c`` sums to zero over the cross-section
    (the mapped velocities integrate to the same ``Q`` at every station), so it
    is a *redistribution*, not a source.  Columns losing volume axially shed it
    laterally carrying their own composition; columns gaining volume take in
    the donor mixing-cup composition.  Both invariants then hold to round-off.

    This is transverse redistribution *imposed algorithmically*, in the same
    spirit as the paper's own segregation step - not a solved transverse
    velocity.  Its physical content is an explicit assumption: lateral
    redistribution is instantaneous and well-mixed across the section.  The
    closure is identically inert when ``div(u) = 0``, so it changes nothing in
    every single-rheology case.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "CFLViolation",
    "BoundednessViolation",
    "upwind_faces",
    "cfl_timestep",
    "advect",
    "advect_multi",
    "face_velocity_stack",
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
    u_faces: np.ndarray | None = None,
    face_area: np.ndarray | None = None,
    cell_volume: np.ndarray | None = None,
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
    divergence_correction : subtract ``f * div(u)``, i.e. integrate the
        advective form.  This is what Eq. A.20 states for the mixing status
        ``s``; for volume fractions see the transverse closures in
        :func:`advect_multi` (assumption A-07).

    u_faces : optional precomputed ``(n_axial + 1, ...)`` face velocity stack
        from :func:`face_velocity_stack`.  Pass it when advecting several
        fields with the same velocity field, so the faces are built once.
    face_area : ``(n_axial + 1, ...)`` axial face areas [m^2].
    cell_volume : ``(n_axial, ...)`` cell volumes [m^3].

    Geometry
    --------
    For a **uniform** cross-section, ``face_area`` and ``cell_volume`` may be
    omitted: the same area divides out of the face flux and the cell volume, so
    the update reduces to ``dt/dz`` times a velocity difference.

    For a **varying** cross-section - an open hole with washouts - it does not
    cancel, and both must be supplied.  The update is then the full
    finite-volume form of Eq. A.8, ``f^{n+1} = f^n - (dt/dV) sum_j u_j A_j
    f_{s,j}``, which is what the paper writes and what the uniform case is a
    special case of.

    Returns the updated field; ``f`` is not modified.
    """
    scheme = resolve_face_scheme(face_scheme)
    if u_faces is None:
        u_cells = np.broadcast_to(np.asarray(u_cells, dtype=float), f.shape)
        u_faces = face_velocity_stack(u_cells)
    return _advect_with_faces(
        f, u_faces, dz, dt, scheme, inlet_value, divergence_correction,
        face_area, cell_volume,
    )


def _flux_weights(uf, dz, face_area, cell_volume):
    """Return ``(u*A at faces, cell volume)`` in a form the update can divide by.

    With no geometry given the areas cancel and the pair degenerates to
    ``(uf, dz)`` - the uniform-cross-section form, bit-identical to what it was
    before variable geometry existed.
    """
    if face_area is None:
        return uf, dz
    if cell_volume is None:
        raise ValueError("face_area was given without cell_volume")
    return uf * face_area, cell_volume


def _advect_with_faces(f, uf, dz, dt, scheme, inlet_value, divergence_correction,
                       face_area=None, cell_volume=None):
    """The update itself, given a precomputed face velocity stack ``uf``.

    Face values are written into one preallocated buffer rather than
    concatenated, so advecting several fields with a shared ``uf`` costs one
    allocation per field instead of three.
    """
    # Boundary faces (assumption A-18).  Inlet at z = 0: Dirichlet on inflow,
    # upwind on outflow.  Outlet at z = L: zero-gradient in both directions, so
    # a reversed outlet face draws the last cell's own composition rather than
    # injecting anything new.
    ff = np.empty_like(uf)
    ff[1:-1] = scheme(f, uf[1:-1], dz, dt)
    ff[0] = np.where(uf[0] >= 0.0, inlet_value, f[0])
    ff[-1] = f[-1]

    ua, volume = _flux_weights(uf, dz, face_area, cell_volume)
    flux = ua * ff
    div_flux = (flux[1:] - flux[:-1]) / volume

    if divergence_correction:
        div_flux -= f * ((ua[1:] - ua[:-1]) / volume)

    return f - dt * div_flux


def face_velocity_stack(u_cells: np.ndarray) -> np.ndarray:
    """Axial face velocities including both boundaries, ``(n_axial + 1, ...)``."""
    return np.concatenate(
        [u_cells[0][None, ...], _face_velocities(u_cells), u_cells[-1][None, ...]],
        axis=0,
    )


def advect_multi(
    fields: np.ndarray,
    u_cells: np.ndarray,
    dz: float,
    dt: float,
    face_scheme="upwind",
    inlet_values=None,
    closure: str = "redistribute",
    area: np.ndarray | None = None,
    u_faces: np.ndarray | None = None,
    face_area: np.ndarray | None = None,
    cell_volume: np.ndarray | None = None,
) -> np.ndarray:
    """Advect a stack of volume fractions, ``(n_fluids, n_axial, n_layer, n_azimuth)``.

    ``inlet_values`` is a sequence of length ``n_fluids``.  ``closure`` selects
    the transverse closure described in the module docstring; ``area`` is the
    cell cross-sectional area, required by ``"redistribute"``.  ``face_area``
    and ``cell_volume`` carry a varying cross-section; omit both for a uniform
    one (see :func:`advect`).
    """
    fields = np.asarray(fields, dtype=float)
    n_fluids = fields.shape[0]
    if inlet_values is None:
        inlet_values = np.zeros(n_fluids)
    if closure not in ("none", "local", "redistribute"):
        raise ValueError(
            f"unknown transverse closure {closure!r}; "
            "expected 'none', 'local' or 'redistribute'"
        )

    if closure == "redistribute" and area is None:
        raise ValueError("closure='redistribute' needs the cell area array")

    # Build the face velocities once and share them across every fluid and the
    # redistribution step, rather than rebuilding them per field.
    if u_faces is None:
        u_cells = np.broadcast_to(np.asarray(u_cells, dtype=float), fields.shape[1:])
        u_faces = face_velocity_stack(u_cells)

    scheme = resolve_face_scheme(face_scheme)
    out = np.empty_like(fields)
    for i in range(n_fluids):
        out[i] = _advect_with_faces(
            fields[i], u_faces, dz, dt, scheme, inlet_values[i],
            divergence_correction=(closure == "local"),
            face_area=face_area, cell_volume=cell_volume,
        )
    if closure != "redistribute":
        return out

    ua, volume = _flux_weights(u_faces, dz, face_area, cell_volume)
    div_u = (ua[1:] - ua[:-1]) / volume  # (n_axial, n_layer, n_azimuth)
    weight = area if cell_volume is None else cell_volume
    return _apply_transverse_redistribution(out, fields, div_u, weight, dt)


def _apply_transverse_redistribution(
    out: np.ndarray, fields: np.ndarray, div_u: np.ndarray, weight: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Move the divergence-induced imbalance sideways instead of destroying it.

    Per axial station, ``D_c = div(u)_c``.  Columns with ``D_c < 0`` gain volume
    axially and must shed it laterally, carrying their own composition; columns
    with ``D_c > 0`` must take volume in, and receive the donor mixing-cup
    composition.  Because ``sum_c D_c A_c = 0`` the exchange balances exactly,
    so every fluid's total volume and the partition ``sum_i f_i = 1`` are both
    preserved to round-off.

    That balance is what the solver's ``enforce_discrete_continuity`` option
    guarantees (assumption A-22).  Without it the mapped velocities integrate to
    ``Q`` only to ~4e-6, and this closure conserves only to the same order.
    """
    donor = np.clip(-div_u, 0.0, None)  # lateral outflow rate
    receiver = np.clip(div_u, 0.0, None)  # lateral inflow rate

    # Weighted by cell volume (or, for a uniform section, equivalently by
    # area, since the volumes are then all A*dz).
    w = donor * weight
    supply = w.sum(axis=(1, 2))  # per station
    active = supply > 0.0
    if not np.any(active):
        return out

    # Donor mixing-cup composition per station, (n_fluids, n_axial).
    cup = np.einsum("iklm,klm->ik", fields, w)
    cup[:, active] /= supply[active]

    out -= dt * donor[None, ...] * fields
    out += dt * receiver[None, ...] * cup[:, :, None, None]
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
