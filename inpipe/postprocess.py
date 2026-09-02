"""Plotting and reporting.

The headline figure is :func:`plot_centre_plane`, which reproduces the
*view* of the paper's Fig. 5(a): fluid concentration in the vertical centre
plane, flow left to right.

Scope note (assumption A-19): the paper's Fig. 5 is an 83-degrees-from-vertical
case dominated by segregation and backflow.  Phase 1 is a vertical-well model
with segregation deliberately omitted, so the figure below reproduces Fig. 5's
presentation and the concentric parabolic-stretching mechanism, not its
buoyancy-driven asymmetry.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "centreline_velocity",
    "cross_section_average",
    "radial_concentration",
    "save_results",
    "plot_centre_plane",
    "plot_velocity_profile",
    "plot_outlet_history",
    "plot_diagnostics",
    "summary_table",
]

_CMAP = "RdYlBu_r"  # red = displacing fluid, blue = displaced, as in Fig. 5


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_centre_plane(result, fluid_index=0, ax=None, title=None, show_front=True):
    """Concentration in the vertical centre plane (paper Fig. 5 view).

    The horizontal axis is measured depth (flow direction, ``+z``); the
    vertical axis is the chord coordinate ``y`` across the pipe.
    """
    plt = _mpl()
    g = result.grid
    field = result.centre_plane(fluid_index)  # (n_layer, n_axial)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.6))

    im = ax.pcolormesh(
        g.z_faces,
        g.y_layer_faces,
        field,
        cmap=_CMAP,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    # Mask the pipe wall so the round cross-section reads correctly.
    zz = np.linspace(0.0, g.length, 400)
    ax.fill_between(zz, np.full_like(zz, g.radius), np.full_like(zz, g.radius * 1.15),
                    color="0.75", lw=0)
    ax.fill_between(zz, np.full_like(zz, -g.radius * 1.15), np.full_like(zz, -g.radius),
                    color="0.75", lw=0)

    if show_front:
        front = result.diagnostics.front_position[-1]
        j = g.n_azimuth // 2
        zf = front[:, j] if front.ndim == 2 else front
        yc = 0.5 * (g.y_layer_faces[:-1] + g.y_layer_faces[1:])
        ok = np.isfinite(zf)
        if ok.sum() > 2:
            ax.plot(zf[ok], yc[ok], "k--", lw=1.0, label="f = 0.5 contour")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    ax.set_xlim(0.0, g.length)
    ax.set_ylim(-g.radius * 1.15, g.radius * 1.15)
    ax.set_xlabel("measured depth $z$ [m]  (flow $\\rightarrow$)")
    ax.set_ylabel("$y$ [m]")
    ax.set_title(title or f"Centre-plane concentration of {result.fluids[fluid_index].name}"
                          f"  ($t$ = {result.time:.1f} s)")
    cb = ax.figure.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("volume fraction $f$")
    return ax


def plot_velocity_profile(profile, ax=None, n=400, label=None):
    """Axial velocity along the vertical centreline (paper Fig. 3c view)."""
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 4.0))
    r = np.linspace(0.0, profile.radius, n)
    y = np.concatenate([-r[::-1], r])
    u = np.concatenate([profile(r)[::-1], profile(r)])
    ax.plot(u, y, label=label or profile.fluid.name)
    ax.axhline(profile.radius, color="0.5", lw=0.8)
    ax.axhline(-profile.radius, color="0.5", lw=0.8)
    ax.set_xlabel("axial velocity $u$ [m/s]")
    ax.set_ylabel("$y$ [m]")
    ax.grid(alpha=0.3)
    return ax


def plot_outlet_history(result, ax=None):
    """Fluid fractions leaving the shoe - the quantity that feeds an annulus model."""
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 3.2))
    for i, fl in enumerate(result.fluids):
        ax.plot(result.outlet_time, result.outlet_fractions[:, i], label=fl.name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("area-weighted outlet fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_diagnostics(result, path=None):
    """Four-panel conservation and boundedness report."""
    plt = _mpl()
    d = result.diagnostics
    t = np.asarray(d.time)
    vols = np.asarray(d.fluid_volumes)
    influx = np.asarray(d.influx)
    outflux = np.asarray(d.outflux)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    ax = axes[0, 0]
    for i, fl in enumerate(result.fluids):
        ax.plot(t, vols[:, i], label=fl.name)
    ax.set_ylabel("volume in pipe [m$^3$]")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Fluid inventory")

    ax = axes[0, 1]
    budget = vols - (vols[0] + influx - outflux)
    scale = max(vols[0].sum(), 1e-30)
    ax.plot(t, np.max(np.abs(budget), axis=1) / scale)
    ax.set_yscale("log")
    ax.set_ylabel("relative mass-budget error")
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.3)
    ax.set_title("Conservation: $|V - (V_0 + \\int q_{in} - \\int q_{out})| / V_0$")

    ax = axes[1, 0]
    ax.plot(t, d.f_min, label="min $f$")
    ax.plot(t, d.f_max, label="max $f$")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Boundedness")

    ax = axes[1, 1]
    ax.semilogy(t, np.maximum(d.sum_to_one_error, 1e-18))
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\max |\sum_i f_i - 1|$")
    ax.grid(alpha=0.3)
    ax.set_title("Sum-to-one")

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def summary_table(result) -> str:
    """A plain-text run report."""
    d = result.diagnostics
    vols = np.asarray(d.fluid_volumes)
    influx = np.asarray(d.influx)
    outflux = np.asarray(d.outflux)
    budget = vols - (vols[0] + influx - outflux)
    rel = np.max(np.abs(budget)) / max(vols[0].sum(), 1e-30)
    lines = [
        f"grid            : {result.grid.shape[0]} x {result.grid.shape[1]} x {result.grid.shape[2]}"
        f"  ({np.prod(result.grid.shape)} cells)",
        f"dz              : {result.grid.dz:.4g} m",
        f"simulated time  : {result.time:.4g} s over {result.n_steps} steps",
        f"wall time       : {result.wall_time:.2f} s "
        f"({1e3 * result.wall_time / max(result.n_steps, 1):.2f} ms/step)",
        f"mass budget err : {rel:.3e} (relative)",
        f"sum-to-one err  : {max(d.sum_to_one_error):.3e} (absolute)",
        f"f range         : [{min(d.f_min):.3e}, {max(d.f_max):.6f}]",
    ]
    if d.dm_num_paper:
        lines += [
            f"Dm_num (dx^2/dt): {np.mean(d.dm_num_paper):.4e} m^2/s   [paper's estimate]",
            f"Dm_num (scheme) : {np.mean(d.dm_num_scheme):.4e} m^2/s   [upwind modified equation]",
        ]
    for i, fl in enumerate(result.fluids):
        lines.append(f"  volume[{fl.name}] : {vols[-1, i]:.6e} m^3")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export - the interface to an external CFD comparison
# ---------------------------------------------------------------------------


def centreline_velocity(result, station=0, include_wall=True):
    """Axial velocity along the vertical centreline at one depth station.

    Returns ``(y, u)`` with ``y`` in ``[-R, R]``.  This is the quantity the
    paper plots in Fig. 4's right column and the natural thing to lay a CFD
    profile over.

    Values are cell-centre velocities, so they stop short of the wall; with
    ``include_wall`` the exact no-slip endpoints ``u(+-R) = 0`` are appended so
    the profile spans the full diameter.
    """
    g = result.grid
    j = g.n_azimuth // 2
    y = g.cell_y[:, j]
    u = result.velocity[station, :, j]
    order = np.argsort(y)
    y, u = y[order], u[order]
    if include_wall:
        y = np.concatenate([[-g.radius], y, [g.radius]])
        u = np.concatenate([[0.0], u, [0.0]])
    return y, u


def radial_concentration(result, fluid_index, z):
    """Concentration versus chord coordinate ``y`` at measured depth ``z`` [m].

    Returns ``(y, f)`` on the centre plane - a radial cut for comparison with a
    CFD line probe.
    """
    g = result.grid
    k = int(np.clip(np.searchsorted(g.z_faces, z) - 1, 0, g.n_axial - 1))
    j = g.n_azimuth // 2
    y = g.cell_y[:, j]
    if g.n_azimuth % 2 == 0:
        f = result.fractions[fluid_index][k, :, j - 1 : j + 1].mean(axis=1)
    else:
        f = result.fractions[fluid_index][k, :, j]
    order = np.argsort(y)
    return y[order], f[order]


def cross_section_average(result, fluid_index):
    """Area-averaged concentration versus depth, shape ``(n_axial,)``.

    The 1D profile a CFD run is most cheaply compared against.
    """
    a = result.grid.cell_area
    return (result.fractions[fluid_index] * a).sum(axis=(1, 2)) / a.sum()


def save_results(result, directory, prefix="run"):
    """Write the run to disk for external post-processing.

    Produces
    --------
    ``<prefix>_fields.npz``
        Full state: volume fractions ``(n_fluids, n_axial, n_layer, n_azimuth)``,
        mixing status, cell velocities, cell geometry (``x``, ``y``, ``r``,
        ``area``, ``volume``), axial and layer face positions, fluid names and
        rheology, and the run metadata.
    ``<prefix>_centreplane_<fluid>.csv``
        Centre-plane concentration as a ``z`` x ``y`` table - the Fig. 5 view.
    ``<prefix>_axial_profile.csv``
        Area-averaged concentration of every fluid versus depth.
    ``<prefix>_centreline_velocity.csv``
        Axial velocity versus ``y`` on the vertical centreline.
    ``<prefix>_outlet_history.csv``
        Fluid fractions leaving the shoe versus time.

    Returns the list of paths written.
    """
    from pathlib import Path

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    g = result.grid
    written = []

    npz = out / f"{prefix}_fields.npz"
    np.savez_compressed(
        npz,
        fractions=result.fractions,
        mixing_status=result.mixing_status,
        velocity=result.velocity,
        z_centers=g.z_centers,
        z_faces=g.z_faces,
        y_layer_faces=g.y_layer_faces,
        x_layer_faces=g.x_layer_faces,
        cell_x=g.cell_x,
        cell_y=g.cell_y,
        cell_r=g.cell_r,
        cell_area=g.cell_area,
        cell_volume=g.cell_volume,
        fluid_names=np.array([f.name for f in result.fluids]),
        fluid_rheology=np.array([[f.rho, f.tau0, f.k, f.n] for f in result.fluids]),
        outlet_time=result.outlet_time,
        outlet_fractions=result.outlet_fractions,
        radius=g.radius,
        length=g.length,
        time=result.time,
        n_steps=result.n_steps,
    )
    written.append(npz)

    yc = 0.5 * (g.y_layer_faces[:-1] + g.y_layer_faces[1:])
    for i, fl in enumerate(result.fluids):
        path = out / f"{prefix}_centreplane_{fl.name}.csv"
        field = result.centre_plane(i)  # (n_layer, n_axial)
        header = "y_m\\z_m," + ",".join(f"{z:.6g}" for z in g.z_centers)
        rows = [
            f"{yc[l]:.6g}," + ",".join(f"{v:.6g}" for v in field[l])
            for l in range(g.n_layer)
        ]
        path.write_text(header + "\n" + "\n".join(rows) + "\n")
        written.append(path)

    path = out / f"{prefix}_axial_profile.csv"
    cols = [cross_section_average(result, i) for i in range(len(result.fluids))]
    header = "z_m," + ",".join(f.name for f in result.fluids)
    rows = [
        f"{g.z_centers[k]:.6g}," + ",".join(f"{c[k]:.6g}" for c in cols)
        for k in range(g.n_axial)
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    written.append(path)

    path = out / f"{prefix}_centreline_velocity.csv"
    y, u = centreline_velocity(result)
    path.write_text(
        "y_m,u_mps\n" + "\n".join(f"{a:.6g},{b:.6g}" for a, b in zip(y, u)) + "\n"
    )
    written.append(path)

    path = out / f"{prefix}_outlet_history.csv"
    header = "t_s," + ",".join(f.name for f in result.fluids)
    rows = [
        f"{t:.6g}," + ",".join(f"{v:.6g}" for v in row)
        for t, row in zip(result.outlet_time, result.outlet_fractions)
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    written.append(path)

    return written
