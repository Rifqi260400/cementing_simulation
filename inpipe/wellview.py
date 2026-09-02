"""Well-section rendering and animation of a circulating job.

Draws the well the way a cementing engineer reads it: depth down the page, a
true-to-scale radial section across it, with the casing bore in the middle, the
steel wall either side, and the annulus out to the caliper-measured hole wall.
Concentration colours both fluid-filled regions, so cement can be watched
falling down the casing, turning at the shoe, and climbing the annulus past
the washouts.
"""

from __future__ import annotations

import numpy as np

__all__ = ["well_section_image", "plot_well_section", "animate_circulation"]

#: Colours for the parts of the section that hold no fluid.
STEEL_COLOR = "#5a5f66"
FORMATION_COLOR = "#c9b79c"


def _configure_ffmpeg():
    """Point matplotlib at a bundled ffmpeg if the system has none."""
    import matplotlib

    try:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional dependency
        pass


def well_section_image(result, fluid_index, casing_f=None, annulus_f=None, n_x=420):
    """Rasterise one instant of the well onto an ``(n_depth, n_x)`` image.

    Returns ``(image, extent)``.  ``image`` holds the volume fraction of the
    chosen fluid where there is fluid and ``NaN`` in the steel and the
    formation, so those can be painted separately.  ``extent`` is
    ``(x_min, x_max, z_max, z_min)`` for ``imshow`` with depth increasing down.

    ``casing_f`` and ``annulus_f`` default to the final state; pass a snapshot's
    arrays to render an earlier instant.
    """
    cg, ag = result.casing_grid, result.annulus_grid
    cf = result.casing_fractions if casing_f is None else casing_f
    af = result.annulus_fractions if annulus_f is None else annulus_f

    r_ci = cg.radius
    r_co = 0.5 * ag.casing_od
    r_hole_max = float(ag.r_outer.max())
    x = np.linspace(-r_hole_max, r_hole_max, n_x)
    ax_abs = np.abs(x)

    # Depth axis: the casing grid's, with the annulus sampled onto it.
    z = cg.z_centers
    img = np.full((z.size, n_x), np.nan)

    # --- casing bore: the centre-plane cut, f against the chord coordinate ---
    j = cg.n_azimuth // 2
    if cg.n_azimuth % 2 == 0:
        casing_plane = cf[fluid_index][:, :, j - 1 : j + 1].mean(axis=2)
    else:
        casing_plane = cf[fluid_index][:, :, j]
    y_edges = cg.y_layer_faces
    inside = ax_abs <= r_ci
    layer_of_x = np.clip(np.searchsorted(y_edges, x, side="right") - 1,
                         0, cg.n_layer - 1)
    img[:, inside] = casing_plane[:, layer_of_x[inside]]

    # --- annulus: rings between the casing OD and the hole wall -------------
    # The annulus grid is in flow order (shoe first); put it on ascending depth
    # and interpolate onto the casing depth axis.
    order = np.argsort(ag.z_centers)
    z_a = ag.z_centers[order]
    ann_rings = af[fluid_index].mean(axis=2)[order]      # (n_axial, n_layer)
    ring_faces = ag.ring_faces[order]                    # (n_axial, n_layer+1)
    r_outer = ag.r_outer[order]

    ann_on_z = np.empty((z.size, ag.n_layer))
    for l in range(ag.n_layer):
        ann_on_z[:, l] = np.interp(z, z_a, ann_rings[:, l])
    faces_on_z = np.empty((z.size, ag.n_layer + 1))
    for l in range(ag.n_layer + 1):
        faces_on_z[:, l] = np.interp(z, z_a, ring_faces[:, l])
    r_outer_on_z = np.interp(z, z_a, r_outer)

    for k in range(z.size):
        band = (ax_abs > r_co) & (ax_abs <= r_outer_on_z[k])
        if not band.any():
            continue
        ring = np.clip(
            np.searchsorted(faces_on_z[k], ax_abs[band], side="right") - 1,
            0, ag.n_layer - 1,
        )
        img[k, band] = ann_on_z[k, ring]

    extent = (-r_hole_max, r_hole_max, z[-1], z[0])
    return img, extent


def plot_well_section(result, fluid_index=None, ax=None, casing_f=None,
                      annulus_f=None, title=None, cmap="RdYlBu_r", colorbar=True):
    """Draw one instant of the well section."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if fluid_index is None:
        fluid_index = len(result.fluids) - 1
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 8.0))

    img, extent = well_section_image(result, fluid_index, casing_f, annulus_f)
    ag = result.annulus_grid

    # Formation behind everything, then the steel, then the fluids on top.
    ax.set_facecolor(FORMATION_COLOR)
    z = result.casing_grid.z_centers
    order = np.argsort(ag.z_centers)
    r_out = np.interp(z, ag.z_centers[order], ag.r_outer[order])
    r_co = 0.5 * ag.casing_od
    ax.fill_betweenx(z, -r_co, r_co, color=STEEL_COLOR, zorder=1)

    im = ax.imshow(img, extent=extent, cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest", zorder=2)
    # Hole wall, so washouts read clearly.
    ax.plot(r_out, z, color="0.25", lw=0.8, zorder=3)
    ax.plot(-r_out, z, color="0.25", lw=0.8, zorder=3)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("radial position [m]")
    ax.set_ylabel("measured depth [m]")
    ax.set_title(title or f"{result.fluids[fluid_index].name} volume fraction")
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("volume fraction")
    return ax, im


def animate_circulation(result, path, fluid_index=None, fps=12, dpi=120,
                        cmap="RdYlBu_r", extra_panels=True):
    """Write an animation of the job to ``path`` (``.mp4`` or ``.gif``).

    Needs snapshots: run the solver with ``n_snapshots > 0``.  Returns the path
    written.

    Alongside the section, two panels track the job: the area-averaged cement
    fraction against depth in each leg, and annular displacement efficiency
    against time.
    """
    import matplotlib

    matplotlib.use("Agg")
    _configure_ffmpeg()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    if not result.snapshots:
        raise ValueError(
            "no snapshots recorded - run the solver with n_snapshots > 0"
        )
    if fluid_index is None:
        fluid_index = len(result.fluids) - 1
    name = result.fluids[fluid_index].name

    if extra_panels:
        fig = plt.figure(figsize=(11.5, 7.6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5], hspace=0.32, wspace=0.22)
        ax_sec = fig.add_subplot(gs[:, 0])
        ax_prof = fig.add_subplot(gs[0, 1])
        ax_eff = fig.add_subplot(gs[1, 1])
    else:
        fig, ax_sec = plt.subplots(figsize=(4.4, 8.0))
        ax_prof = ax_eff = None

    snap0 = result.snapshots[0]
    _, im = plot_well_section(result, fluid_index, ax=ax_sec,
                              casing_f=snap0["casing"], annulus_f=snap0["annulus"],
                              cmap=cmap)

    cg, ag = result.casing_grid, result.annulus_grid
    order = np.argsort(ag.z_centers)
    z_a = ag.z_centers[order]
    a_a = ag.cell_area[order]

    def profiles(snap):
        c = (snap["casing"][fluid_index] * cg.cell_area).sum(axis=(1, 2)) / cg.cell_area.sum()
        a = (snap["annulus"][fluid_index] * ag.cell_area).sum(axis=(1, 2))
        a = a[order] / a_a.sum(axis=(1, 2))
        return c, a

    if ax_prof is not None:
        c0, a0 = profiles(snap0)
        (line_c,) = ax_prof.plot(c0, cg.z_centers, label="casing")
        (line_a,) = ax_prof.plot(a0, z_a, label="annulus")
        ax_prof.set_ylim(cg.z_faces[-1], cg.z_faces[0])
        ax_prof.set_xlim(-0.03, 1.03)
        ax_prof.set_xlabel(f"{name} fraction (area-averaged)")
        ax_prof.set_ylabel("depth [m]")
        ax_prof.grid(alpha=0.3)
        ax_prof.legend(fontsize=8, loc="lower right")

        t_hist = result.history["time"] / 60.0
        eff = result.history["annular_efficiency"]
        ax_eff.plot(t_hist, eff, color="0.6", lw=1.0)
        (marker,) = ax_eff.plot([], [], "o", color="C3", ms=6)
        ax_eff.set_xlabel("time [min]")
        ax_eff.set_ylabel("annular displacement\nefficiency")
        ax_eff.set_ylim(-0.03, 1.03)
        ax_eff.grid(alpha=0.3)

    def update(i):
        snap = result.snapshots[i]
        img, _ = well_section_image(result, fluid_index,
                                    snap["casing"], snap["annulus"])
        im.set_data(img)
        ax_sec.set_title(f"{name}   t = {snap['time'] / 60:.1f} min")
        if ax_prof is not None:
            c, a = profiles(snap)
            line_c.set_xdata(c)
            line_a.set_xdata(a)
            j = int(np.searchsorted(result.history["time"], snap["time"]))
            j = min(j, len(result.history["time"]) - 1)
            marker.set_data([result.history["time"][j] / 60.0],
                            [result.history["annular_efficiency"][j]])
        return ()

    anim = FuncAnimation(fig, update, frames=len(result.snapshots), blit=False)
    path = str(path)
    if path.lower().endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(fig)
    return path
