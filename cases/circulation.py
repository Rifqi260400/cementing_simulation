"""Full cementing circulation - cement down the casing, round the shoe, up the annulus.

Defaults to the real caliper log in ``data/`` (well K-GEP-1, a 390 m composite
log with the borehole diameter curve in inches).  The well depth, the gauge hole
size and the annular geometry all come from that log.

Run::

    python -m cases.circulation                       # 175 m to the shoe
    python -m cases.circulation --top-depth 0         # the whole logged interval
    python -m cases.circulation --caliper my.las      # your own log
    python -m cases.circulation --synthetic           # no log needed

Modelled interval
-----------------
Only the open hole below ``--top-depth`` (175 m by default) is modelled.  Above
it the bundled log reads a near-constant 10.43 in - a standard deviation of
0.007 in over 17 000 samples, against 3.05 in in the section below - which is
not rock, so that interval is very likely cased and its annulus is not an open
hole (assumption A-33).

The column above the interval is still accounted for hydrostatically, so shoe
pressure and ECD stay true-depth quantities: the annulus above is taken to
remain mud, and the casing above is volume-averaged over what has been pumped
through it.  Friction above the interval is *not* included, so pump pressure is
a lower bound by that amount.

Caliper handling
----------------
The log's own ``NULL`` value and its ``~C`` curve units are honoured, and the
file may be wrapped (``WRAP. YES``), which is how the bundled log is written.

The bundled log's caliper collapses over the last 3.2 m - from 13.3 in at
386.75 m to 2.1 in below it - which is the tool bottoming out, not geometry.
That block is cut, and the cut is reported rather than applied silently;
``--keep-tail`` leaves it in.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from inpipe.annulus_grid import AnnulusGrid
from inpipe.caliper import implausible_tail, read_caliper, synthetic_caliper
from inpipe.circulation import CirculationSolver, WellConfig
from inpipe.config import GridConfig, NumericsConfig, bpm_to_m3s, inch_to_m, m_to_inch
from inpipe.fluid import Fluid, PumpSchedule, PumpStage

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
DEFAULT_LOG = ROOT / "data" / "K-GEP-1_composite.las"

#: 7 in, 29 lb/ft casing - the standard string for a 10-5/8 in hole, which is
#: what this log's 10.43 in gauge is.  Nominal annular clearance 1.71 in.
CASING_OD = inch_to_m(7.0)
CASING_ID = inch_to_m(6.184)

FLOW_RATE = bpm_to_m3s(5.0)
MUD = Fluid("mud", rho=1198.0, tau0=2.0, k=0.30, n=0.72)
CEMENT = Fluid("cement", rho=1870.0, tau0=6.0, k=0.55, n=0.65)

N_LAYER = 9
N_AZIMUTH = 8

#: Model only the open hole below this depth; see the module docstring.
TOP_DEPTH = 175.0


def load_caliper(path=None, synthetic=False, keep_tail=False, verbose=True):
    """Read (or synthesise) the caliper and trim any collapsed tail."""
    if synthetic:
        return synthetic_caliper(200.0, inch_to_m(8.5)), None

    log = read_caliper(path or DEFAULT_LOG)
    tail = implausible_tail(log)
    if tail is None or keep_tail:
        return log, tail

    start, end, n = tail
    trimmed = read_caliper(path or DEFAULT_LOG, depth_max=start - 1e-9)
    if verbose:
        print(f"caliper tail cut: {start:.2f} - {end:.2f} m "
              f"({n} samples, down to {m_to_inch(log.diameter.min()):.2f} in) "
              f"- the tool bottoming out, not geometry")
    return trimmed, tail


def build(caliper, n_axial=250, excess=1.05, casing_od=CASING_OD,
          casing_id=CASING_ID, flow_rate=FLOW_RATE, top_depth=TOP_DEPTH,
          cement_volume=None):
    shoe = float(caliper.depth[-1])
    length = shoe - top_depth
    if length <= 0.0:
        raise ValueError(
            f"top_depth {top_depth} m is at or below the shoe at {shoe:.2f} m"
        )
    well = WellConfig(
        length, casing_id, casing_od, caliper,
        top_depth=top_depth,
        rho_above_casing="auto",     # turns over as cement is pumped through
        rho_above_annulus=MUD.rho,   # returns above the interval stay mud
    )
    grid = GridConfig(n_axial=n_axial, n_layer=N_LAYER, n_azimuth=N_AZIMUTH)

    v_casing = math.pi * (0.5 * casing_id) ** 2 * length
    v_annulus = AnnulusGrid(length, casing_od, caliper, n_axial,
                            N_LAYER, N_AZIMUTH, z_offset=top_depth).total_volume
    # ``cement_volume`` overrides the excess rule, so a job can be sized off a
    # different hole than the one it is pumped into - which is exactly the
    # mistake of designing on bit size and ignoring the caliper.
    pumped = (v_casing + v_annulus) * excess if cement_volume is None else cement_volume
    schedule = PumpSchedule([PumpStage(CEMENT, pumped, flow_rate)])
    solver = CirculationSolver(
        well, schedule, initial_fluid=MUD, grid=grid,
        numerics=NumericsConfig(diagnostics_every=40),
    )
    return solver, schedule, length, v_casing, v_annulus, shoe


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caliper", type=Path, default=None, help="caliper log (CSV or LAS)")
    p.add_argument("--synthetic", action="store_true", help="use a synthetic log")
    p.add_argument("--keep-tail", action="store_true", help="do not cut the collapsed tail")
    p.add_argument("--n-axial", type=int, default=250)
    p.add_argument("--snapshots", type=int, default=110)
    p.add_argument("--casing-od-in", type=float, default=7.0)
    p.add_argument("--casing-id-in", type=float, default=6.184)
    p.add_argument("--rate-bpm", type=float, default=5.0)
    p.add_argument("--top-depth", type=float, default=TOP_DEPTH,
                   help="model only the open hole below this depth [m]")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    OUT.mkdir(exist_ok=True)

    caliper, _ = load_caliper(args.caliper, args.synthetic, args.keep_tail)
    print(caliper.summary(casing_od=inch_to_m(args.casing_od_in)))

    top_depth = 0.0 if args.synthetic else args.top_depth
    solver, schedule, length, v_casing, v_annulus, shoe = build(
        caliper, n_axial=args.n_axial,
        casing_od=inch_to_m(args.casing_od_in),
        casing_id=inch_to_m(args.casing_id_in),
        flow_rate=bpm_to_m3s(args.rate_bpm),
        top_depth=top_depth,
    )
    ag = solver.annulus_grid
    gap = ag.r_outer - ag.r_inner

    smooth = math.pi * ((0.5 * caliper.gauge) ** 2
                        - (0.5 * inch_to_m(args.casing_od_in)) ** 2) * length
    print(f"\nmodelled interval: {top_depth:.2f} - {shoe:.2f} m "
          f"({length:.2f} m of open hole)")
    print(f"gauge hole       : {m_to_inch(caliper.gauge):.2f} in "
          f"({caliper.gauge * 1e3:.1f} mm)")
    print(f"casing           : {args.casing_od_in:.3f} in OD / "
          f"{args.casing_id_in:.3f} in ID")
    print(f"annular gap      : {gap.min() * 1e3:.1f} - {gap.max() * 1e3:.1f} mm "
          f"(gauge {(caliper.gauge - inch_to_m(args.casing_od_in)) / 2 * 1e3:.1f} mm)")
    print(f"casing volume    : {v_casing:.3f} m^3")
    print(f"annulus volume   : {v_annulus:.3f} m^3 "
          f"({100 * (v_annulus / smooth - 1):+.1f} % vs an in-gauge hole)")
    print(f"cement pumped    : {schedule.total_volume:.3f} m^3")
    print(f"job duration     : {schedule.total_time / 60:.1f} min at "
          f"{args.rate_bpm:.1f} bpm")

    result = solver.run(t_end=schedule.total_time, n_snapshots=args.snapshots,
                        progress=False)
    i_cem = result.fluids.index(CEMENT)
    h = result.history

    print(f"\nsteps            : {result.n_steps} in {result.wall_time:.1f} s "
          f"({1e3 * result.wall_time / result.n_steps:.1f} ms/step)")
    print(f"sum-to-one error : {h['sum_to_one_error'].max():.2e}")
    print(f"volume error     : {h['mass_error'].max():.2e}")
    print(f"annular displacement efficiency: "
          f"{result.annular_displacement_efficiency(i_cem):.4f}")

    print("\n--- hydraulics at end of job ---")
    print(result.hydraulics.summary())
    ff = h["pump_pressure"] < 0.0
    print(f"\nU-tube imbalance peaks at {h['utube_imbalance'].max() / 1e5:.2f} bar "
          f"({h['utube_imbalance'].max() / 6894.757:.0f} psi)")
    print(f"free-fall for {100 * ff.mean():.0f} % of the job")
    print(f"ECD at shoe {h['ecd_at_shoe'].min():.0f} - {h['ecd_at_shoe'].max():.0f} kg/m^3")

    yd = result.yield_diagnostic(MUD)
    print(f"annular wall shear {yd['tau_w'].min():.2f} - {yd['tau_w'].max():.2f} Pa "
          f"vs mud yield stress {MUD.tau0:.2f} Pa; "
          f"{100 * yd['volume_fraction_below']:.1f} % of the annulus below it")

    # --- where does cement end up, against the hole size? ------------------
    order = np.argsort(ag.z_centers)
    z = ag.z_centers[order]
    vol = ag.cell_volume.sum(axis=(1, 2))[order]
    cem = (result.annulus_fractions[i_cem] * ag.cell_volume).sum(axis=(1, 2))[order]
    local_eff = cem / vol
    hole = ag.hole_diameter[order]
    wide = hole > 1.3 * caliper.gauge
    if wide.any():
        print(f"\nwashed-out sections (> 1.3 x gauge): {100 * wide.mean():.0f} % of "
              f"the interval")
        print(f"  raw comparison   : {local_eff[wide].mean():.4f} in washouts "
              f"vs {local_eff[~wide].mean():.4f} elsewhere")
        # The raw comparison confounds geometry with depth: annular flow is
        # upward, so shallow cells are simply reached last and read low at the
        # end of the job whatever their diameter.  Compare within depth bands.
        edges = np.linspace(z.min(), z.max(), 6)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            band = (z >= lo) & (z < hi)
            bw, bg = band & wide, band & ~wide
            if not (bw.any() and bg.any()):
                continue
            w_eff = float(np.average(local_eff[bw], weights=vol[bw]))
            g_eff = float(np.average(local_eff[bg], weights=vol[bg]))
            rows.append((lo, hi, w_eff, g_eff, float(vol[band].sum())))
        if rows:
            print("  within depth bands (a paired comparison - the only one that")
            print("  separates geometry from arrival order):")
            for lo, hi, w_eff, g_eff, _ in rows:
                mark = "washout better" if w_eff > g_eff else "gauge better"
                print(f"    {lo:6.0f}-{hi:6.0f} m   washout {w_eff:.4f}   "
                      f"gauge {g_eff:.4f}   ({w_eff - g_eff:+.4f}, {mark})")
            # Average the *differences*, not the levels: averaging levels
            # re-imports the depth trend, because washout volume and gauge
            # volume are not spread the same way over the bands.
            diffs = np.array([w - g for _, _, w, g, _ in rows])
            weights = np.array([v for *_, v in rows])
            print(f"  mean difference  : {np.average(diffs, weights=weights):+.4f} "
                  f"(volume-weighted over bands; "
                  f"{(diffs > 0).sum()}/{len(diffs)} bands favour the washout)")

    # --- figures -----------------------------------------------------------
    from inpipe.wellview import animate_circulation, plot_well_section

    fig, axes = plt.subplots(1, 4, figsize=(15, 9.5), sharey=True)
    picks = [int(f * (len(result.snapshots) - 1)) for f in (0.2, 0.45, 0.7, 1.0)]
    for ax, i in zip(axes, picks):
        snap = result.snapshots[i]
        plot_well_section(result, i_cem, ax=ax, casing_f=snap["casing"],
                          annulus_f=snap["annulus"],
                          title=f"t = {snap['time'] / 60:.1f} min",
                          colorbar=(ax is axes[-1]))
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.suptitle(f"K-GEP-1: cement displacing mud, open hole "
                 f"{top_depth:.0f}-{shoe:.0f} m, {args.casing_od_in:.0f} in casing "
                 f"in a {m_to_inch(caliper.gauge):.1f} in hole")
    fig.tight_layout()
    fig.savefig(OUT / "field_circulation_sections.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT / 'field_circulation_sections.png'}")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    axes[0].plot(m_to_inch(hole), z, lw=0.6)
    axes[0].axvline(m_to_inch(caliper.gauge), color="0.5", ls="--", lw=0.9, label="gauge")
    axes[0].axvline(args.casing_od_in, color="C3", ls=":", lw=1.0, label="casing OD")
    axes[0].set_ylim(shoe, top_depth)
    axes[0].set_xlabel("hole diameter [in]")
    axes[0].set_ylabel("depth [m]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(local_eff, z, lw=0.8)
    axes[1].set_ylim(shoe, top_depth)
    axes[1].set_xlim(-0.03, 1.03)
    axes[1].set_xlabel("local cement fraction")
    axes[1].grid(alpha=0.3)

    axes[2].plot(h["time"] / 60, h["annular_efficiency"])
    axes[2].set_xlabel("time [min]")
    axes[2].set_ylabel("annular displacement efficiency")
    axes[2].grid(alpha=0.3)

    axes[3].plot(h["time"] / 60, h["pump_pressure"] / 1e5, label="pump pressure")
    axes[3].plot(h["time"] / 60, h["utube_imbalance"] / 1e5, label="U-tube imbalance")
    axes[3].axhline(0.0, color="k", lw=0.8)
    axes[3].set_xlabel("time [min]")
    axes[3].set_ylabel("pressure [bar]")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "field_circulation_history.png", dpi=140)
    print(f"wrote {OUT / 'field_circulation_history.png'}")

    path = animate_circulation(result, OUT / "field_circulation.mp4",
                               fluid_index=i_cem, fps=14)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
