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
from inpipe.caseio import load_case
from inpipe.circulation import CirculationSolver, WellConfig
from inpipe.config import (
    GridConfig,
    NumericsConfig,
    bpm_to_m3s,
    inch_to_m,
    m3s_to_bpm,
    m_to_inch,
)
from inpipe.fluid import PumpSchedule, PumpStage

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
DEFAULT_LOG = ROOT / "data" / "K-GEP-1_composite.las"

#: 7 in, 29 lb/ft casing - the standard string for a 10-5/8 in hole, which is
#: what this log's 10.43 in gauge is.  Nominal annular clearance 1.71 in.
#: Fluid properties, casing geometry, rate and the modelled interval all come
#: from a case file, so they can be changed without editing code - the mud and
#: slurry properties are placeholders until the real ones are available.  See
#: :mod:`inpipe.caseio`; ``--case`` selects a different file.
DEFAULT_CASE = ROOT / "cases" / "kgep1.json"
CASE = load_case(DEFAULT_CASE)

CASING_OD = CASE.geometry["casing_od"]
CASING_ID = CASE.geometry["casing_id"]
FLOW_RATE = CASE.flow["flow_rate"]
MUD = CASE.displaced
CEMENT = CASE.displacing

N_LAYER = 9
N_AZIMUTH = 8

#: Model only the open hole below this depth; see the module docstring.
TOP_DEPTH = CASE.geometry["top_depth"]


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


def build(caliper, n_axial=250, excess=None, casing_od=None,
          casing_id=None, flow_rate=None, top_depth=None,
          cement_volume=None, spec=None, rat_hole_length=None):
    """Build the solver for one job.

    Every property defaults to the case file (``spec``, or ``cases/kgep1.json``);
    an explicit argument overrides it.
    """
    spec = CASE if spec is None else spec
    casing_od = spec.geometry["casing_od"] if casing_od is None else casing_od
    casing_id = spec.geometry["casing_id"] if casing_id is None else casing_id
    flow_rate = spec.flow["flow_rate"] if flow_rate is None else flow_rate
    top_depth = spec.geometry["top_depth"] if top_depth is None else top_depth
    excess = spec.flow.get("excess", 1.05) if excess is None else excess
    rat_hole = (spec.geometry.get("rat_hole_length", 0.0)
                if rat_hole_length is None else rat_hole_length)
    mud, cement = spec.displaced, spec.displacing

    # The casing does not land on bottom: the rat hole is open hole below the
    # shoe, and it is the space the cement turns around in on its way up.
    total_depth = float(caliper.depth[-1])
    shoe = total_depth - rat_hole
    length = shoe - top_depth
    if length <= 0.0:
        raise ValueError(
            f"top_depth {top_depth} m is at or below the shoe at {shoe:.2f} m "
            f"(total depth {total_depth:.2f} m less a {rat_hole:.2f} m rat hole)"
        )
    well = WellConfig(
        length, casing_id, casing_od, caliper,
        top_depth=top_depth,
        rat_hole_length=rat_hole,
        rho_above_casing="auto",     # turns over as cement is pumped through
        rho_above_annulus=mud.rho,   # returns above the interval stay mud
    )
    grid = GridConfig(n_axial=n_axial, n_layer=N_LAYER, n_azimuth=N_AZIMUTH)

    v_casing = math.pi * (0.5 * casing_id) ** 2 * length
    v_annulus = AnnulusGrid(length, casing_od, caliper, n_axial,
                            N_LAYER, N_AZIMUTH, z_offset=top_depth).total_volume
    # ``cement_volume`` overrides the excess rule, so a job can be sized off a
    # different hole than the one it is pumped into - which is exactly the
    # mistake of designing on bit size and ignoring the caliper.
    # The rat hole has to be filled too, so it belongs in the job volume.
    pumped = ((v_casing + v_annulus + well.rat_hole_volume) * excess
              if cement_volume is None else cement_volume)
    schedule = PumpSchedule([PumpStage(cement, pumped, flow_rate)])
    solver = CirculationSolver(
        well, schedule, initial_fluid=mud, grid=grid,
        numerics=NumericsConfig(
            diagnostics_every=40,
            regularisation_shear_rate=spec.regularisation_shear_rate,
            normalise_consistency=spec.normalise_consistency,
        ),
    )
    return solver, schedule, length, v_casing, v_annulus, shoe


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caliper", type=Path, default=None, help="caliper log (CSV or LAS)")
    p.add_argument("--synthetic", action="store_true", help="use a synthetic log")
    p.add_argument("--keep-tail", action="store_true", help="do not cut the collapsed tail")
    p.add_argument("--n-axial", type=int, default=250)
    p.add_argument("--snapshots", type=int, default=110)
    p.add_argument("--case", type=Path, default=DEFAULT_CASE,
                   help="JSON case file with fluid properties, casing geometry "
                        "and rate (default cases/kgep1.json)")
    # These default to the case file rather than to a number, so editing the
    # case file is enough and a flag still wins when one is given.
    p.add_argument("--casing-od-in", type=float, default=None)
    p.add_argument("--casing-id-in", type=float, default=None)
    p.add_argument("--rate-bpm", type=float, default=None)
    p.add_argument("--top-depth", type=float, default=None,
                   help="model only the open hole below this depth [m]")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    OUT.mkdir(exist_ok=True)

    spec = CASE if args.case == DEFAULT_CASE else load_case(args.case)
    casing_od = (spec.geometry["casing_od"] if args.casing_od_in is None
                 else inch_to_m(args.casing_od_in))
    casing_id = (spec.geometry["casing_id"] if args.casing_id_in is None
                 else inch_to_m(args.casing_id_in))
    flow_rate = (spec.flow["flow_rate"] if args.rate_bpm is None
                 else bpm_to_m3s(args.rate_bpm))

    print(spec.summary())
    print()
    caliper, _ = load_caliper(args.caliper, args.synthetic, args.keep_tail)
    print(caliper.summary(casing_od=casing_od))

    top_depth = 0.0 if args.synthetic else (
        spec.geometry["top_depth"] if args.top_depth is None else args.top_depth)
    solver, schedule, length, v_casing, v_annulus, shoe = build(
        caliper, n_axial=args.n_axial, spec=spec,
        casing_od=casing_od, casing_id=casing_id, flow_rate=flow_rate,
        top_depth=top_depth,
    )
    ag = solver.annulus_grid
    gap = ag.r_outer - ag.r_inner

    smooth = math.pi * ((0.5 * caliper.gauge) ** 2
                        - (0.5 * casing_od) ** 2) * length
    well = solver.well
    print(f"\nmodelled interval: {top_depth:.2f} - {shoe:.2f} m "
          f"({length:.2f} m of open hole)")
    if well.rat_hole_length > 0.0:
        print(f"rat hole         : {shoe:.2f} - {well.total_depth:.2f} m "
              f"({well.rat_hole_length:.2f} m, {well.rat_hole_volume:.3f} m^3) "
              f"- open hole below the shoe, where the cement turns around")
    print(f"gauge hole       : {m_to_inch(caliper.gauge):.2f} in "
          f"({caliper.gauge * 1e3:.1f} mm)")
    print(f"casing           : {m_to_inch(casing_od):.3f} in OD / "
          f"{m_to_inch(casing_id):.3f} in ID")
    print(f"annular gap      : {gap.min() * 1e3:.1f} - {gap.max() * 1e3:.1f} mm "
          f"(gauge {(caliper.gauge - casing_od) / 2 * 1e3:.1f} mm)")
    print(f"casing volume    : {v_casing:.3f} m^3")
    print(f"annulus volume   : {v_annulus:.3f} m^3 "
          f"({100 * (v_annulus / smooth - 1):+.1f} % vs an in-gauge hole)")
    print(f"cement pumped    : {schedule.total_volume:.3f} m^3")
    print(f"job duration     : {schedule.total_time / 60:.1f} min at "
          f"{m3s_to_bpm(flow_rate):.1f} bpm")

    result = solver.run(t_end=schedule.total_time, n_snapshots=args.snapshots,
                        progress=False,
                        gauge_diameter=spec.geometry.get("bit_diameter"))
    i_cem = result.fluids.index(CEMENT)
    h = result.history

    print(f"\nsteps            : {result.n_steps} in {result.wall_time:.1f} s "
          f"({1e3 * result.wall_time / result.n_steps:.1f} ms/step)")
    print(f"sum-to-one error : {h['sum_to_one_error'].max():.2e}")
    print(f"volume error     : {h['mass_error'].max():.2e}")
    print(f"annular displacement efficiency: "
          f"{result.annular_displacement_efficiency(i_cem):.4f}")

    # --- flow regime: is a laminar profile defensible on this job? ----------
    from inpipe.regime import LAMINAR_LIMIT, TURBULENT_LIMIT, regime_from_result

    print("\n--- flow regime (the solver integrates a LAMINAR profile) ---")
    re_c = np.nanmax(h["reynolds_casing"])
    re_a = np.nanmax(h["reynolds_annulus"])
    print(f"peak Reynolds over the job: casing {re_c:.0f}, annulus {re_a:.0f} "
          f"(laminar below {LAMINAR_LIMIT:.0f}, turbulent above {TURBULENT_LIMIT:.0f})")
    if max(re_c, re_a) >= LAMINAR_LIMIT:
        print("  *** the flow leaves the laminar range during this job - the "
              "velocity profile this model solves does not apply there")
    else:
        print("  laminar throughout, so the solved profile holds")
    for reg in regime_from_result(result, flow_rate, solver.numerics):
        print(reg.summary())

    print("\n--- displacement quality (Xue et al. 2022) ---")
    ld = np.asarray(h["interface_length"], dtype=float)
    sw = np.asarray(h["swept_efficiency"], dtype=float)
    good = np.isfinite(ld)
    if good.any():
        print(f"interface length (20-80 % contours): peaks at {np.nanmax(ld):.1f} m, "
              f"ends at {ld[good][-1]:.1f} m")
        print(f"swept efficiency  : {np.nanmin(sw):.3f} -> {sw[np.isfinite(sw)][-1]:.3f}"
              "   (the global figure below is a straight ramp until breakthrough "
              "and says little while the job runs)")

    if result.rathole_volume > 0.0:
        left = result.rathole_fractions[0] * result.rathole_volume
        print(f"rat hole at end  : {100 * result.rathole_fractions[i_cem]:.2f} % cement, "
              f"{left * 1e3:.2f} L of mud left below the shoe")

    print("\n--- rising time ---")
    print(result.arrival.summary())
    arrival_csv = OUT / "field_arrival_time.csv"
    result.arrival.to_csv(arrival_csv)
    print(f"    wrote {arrival_csv}")

    # The efficiency history is a validation curve in its own right - ANSYS
    # produces the same three against time - so it is exported, not only drawn.
    hist_csv = OUT / "field_displacement_history.csv"
    np.savetxt(
        hist_csv,
        np.column_stack([
            h["time"], h["annular_efficiency"], h["swept_efficiency"],
            h["interface_length"], h["interface_front"], h["interface_back"],
            h["reynolds_casing"], h["reynolds_annulus"],
        ]),
        delimiter=",", comments="", fmt="%.6g",
        header="time_s,efficiency_global,efficiency_swept,interface_length_m,"
               "front_20pct_depth_m,back_80pct_depth_m,"
               "reynolds_casing_max,reynolds_annulus_max",
    )
    print(f"    wrote {hist_csv}")

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
                          annulus_f=snap["annulus"], rathole_f=snap["rathole"],
                          title=f"t = {snap['time'] / 60:.1f} min",
                          colorbar=(ax is axes[-1]))
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.suptitle(f"K-GEP-1: cement displacing mud, open hole "
                 f"{top_depth:.0f}-{shoe:.0f} m, {m_to_inch(casing_od):.0f} in casing "
                 f"in a {m_to_inch(caliper.gauge):.1f} in hole")
    fig.tight_layout()
    fig.savefig(OUT / "field_circulation_sections.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT / 'field_circulation_sections.png'}")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    axes[0].plot(m_to_inch(hole), z, lw=0.6)
    axes[0].axvline(m_to_inch(caliper.gauge), color="0.5", ls="--", lw=0.9, label="gauge")
    axes[0].axvline(m_to_inch(casing_od), color="C3", ls=":", lw=1.0, label="casing OD")
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

    # Global efficiency is a straight ramp until breakthrough, so the swept
    # efficiency and the interface length are plotted with it: those two say
    # something while the job is still running (Xue et al. 2022).
    axes[2].plot(h["time"] / 60, h["annular_efficiency"], label="global (all annulus)")
    axes[2].plot(h["time"] / 60, h["swept_efficiency"], lw=1.6,
                 label="swept (behind the front)")
    axes[2].set_xlabel("time [min]")
    axes[2].set_ylabel("displacement efficiency")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].legend(fontsize=7, loc="lower right")
    axes[2].grid(alpha=0.3)
    twin = axes[2].twinx()
    twin.plot(h["time"] / 60, h["interface_length"], color="C3", ls=":", lw=1.3)
    twin.set_ylabel("interface length, 20-80 % [m]", color="C3", fontsize=8)
    twin.tick_params(axis="y", labelcolor="C3", labelsize=7)

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

    # --- arrival time against depth: the fiber optic comparison ------------
    # Laid out like Fig. 2 of Hart et al. (2025), Sci Rep 15:11365 - interface
    # depth against time on the left, rise velocity against depth on the right -
    # so a DAS waterfall can be overlaid on the left panel directly.
    rep = result.arrival
    fig, (ax, axv) = plt.subplots(1, 2, figsize=(11.0, 7.0), sharey=True)
    ax.plot(rep.volumetric / 60.0, rep.depth, color="0.55", ls="--", lw=1.4,
            label="volumetric (pump rate + caliper)")
    ax.plot(rep.at(0.5) / 60.0, rep.depth, color="C0", lw=1.8,
            label="front (cement fraction 0.5)")
    # Only where both contours exist.  Above the depth the 0.9 contour never
    # reaches, an unmasked fill draws a flat band to the edge of the axes and
    # reads as "the mixing zone lasts to the end of the job", when what it
    # actually means is that those depths finish holding more than 10 % mud.
    band = np.isfinite(rep.at(0.1)) & np.isfinite(rep.at(0.9))
    ax.fill_betweenx(rep.depth, rep.at(0.1) / 60.0, rep.at(0.9) / 60.0,
                     where=band, color="C0", alpha=0.18, lw=0,
                     label="mixing zone (0.1 - 0.9)")
    if not np.all(band):
        shallowest = rep.depth[band].min() if np.any(band) else rep.depth.max()
        ax.axhline(shallowest, color="C3", ls=":", lw=1.0)
        ax.text(ax.get_xlim()[0], shallowest, "  above here the job ends with "
                "> 10 % mud", color="C3", fontsize=7.5, va="bottom")
    if rep.in_gauge is not None:
        ax.plot(rep.in_gauge / 60.0, rep.depth, color="C1", ls="-.", lw=1.2,
                label="in-gauge \"fast rise\" (no washouts filled)")
    ax.invert_yaxis()
    ax.set_xlabel("time since start of pumping [min]")
    ax.set_ylabel("measured depth [m]")
    ax.set_title("Cement arrival against depth\n"
                 "imposed pump rate - an upper bound on time", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    axv.plot(rep.rise_velocity(rep.volumetric), rep.depth, color="0.55", ls="--",
             lw=1.4, label="volumetric")
    axv.plot(rep.rise_velocity(rep.front_envelope), rep.depth, color="C0", lw=1.6,
             label="front (leading edge)")
    if rep.in_gauge is not None:
        axv.plot(rep.rise_velocity(rep.in_gauge), rep.depth, color="C1", ls="-.",
                 lw=1.2, label="in-gauge")
    # Log scale: this hole runs 8.15 - 24.0 in against 7 in casing, so the
    # annular area spans a factor of 30 and the velocity two decades.  On a
    # linear axis the tight sections flatten everything else against the wall.
    axv.set_xscale("log")
    axv.set_xlabel("rise velocity [m/min]")
    axv.set_title("Rise velocity\ndips are washouts, spikes are tight hole", fontsize=10)
    axv.legend(fontsize=8, loc="lower right")
    axv.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "field_arrival_time.png", dpi=140)
    print(f"wrote {OUT / 'field_arrival_time.png'}")

    path = animate_circulation(result, OUT / "field_circulation.mp4",
                               fluid_index=i_cem, fps=14)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
