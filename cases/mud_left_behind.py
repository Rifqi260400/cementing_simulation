"""Case 4 - how much mud the washouts leave behind.

Runs the same cementing job twice on the K-GEP-1 open hole: once on the real
caliper, once on an in-gauge hole of the same length and the same casing.  The
difference is the mud left behind that is attributable to the enlargements
rather than to the displacement being imperfect anyway.

Run::

    python -m cases.mud_left_behind
    python -m cases.mud_left_behind --excess 1.5     # pump more
    python -m cases.mud_left_behind --caliper my.las

Read the header of ``inpipe/mudleft.py`` before quoting any number from this:
the model has no mechanism that strands fluid permanently, so "left behind"
always means "not yet swept at the volume actually pumped".  Flow separation in
the cavity, which would strand fluid however long you circulate, is a
two-dimensional effect this model cannot represent, so these figures are a lower
bound on what a washout costs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inpipe.caliper import CaliperLog
from inpipe.caseio import load_case
from inpipe.mudleft import mud_left_behind
from inpipe.config import m_to_inch
from cases.circulation import CASE, DEFAULT_CASE, build, load_caliper

OUT = Path(__file__).resolve().parent.parent / "results"


def in_gauge(caliper: CaliperLog) -> CaliperLog:
    """A smooth hole of the same length at the log's gauge diameter."""
    return CaliperLog(
        caliper.depth.copy(),
        np.full_like(caliper.diameter, caliper.gauge),
        name=f"{caliper.name} (in-gauge)",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caliper", type=Path, default=None)
    p.add_argument("--excess", type=float, default=1.05,
                   help="cement pumped, as a multiple of the well volume")
    p.add_argument("--n-axial", type=int, default=200)
    p.add_argument("--snapshots", type=int, default=90)
    p.add_argument("--washout-threshold", type=float, default=1.3)
    p.add_argument("--case", type=Path, default=DEFAULT_CASE,
                   help="JSON case file with fluid properties (default cases/kgep1.json)")
    return p.parse_args(argv)


def run_one(caliper, args, label, cement_volume=None):
    spec = CASE if getattr(args, "case", DEFAULT_CASE) == DEFAULT_CASE else load_case(args.case)
    solver, schedule, length, v_casing, v_annulus, shoe = build(
        caliper, n_axial=args.n_axial, excess=args.excess,
        cement_volume=cement_volume, spec=spec,
    )
    result = solver.run(t_end=schedule.total_time, n_snapshots=args.snapshots)
    report = mud_left_behind(result, spec.displaced, gauge=caliper.gauge,
                             washout_threshold=args.washout_threshold)
    print(f"\n--- {label} ---")
    print(f"annulus {v_annulus:.4f} m^3, cement pumped {schedule.total_volume:.3f} m^3 "
          f"({args.excess:.2f} x well volume)")
    print(report.summary())
    return result, report, schedule


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    OUT.mkdir(exist_ok=True)
    caliper, _ = load_caliper(args.caliper)
    print(caliper.summary())

    # A: the real hole, with the job sized off the real caliper.
    real, rep, schedule = run_one(caliper, args, "A - real hole, job sized on caliper")
    # B: an in-gauge hole, same excess - the reference for "if the hole were smooth".
    gauge_log = in_gauge(caliper)
    smooth, rep_g, sched_g = run_one(gauge_log, args, "B - in-gauge hole, same excess")
    # C: the real hole, but with the job sized off the *gauge* diameter, which
    #    is what happens when a design ignores the caliper.
    under, rep_u, sched_u = run_one(
        caliper, args, "C - real hole, job sized on GAUGE (caliper ignored)",
        cement_volume=sched_g.total_volume,
    )

    print("\n=== what the washouts actually cost ===")
    print(f"1. Volume.  The real annulus holds "
          f"{rep.total_volume - rep_g.total_volume:+.4f} m^3 more than an in-gauge "
          f"hole ({100 * (rep.total_volume / rep_g.total_volume - 1):+.1f} %). "
          f"A job designed on bit size is short by that much.")
    print(f"   sized on caliper : {schedule.total_volume:.3f} m^3 cement -> "
          f"efficiency {rep.efficiency:.4f}, {rep.total_mud:.4f} m^3 mud left")
    print(f"   sized on gauge   : {sched_u.total_volume:.3f} m^3 cement -> "
          f"efficiency {rep_u.efficiency:.4f}, {rep_u.total_mud:.4f} m^3 mud left")
    print(f"   cost of ignoring the caliper: "
          f"{rep_u.total_mud - rep.total_mud:+.4f} m^3 more mud left behind, "
          f"{100 * (rep.efficiency - rep_u.efficiency):.1f} efficiency points")
    print(f"\n2. Concentration.  Washed-out hole is "
          f"{100 * rep.washout_volume_share:.1f} % of the annular volume but holds "
          f"{100 * rep.washout_mud_share:.1f} % of the leftover mud "
          f"- {rep.concentration_ratio:.2f} x its share.")
    print("   This ratio is a property of THIS well and job, not a law about")
    print("   washouts: it also depends on where the enlargements sit relative to")
    print("   the front at the end of the job. On a log with mid-well washouts the")
    print("   same model gives a ratio below 1.")
    print(f"\n3. Displacement itself is NOT worse.  At the same number of pore "
          f"volumes pumped the washed-out hole reaches {rep.efficiency:.4f} against "
          f"{rep_g.efficiency:.4f} in gauge, because a wider gap flows slower and a "
          f"yield-stress fluid's profile flattens (see docs/assumptions.md, A-24).")
    print("\n   Caveat: this model strands nothing permanently - pumping 2.5 well")
    print("   volumes clears the annulus to 99.98 %. Flow separation in the cavity,")
    print("   which would strand fluid however long you circulate, is a 2D effect it")
    print("   cannot represent, so these are lower bounds.")

    # --- figures -----------------------------------------------------------
    from inpipe.wellview import animate_circulation, plot_well_section

    # The solver's fluid registry starts with the initial in-situ fluid,
    # which is the displaced one whatever the case file calls it.
    i_mud = 0
    z = rep.depth

    fig, axes = plt.subplots(1, 4, figsize=(16, 6.4), sharey=True)
    axes[0].plot(m_to_inch(rep.hole_diameter), z, lw=0.6, color="0.25")
    axes[0].axvline(m_to_inch(rep.gauge), color="C0", ls="--", lw=1.0, label="gauge")
    axes[0].axvline(m_to_inch(args.washout_threshold * rep.gauge), color="C3",
                    ls=":", lw=1.0, label=f"{args.washout_threshold:g} x gauge")
    axes[0].set_xlabel("hole diameter [in]")
    axes[0].set_ylabel("measured depth [m]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    dz = float(np.mean(np.diff(z)))
    axes[1].plot(rep.mud_volume / dz, z, lw=0.8, label="real hole")
    axes[1].plot(rep_g.mud_volume / dz, z, lw=0.8, label="in-gauge")
    axes[1].set_xlabel("residual mud [m$^3$ per m]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(rep.local_fraction, z, lw=0.8, label="real hole")
    axes[2].plot(rep_g.local_fraction, z, lw=0.8, label="in-gauge")
    axes[2].set_xlabel("residual mud / local annular volume")
    axes[2].set_xlim(-0.03, 1.03)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    axes[3].plot(np.cumsum(rep.mud_volume[::-1])[::-1], z, lw=0.9, label="real hole")
    axes[3].plot(np.cumsum(rep_g.mud_volume[::-1])[::-1], z, lw=0.9, label="in-gauge")
    axes[3].set_xlabel("cumulative residual mud below [m$^3$]")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)
    axes[0].set_ylim(z.max(), z.min())

    axes[1].plot(rep_u.mud_volume / dz, z, lw=0.8, ls="--",
                 label="real hole, job sized on gauge")
    axes[1].legend(fontsize=7)
    axes[2].plot(rep_u.local_fraction, z, lw=0.8, ls="--", label="sized on gauge")
    axes[2].legend(fontsize=7)
    axes[3].plot(np.cumsum(rep_u.mud_volume[::-1])[::-1], z, lw=0.9, ls="--",
                 label="sized on gauge")
    axes[3].legend(fontsize=7)
    fig.suptitle(
        f"Mud left behind - washouts hold {100 * rep.washout_mud_share:.0f} % of it "
        f"on {100 * rep.washout_volume_share:.0f} % of the volume "
        f"({rep.concentration_ratio:.2f} x their share)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "mud_left_behind.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT / 'mud_left_behind.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 9.5), sharey=True)
    plot_well_section(real, i_mud, ax=axes[0], title="real hole", colorbar=False,
                      cmap="YlGnBu")
    plot_well_section(smooth, i_mud, ax=axes[1], title="in-gauge hole",
                      colorbar=True, cmap="YlGnBu")
    axes[1].set_ylabel("")
    fig.suptitle("Residual mud fraction at end of job (dark = mud remaining)")
    fig.tight_layout()
    fig.savefig(OUT / "mud_left_section.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT / 'mud_left_section.png'}")

    path = animate_circulation(real, OUT / "mud_left_behind.mp4",
                               fluid_index=i_mud, fps=12, cmap="YlGnBu")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
