"""Case 5 - matched setup for CFD comparison (Tao et al. 2025).

Fluid properties and geometry follow

    Tao, C.; Wang, Q.; Ahmadi, G.; Massoudi, M. (2025).
    "Numerical Analysis of Cement Placement into Drilling Fluid in Oilwell
    Applications." Materials 18, 3098.  https://doi.org/10.3390/ma18133098

so that this reduced-order model and an ANSYS-Fluent VOF run can be laid over
each other.  Exports radial concentration profiles across the annular gap,
which is the comparison that separates the two candidate explanations for mud
left in an enlargement (see the README).

Run::

    python -m cases.validation_tao2025                # all three inlet velocities
    python -m cases.validation_tao2025 --u-inlet 0.5  # just one
    python -m cases.validation_tao2025 --irregular    # Case-2 wavy outer wall

Known mismatches against the CFD - read these before comparing
--------------------------------------------------------------
1. **Yield-stress treatment.**  Fluent regularises Herschel-Bulkley below a
   critical shear rate ``gammadot_c = 5.5 1/s``, capping the viscosity at
   ``tau_y/gammadot_c + k gammadot_c^(n-1) = 0.470 Pa.s`` and making the slurry
   a 470x-water *Newtonian* there.  This model uses the unregularised law with a
   genuine unyielded plug.  The annulus in this geometry runs at a nominal
   shear rate of 0.6 to 6.4 1/s, i.e. *straddling* the cut-off, so much of it
   sits in the regularised regime.  Expect this model to show a markedly larger
   plug and a flatter profile.  It is the single biggest mismatch.

2. **Interfacial tension.**  The CFD uses ``sigma = 0.07 N/m`` between slurry
   and drilling fluid, giving a capillary number of order one here.  This model
   is miscible - volume fractions with no interfacial tension at all.

3. **The "drilling fluid" is water** (998 kg/m^3, 1 cP, Newtonian).  It has no
   yield stress, so it can never be stranded by failing to yield.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from inpipe.annulus_grid import AnnulusGrid
from inpipe.caliper import CaliperLog
from inpipe.circulation import CirculationSolver, WellConfig
from inpipe.config import GridConfig, NumericsConfig
from inpipe.fluid import Fluid, PumpSchedule, PumpStage
from inpipe.mudleft import mud_left_behind

OUT = Path(__file__).resolve().parent.parent / "results"

# --- geometry, Tao et al. Section 3 -----------------------------------------
LENGTH = 1.0            # m, pipe segment
CASING_OD = 0.20        # m, "casing diameter is 20 cm"
CASING_WALL = 0.02      # m, "casing thickness is 2 cm"
CASING_ID = CASING_OD - 2 * CASING_WALL      # 0.16 m
ANNULUS_GAP = 0.10      # m, "annulus thickness is 10 cm"
HOLE_DIAMETER = CASING_OD + 2 * ANNULUS_GAP  # 0.40 m

# --- fluids, Tao et al. Table 1 ---------------------------------------------
CEMENT = Fluid("cement slurry", rho=1200.0, tau0=1.4, k=0.6, n=0.4)
DRILLING_FLUID = Fluid("drilling fluid", rho=998.0, tau0=0.0, k=1.0e-3, n=1.0)

#: Fluent's Herschel-Bulkley regularisation cut-off, Table 1.
GAMMADOT_C = 5.5
INLET_VELOCITIES = (0.5, 0.2, 0.05)


def fluent_capped_viscosity(fluid: Fluid, gammadot_c: float = GAMMADOT_C) -> float:
    """The Newtonian viscosity Fluent falls back to below ``gammadot_c``."""
    return fluid.tau0 / gammadot_c + fluid.k * gammadot_c ** (fluid.n - 1.0)


def smooth_wall(n=201) -> CaliperLog:
    z = np.linspace(0.0, LENGTH, n)
    return CaliperLog(z, np.full(n, HOLE_DIAMETER), name="Case-1 smooth wall")


def wavy_wall(n=401, amplitude=0.25, wavelength=0.2) -> CaliperLog:
    """Case-2: "a layer of irregular (wavey) walls" added outside the smooth one.

    The paper does not give the wave shape, so this is a sinusoid of the stated
    character - always *outward* of the smooth wall, so the annulus is larger,
    as the paper states.
    """
    z = np.linspace(0.0, LENGTH, n)
    bump = 0.5 * (1.0 - np.cos(2.0 * math.pi * z / wavelength))
    return CaliperLog(z, HOLE_DIAMETER * (1.0 + amplitude * bump),
                      name="Case-2 wavy wall")


def build(caliper, u_inlet, n_axial=100, n_layer=13, n_azimuth=8, fill=1.05):
    q = u_inlet * math.pi * (0.5 * CASING_ID) ** 2
    well = WellConfig(LENGTH, CASING_ID, CASING_OD, caliper)
    v_casing = math.pi * (0.5 * CASING_ID) ** 2 * LENGTH
    v_annulus = AnnulusGrid(LENGTH, CASING_OD, caliper, n_axial,
                            n_layer, n_azimuth).total_volume
    schedule = PumpSchedule([PumpStage(CEMENT, (v_casing + v_annulus) * fill, q)])
    solver = CirculationSolver(
        well, schedule, initial_fluid=DRILLING_FLUID,
        grid=GridConfig(n_axial=n_axial, n_layer=n_layer, n_azimuth=n_azimuth),
        numerics=NumericsConfig(diagnostics_every=25),
    )
    return solver, schedule, q, v_annulus


def radial_profile(result, fluid, depth):
    """Cement fraction across the annular gap at one depth.

    Returns ``(s_over_b, fraction)`` with ``s`` the distance from the *inner*
    (casing) wall normalised by the gap, so 0 is the casing and 1 the hole wall
    - the form to overlay a CFD line probe on.
    """
    g = result.annulus_grid
    i = result.fluids.index(fluid)
    k = int(np.argmin(np.abs(g.z_centers - depth)))
    rings = g.ring_faces[k]
    r_mid = 0.5 * (rings[:-1] + rings[1:])
    frac = result.annulus_fractions[i][k].mean(axis=1)
    return (r_mid - g.r_inner) / (g.r_outer[k] - g.r_inner), frac


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u-inlet", type=float, default=None,
                   help="single inlet velocity [m/s]; default runs all three")
    p.add_argument("--irregular", action="store_true", help="Case-2 wavy wall")
    p.add_argument("--n-axial", type=int, default=100)
    p.add_argument("--probe-depth", type=float, default=0.5,
                   help="depth [m] for the radial profile export")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    OUT.mkdir(exist_ok=True)
    caliper = wavy_wall() if args.irregular else smooth_wall()
    label = "Case-2 (wavy)" if args.irregular else "Case-1 (smooth)"
    velocities = (args.u_inlet,) if args.u_inlet else INLET_VELOCITIES

    eta0 = fluent_capped_viscosity(CEMENT)
    print(f"geometry : L = {LENGTH} m, casing {CASING_ID * 100:.0f} cm ID / "
          f"{CASING_OD * 100:.0f} cm OD, hole {HOLE_DIAMETER * 100:.0f} cm, "
          f"gap {ANNULUS_GAP * 100:.0f} cm   [{label}]")
    print(f"cement   : rho {CEMENT.rho:.0f}, tau_y {CEMENT.tau0} Pa, "
          f"k {CEMENT.k} Pa.s^n, n {CEMENT.n}")
    print(f"displaced: rho {DRILLING_FLUID.rho:.0f}, mu {DRILLING_FLUID.k:.0e} Pa.s "
          f"(Newtonian - this is water, not a mud)")
    print(f"Fluent regularisation: below {GAMMADOT_C} 1/s it caps viscosity at "
          f"{eta0:.4f} Pa.s and drops the plug entirely\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    results = {}
    for u_in in velocities:
        solver, schedule, q, v_annulus = build(caliper, u_in, n_axial=args.n_axial)
        result = solver.run(t_end=schedule.total_time, n_snapshots=0)
        rep = mud_left_behind(result, DRILLING_FLUID, gauge=HOLE_DIAMETER)
        prof = solver._annulus_profiles(q)
        mid = prof[len(prof) // 2]
        ub = q / (math.pi * ((0.5 * HOLE_DIAMETER) ** 2 - (0.5 * CASING_OD) ** 2))
        results[u_in] = (result, rep)

        print(f"u_inlet = {u_in:5.2f} m/s -> Q = {q * 1000:6.3f} L/s, "
              f"annulus mean {ub:.4f} m/s, job {schedule.total_time:5.1f} s")
        print(f"    tau_w {mid.tau_w:6.3f} Pa | plug "
              f"{100 * mid.plug_half_width / mid.half_gap:4.1f} % of gap | "
              f"u_max/u_bar {mid.u_max / mid.mean_velocity:5.3f}")
        print(f"    displacement efficiency {1 - rep.total_mud / rep.total_volume:.4f}, "
              f"drilling fluid left {rep.total_mud:.5f} m^3")

        s, f = radial_profile(result, CEMENT, args.probe_depth)
        axes[0].plot(s, f, marker="o", ms=3, label=f"$u_{{in}}$ = {u_in} m/s")
        path = OUT / f"tao2025_radial_u{u_in:g}.csv"
        path.write_text(
            "s_over_gap,cement_fraction\n"
            + "\n".join(f"{a:.6g},{b:.6g}" for a, b in zip(s, f)) + "\n"
        )
        print(f"    wrote {path.name}")

        g = result.annulus_grid
        order = np.argsort(g.z_centers)
        a = g.cell_area
        i_cem = result.fluids.index(CEMENT)
        ax_prof = ((result.annulus_fractions[i_cem] * a).sum(axis=(1, 2))
                   / a.sum(axis=(1, 2)))[order]
        axes[1].plot(ax_prof, g.z_centers[order], label=f"{u_in} m/s")

        h = result.history
        axes[2].plot(h["time"], h["annular_efficiency"], label=f"{u_in} m/s")

    axes[0].set_xlabel("distance from casing wall / gap")
    axes[0].set_ylabel("cement volume fraction")
    axes[0].set_title(f"radial profile at z = {args.probe_depth} m\n"
                      "(0 = casing wall, 1 = hole wall)")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("cement fraction (area-averaged)")
    axes[1].set_ylabel("depth [m]")
    axes[1].set_ylim(LENGTH, 0)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("annular displacement efficiency")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Matched to Tao et al. (2025) - {label}")
    fig.tight_layout()
    fig.savefig(OUT / "tao2025_validation.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT / 'tao2025_validation.png'}")


if __name__ == "__main__":
    main()
