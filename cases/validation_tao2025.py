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

Everything editable lives in ``cases/tao2025.json`` - fluid properties,
geometry, flow rate, interfacial tension and the rheology treatment.  Change a
number there and rerun; nothing in this file needs touching.  A different file
can be passed with ``--case``.

Yield stress follows Fluent
---------------------------
By default this case sets ``regularisation_shear_rate = 5.5 1/s``, so the
slurry is integrated the way Fluent integrates it: viscosity capped below that
shear rate and **no rigid plug anywhere**.  ``--exact`` switches to the
unregularised law with a true plug, which is what the in-pipe paper writes, to
see how much the treatment is worth.  See :mod:`inpipe.rheology` for why the
published Eqs. (15)-(16) need their inequalities swapped, and why the
consistency index is kept literal.

Remaining mismatches against the CFD
------------------------------------
1. **Interfacial tension.**  The CFD uses ``sigma = 0.07 N/m``, giving a
   capillary number of order one here.  This model is miscible - volume
   fractions, no momentum equation, so no interfacial tension term can enter.
   The value is carried and its dimensionless groups reported, so it is visible
   when the miscible assumption stops being defensible; it is *not* modelled.

2. **The "drilling fluid" is water** (998 kg/m^3, 1 cP, Newtonian).  It has no
   yield stress, so it can never be stranded by failing to yield.

3. **No buoyancy.**  The densimetric Froude number here is below one at every
   inlet velocity, so the CFD is buoyancy-influenced and this model is not.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from inpipe.annulus_grid import AnnulusGrid
from inpipe.caliper import CaliperLog
from inpipe.caseio import DEFAULT_CASE, load_case
from inpipe.circulation import CirculationSolver, WellConfig
from inpipe.config import GridConfig, NumericsConfig
from inpipe.fluid import PumpSchedule, PumpStage
from inpipe.mudleft import mud_left_behind
from inpipe.rheology import critical_stress, plateau_viscosity

OUT = Path(__file__).resolve().parent.parent / "results"

INLET_VELOCITIES = (0.5, 0.2, 0.05)


def smooth_wall(spec, n=201) -> CaliperLog:
    z = np.linspace(0.0, spec.geometry["length"], n)
    return CaliperLog(z, np.full(n, spec.geometry["hole_diameter"]),
                      name="Case-1 smooth wall")


def wavy_wall(spec, n=401, amplitude=0.25, wavelength=0.2) -> CaliperLog:
    """Case-2: "a layer of irregular (wavey) walls" added outside the smooth one.

    The paper does not give the wave shape, so this is a sinusoid of the stated
    character - always *outward* of the smooth wall, so the annulus is larger,
    as the paper states.
    """
    z = np.linspace(0.0, spec.geometry["length"], n)
    bump = 0.5 * (1.0 - np.cos(2.0 * math.pi * z / wavelength))
    return CaliperLog(z, spec.geometry["hole_diameter"] * (1.0 + amplitude * bump),
                      name="Case-2 wavy wall")


def build(spec, caliper, u_inlet, n_axial=100, n_layer=13, n_azimuth=8,
          exact=False):
    length = spec.geometry["length"]
    casing_id, casing_od = spec.geometry["casing_id"], spec.geometry["casing_od"]
    fill = spec.flow.get("excess", 1.05)
    q = u_inlet * math.pi * (0.5 * casing_id) ** 2
    well = WellConfig(length, casing_id, casing_od, caliper)
    v_casing = math.pi * (0.5 * casing_id) ** 2 * length
    v_annulus = AnnulusGrid(length, casing_od, caliper, n_axial,
                            n_layer, n_azimuth).total_volume
    schedule = PumpSchedule(
        [PumpStage(spec.displacing, (v_casing + v_annulus) * fill, q)])
    solver = CirculationSolver(
        well, schedule, initial_fluid=spec.displaced,
        grid=GridConfig(n_axial=n_axial, n_layer=n_layer, n_azimuth=n_azimuth),
        numerics=NumericsConfig(
            diagnostics_every=25,
            regularisation_shear_rate=(
                None if exact else spec.regularisation_shear_rate),
            normalise_consistency=spec.normalise_consistency,
        ),
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
    p.add_argument("--case", type=Path, default=DEFAULT_CASE,
                   help="JSON case file with fluids, geometry and rheology")
    p.add_argument("--exact", action="store_true",
                   help="use the exact Herschel-Bulkley law (rigid plug) instead "
                        "of Fluent's regularisation")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    OUT.mkdir(exist_ok=True)
    spec = load_case(args.case)
    caliper = wavy_wall(spec) if args.irregular else smooth_wall(spec)
    label = "Case-2 (wavy)" if args.irregular else "Case-1 (smooth)"
    velocities = (args.u_inlet,) if args.u_inlet else INLET_VELOCITIES
    length = spec.geometry["length"]
    casing_od, hole = spec.geometry["casing_od"], spec.geometry["hole_diameter"]
    cement, displaced = spec.displacing, spec.displaced
    gc = None if args.exact else spec.regularisation_shear_rate

    print(spec.summary())
    print(f"\nmesh     : {args.n_axial} axial x 13 x 8   [{label}]")
    if gc is not None:
        print(f"rheology : tau_c = {critical_stress(cement, gc, spec.normalise_consistency):.3f} Pa, "
              f"plateau viscosity = "
              f"{plateau_viscosity(cement, gc, spec.normalise_consistency):.4f} Pa.s, "
              f"no plug anywhere")
    else:
        print("rheology : exact Herschel-Bulkley, rigid plug where tau < tau0")
    print()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for u_in in velocities:
        solver, schedule, q, v_annulus = build(spec, caliper, u_in,
                                               n_axial=args.n_axial,
                                               exact=args.exact)
        result = solver.run(t_end=schedule.total_time, n_snapshots=0)
        rep = mud_left_behind(result, displaced, gauge=hole)
        prof = solver._annulus_profiles(q)
        mid = prof[len(prof) // 2]
        ub = q / (math.pi * ((0.5 * hole) ** 2 - (0.5 * casing_od) ** 2))

        print(f"u_inlet = {u_in:5.2f} m/s -> Q = {q * 1000:6.3f} L/s, "
              f"annulus mean {ub:.4f} m/s, job {schedule.total_time:5.1f} s")
        print(f"    tau_w {mid.tau_w:6.3f} Pa | plug "
              f"{100 * mid.plug_half_width / mid.half_gap:4.1f} % of gap | "
              f"u_max/u_bar {mid.u_max / mid.mean_velocity:5.3f}")
        ca = spec.interface.capillary_number(
            plateau_viscosity(cement, gc, spec.normalise_consistency)
            if gc is not None else cement.k, ub)
        at = ((cement.rho - displaced.rho)
              / (cement.rho + displaced.rho))
        fr = ub / math.sqrt(max(at, 1e-30) * 9.80665 * (hole - casing_od) / 2)
        print(f"    Ca {ca:6.3f} (interfacial tension: reported, not modelled) | "
              f"Fr {fr:6.3f} (buoyancy: not modelled)")
        print(f"    displacement efficiency {1 - rep.total_mud / rep.total_volume:.4f}, "
              f"drilling fluid left {rep.total_mud:.5f} m^3")

        s, f = radial_profile(result, cement, args.probe_depth)
        axes[0].plot(s, f, marker="o", ms=3, label=f"$u_{{in}}$ = {u_in} m/s")
        tag = "exact" if args.exact else "fluent"
        path = OUT / f"tao2025_radial_u{u_in:g}_{tag}.csv"
        path.write_text(
            "s_over_gap,cement_fraction\n"
            + "\n".join(f"{a:.6g},{b:.6g}" for a, b in zip(s, f)) + "\n"
        )
        print(f"    wrote {path.name}")

        g = result.annulus_grid
        order = np.argsort(g.z_centers)
        a = g.cell_area
        i_cem = result.fluids.index(cement)
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
    axes[1].set_ylim(length, 0)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("annular displacement efficiency")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    treatment = "exact HB (plug)" if args.exact else "Fluent regularisation (no plug)"
    fig.suptitle(f"{spec.name} - {label} - {treatment}")
    fig.tight_layout()
    out_png = OUT / f"tao2025_validation_{'exact' if args.exact else 'fluent'}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
