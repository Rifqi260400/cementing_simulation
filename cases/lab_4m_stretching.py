"""Case 1 - the paper's Section 3.1 geometry: 4 m pipe, 19 mm ID, vertical.

Reproduces the centre-plane concentration view of the paper's Fig. 5(a) and
quantifies the numerical smearing of the first-order upwind baseline.

Scope note (assumption A-19): the paper's own Fig. 5 is at 83 degrees from
vertical, where segregation and backflow dominate.  Phase 1 omits segregation
deliberately (it is structurally inactive at beta = 0), so this reproduces the
figure's *presentation* and the concentric parabolic-stretching mechanism, not
its buoyancy-driven asymmetry.

Run:  python -m cases.lab_4m_stretching
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from inpipe.config import GeometryConfig, GridConfig, NumericsConfig, SimulationConfig
from inpipe.fluid import Fluid, PumpSchedule, PumpStage
from inpipe.postprocess import plot_centre_plane, plot_diagnostics, summary_table
from inpipe.solver import InPipeSolver
from inpipe.transport import upwind_diffusivity
from inpipe.velocity import solve_profile

OUT = Path(__file__).resolve().parent.parent / "results"

LENGTH = 4.0
INNER_D = 0.019
U_BAR = 0.05
Z0 = 1.0
SNAPSHOTS = (5.0, 15.0, 30.0)


def build(n_axial=400, n_layer=13, n_azimuth=18):
    area = math.pi * (0.5 * INNER_D) ** 2
    q = U_BAR * area
    displaced = Fluid.newtonian(1000.0, 0.001, "displaced")
    displacing = Fluid.newtonian(1000.0, 0.001, "displacing")
    config = SimulationConfig(
        geometry=GeometryConfig(length=LENGTH, inner_diameter=INNER_D, inclination=0.0),
        grid=GridConfig(n_axial=n_axial, n_layer=n_layer, n_azimuth=n_azimuth),
        numerics=NumericsConfig(diagnostics_every=20),
    )
    schedule = PumpSchedule([PumpStage(displacing, q * 1000.0, q)])
    solver = InPipeSolver(config, schedule, initial_fluid=displaced)
    solver.set_initial_interface(Z0, displacing, displaced)
    return solver, displacing, q


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    solver, displacing, q = build()
    i_disp = solver.fluids.index(displacing)

    fig, axes = plt.subplots(len(SNAPSHOTS), 1, figsize=(10, 2.5 * len(SNAPSHOTS)))
    for ax, t_snap in zip(np.atleast_1d(axes), SNAPSHOTS):
        result = solver.run(t_end=t_snap)
        plot_centre_plane(
            result, fluid_index=i_disp, ax=ax,
            title=f"$t$ = {result.time:.0f} s   ($\\bar u$ = {U_BAR} m/s, "
                  f"$\\Delta z$ = {result.grid.dz * 1e3:.1f} mm)",
        )
    fig.suptitle(
        "Centre-plane concentration of the displacing fluid "
        "(paper Fig. 5(a) view; vertical pipe, 4 m x 19 mm)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig5a_centre_plane.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'fig5a_centre_plane.png'}")

    print("\n" + summary_table(result))

    # -- analytical comparison ---------------------------------------------
    g = result.grid
    front = result.diagnostics.front_position[-1]
    exact = Z0 + result.velocity[0] * result.time
    ok = np.isfinite(front) & (exact < g.length - 6 * g.dz)
    print(f"\nfront vs z0 + u(r) t : max error = "
          f"{np.abs(front - exact)[ok].max() / g.dz:.2f} cells "
          f"({np.abs(front - exact)[ok].max() * 1e3:.2f} mm)")

    prof = solve_profile(q, solver.fluids[0], g.radius)
    print(f"tip speed / u_bar    : "
          f"{(np.nanmax(front) - Z0) / (U_BAR * result.time):.4f}  (analytical 2.0)")

    dt = float(np.mean(result.diagnostics.dt))
    u_c = float(result.velocity[0].max())
    print(f"\nnumerical diffusion at dz = {g.dz:.4g} m, dt = {dt:.4g} s:")
    print(f"  paper's estimate dx^2/dt      = {g.dz**2 / dt:.4e} m^2/s")
    print(f"  upwind modified equation      = {upwind_diffusivity(u_c, g.dz, dt):.4e} m^2/s")
    print(f"  physical Dm (Debacq/Alba)     = 1e-4 to 1e-3 m^2/s")

    plot_diagnostics(result, path=OUT / "lab_diagnostics.png")
    print(f"wrote {OUT / 'lab_diagnostics.png'}")

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    from inpipe.postprocess import plot_velocity_profile

    plot_velocity_profile(prof, ax=ax, label="Newtonian")
    ax.set_title("Axial velocity along the\nvertical centreline (Fig. 3c view)")
    fig.tight_layout()
    fig.savefig(OUT / "velocity_profile.png", dpi=150)
    print(f"wrote {OUT / 'velocity_profile.png'}")


if __name__ == "__main__":
    main()
