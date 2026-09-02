"""Case 2 - field scale: 200 m of 5 in (127 mm) casing, vertical.

Three Herschel-Bulkley fluids (mud, spacer, cement) pumped at 5 bpm.  This is
the Phase 1 definition-of-done case: it must run in under five minutes with
mass conservation to rel 1e-10.

Run:  python -m cases.field_200m
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from inpipe.config import (
    GeometryConfig,
    GridConfig,
    NumericsConfig,
    SimulationConfig,
    bpm_to_m3s,
    inch_to_m,
)
from inpipe.fluid import Fluid, PumpSchedule, PumpStage
from inpipe.postprocess import (
    plot_centre_plane,
    plot_diagnostics,
    plot_outlet_history,
    summary_table,
)
from inpipe.solver import InPipeSolver

OUT = Path(__file__).resolve().parent.parent / "results"

LENGTH = 200.0
INNER_D = inch_to_m(5.0)
FLOW_RATE = bpm_to_m3s(5.0)

MUD = Fluid("mud", rho=1198.0, tau0=2.0, k=0.30, n=0.72)
SPACER = Fluid("spacer", rho=1318.0, tau0=1.2, k=0.18, n=0.80)
CEMENT = Fluid("cement", rho=1870.0, tau0=6.0, k=0.55, n=0.65)


def build(n_axial=200, n_layer=13, n_azimuth=18):
    area = math.pi * (0.5 * INNER_D) ** 2
    pipe_volume = area * LENGTH
    schedule = PumpSchedule(
        [
            PumpStage(SPACER, 0.35 * pipe_volume, FLOW_RATE),
            PumpStage(CEMENT, 1.20 * pipe_volume, FLOW_RATE),
        ]
    )
    config = SimulationConfig(
        geometry=GeometryConfig(length=LENGTH, inner_diameter=INNER_D, inclination=0.0),
        grid=GridConfig(n_axial=n_axial, n_layer=n_layer, n_azimuth=n_azimuth),
        numerics=NumericsConfig(diagnostics_every=25),
    )
    return InPipeSolver(config, schedule, initial_fluid=MUD), schedule


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    solver, schedule = build()
    print(f"pipe volume  : {math.pi * (0.5 * INNER_D) ** 2 * LENGTH:.4f} m^3")
    print(f"job duration : {schedule.total_time:.1f} s "
          f"({schedule.total_time / 60:.1f} min) at {FLOW_RATE * 60:.4f} m^3/min")
    result = solver.run(progress=False)
    print("\n" + summary_table(result))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    for ax, fl in zip(axes, result.fluids):
        plot_centre_plane(
            result, fluid_index=result.fluids.index(fl), ax=ax,
            title=f"{fl.name}  ($t$ = {result.time:.0f} s)", show_front=False,
        )
    fig.suptitle("200 m x 5 in vertical casing, 5 bpm - centre-plane concentrations", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "field_centre_plane.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {OUT / 'field_centre_plane.png'}")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    plot_outlet_history(result, ax=ax)
    ax.set_title("Fluid fractions leaving the shoe (the input an annulus model would take)")
    fig.tight_layout()
    fig.savefig(OUT / "field_outlet_history.png", dpi=150)
    print(f"wrote {OUT / 'field_outlet_history.png'}")

    plot_diagnostics(result, path=OUT / "field_diagnostics.png")
    print(f"wrote {OUT / 'field_diagnostics.png'}")

    # Displacement efficiency: fraction of the pipe swept by the lead fluid.
    i_cem = result.fluids.index(CEMENT)
    cv = result.grid.cell_volume
    swept = float((result.fractions[i_cem] * cv).sum()) / (cv.sum() * result.grid.n_axial)
    print(f"\ncement fraction in pipe at end of job: {swept:.4f}")


if __name__ == "__main__":
    main()
