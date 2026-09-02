"""Case 3 - a full cementing circulation: cement down the casing, up the annulus.

200 m of 5 in casing in an open hole whose diameter comes from a caliper log.
The well starts full of mud; cement is pumped from surface with no spacer,
turns at the shoe, and displaces the mud up the annulus.

Run:
    python -m cases.circulation_200m                 # synthetic caliper
    python -m cases.circulation_200m path/to/cal.csv # your own caliper log

The caliper file may be CSV or LAS.  Units are resolved from the column names
or the LAS ~C section, and otherwise from the diameter magnitude; see
``inpipe/caliper.py``.  Depth is never guessed from its own magnitude.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from inpipe.annulus_grid import AnnulusGrid
from inpipe.caliper import read_caliper, synthetic_caliper
from inpipe.circulation import CirculationSolver, WellConfig
from inpipe.config import GridConfig, NumericsConfig, bpm_to_m3s, inch_to_m
from inpipe.fluid import Fluid, PumpSchedule, PumpStage

OUT = Path(__file__).resolve().parent.parent / "results"

LENGTH = 200.0
CASING_ID = inch_to_m(5.0)      # as used in the in-pipe cases
CASING_OD = inch_to_m(5.5)
GAUGE_HOLE = inch_to_m(8.5)
FLOW_RATE = bpm_to_m3s(5.0)

MUD = Fluid("mud", rho=1198.0, tau0=2.0, k=0.30, n=0.72)
CEMENT = Fluid("cement", rho=1870.0, tau0=6.0, k=0.55, n=0.65)

N_AXIAL = 200
N_LAYER = 9
N_AZIMUTH = 8
N_SNAPSHOTS = 90


def build(caliper_path=None, n_axial=N_AXIAL, excess=1.05):
    caliper = (
        read_caliper(caliper_path)
        if caliper_path
        else synthetic_caliper(LENGTH, GAUGE_HOLE)
    )
    well = WellConfig(LENGTH, CASING_ID, CASING_OD, caliper)
    grid = GridConfig(n_axial=n_axial, n_layer=N_LAYER, n_azimuth=N_AZIMUTH)

    casing_volume = math.pi * (0.5 * CASING_ID) ** 2 * LENGTH
    annulus_volume = AnnulusGrid(
        LENGTH, CASING_OD, caliper, n_axial, N_LAYER, N_AZIMUTH
    ).total_volume
    # Pump enough cement to fill both legs, plus an excess.
    schedule = PumpSchedule(
        [PumpStage(CEMENT, (casing_volume + annulus_volume) * excess, FLOW_RATE)]
    )
    solver = CirculationSolver(
        well, schedule, initial_fluid=MUD, grid=grid,
        numerics=NumericsConfig(diagnostics_every=25),
    )
    return solver, schedule, caliper, casing_volume, annulus_volume


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    caliper_path = sys.argv[1] if len(sys.argv) > 1 else None
    solver, schedule, caliper, v_casing, v_annulus = build(caliper_path)

    print(caliper.summary(casing_od=CASING_OD))
    smooth = math.pi * ((0.5 * caliper.gauge) ** 2 - (0.5 * CASING_OD) ** 2) * LENGTH
    print(f"\ncasing volume    : {v_casing:.3f} m^3")
    print(f"annulus volume   : {v_annulus:.3f} m^3 "
          f"({100 * (v_annulus / smooth - 1):+.1f} % vs an in-gauge hole)")
    print(f"cement pumped    : {schedule.total_volume:.3f} m^3")
    print(f"job duration     : {schedule.total_time / 60:.1f} min at "
          f"{FLOW_RATE * 60:.4f} m^3/min")

    result = solver.run(t_end=schedule.total_time, n_snapshots=N_SNAPSHOTS)
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

    free_fall = h["pump_pressure"] < 0.0
    print(f"\nU-tube imbalance peaks at {h['utube_imbalance'].max() / 1e5:.2f} bar "
          f"({h['utube_imbalance'].max() / 6894.757:.0f} psi)")
    print(f"the well would FREE-FALL for {100 * free_fall.mean():.0f} % of the job "
          f"- see the note in the README on what that means for these results")
    print(f"ECD at shoe ranges {h['ecd_at_shoe'].min():.0f} to "
          f"{h['ecd_at_shoe'].max():.0f} kg/m^3")

    # --- figures -----------------------------------------------------------
    from inpipe.wellview import animate_circulation, plot_well_section

    fig, axes = plt.subplots(1, 4, figsize=(15, 8.5), sharey=True)
    picks = [int(f * (len(result.snapshots) - 1)) for f in (0.15, 0.4, 0.65, 1.0)]
    for ax, i in zip(axes, picks):
        snap = result.snapshots[i]
        plot_well_section(
            result, i_cem, ax=ax, casing_f=snap["casing"], annulus_f=snap["annulus"],
            title=f"t = {snap['time'] / 60:.1f} min", colorbar=(ax is axes[-1]),
        )
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.suptitle("Cement displacing mud - casing down, annulus up (colour = cement fraction)")
    fig.tight_layout()
    fig.savefig(OUT / "circulation_sections.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT / 'circulation_sections.png'}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(h["time"] / 60, h["annular_efficiency"])
    axes[0].set_xlabel("time [min]")
    axes[0].set_ylabel("annular displacement efficiency")
    axes[0].grid(alpha=0.3)

    axes[1].plot(h["time"] / 60, h["ecd_at_shoe"], label="ECD at shoe")
    axes[1].axhline(MUD.rho, color="0.6", ls="--", lw=0.9, label="mud density")
    axes[1].axhline(CEMENT.rho, color="0.3", ls=":", lw=0.9, label="cement density")
    axes[1].set_xlabel("time [min]")
    axes[1].set_ylabel("equivalent circulating density [kg/m$^3$]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(h["time"] / 60, h["pump_pressure"] / 1e5, label="pump pressure")
    axes[2].plot(h["time"] / 60, h["utube_imbalance"] / 1e5, label="U-tube imbalance")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].fill_between(h["time"] / 60, 0, h["pump_pressure"].min() / 1e5,
                         where=h["pump_pressure"] < 0, color="C3", alpha=0.12,
                         label="free-fall")
    axes[2].set_xlabel("time [min]")
    axes[2].set_ylabel("pressure [bar]")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "circulation_history.png", dpi=140)
    print(f"wrote {OUT / 'circulation_history.png'}")

    path = animate_circulation(result, OUT / "circulation.mp4", fluid_index=i_cem, fps=12)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
