# Reduced-order in-pipe displacement solver

A Python reconstruction of the in-pipe displacement model of

> Dai, H., Eslami, A., Schneider, J., Liu, G., Schwering, F. (2023).
> *Modeling displacement flow inside a full-length casing string for well cementing.*
> **Petroleum Research 9**, 1–16. [doi:10.1016/j.ptlrs.2023.08.004](https://doi.org/10.1016/j.ptlrs.2023.08.004)

**This is not a CFD solver.** It couples an analytical 1D axial velocity
profile, solved per depth station, to a 3D scalar transport of fluid volume
fractions advected by that axial velocity only. No Navier–Stokes, no pressure
Poisson solve, no transverse velocity — matching the paper's own reduced-order
strategy.

---

## Phase 1 scope

Target application: a **vertical** wellbore, ~200 m of 5 in (127 mm) casing.

Because the well is vertical (`β = 0`), every buoyancy criterion in the source
paper degenerates: the inertial velocity `v_t = √(At·g·sin β·D)` is zero, so the
Froude number `Fr = u/v_t` is undefined and the inertial Reynolds number
`Re_t = ρ v_t D/μ` is zero, which can never satisfy the segregation condition
`Re_t > 1` (Eq. A.18). Segregation, flow-regime switching and pipe rotation are
therefore **structurally inactive** and are deliberately not implemented.

| | status |
|---|---|
| Herschel–Bulkley rheology (Eq. 1), pump schedule | built |
| Concentric axial velocity profile (Eqs. A.3–A.7) | built |
| Stratified 3D mesh (Figs. 2b, 3a) | built |
| VOF transport of `f_i` (Eq. 2 / A.8, A.9), upwind baseline | built |
| Mixing-status variable `s` (Eqs. A.19–A.20) | plumbing built, nothing sets `s = 1` |
| Interface reconstruction (donor–acceptor, THINC) | Phase 2 |
| Finite-rate diffusive mixing (RHS of Eq. 2) | Phase 3 |
| Segregation (A.17–A.18), instability criteria (A.10, A.14–A.16), rotation (Eqs. 3–6) | out of scope — inactive at `β = 0` |
| Annulus model, free-fall / U-tubing, field validation | out of scope |

---

## Install and run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q -m "not slow"   # test gates 1-4 and integration gates 1-3
.venv/bin/python -m pytest -q -m slow         # the 200 m field-scale gate
.venv/bin/python -m cases.lab_4m_stretching   # paper Fig. 5(a) view + smearing study
.venv/bin/python -m cases.field_200m          # 200 m x 5 in cementing job
```

## Layout

```
inpipe/
  config.py       unit conversions (bpm, ft, cP, ppg, psi) and config dataclasses
  fluid.py        Module 1 - Herschel-Bulkley fluid, pump schedule
  velocity.py     Module 2 - concentric axial velocity profile, tau_w inverse solve
  grid.py         Module 3 - stratified cross-section mesh, exact segment geometry
  transport.py    Module 4 - axial VOF advection, CFL, numerical-diffusion metrics
  interface.py    Module 5 - Phase 2 placeholder (see the module docstring)
  solver.py       Module 6 - the coupled time loop and diagnostics
  postprocess.py  plotting: centre-plane view, outlet history, conservation report
cases/            runnable lab-scale and field-scale cases
tests/            the five test gates
docs/assumptions.md   the assumption register - read this first
```

## Sign convention

`+z` is the **flow direction**, i.e. downward inside the casing for a vertical
well. `β` is inclination from vertical, so `β = 0` gives `cos β = 1`. The
frictional pressure gradient is `dp/dz − ρ g cos β = 2 τ_w / R` (Eq. A.7).
SI units are used everywhere internally; conversion happens only at I/O.

---

## What the reconstruction found

The paper's PDF text is OCR-corrupted in places, and some of its stated
numerics do not survive scrutiny. Every departure is logged in
[`docs/assumptions.md`](docs/assumptions.md); four are worth stating up front.

**1. The exponent in Eqs. A.3/A.6 is `1/n + 1`, not `1/(n+1)`.**
Redoing the integration `−du/dr = ((τ−τ0)/k)^(1/n)` from `r` to `R` gives
`1/n + 1`. The corrected form reproduces Poiseuille (rel 1e-10), the power-law
peak `(3n+1)/(n+1)` (rel 1e-8) and Buckingham–Reiner (rel 1e-8); the printed
form reproduces none of them.

**2. Eq. A.9 as printed is dimensionally inconsistent and has the wrong sign.**
The face area is missing (compare Eq. A.8's `∫_s u_n f_s dS`), and with
outward-positive normals an outflow must *decrease* cell content. The correct
update is `f^{n+1} = f^n − (Δt/ΔV) Σ_j (u_j A_j f_{s,j})`.

**3. Under mixed rheology, conservation and sum-to-one become mutually
incompatible.** This is a genuine inconsistency in the source model, not a
transcription error. With no transverse velocity, each `(layer, azimuth)` column
carries its own axial velocity; when neighbouring stations hold different
effective fluids their profile *shapes* differ, so `∂u/∂z ≠ 0` per column even
though every station passes the same `Q`. The discrete continuity condition
`Σ_j u_j A_j = 0` then fails per cell. Measured on the 200 m field case:

| closure | max &#124;Σᵢfᵢ − 1&#124; | per-fluid mass error | max `f` |
|---|---|---|---|
| Eq. A.9 as written | **0.39** | 1.6e-15 | **1.101** |
| advective form of Eq. (2) (`−f∇·u`) | 2.8e-14 | **1.8e-2** | 1.000 |
| transverse redistribution (default) | 1.8e-15 | 1.8e-15 | 1.000 |

Neither of the first two is usable: one lets the fractions stop being a
partition of the cell (`f` reaching 1.10), the other creates and destroys 1.8 %
of each fluid. The resolution is that `Σ_c (∇·u)_c A_c = 0` across the
cross-section, so the imbalance is a *redistribution*, not a source: columns
losing volume axially shed it laterally carrying their own composition, and
columns gaining volume receive the donor mixing-cup composition. Both invariants
then hold to round-off.

This is transverse redistribution *imposed algorithmically*, in the same spirit
as the paper's own segregation step — not a solved transverse velocity. Its
physical content is an explicit assumption (lateral redistribution is
instantaneous and well-mixed across the section) and it is logged as such. It is
inert wherever `∇·u = 0`, which is why the inconsistency is invisible in the
paper's own iso-fluid validation cases (§3.1, Figs. 4–5) and surfaces only once
several cementing fluids are in the pipe at once — the model's actual intended
application.

**4. The justification for dropping the diffusion term does not survive mesh
refinement.** The paper drops the `D_m ∇²f` term because at `Δx = 30 m`,
`Δt = 0.1 min` the numerical diffusion `Δx²/Δt ≈ 150 m²/s` swamps the physical
`D_m ≈ 1e-3…1e-4 m²/s` by ~1e5. Two things are wrong with carrying that forward:

- `Δx²/Δt` is the wrong scaling for this scheme. For upwind plus explicit Euler
  at fixed Courant number `C`, the true leading-order numerical diffusivity is
  `½ u Δz (1 − C)` — linear in `Δz`, not `Δx²/Δt`. Measured by fitting `D` from
  the variance of the front, the two agree to ~2 %, while `Δx²/Δt` over-states
  the smearing by `2/(C(1−C))` — about 50× at `C = 0.04`.
- Because `Δt` is CFL-limited, `Δt ∝ Δz`, so even `Δx²/Δt` falls only *linearly*
  with `Δx`. The build spec's expectation of `Δx²` decay is unreachable for any
  explicit CFL-limited scheme; a test demonstrates both the arithmetic and the
  stability wall.

  At `Δz = 5 mm` on the lab-scale case the measured numerical diffusivity is
  `2.6e-5 m²/s` — *below* the physical `D_m` the paper discards. So Phase 3
  (restoring the right-hand side of Eq. 2) is not optional polish for a refined
  model; it is required for the refinement to mean anything.

**Also worth noting:** centroid-radius velocity mapping — the spec's baseline —
fails its own 1 % flow-rate gate at the paper's 13 × 18 cross-section for a
yield-stress fluid (+1.38 %; Newtonian +0.54 %, power-law `n = 0.4` +0.71 %).
Exact per-cell area averaging is the default instead, at 4e-6 relative error and
negligible run-time cost.

---

## Test gates

| Gate | Module | What it pins down |
|---|---|---|
| 1 | `fluid.py` | unit round-trips, pump-stage boundaries |
| 2 | `velocity.py` | Poiseuille shape and peak, power-law peak `(3n+1)/(n+1)`, Buckingham–Reiner, no-slip, plug flatness, `Q` round-trip, yield limit, closed form vs. quadrature |
| 3 | `grid.py` | areas sum to `πR²` and volumes to `πR²L` *exactly* at any resolution, mirror symmetry, equal-area layer alternative, velocity-mapping error |
| 4 | `transport.py` | uniform-field invariance, mass budget over 1000 steps, sum-to-one, monotone square wave, parabolic stretching, CFL enforcement, numerical-diffusion scaling |
| 5 | `solver.py` | stationary single fluid, the paper's 4 m × 19 mm geometry, grid convergence, the 200 m field case under 5 min |

All geometric identities are **exact**, not convergent: cell areas and centroids
come from closed-form circular-segment integrals, so refining the mesh does not
improve the area sum — it was already at round-off.
