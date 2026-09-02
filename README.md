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

**And one thing that held up better than expected.** Uniform spacing in the
chord coordinate `y` leaves the near-wall layers thin and under-resolved, which
looked like a threat to displacement efficiency — the residual sits near the
wall. It is not: on the 200 m case, efficiency moves only 87.65 → 87.37 →
87.29 % across a 4× refinement in `n_layer`, and an entirely different layering
rule (`equal_area`) lands at 87.42 %. The residual volume is set by the integral
of the parabolic profile, not by how the wall region is cut. What *does* stay
resolution-sensitive is where the residual sits: at `n_layer = 13` the two
outermost layers hold 31 % of the leftover mud on 7 % of the area.

---

## Test gates

| Gate | Module | What it pins down |
|---|---|---|
| 1 | `fluid.py` | unit round-trips against values quoted in the paper, Herschel-Bulkley constructors and validation, volume-weighted mixing, pump-stage boundaries at `t = 0`, exactly on a boundary, and past the end |
| 2 | `velocity.py` | Poiseuille shape and peak, power-law peak `(3n+1)/(n+1)`, Buckingham–Reiner, no-slip, plug flatness, `Q` round-trip, yield limit, closed form vs. quadrature |
| 3 | `grid.py` | areas sum to `πR²` and volumes to `πR²L` *exactly* at any resolution, mirror symmetry, equal-area layer alternative, velocity-mapping error |
| 4 | `transport.py` | uniform-field invariance, mass budget over 1000 steps, sum-to-one, monotone square wave, parabolic stretching, CFL enforcement, numerical-diffusion scaling |
| 5 | `solver.py` | stationary single fluid, the paper's 4 m × 19 mm geometry, grid convergence, the 200 m field case under 5 min |

All geometric identities are **exact**, not convergent: cell areas and centroids
come from closed-form circular-segment integrals, so refining the mesh does not
improve the area sum — it was already at round-off.

`pytest -q` runs the whole suite, field gate included, in a few minutes.

### Robustness

Beyond the physics gates, `tests/test_robustness.py` sweeps the input space and
asserts the four invariants — per-fluid mass budget, sum-to-one, boundedness,
finiteness — on every case:

| axis | covered |
|---|---|
| rheology | Newtonian, thick Newtonian, power-law `n ∈ {0.2, 0.4}`, Bingham, Herschel–Bulkley with `τ0 = 30 Pa`, `n = 0.25` |
| flow rate | four decades, down to barely above the yield stress |
| grid | `1 × 7 × 8` (single axial cell), `5 × 1 × 1` (single cell per section), odd azimuth counts, up to `400 × 52 × 72` |
| schedule | single stage, and three stages with a 3× rate step up and a 6× step down |
| geometry | 0.5 m × 5 mm bench scale up to 1524 m × 127 mm full-length casing |
| numerics | `CFL ∈ {0.05, 0.4, 0.9}`, both velocity mappings, all three transverse closures |

**All cases pass on the defaults.** The only two configurations that fail are
the two non-default modes the register already documents as deficient
(`transverse_closure="local"`, and `enforce_discrete_continuity=False`); both
have tests asserting they fail *in the documented way*, so nobody switches to
them expecting conservation.

### Performance

Measured on the field case (three Herschel–Bulkley fluids, full job):

| mesh | cells | ms/step |
|---|---|---|
| 40 × 7 × 8 | 2 240 | 0.9 |
| 100 × 13 × 18 | 23 400 | 8 |
| 200 × 13 × 18 | 46 800 | 28 |
| 200 × 26 × 36 | 187 200 | 42 |
| 400 × 52 × 72 | 1 497 600 | 266 |

The paper's own mesh is 23 400 cells. The model stays comfortable to a few
hundred thousand; at 1.5 M cells it still runs and still conserves, but it is
well outside the point of a reduced-order model — if you need that resolution,
you want the CFD run, not this.

---

## Phase 1 definition of done

| requirement | status |
|---|---|
| all five test gates green | 163 tests pass |
| parabolic stretching reproduced | front matches `z₀ + u(r)t` to **0.16 cells** (1.6 mm at `Δz = 10 mm`); tip speed **1.9876 ū** against the analytical 2 |
| smearing quantified as `Dm_num` | measured at four resolutions, matching the upwind modified equation to ~2 %; see the table above |
| 200 m case under 5 min | **77 s** of the 300 s budget (200 × 13 × 18, 46 800 cells, 1258 steps, 61 ms/step) |
| mass conservation to rel 1e-10 on that case | **1.9e-15**, with sum-to-one at 2.0e-15 and `f` inside [0, 1] |
| `docs/assumptions.md` populated | 22 entries, each with justification and sensitivity status |
| plot reproducing the Fig. 5(a) centre-plane view | `results/fig5a_centre_plane.png` |

### Figures produced

All of it can be written out for external comparison:

```python
from inpipe.postprocess import save_results
save_results(result, "out/", prefix="case")
```

which produces a `.npz` of the full state (fractions, mixing status, velocity,
cell geometry, rheology) plus CSVs of the centre-plane field, the area-averaged
axial profile, the centreline velocity (spanning the full diameter, with the
exact no-slip endpoints so a CFD profile overlays directly), and the outlet
history. `centreline_velocity`, `radial_concentration` and
`cross_section_average` give the same quantities as arrays for line probes.

| file | what it shows |
|---|---|
| `results/fig5a_centre_plane.png` | the paper's Fig. 5(a) view — parabolic front at three times |
| `results/velocity_profile.png` | axial velocity along the centreline (Fig. 3c view) |
| `results/lab_diagnostics.png` | conservation, boundedness, sum-to-one for the lab case |
| `results/field_cement_snapshots.png` | cement front developing through the 200 m job |
| `results/field_centre_plane.png` | all three fluids at end of job |
| `results/field_outlet_history.png` | fluid fractions leaving the shoe |
| `results/field_diagnostics.png` | field-scale conservation report |

---

## Where Phase 2 plugs in

`inpipe/interface.py` is deliberately empty. To add a face-value scheme,
register a function with the signature of `transport.upwind_faces` in
`transport.FACE_SCHEMES`; the solver picks it up from
`NumericsConfig.face_scheme` with no other change. The measured `Dm_num` table
above is the baseline any new scheme has to beat.

## Open question for the next phase

The paper's Appendix A.2 says a sharp interface is "maintained by applying axial
interface reconstruction to suppress numerical diffusion because of large axial
grid size" — but never describes the scheme. That gap is the reason Phase 1
stops at the upwind baseline rather than guessing at a reconstruction and
calling it a replication. Deciding what to put in its place (donor–acceptor,
THINC, or something else) is a modelling decision, not a transcription one, and
it is the Phase 2 question.
