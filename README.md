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

## Full circulation: cement down the casing, up the annulus

```bash
.venv/bin/python -m cases.circulation                     # 175 m to the shoe
.venv/bin/python -m cases.circulation --top-depth 0       # whole logged interval
.venv/bin/python -m cases.circulation --caliper my.las    # your own log
.venv/bin/python -m cases.circulation --synthetic         # no log needed
.venv/bin/python -m cases.circulation --case my.json      # your own fluids
```

Runs on the real caliper log in `data/K-GEP-1_composite.las` — a 390 m
composite log with the borehole-diameter curve in inches. The well depth, the
gauge hole size and the annular geometry all come from it.

**Modelled interval: 175 m to the shoe at 386.75 m** — the open hole only.
Above 175 m the log reads a near-constant 10.43 in (standard deviation 0.007 in
over 17 000 samples, against 3.05 in below 195 m), which is not rock, so that
section is very likely cased and its annulus is not an open hole. `--top-depth 0`
models the whole logged interval instead.

**7 in casing** (6.184 in ID, 29 lb/ft — the standard string for a 10.43 in
gauge hole) in an open hole that ranges 8.15–24.0 in. The well starts full of
mud; cement is pumped with no spacer, turns at the shoe, and displaces mud up
the annulus.

**Both fluids' properties live in `cases/kgep1.json`**, not in the code — the
mud and slurry numbers there are placeholders until the real ones are measured,
so change them and rerun. The same file carries the casing sizes, the pump
rate, the excess and the modelled interval, and can switch this case to the
Fluent yield-stress treatment as well (`rheology.regularisation_shear_rate`).
A command-line flag still overrides the file when one is given. To make the
drilling fluid water, as in the CFD case, set `tau0` to 0, `k` to `1e-3` and
`n` to 1.

The column above the modelled interval is still accounted for hydrostatically,
so shoe pressure and ECD stay true-depth quantities: the annulus above is taken
to remain mud, the casing above is volume-averaged over what has been pumped
through it. Friction above the interval is not included, so pump pressure is a
lower bound by that amount.

The log is **wrapped** (`WRAP. YES`), declares `NULL = -99999`, and carries
non-UTF8 bytes in a curve description — all handled. Its units come from the
`~C` section (depth in m, diameter in inches), not from magnitudes.

**The collapsed tail is cut, and the cut is reported.** The caliper drops from
13.3 in at 386.75 m to 2.1 in below it and stays there for 3.2 m: the tool
bottoming out, not geometry. `implausible_tail` finds it; the case cuts it and
says so. Nothing is trimmed silently — `--keep-tail` leaves it in.

### Results on this well

| | 175–386.75 m (default) | whole logged interval |
|---|---|---|
| annular displacement efficiency | **89.9 %** | 87.4 % |
| annulus volume | 10.19 m³, **+59 %** over an in-gauge hole | 15.15 m³, +29 % |
| job | 15.0 m³ cement, 18.9 min at 5 bpm | 23.8 m³, 29.9 min |
| ECD at shoe | 1222 → 1582 kg/m³ | 1242 → 1847 kg/m³ |
| U-tube imbalance | peaks at **21.8 bar (316 psi)**, free-fall **99 %** of the job | 18.6 bar, 92 % |
| conservation | sum-to-one 1.6e-13, volume 1.1e-15 | 1.3e-13, 7.9e-16 |

## CFD comparison — matched to Tao et al. (2025)

```bash
.venv/bin/python -m cases.validation_tao2025              # all three inlet velocities
.venv/bin/python -m cases.validation_tao2025 --exact      # exact HB instead (rigid plug)
.venv/bin/python -m cases.validation_tao2025 --irregular  # Case-2 wavy wall
.venv/bin/python -m cases.validation_tao2025 --case my.json
```

**Everything editable lives in `cases/tao2025.json`** — both fluids' density and
rheology, the geometry, the flow rate, the interfacial tension and the yield-stress
treatment. Change a number there and rerun; no code needs touching. Unknown keys
are *rejected*, so a mistyped property fails loudly rather than silently leaving
a default in place.

Fluids and geometry follow Tao, Wang, Ahmadi & Massoudi (2025), *Materials* 18,
3098 — cement slurry `ρ = 1200`, Herschel–Bulkley `τ_y = 1.4 Pa`, `k = 0.6`,
`n = 0.4`; drilling fluid `ρ = 998`, Newtonian `μ = 1 cP`; 1 m of 16 cm ID /
20 cm OD casing in a 40 cm hole. Exports
`results/tao2025_radial_u*.csv` — cement fraction across the annular gap,
ready to overlay on a CFD line probe.

### Where the two agree, and where they do not

Displacement efficiency, all twelve runs (100 x 13 x 8, 1.05 x annulus volume):

| geometry | yield-stress law | 0.5 m/s | 0.2 m/s | 0.05 m/s |
|---|---|---|---|---|
| smooth | exact HB (rigid plug) | 0.8717 | 0.8884 | **0.9133** |
| smooth | Fluent regularisation | **0.8424** | 0.8194 | 0.8087 |
| wavy | exact HB (rigid plug) | 0.8905 | 0.9072 | **0.9304** |
| wavy | Fluent regularisation | **0.8440** | 0.8247 | 0.8186 |

**Irregular beats smooth in both treatments** — the same direction Tao et al.
report ("efficiency … higher in Case-2"). Independent corroboration of the
profile-flattening mechanism, from a completely different method. The *margin*
is treatment-dependent though: +0.019 under the exact law, only +0.002 to
+0.010 under the regularisation, because with no plug there is less profile to
flatten.

**The velocity trend reverses with the yield-stress treatment, and this is the
result worth taking away.** Under the exact law the model is monotonic the
wrong way: slower flow means a larger plug (40 % → 50 % → 63 % of the gap) and
a flatter profile, so better displacement. Under Fluent's regularisation there
is no plug to grow, the profile only sharpens as the flow slows
(`u_max/ū` 1.373 → 1.469 → 1.494), and the ranking becomes
**0.5 > 0.2 > 0.05 — which is Tao et al.'s ranking.**

An earlier version of this README attributed that disagreement to buoyancy. On
this evidence that was premature: the discrepancy was the constitutive
treatment, and it goes away when this model integrates the yield stress the way
the CFD's solver does. Three points is not a validation, and buoyancy may still
matter — the densimetric Froude number `Fr = ū/√(At·g·h)` is **0.36, 0.14 and
0.036**, below one everywhere and falling, and this model has no buoyancy
mechanism at all (assumption A-29). But the yield-stress treatment now accounts
for the sign, so buoyancy is no longer needed to explain it.

The practical consequence for comparing against Fluent: **compare against the
regularised branch.** That is the law Fluent solves.

### Yield stress now follows Fluent

`regularisation_shear_rate` switches the constitutive law from the exact
Herschel–Bulkley (rigid plug, the in-pipe paper's form) to Fluent's regularised
one: viscosity capped below `γ̇_c` and **no plug anywhere**. On the Tao et al.
annulus at 0.5 m/s that is a first-order change, not a detail —

| | exact HB | Fluent regularisation |
|---|---|---|
| plug | **40 % of the gap** | none |
| `u_max/ū` | 1.175 | **1.373** |
| efficiency | 0.8717 | 0.8424 |

— and it acts precisely in the slow region where fluid is left behind. Two
findings about the published equations, both of which change numbers:

- **Eqs. (15)–(16) have their inequalities the wrong way round.** As printed the
  `τ_y/γ̇` branch is assigned to `γ̇ < γ̇_c`, where it *diverges*; the bi-viscosity
  branch is assigned to `γ̇ > γ̇_c`, where it goes *negative* above `2γ̇_c`.
- **Eq. (15) contradicts Eq. (14) and Table 1 by a factor of 2.8.** Normalising
  the consistency by `γ̇_c`, as Eq. (15) prints it, implies an effective
  `k = 1.669 Pa·sⁿ` rather than the 0.6 of Table 1. *Continuity does not settle
  this — both readings join continuously at `γ̇_c`.* What settles it is that only
  the literal form reproduces Eq. (14), and only it reduces to the exact law as
  `γ̇_c → 0`. Literal is the default; `normalise_consistency` selects the other.

### Two mismatches that remain

1. **Interfacial tension is carried but not modelled.** `σ` is now an input and
   its dimensionless groups are reported, but the transport is still miscible —
   there is no momentum equation for a surface-tension term to enter, and
   storing the number does not change that. It matters here: `Ca = μU/σ` is
   **0.07–1.3**, order one, so interfacial tension is shaping the CFD's
   interface. The value is also worth questioning — 0.07 N/m is the *water–air*
   surface tension; two aqueous wellbore fluids are nearer 0–1 mN/m.
2. **The "drilling fluid" is water** — 998 kg/m³, 1 cP, Newtonian. No yield
   stress, so it can never be stranded by failing to yield, and the viscosity
   ratio (thick displacing thin) is unconditionally stable.

## Mud left behind by the washouts

```bash
.venv/bin/python -m cases.mud_left_behind
```

Runs the same job three ways on the K-GEP-1 open hole and differences them.
Produces `results/mud_left_behind.png`, `mud_left_section.png` and
`mud_left_behind.mp4`.

| scenario | cement | efficiency | mud left |
|---|---|---|---|
| A — real hole, job sized on the caliper | 14.71 m³ | **89.5 %** | 1.04 m³ |
| B — in-gauge hole, same excess | 11.05 m³ | 82.1 % | 1.15 m³ |
| C — real hole, job sized on **gauge** (caliper ignored) | 11.05 m³ | **69.8 %** | **2.99 m³** |

**1. The cost of a washout is volume, not displacement.** The real annulus holds
+3.49 m³ (+54 %) more than an in-gauge hole. Design the job on bit size and you
are short by that much: efficiency drops 89.5 % → 69.8 % and the mud left behind
triples, from 1.04 to 2.99 m³.

**2. On this well the leftover mud concentrates in the enlargements** — washed-out
hole is 43 % of the annular volume but holds 66 % of the residual, 1.55× its
share, and that ratio is stable as more is pumped. **This is a property of this
well and this job, not a law about washouts:** on a synthetic log whose
enlargements sit mid-well instead of at the shallow end, the same model gives a
ratio below 1. Where the front has reached by the end of the job matters as much
as the geometry.

**3. Displacement itself is not worse in a washout** — at the same pore volumes
pumped the real hole reaches 89.5 % against 82.1 % in gauge, by the
profile-flattening mechanism below.

**Nothing is stranded permanently.** Pumping 2.5 well volumes clears the annulus
to 99.98 %, and 4× clears it exactly. Residual mud here always means *"not yet
swept at the volume pumped"*, never *"stuck"*. In a real concentric washout the
flow separates at the expansion and recirculates in the cavity, which strands
fluid however long you circulate — a two-dimensional effect a reduced-order model
with a fully-developed profile at every station cannot represent. **These numbers
are lower bounds on what a washout costs.**

### Do not read the raw washout comparison

Local efficiency correlates strongly with depth — annular flow is upward, so a
shallow cell is simply *reached last* and reads low at the end of the job
whatever its diameter. On this interval efficiency runs from 0.62 at 175 m to
0.997 at 373 m for that reason alone.

So a raw washout-vs-gauge comparison measures where the front is, not the
geometry, **and it flips the sign of the conclusion**: raw, it says washouts are
worse (0.880 vs 0.939); within depth bands the picture is mixed and dominated by
the shallowest band, where the washout is far *ahead* (0.812 vs 0.669). The case
prints both, and a test pins the confound so the raw number is never read alone.

The one exact statement is the mechanism, verified analytically: a wider gap
flows slower, the yield stress takes a larger share of the stress budget, the
plug grows from 30 % to 63 % of the gap, and the profile flattens from
`u_max/ū` = 1.25 to 1.12 — which displaces *better*. Whether that outweighs the
extra volume a washout holds depends on the well, and on this one it is close.
**Either way, the mechanisms that make real washouts fail — eccentricity,
density segregation into the cavity, mud stranded below its yield stress — are
all outside this phase.**

The annulus is solved by the **parallel-plate (slot) approximation**, which is
what the source paper says its own annulus model uses (Appendix A.1:
`τ_w = h/2·P` for plates against `τ_w = R/2·P` for a pipe). Its error against
the exact concentric-annulus solution is measured, not assumed: −0.31 % at
gauge, −1.7 % at a 400 mm washout.

**Caliper input.** CSV or LAS. Units come from an explicit argument, then from
the column name or the LAS `~C` unit field, and only then from the diameter
magnitude — which is unambiguous, since a borehole is either ~0.2 m or ~8.5 in.
Depth is *never* inferred from its own magnitude: a 656 ft log and a 656 m log
are indistinguishable that way, and guessing wrong scales the whole well by
3.28. A diameter that resolves outside 0.02–2 m is refused rather than returned.

### Three things the circulation model shows that are worth knowing

**The well would free-fall for 84 % of this job.** Gravity enters as hydrostatic
head and friction; the rate is the pump rate. But the report says that
assumption fails: once cement fills the casing, the required pump pressure goes
*negative*, peaking at a 9.4 bar (136 psi) U-tube imbalance. A real well would
take fluid on its own and return faster than pumped. `is_free_falling` flags it
every step. The swept geometry at a given *pumped volume* is still meaningful;
the *timeline* is not.

**Washouts do not degrade displacement in this model** — they marginally improve
it (86–88 % either way). The mechanism is real: a wider gap flows slower, so the
yield stress takes a bigger share of the stress budget, the plug grows from 30 %
to 63 % of the gap, and the profile flattens from `u_max/ū` = 1.25 to 1.12. A
flatter profile displaces better. Every mechanism that makes real washouts bad —
eccentricity, density segregation into the cavity, mud stranded below its yield
stress — is outside this phase. **If the point of modelling enlargements is to
predict where cement fails, eccentricity is the missing piece**, and it is the
one the paper's stratified grid exists to represent.

**Eq. A.7's printed sign is wrong for the casing.** Wall shear opposes motion, so
friction costs pressure *along the flow*: `dp/dz = ρg cos β − 2τ_w/R` going down,
`+` going up. The paper prints the upward-flow sign; using it for a downward
casing over-states the shoe pressure by twice the friction.

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
  solver.py       Module 6 - the in-pipe time loop and diagnostics
  slot.py         annular flow by the parallel-plate approximation
  caliper.py      caliper log reading (CSV/LAS) and unit resolution
  annulus_grid.py annular mesh with a depth-varying cross-section
  hydraulics.py   hydrostatic head, friction, ECD, the U-tube imbalance
  circulation.py  the coupled casing-and-annulus solver
  wellview.py     well-section rendering and job animation
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

| `results/field_circulation.mp4` | the K-GEP-1 job as an animation - cement down, round the shoe, up the annulus |
| `results/field_circulation_sections.png` | well sections at four times, on the real caliper |
| `results/field_circulation_history.png` | hole size, local efficiency, efficiency and pressure against time |

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
