# Assumption register

Reconstruction of Dai, H., Eslami, A., Schneider, J., Liu, G., Schwering, F.
(2023), *Modeling displacement flow inside a full-length casing string for well
cementing*, **Petroleum Research 9**, 1–16.

Every decision that the paper does not pin down is recorded here. This register
is what separates a defensible reconstruction from an undocumented guess.

Legend for **Sensitivity tested?**: `no` = not yet; `yes` = a test or study
quantifies the impact; `n/a` = the choice is exact, so there is nothing to vary.

---

## A. Choices carried over from the build spec

| ID | Paper ref | What the paper specifies | What is missing | Our choice | Justification | Sensitivity tested? |
|----|-----------|--------------------------|-----------------|------------|----------------|---------------------|
| A-01 | Eq. A.2 | `Q = ∫_A u(r) dA` | No quadrature rule; the integrand has a derivative discontinuity at the plug radius `r0` | Split the integral at `r0`: analytic plug term `B·π·r0²` plus `scipy.integrate.quad` on `r ∈ [r0, R]` (`epsrel=1e-13`) | Splitting keeps each piece smooth, so adaptive quadrature converges to machine precision. Verified against Buckingham–Reiner to rel 1e-8 and against Poiseuille / power-law peaks to rel 1e-10 | n/a (exact to stated tolerance) |
| A-02 | Eq. A.7 | `dp/dz − ρ g cos β = 2 τ_w / R` | Sign convention for `z` is never stated | `+z` is the flow direction (downward inside casing for a vertical well); `β` is inclination from vertical, so `β = 0` ⇒ `cos β = 1` | A single convention stated in `velocity.py`'s module docstring and used everywhere. Getting this wrong silently inverts the hydrostatic term | n/a |
| A-03 | §2.2, Fig. 2b/3a | Cross-section is cut by straight horizontal lines into layers | Layer spacing rule is not given | Uniform spacing in the vertical chord coordinate `y ∈ [−R, R]` | Simplest rule consistent with the figures. **Known consequence:** near-wall layers are thin in *area*, so they are under-resolved relative to the core. The rule is injected as a strategy (`GridConfig.layer_rule`) so an equal-area alternative can be swapped in without touching the solver | yes — see the convergence table below |
| A-04 | §2.3 | 1D radial profile is applied "to the entire section axisymmetrically" | How the 1D `u(r)` is sampled onto finite-volume cells | **Exact per-cell area averaging** (`NumericsConfig.velocity_mapping = "area_average"`, the default). Centroid-radius evaluation is retained as `"centroid"` | The spec's baseline was centroid evaluation, with the rule "if the flow-rate error is worse than ~1 % at your working resolution, switch to area-averaged velocity". **It is worse.** Measured `Q` error at the paper's own 13 × 18 cross-section (Dai et al. §3.1 used a 100 × 13 × 18 mesh): Newtonian +0.54 %, power-law `n = 0.4` +0.71 %, Herschel-Bulkley `τ0 = 3 Pa` **+1.38 %** — the yield-stress case fails the gate, and it fails in the direction that matters (a systematic *over*-estimate of `Q`, because the centroid radius of a cell under-states its area-weighted mean radius while `u` is convex in `r`). Area averaging brings the error to 4 × 10⁻⁶ relative. Cost is kept negligible by precomputing an exact cell-by-annulus area matrix once (see A-20), so the mapping is one profile evaluation plus a matvec (~0.16 ms) per station per step | yes — both mappings measured at three resolutions for three rheologies; `tests/test_grid.py` prints the centroid errors on every run |
| A-05 | Eq. A.9 | Explicit Euler in time | No stability constraint or Courant number is given | `Δt < CFL · Δz / max|u|` with `CFL = 0.4`, configurable via `NumericsConfig.cfl`; the condition is asserted every step and raises on violation | Explicit upwind advection is stable for `CFL ≤ 1`; 0.4 leaves margin for the velocity changing between steps as the effective fluid changes. Never clipped silently — a violation is a modelling error, not something to hide | no |
| A-06 | §A.1 | "averaged rheological parameters and density of fluids are used" | The averaging rule is not specified | Volume-fraction-weighted arithmetic mean of `ρ`, `τ0`, `k` and `n` (`fluid.mix_fluids`) | The only rule the wording plainly supports. **Flagged as not physically rigorous for `n`:** the flow index is an exponent, not an extensive property, so a volume-weighted mean of `n` has no constitutive justification. A mixture of an `n = 0.4` and an `n = 1.0` fluid is *not* an `n = 0.7` fluid. Prime candidate for a sensitivity study | no |
| A-08 | Eq. A.9 | First-order face values are implied by "assuming uniform distribution of fluid concentration ... on each face" | Nothing beyond first order | Phase 1 baseline is first-order upwind, deliberately diffusive. The face-value function is injected (`advect(..., face_scheme=...)`) so Phase 2 can drop in donor–acceptor and THINC without touching the solver | The paper achieves sharp interfaces by "axial interface reconstruction", which it does not describe. Reproducing an undescribed scheme would be a guess; reproducing the *baseline* and measuring its numerical diffusion is the honest starting point, and it is the entry point to the Phase 2 contribution | yes — `Dm_num` measured vs. `Δx` in `tests/test_transport.py` and `test_integration.py` |
| A-10 | Eqs. A.10, A.14–A.18, Eqs. 3–6 | Segregation, flow-regime instability criteria and pipe rotation | — | **Deliberately not implemented in Phase 1** | The target application is a vertical well (`β = 0`). Every buoyancy criterion degenerates there: `v_t = √(At·g·sin β·D) = 0`, so `Fr = u/v_t` is undefined and `Re_t = ρ v_t D / μ = 0`, which fails `Re_t > 1` (Eq. A.18) — segregation can never activate. Eqs. A.10/A.14/A.15 all carry `cos β = 1` on the left and a degenerate `Fr` on the right. Pipe rotation acts only through segregation and mixing, both inactive | n/a (structurally inactive at `β = 0`) |

## B. Additional decisions made during implementation

| ID | Paper ref | What the paper specifies | What is missing | Our choice | Justification | Sensitivity tested? |
|----|-----------|--------------------------|-----------------|------------|----------------|---------------------|
| A-07 | Eq. A.8/A.9 | Conservative finite-volume update of `f_i` | Nothing about what happens when the axial velocity profile varies with depth | **Transverse redistribution closure** (`NumericsConfig.transverse_closure = "redistribute"`, the default). The two alternatives, `"none"` (Eq. A.9 verbatim) and `"local"` (subtract `f·∇·u`, i.e. the advective form of Eq. 2), remain selectable | **This resolves a genuine inconsistency in the source model, not a transcription error.** With no transverse velocity, each `(layer, azimuth)` column carries its own axial velocity; when neighbouring stations hold different effective fluids their *profile shapes* differ, so a column has `∂u/∂z ≠ 0` even though every station passes the same `Q`. The discrete continuity condition `Σ_j u_j A_j = 0` then fails per cell, and the conservative update and the sum-to-one constraint become mutually incompatible. See the measured table below. The resolution: because `Σ_c (∇·u)_c A_c = 0` across the cross-section, the imbalance is a *redistribution*, not a source — columns losing volume axially shed it laterally carrying their own composition, and columns gaining volume receive the donor mixing-cup composition. Both invariants then hold to round-off. **Physical content, stated explicitly:** this assumes lateral redistribution is instantaneous and well-mixed across the section. That is a strong assumption, of the same character as the paper's own algorithmically-imposed segregation and instantaneous mixing — it is *not* a solved transverse velocity, and no momentum equation is involved. It is inert wherever `∇·u = 0`, so it changes nothing in any single-rheology case | yes — all three closures measured, see below |

### Layer-spacing sensitivity (the A-03 result)

Displacement efficiency on the 200 m field case (100 axial cells, mud →
spacer → cement), varying both the layer count and the layering rule:

| `n_layer` | rule | outermost two layers, % of area | residual mud [m³] | efficiency |
|---|---|---|---|---|
| 13 | `uniform_y` | 7.07 % | 0.3128 | 87.65 % |
| 26 | `uniform_y` | 2.53 % | 0.3200 | 87.37 % |
| 52 | `uniform_y` | 0.90 % | 0.3221 | 87.29 % |
| 13 | `equal_area` | 15.38 % | 0.2921 | 88.47 % |
| 26 | `equal_area` | 7.69 % | 0.3187 | 87.42 % |

The headline number is **mesh-converged and rule-insensitive**: efficiency
moves by 0.36 points across a 4× refinement in `n_layer` and lands in the same
place from a completely different layering rule. That is because the residual
volume is set by the integral of the parabolic profile, not by how the near-wall
region is cut. So the concern flagged against `uniform_y` — thin, under-resolved
near-wall layers — does **not** propagate to the aggregate result.

What it does affect is the *radial distribution* of the residual. At
`n_layer = 13` the two outermost layers hold 31 % of the leftover mud while
covering only 7 % of the area; at `n_layer = 52` that concentration is spread
over thinner layers. Any downstream use that cares where the residual sits
(rather than how much there is) should refine `n_layer` or switch to
`equal_area`; anything reading off a single efficiency number can use 13.

### The conservation dilemma and its resolution (the A-07 result)

Measured on the 200 m × 5 in field case (mud → spacer → cement, three distinct
Herschel–Bulkley rheologies, 1257 steps):

| closure | max &#124;Σᵢfᵢ − 1&#124; | per-fluid mass-budget error | max `f` |
|---|---|---|---|
| `"none"` — Eq. A.9 as written | **0.39** | 1.6e-15 | **1.101** |
| `"local"` — advective form of Eq. (2) | 2.8e-14 | **1.8e-2** | 1.000 |
| `"redistribute"` — default | 2.8e-14 | < 1e-12 | 1.000 |

Neither of the first two is usable at field scale. `"none"` conserves each
fluid perfectly but the fractions stop being a partition of the cell — a 39 %
violation, with `f` exceeding 1 by 10 %, which would corrupt any displacement-
efficiency number read off the result. `"local"` keeps a valid partition but
creates and destroys 1.8 % of each fluid. Both are *exact* in every
single-rheology case, which is why the inconsistency is invisible in the
paper's own iso-fluid validation cases (§3.1, Figs. 4–5) and only appears once
several cementing fluids are in the pipe at once — i.e. in the model's actual
intended application.

| A-09 | §2.2 | "in a vertical pipe, [the central longitudinal] grid line becomes horizontal in space and its direction is not uniquely defined" | Which direction to pick | Fixed reference azimuth `0` rad (the `+x` axis), exposed as `GridConfig.reference_azimuth` | For a vertical well the choice is physically arbitrary, exactly as the paper says: with `β = 0` nothing in the model depends on the cross-sectional orientation. The mesh remains valid; only its labelling is a convention. Fixing it makes runs reproducible | n/a (physically arbitrary at `β = 0`) |
| A-11 | Fig. 3a | Each layer is cut by "longitudinal lines" | Whether the longitudinal lines are global (shared `x` positions across all layers) or per-layer | Per-layer: within each layer, the layer's full `x`-extent `[−w, +w]` is divided into `n_azimuth` equal-width columns, where `w` is the layer's half-width | Required by the `n_axial × n_layer × n_azimuth` structured array shape the spec mandates — global lines would give a ragged number of cells per layer. Guarantees the union of cells tiles the disc exactly, hence exact area/volume sums | n/a |
| A-12 | Fig. 3a | Cell cross-sectional areas | Method of computation | Closed-form circular-segment integration: `A = ∫ [min(y_hi, √(R²−x²)) − max(y_lo, −√(R²−x²))]₊ dx`, evaluated analytically on sub-intervals split at the circle-intersection breakpoints. First and second moments (for centroids) use the same closed forms | The spec forbids Monte Carlo. Analytic evaluation makes `Σ A = πR²` exact to round-off at *any* resolution, which is the stated test gate (exact, not merely convergent) | n/a (exact) |
| A-13 | Eq. A.3, A.6 | `u(r) = A[T(r)/k]^{…} + B` | The PDF renders the exponent as `1/(n+1)` | Use `1/n + 1` | The paper's PDF text of Eqs. A.3–A.6 is OCR-corrupted. Carrying out the integration `−du/dr = ((τ−τ0)/k)^{1/n}` from `r` to `R` yields the exponent `1/n + 1`. Confirmed empirically: `1/n + 1` reproduces Poiseuille (rel 1e-10), the power-law peak `(3n+1)/(n+1)` (rel 1e-8) and Buckingham–Reiner (rel 1e-8); `1/(n+1)` reproduces none of them | n/a (verified against three closed forms) |
| A-14 | Eq. A.9 | `f^{n+1} = f^n + (Δt/ΔV) Σ_j u_{n,j} f_{s,j}` | The equation as printed is dimensionally wrong (no face area `A_j`) and has the wrong sign for outward normals | `f^{n+1} = f^n − (Δt/ΔV) Σ_j (u_{n,j} · A_j · f_{s,j})` | OCR corruption again. The face area is required for dimensional consistency with Eq. A.8 (`∫_s u_n f_s dS`), and with outward-positive face normals an outflow must *decrease* the cell content, so the sign is negative | n/a |
| A-15 | §A.1 | Brent's root find for the pressure gradient | Brackets and tolerances | Lower bracket `τ0·(1+1e-9)`; upper bracket expanded geometrically (×2, ≤200 iterations) from an analytical power-law inversion; `xtol = rtol = 1e-12` | `Q(τ_w)` is zero for `τ_w ≤ τ0` and strictly increasing above it, so a sign change is guaranteed once the upper bracket is large enough. A failed bracket raises `NoBracketError` rather than returning a silent wrong answer | n/a |
| A-16 | §2.4, Eq. (2) | `∂f_i/∂t + u·∇f_i = D_m ∇²f_i`, with `D_m` term dropped | — | Right-hand side dropped, matching the paper. Physical diffusion is *not* modelled in Phase 1 | The paper's own justification is quantitative: at their resolution (`Δx = 30 m`, `Δt = 0.1 min`) numerical diffusion `Δx²/Δt ≈ 150 m²/s` swamps physical `D_m ≈ 1e-3…1e-4 m²/s` by ~1e5, so the physical term is meaningless. **This justification weakens as `Δx` falls** — at `Δx ≈ 1 m` the ratio drops by ~1e3 — which is precisely why `Dm_num` is measured and reported here. Restoring the term is Phase 3 | yes — see the measured numbers below |

### Measured numerical diffusion (the A-08/A-16 result)

The paper discards the physical diffusion term because at *their* resolution it
is swamped by numerical diffusion. Reconstructed at fine resolution, that
justification inverts. Measured on the 4 m / 19 mm parabolic-stretching case,
first-order upwind at `CFL = 0.4`, fitting `D` from the variance of `−∂f/∂z`
via `Var = 2Dt`:

| `Δz` [m] | `Δt` [s] | Courant | `Δx²/Δt` (paper's estimate) | upwind modified equation `½u Δz(1−C)` | fitted from the front |
|---|---|---|---|---|---|
| 0.0200 | 0.0805 | 0.043 | 4.97e-3 | 1.01e-4 | 1.06e-4 |
| 0.0100 | 0.0402 | 0.043 | 2.48e-3 | 5.07e-5 | 5.20e-5 |
| 0.0050 | 0.0201 | 0.043 | 1.24e-3 | 2.54e-5 | 2.57e-5 |

Two things follow, and both matter for the thesis contribution:

1. **The paper's `Δx²/Δt` is the wrong scaling for this scheme.** For upwind
   plus explicit Euler at a *fixed* Courant number `C`, the true leading-order
   numerical diffusivity is `½u Δz(1−C)`, which is linear in `Δz`, not
   `Δx²/Δt`. The two agree only in order of magnitude at the paper's own coarse
   operating point; `Δx²/Δt` over-states the smearing by a factor of `2/(C(1−C))`
   — here roughly 50×. Measured and predicted values agree to ~2 %.
2. **At `Δz ≈ 1 cm` the numerical diffusivity has fallen *below* the physical
   `D_m ≈ 1e-3 … 1e-4 m²/s`** (ratio 0.026 at `Δz = 5 mm` against `D_m = 1e-3`).
   The paper's ratio is ~1e5 the other way. So the grounds for dropping the
   right-hand side of Eq. (2) do not survive mesh refinement, and Phase 3
   (restoring finite-rate diffusive mixing) is not optional polish — it is
   required for the refined model to mean anything.
| A-17 | §2.1, A.4 | Mixing status `s` is set to 1 where instability is detected | — | `s` is advected with the same kernel and scheme as concentration, but **nothing sets `s = 1` in Phase 1**; it stays at its initial value | Instability detection (Eqs. A.10, A.14–A.16) is out of scope for Phase 1, and at `β = 0` two of the three criteria degenerate anyway. The plumbing exists so Phase 3 has somewhere to write | n/a |
| A-18 | §6.3 (spec) | — | Outlet condition | Zero-gradient (`∂f/∂z = 0`) outflow at the shoe; inlet is Dirichlet on `f` from the pump schedule with `Q` imposed | Standard outflow condition for pure advection with `u > 0`: the outlet face value equals the last interior cell, so no information propagates upstream | n/a |
| A-20 | §2.3 | — | How to make exact area averaging cheap enough for the time loop | Because `u = u(r)`, the cell integral is a linear functional of the 1D profile: `∫_cell u dA = ∫₀ᴿ u(r)·(dA_cell/dr) dr`. Precompute `W[cell, k] = area(cell ∩ disc(r_{k+1})) − area(cell ∩ disc(r_k))` exactly, once per grid, with `n_radial = 1024` annular bins; at run time apply `W` by matvec | The weights are pure geometry, so they are built once and reused for every station and every timestep. Verified against direct 2D Gauss–Legendre quadrature and shown to converge second-order in `n_radial` (`tests/test_grid.py`). The annulus areas `Σ_cells W[:, k]` reproduce `π(r_{k+1}² − r_k²)` to 1e-13 | yes — `n_radial` convergence measured |
| A-22 | §A.1 | `Q(z)` is imposed and uniform | Nothing about the discrete consistency of the mapped cell velocities | Rescale each station's mapped velocity field by a single scalar so `Σ_cells u·A = Q` exactly (`NumericsConfig.enforce_discrete_continuity`) | The area-averaged mapping is already accurate to ~4e-6 relative (A-04), so this is a tiny correction — but it makes `Σ_c (∇·u)_c A_c` vanish to round-off rather than to 4e-6, which is exactly what the `"redistribute"` closure needs in order to conserve to 1e-12 instead of 1e-6. A scalar rescale cannot change the profile shape, so it introduces no new physics | n/a |
| A-23 | Eq. A.2 | `Q = ∫_A u dA`, evaluated by Brent iteration on the pressure gradient | — | Use the **closed-form** `Q(τ_w)` for Herschel–Bulkley obtained by integrating Eq. A.2 by parts, rather than the spec's adaptive quadrature. The quadrature version is kept as `flow_rate_quad` | The `τ_w` root find sits inside the time loop at every station holding mixed fluids; the quadrature version dominated field-scale runtime (`solve_tau_w` went from ~10 ms to 24 µs, ~400×). The closed form is *exact*, not an approximation: `Q = πR³(Tm/k)^{1/n}·a·[a²n/(3n+1) + 2xa·n/(2n+1) + x²n/(n+1)]` with `Tm = τ_w − τ0`, `x = τ0/τ_w`, `a = 1 − x`. Asserted equal to the quadrature to rel 1e-10 across six rheologies and five stress levels, and shown to reduce to Hagen–Poiseuille and to the power-law result in closed form | n/a (exact, cross-checked) |
| A-19 | §3.1 | The paper's CFD comparison case is at `β = 83°` from vertical | — | The 4 m / 19 mm integration case here is run **vertical** (`β = 0`), mirroring the paper's *geometry* only | At `β = 83°` the paper's result is dominated by segregation and backflow (Fig. 5), which requires the buoyancy machinery that Phase 1 deliberately omits (A-10). **Consequence: paper Fig. 5 is not quantitatively reproducible in Phase 1** — only its centre-plane *view* and the concentric parabolic-stretching mechanism are. This is a scope limit, not a validation failure | n/a |


---

## C. The annulus leg and the irregular wellbore (added after Phase 1)

The circulation model — cement down the casing, round the shoe, up the annulus
past a caliper-measured hole — goes beyond the source paper, which models the
casing only and states that it "connects an existing annulus model" (Dai & Liu,
2018) that it does not describe. These rows record what was built in its place.

| ID | Paper ref | What the paper specifies | What is missing | Our choice | Justification | Sensitivity tested? |
|----|-----------|--------------------------|-----------------|------------|----------------|---------------------|
| A-24 | §A.1 | "Very similar equations … are used for calculating the flow in annulus (assuming parallel plates) … we have `τ_w = h/2·P` for flow between plates with a gap width `h`" | The annulus model itself is in a separate, undescribed paper | **Parallel-plate (slot) approximation**, half-gap `b = (r_o − r_i)/2`, width `W = π(r_o + r_i)`. The velocity profile is then *identical in form* to the pipe profile with `R → b`, so `velocity_profile` is reused unchanged; only the flow-rate integral differs and it has the same kind of closed form | This is the one sentence the paper gives about its annulus, so it is the faithful choice. The slot area `2bW` equals the true annular area `π(r_o²−r_i²)` exactly, by construction. The approximation's own error is measured, not assumed: **−0.31 %** against the exact concentric-annulus solution at 5½ in casing in an 8½ in hole, and still only **−1.7 %** at a 400 mm washout | yes — `slot_error_estimate` measures it at four hole sizes |
| A-25 | Eq. A.7 | `dp/dz − ρ g cos β = 2 τ_w / R` | — | `dp/dz = ρ g cos β − flow_sign · 2τ_w/R`, with `flow_sign = +1` down the depth axis (casing) and `−1` up it (annulus) | **The paper's printed sign is wrong for its own geometry.** Wall shear opposes the motion, so it costs pressure *along the flow*: a force balance gives `−dp/dz + ρg − 2τ_w/R = 0` for downward flow. The printed form is the upward-flow (annulus) sign, and taking it literally for the casing over-states the shoe pressure by twice the friction. The two legs now carry opposite signs explicitly | n/a (force balance) |
| A-26 | — | — | What happens to the radial structure at the shoe | The casing outlet's **flux-weighted mixing cup** becomes a uniform annulus inlet | The flow reverses through the shoe and float equipment over a length no reduced-order model resolves. Any radial structure in the casing is lost at the turn, and the annulus develops its own profile from a uniform inlet. This under-states contamination if the casing front is strongly stretched at breakthrough | no |
| A-27 | — | Uniform pipe diameter throughout | Nothing about a varying cross-section | Annular cell areas, volumes and axial face areas all vary with depth from the caliper; the transport kernel carries face areas explicitly rather than cancelling them | The paper's Eq. A.9 is written with face areas, so this is the general form and the uniform pipe is its special case. **The face areas are the subtle part:** flux crosses *faces*, whose areas differ from the cell areas either side where the hole diameter varies, so a velocity field normalised at cell centres does not carry `Q` through the faces between them. Left uncorrected this broke `Σ f_i = 1` by 2.4 %. `AnnulusGrid.normalise_face_flux` enforces it — physics, since the well is incompressible and every axial face passes the same rate | yes — `test_face_flux_normalisation_is_exact` |
| A-28 | Eq. A.7 | — | Whether gravity feeds back into the flow | Gravity enters as **hydrostatic head and friction only**. The flow rate is the pump rate; the U-tube imbalance is computed and reported but does not drive the flow | Chosen scope for this phase. **The report is not decoration — it says the assumption fails here.** On the 200 m case the casing column becomes heavy enough that the required pump pressure goes *negative* for **84 % of the job**, peaking at a 9.4 bar (136 psi) imbalance: a real well would free-fall and return faster than pumped. `HydraulicsReport.is_free_falling` flags it every step. Until U-tube hydraulics are added, the displacement *timeline* from this model is not the real one, though the swept geometry at a given pumped volume still is | yes — measured over the whole job |
| A-29 | §A.3 | Segregation is inactive at `β = 0` because every criterion carries `sin β` | That is only true of *transverse* buoyancy | Axial density instability is **not** modelled, and is flagged rather than assumed away | At `β = 0` transverse buoyancy vanishes but **axial buoyancy is maximal**. Heavy cement over lighter mud flowing *down* the casing is Rayleigh–Taylor unstable and should finger; the same pair flowing *up* the annulus is stable, which is why cementing works at all. The paper's criteria are blind to this because they are built on `sin β`. Correcting the earlier claim in this register that "every buoyancy criterion degenerates" — the transverse ones do, the axial one does not | no |
| A-30 | — | — | Whether the displaced fluid can yield at all | Not enforced; **reported** by `CirculationResult.yield_diagnostic` | A displaced fluid only moves where the flow yields it; below its yield stress it is immobile however long the job runs — the unyielded-mud channel of annular displacement theory. This model averages rheology per station and solves one profile, so it will "displace" fluid a real well would strand. On the case here the cement's own yield stress floors `τ_w` at 6.8–20 Pa, well above the mud's 2 Pa, so nothing is stranded; with a stiffer mud or a lower rate it would be | yes — diagnostic tested at both extremes |

| A-31 | — | — | How to size a casing string against a measured hole | **7 in OD (177.8 mm) / 6.184 in ID (157.1 mm), 29 lb/ft**, against the log's 10.43 in (265 mm) gauge hole | The standard string for a 10⅝ in hole; nominal annular clearance 1.71 in. Checked against the log rather than assumed: the annulus stays open everywhere, 15.1 mm at the tightest point (a 8.15 in under-gauge interval at 365–375 m) and 43.6 mm at gauge. A 6⅝ in string would give 19.4 mm at the tightest, a 5½ in string 33.7 mm — both selectable via `--casing-od-in` | yes — three sizes compared |
| A-32 | — | — | Where a real caliper log stops being geometry | The bundled log's last **3.2 m (386.76–389.95 m)** is cut: the reading drops abruptly from 13.3 in to 2.1 in and stays there, which is the tool bottoming out on fill. Well depth is therefore **386.75 m** | Detected by `implausible_tail`, applied explicitly by the case and printed on every run, never silently. Above the cut nothing else in 38 557 samples falls below 8.15 in, so this is one clean block rather than a threshold sweeping up scattered noise | n/a |
| A-33 | — | — | Whether the whole logged interval is open hole | Treated as open hole throughout, **as instructed** — but flagged, because the log says otherwise | **The top 175 m is almost certainly not open hole.** Its diameter has a standard deviation of **0.0072 in (0.18 mm)** across 17 381 samples, against **3.05 in** in the section below 195 m — a factor of **424**. Rock does not do that. Either the interval is cased and the tool is reading a casing ID, or the caliper is clipped at bit gauge. If it is cased, the annulus there is between the 7 in string and that casing's ID, not an open hole, and the annular volume and efficiency over the top 45 % of the well are wrong. Resolving this needs the well's casing record, not the log | no |

| A-34 | — | — | Which part of the logged well to model | **175 m to the shoe at 386.75 m**, the open hole only (`--top-depth`) | Follows A-33: above 175 m the log is not reading rock, so its annulus is not an open hole. Modelling only the interval keeps the geometry honest. The column above is still carried hydrostatically so shoe pressure and ECD stay true-depth quantities — the annulus above is taken to remain mud (true while cement fills only the open-hole annulus), the casing above is volume-averaged over what has been pumped through it. **Friction above the interval is not included**, so pump pressure is a lower bound by that amount | yes — both intervals run and compared |
| A-35 | — | — | How to compare displacement in washed-out against in-gauge hole | Report the raw comparison **and** a paired within-depth-band comparison, never the raw one alone | **The raw comparison flips the sign of the conclusion.** Annular flow is upward, so a shallow cell is reached last and reads low at the end of the job whatever its diameter — local efficiency runs 0.62 at 175 m to 0.997 at 373 m on that account alone. Raw, washouts look worse (0.880 vs 0.939); within depth bands the shallowest band has the washout far ahead (0.812 vs 0.669) and the rest are near-ties. Averaging *levels* across bands re-imports the trend, because washout and gauge volume are not spread the same way over depth, so the aggregate averages the per-band *differences* instead. The honest summary is that the field-scale sign is well-specific and close; only the mechanism (A-24 table below) is exact | yes — pinned by `test_local_efficiency_is_confounded_by_arrival_order` |

| A-36 | — | — | Whether anything strands fluid permanently | Nothing does. Residual displaced fluid is always *"not yet swept at the volume pumped"* | Verified rather than assumed: on the K-GEP-1 open hole, pumping 1.05 well volumes leaves 1.27 m³ of mud, 1.5× leaves 0.22 m³, 2.5× leaves 0.0019 m³ and 4× leaves exactly zero. Transport is axial only and every column moves at a finite speed, so given enough circulation everything flushes. **In a real concentric washout the flow separates at the expansion and recirculates in the cavity, stranding fluid however long you circulate.** That is a two-dimensional (radial and axial) effect, and a reduced-order model carrying a fully-developed profile at every station cannot represent it. Every residual-mud number this model produces is therefore a *lower bound* | yes — measured at four pumped volumes |
| A-37 | — | — | Whether enlargements hold more than their share of the residual | Reported per run; **not claimed as a general result** | On K-GEP-1 washed-out hole is 43 % of the annular volume and holds 66 % of the residual mud (1.55×), and the ratio is stable as more is pumped, so it is not purely an arrival artefact. But on a synthetic log whose enlargements sit mid-well rather than at the shallow end the same model gives **0.54** — washouts hold *less* than their share. The ratio is a property of a well *and* a job, because where the front has reached by the end matters as much as the geometry. Both behaviours are pinned by tests so neither can be quoted as the general case | yes — two logs, three pumped volumes |

### Washouts do not degrade displacement in this model — and why that matters

Field experience says washed-out intervals cement badly. This model says the
opposite, by a small margin (86–88 % efficiency either way). The mechanism is
real and worth understanding before dismissing it: for a fixed pumped rate a
wider gap flows *slower*, so the yield stress takes a larger share of the stress
budget, the plug grows, and the profile flattens toward slug flow —

| hole | gap | `ū` [m/s] | `u_max/ū` | plug / gap |
|---|---|---|---|---|
| 8½ in (gauge) | 38 mm | 0.62 | 1.247 | 0.30 |
| 10 in | 55 mm | 0.39 | 1.203 | 0.40 |
| 12 in | 80 mm | 0.24 | 1.159 | 0.51 |
| 14.8 in | 118 mm | 0.14 | 1.118 | 0.63 |

— and a flatter profile displaces better. So a concentric, buoyancy-free
washout is *locally* easier to displace, not harder. It still holds more mud in
absolute terms, simply by being bigger.

**Every mechanism that makes real washouts bad is outside this phase:**
eccentricity (the casing is not centred, and the narrow side is bypassed),
density segregation into the enlarged cavity, and mud left below its yield
stress in the low-shear pocket (A-30). If the purpose of modelling
enlargements is to predict where cement fails, at least eccentricity is
needed — it is the one the paper's own stratified grid exists to represent.


---

## D. Yield-stress treatment and the CFD comparison

| ID | Ref | What the source specifies | What is missing | Our choice | Justification | Sensitivity tested? |
|----|-----|---------------------------|-----------------|------------|----------------|---------------------|
| A-38 | Tao et al. (2025) Eqs. 14–16 | Herschel–Bulkley with a critical shear rate `γ̇_c = 5.5 s⁻¹` | Which branch applies where, and whether `k` is normalised | **Fluent-style regularisation**, selectable via `NumericsConfig.regularisation_shear_rate`; `None` keeps the exact law with a rigid plug (the Dai et al. form and still the default for well cases) | Under regularisation the viscosity is capped below `γ̇_c` instead of infinite, so **there is no plug**: every point shears at any stress. On the Tao et al. annulus at 0.5 m/s that changes `u_max/ū` from **1.175 to 1.373**, removes a plug occupying **40 % of the gap**, and drops displacement efficiency from 0.8717 to 0.8424 — a first-order change in the profile, concentrated exactly in the slow region where fluid is left behind | yes — both treatments run and compared |
| A-39 | Tao et al. Eqs. 15–16 | The two viscosity branches | The inequalities are printed the wrong way round | Swap them | As printed, the `τ_y/γ̇` branch is assigned to `γ̇ < γ̇_c`, where it **diverges** as `γ̇ → 0` and so cannot be a regularisation; and the bi-viscosity branch is assigned to `γ̇ > γ̇_c`, where it goes **negative above 2γ̇_c**. Neither is usable as printed; swapping is the only reading that gives a working model | n/a |
| A-40 | Tao et al. Eq. 15 vs Eq. 14 | `k(γ̇/γ̇_c)^(n−1)` in Eq. 15; `τ = τ_y + kγ̇ⁿ` with `k = 0.6` in Eq. 14 and Table 1 | Which is authoritative | Keep **`k` literal** (`k γ̇^(n−1)`), matching Eq. 14 and Table 1; `normalise_consistency=True` follows Eq. 15 instead | **I first assumed continuity settled this. It does not — both forms join continuously at `γ̇_c`**, so continuity is no discriminator. What settles it is Eq. 14: the normalised reading implies an effective consistency of `k γ̇_c^(1−n)` = **1.669 Pa·sⁿ against the 0.6 of Table 1**, a factor of 2.8, so the paper's Eqs. 15–16 contradict its own Eq. 14. The literal form is also the only one that reduces to the exact law as `γ̇_c → 0` (verified to 3e-8); under the normalised form `k γ̇_c^(1−n) → 0` for `n < 1`, so shrinking `γ̇_c` thins the fluid away instead of sharpening the plug | yes — both readings selectable, limit behaviour tested |
| A-41 | Tao et al. Table 1 | `σ = 0.07 N/m` between slurry and drilling fluid | — | **Carried and reported, not modelled.** `InterfaceConfig.surface_tension` stores it and reports `Ca` and `Bo`; the transport remains miscible | This solver advects volume fractions with no momentum equation, so no interfacial-tension term can enter — adding the number does not make the model immiscible, and it would be wrong to imply otherwise. Reporting it makes visible *when* the miscible assumption stops being defensible: on the Tao et al. case `Ca = μU/σ` is **0.07–1.3**, order one, so interfacial tension is shaping their interface and this model is blind to it. The value itself is also questionable — 0.07 N/m is the *water–air* surface tension; two aqueous wellbore fluids are nearer 0–1 mN/m and largely miscible | n/a |
| A-44 | — | — | What "rising time" means when the flow rate is imposed | Report **two** curves: the front (0.5 contour of the local cement fraction) and the volumetric arrival (cumulative pumped volume against casing plus annulus volume below that depth, i.e. plug displacement) | They are not the same number and the difference is the model's contribution. The front runs **9–10 % ahead** of the volume balance on both the synthetic and the K-GEP-1 case, because roughly half the annular area moves faster than the mean. That lead does **not** shrink under mesh refinement (−10.2 % at 60 axial stations, −11.1 % at 120, −11.7 % at 240), so it is the velocity profile, not numerical diffusion. Reporting only one of the two would hide either the caliper effect or the profile effect | yes — mesh refinement, and the volumetric curve is pinned against volume/rate to 1e-9 |
| A-50 | Xue et al. (2022) *J. Pet. Sci. Eng.* 208:109393 | Wide side turbulent, narrow side laminar in a cementing annulus; assuming one regime everywhere "will lead to serious model error" | This solver integrates a **laminar** Herschel–Bulkley profile everywhere and has no turbulence model | **Report the regime, do not model it.** `inpipe/regime.py` computes `Re = ρVD_h/μ_eff` with `μ_eff = τ_w/γ̇(τ_w)` from the solver's own law, per station, at every diagnostic step; laminar below 2100, turbulent above 4000 | The answer depends entirely on the mud, which is not yet known: with the placeholder mud the peak over the job is **Re 875 casing / 678 annulus** and the laminar profile is sound; with the mud taken as water — which is how this well's mud has been described — it is **107 000 / 40 800**, and the profile is simply the wrong one. Building a turbulence model is a different solver; running a laminar profile at Re = 10⁵ without saying so is worse than either. Tracked through the job, not read off the end: at the end the annulus is full of cement and reads laminar, and the mud it started with is what decides the question | yes — both fluids run, and a test pins the flag |
| A-51 | Xue et al. (2022) Sections 3.1.1–3.1.2 | Displacement interface length (20 %–80 % contours) and a "dynamic" displacement efficiency measured over the swept region | The usual volumetric efficiency is uninformative during the job | **Both adopted**, in `inpipe/displacement.py`, recorded in the run history and plotted | Their criticism holds against this model exactly: before breakthrough `η = V_cement/V_annulus` is just pumped volume over annulus volume, a straight ramp identical for every rheology and rate — all the physics shows up only as a shift in arrival time. On K-GEP-1 the swept efficiency starts at 0.43 and climbs to 0.87 while the global figure ramps linearly from zero, and the interface length peaks at **121 m** — over half the modelled annulus — then re-sharpens. **Deviation from their definition:** they take the ratio of the volume behind the 80 % contour to that behind the 20 % contour, which is what a CFD post-processor can extract from two contours; this model holds the whole field, so the actual cement volume in the swept region is used, which does not assume the region behind the back edge is clean | yes |
| A-48 | — | Field practice: the casing shoe does not land on bottom | How to treat the open hole below the shoe | The **rat hole is real geometry**: `geometry.rat_hole_length` sets the shoe above total depth, the annulus starts at the shoe, and the gap is carried as one **well-mixed volume** between the casing outlet and the annulus inlet | It is a *dead end* - fluid enters and leaves it at the same level - so meshing it as a flow path the cement passes through would have it swept clean, and a dead-end pocket is the opposite of that. Well-mixed purges exponentially and never quite finishes, which is the right shape. **Its limitation is the other way, though:** with 0.29 m³ against 0.0132 m³/s the rat hole turns over ~50 times in this job, so well-mixed predicts it ends 99.6 % cement. A real rat hole is partly stagnant and would hold more. Read the residual there as a *lower bound* on mud left below the shoe | partly — the volume is exact, the mixing model is not calibrated |
| A-49 | — | — | Whether the rat hole delay is applied twice | Only the **volumetric** curves carry it as an offset; the front is left alone | Before the rat hole was meshed, the arrival curves were shifted by its pumping time. Now the solver holds the volume and the front waits for it by itself, so the shift was removed — keeping both would have double-counted it. The volumetric curves still need it explicitly, being a plug-displacement hand calculation with no rat hole in them, which is also how Hart et al. (2025) write it in their Eq. 2 | yes — pinned by `test_a_rat_hole_delays_the_front_without_being_counted_twice` |
| A-46 | Hart et al. (2025) *Sci Rep* 15:11365, Eqs. 2–4 | Displacement model: cumulative pumped volume against cumulative annular volume from the caliper, stinger and rat hole filled first, rise velocity `Q/A`; run for the minimum (bit diameter) and maximum (caliper) annular volume | Nothing — it is fully specified | **Reproduced exactly** as the `volumetric` and `in_gauge` curves; `geometry.rat_hole_volume` and `geometry.bit_diameter` are inputs | Their model is the reference this one has to contain before it can claim to add anything. The volumetric curve is pinned against volume over rate to 1e-9. Their field result also fixes which curve to expect: cement, being denser than the mud, filled the whole annulus and tracked the caliper curve, while their lighter freshwater spacer took the path of least resistance and tracked the in-gauge one — so a no-spacer job should track the caliper curve | yes |
| A-47 | — | — | The front's arrival is not monotonic in depth | Report the raw `front` **and** its monotone leading edge (`front_envelope`, a running minimum); rise velocity is computed from the envelope and the out-of-order depths are counted | Cement channels through the narrow side of a washout and reaches a shallower station before the wide one is half displaced, so a station can log an earlier arrival than the one below it. On K-GEP-1 that happens at 1 of 250 depths — small, but differencing the raw curve gives rise velocities of **−167 m/min**, which would have gone into a figure as fact. The envelope is also the curve an operator actually traces on a DAS waterfall, so it is the right thing to compare, not a smoothing convenience | yes — pinned monotone and non-negative |
| A-45 | — | — | Whether the U-tube imbalance drives the flow | **No** — the rate is the volume entering the inlet per unit time, as chosen by the user. The imbalance is computed and reported, not fed back | Deliberate and the user's call. The consequence must travel with every rising time this model produces: on K-GEP-1 the imbalance reaches 21.8 bar for 99 % of the job, and a well that free-falls returns faster than the pump imposes. **So the rising time here is an upper bound**; if a fiber optic log shows cement arriving early, this assumption is the first suspect, ahead of the rheology. The bound is stated in `inpipe/timing.py`, in the printed summary and on the figure | no — one side only, by choice |
| A-42b | — | — | How to vary the *field* case's fluids without editing code | `cases/kgep1.json`, read by `cases/circulation.py` and `cases/mud_left_behind.py`; a command-line flag still overrides it | The mud and slurry properties for this well are not yet measured, so they will be changed repeatedly. A case script holding its own constants would make an edit *look* like it worked - the file parses, the summary prints the new numbers - while the simulation quietly ran the old ones. `test_the_field_case_reads_its_fluids_from_the_case_file` pins that the edited values reach the solver | n/a |
| A-42 | — | — | How to vary fluid properties without editing code | JSON case files (`cases/tao2025.json`, `inpipe/caseio.py`) carrying fluids, geometry, flow, interfacial tension and rheology treatment | Unknown keys are **rejected rather than ignored**, so a mistyped property name fails loudly instead of silently leaving a default in place — which in a validation study is the worst kind of quiet error | n/a |

### The cost of the regularised path, and what it is worth

The regularised branch has no closed-form flow rate, so every wall-stress
solve and every velocity profile goes through numerical quadrature where the
exact law evaluates an algebraic expression. Measured on the Tao et al. case
(100 x 13 x 8, warmed, per step):

| Path | ms/step |
|------|---------|
| exact Herschel–Bulkley (closed form) | 10.3 |
| regularised, as it now stands | 52.4 |

**A tabulation attempt was reverted, and the record of it is worth keeping.**
The cumulative moment integrals depend only on the rheology, so caching one
table per fluid looks obviously right, and on a single-fluid microbenchmark it
was ~50× faster than quadrature. In the solver it was **4047 ms/step — 80×
slower than doing nothing**. The reason is that the solver mixes: every station
holding an interface has its own effective rheology, so nearly every lookup was
a fresh table build, and the builds grow with the stress range. A cache whose
key space is created by the thing it is meant to accelerate cannot amortise.
The microbenchmark could not show this because it held the fluid fixed.

Two things did work, and are what the 52.4 ms rests on:

- the Gauss–Legendre nodes are computed once rather than by an eigenvalue
  decomposition on every stress moment — that alone was 3.1 s of every 4.0 s;
- `shear_rate` evaluates each branch only where it applies. It previously
  computed both over the whole array and selected afterwards, paying a
  fractional power and a square root on every point regardless; inside a
  quadrature piece, which lies entirely on one side of the kink, the other
  branch is now skipped outright (69.5 → 52.4 ms/step, results bitwise equal).

A third defect was worse than either, because it was invisible in every number
the case reported: `AnnulusGrid.map_velocity` evaluated the slot profile
**without** the regularisation. The wall stress was solved under Fluent's law
and the velocity field the solver then advected was built under the exact one.
The reported wall stress, plug fraction and `u_max/ū` all came from the
profile object and so all read correctly; only the field was wrong. It
surfaced at the lowest inlet velocity, where the trace of slurry in a
freshly-mixed station lifts the mixture's yield stress *above* the wall stress
that station's flow rate needs: the exact law then calls the whole section
rigid and returns zero velocity, discrete continuity fails at that station,
and `sum_i f_i` drifts 5.5e-2. `InPipeSolver` ignored the setting the same
way. Both now thread it through, pinned by
`test_annulus_mapping_carries_the_regularisation` and
`test_the_in_pipe_solver_honours_the_regularisation`.

Two further defects were found on the way, both of which produced wrong numbers
rather than slow ones:

- `stress_moment` split its interval at the critical stress without clamping
  the pieces into the interval, so an interval lying **wholly below** `τ_c`
  got a yielded piece of negative width and returned a **negative** moment.
  Every station below the critical stress had a negative flow rate.
- the flow rate is `Q = π (R/τ_w)³ ∫ τ² γ̇ dτ`. Anything that misrepresents
  the shape of that moment near the origin is amplified by the cube: the
  interpolated table reported it as linear in `τ` where it truly goes as `τ⁴`,
  so `Q` **diverged** as `τ_w → 0` instead of vanishing, and the wall-stress
  bracket search overflowed walking down. Both are now pinned by tests
  (`test_stress_moment_is_positive_below_the_critical_stress`,
  `test_flow_rate_vanishes_with_the_wall_stress`).

| ID | Ref | What the source specifies | What is missing | Our choice | Justification | Sensitivity tested? |
|----|-----|---------------------------|-----------------|------------|----------------|---------------------|
| A-43 | — | — | Quadrature order for the regularised stress moments | **24 Gauss–Legendre nodes per smooth piece** | Convergence is algebraic, not spectral: the yielded branch carries a `(τ − τ₀)^(1/n)` branch point just outside its interval. Measured worst-case relative error against a 96-node rule, over four fluids × three `γ̇_c` × three moment orders × six stresses: 6 nodes 5.5e-4, 12 nodes 2.3e-5, **24 nodes 1.6e-7**, 48 nodes 1.1e-9. 24 is the knee — 48 costs 1.7× the runtime for accuracy already far below the discretisation error | yes — pinned against a 160-node rule in `tests/test_rheology.py` |
