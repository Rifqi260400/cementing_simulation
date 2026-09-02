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
| A-03 | §2.2, Fig. 2b/3a | Cross-section is cut by straight horizontal lines into layers | Layer spacing rule is not given | Uniform spacing in the vertical chord coordinate `y ∈ [−R, R]` | Simplest rule consistent with the figures. **Known consequence:** near-wall layers are thin in *area*, so they are under-resolved relative to the core. The rule is injected as a strategy (`GridConfig.layer_rule`) so an equal-area alternative can be swapped in without touching the solver | partially — `equal_area` alternative implemented and compared in `tests/test_grid.py` |
| A-04 | §2.3 | 1D radial profile is applied "to the entire section axisymmetrically" | How the 1D `u(r)` is sampled onto finite-volume cells | **Exact per-cell area averaging** (`NumericsConfig.velocity_mapping = "area_average"`, the default). Centroid-radius evaluation is retained as `"centroid"` | The spec's baseline was centroid evaluation, with the rule "if the flow-rate error is worse than ~1 % at your working resolution, switch to area-averaged velocity". **It is worse.** Measured `Q` error at the paper's own 13 × 18 cross-section (Dai et al. §3.1 used a 100 × 13 × 18 mesh): Newtonian +0.54 %, power-law `n = 0.4` +0.71 %, Herschel-Bulkley `τ0 = 3 Pa` **+1.38 %** — the yield-stress case fails the gate, and it fails in the direction that matters (a systematic *over*-estimate of `Q`, because the centroid radius of a cell under-states its area-weighted mean radius while `u` is convex in `r`). Area averaging brings the error to 4 × 10⁻⁶ relative. Cost is kept negligible by precomputing an exact cell-by-annulus area matrix once (see A-20), so the mapping is one profile evaluation plus a matvec (~0.16 ms) per station per step | yes — both mappings measured at three resolutions for three rheologies; `tests/test_grid.py` prints the centroid errors on every run |
| A-05 | Eq. A.9 | Explicit Euler in time | No stability constraint or Courant number is given | `Δt < CFL · Δz / max|u|` with `CFL = 0.4`, configurable via `NumericsConfig.cfl`; the condition is asserted every step and raises on violation | Explicit upwind advection is stable for `CFL ≤ 1`; 0.4 leaves margin for the velocity changing between steps as the effective fluid changes. Never clipped silently — a violation is a modelling error, not something to hide | no |
| A-06 | §A.1 | "averaged rheological parameters and density of fluids are used" | The averaging rule is not specified | Volume-fraction-weighted arithmetic mean of `ρ`, `τ0`, `k` and `n` (`fluid.mix_fluids`) | The only rule the wording plainly supports. **Flagged as not physically rigorous for `n`:** the flow index is an exponent, not an extensive property, so a volume-weighted mean of `n` has no constitutive justification. A mixture of an `n = 0.4` and an `n = 1.0` fluid is *not* an `n = 0.7` fluid. Prime candidate for a sensitivity study | no |
| A-08 | Eq. A.9 | First-order face values are implied by "assuming uniform distribution of fluid concentration ... on each face" | Nothing beyond first order | Phase 1 baseline is first-order upwind, deliberately diffusive. The face-value function is injected (`advect(..., face_scheme=...)`) so Phase 2 can drop in donor–acceptor and THINC without touching the solver | The paper achieves sharp interfaces by "axial interface reconstruction", which it does not describe. Reproducing an undescribed scheme would be a guess; reproducing the *baseline* and measuring its numerical diffusion is the honest starting point, and it is the entry point to the Phase 2 contribution | yes — `Dm_num` measured vs. `Δx` in `tests/test_transport.py` and `test_integration.py` |
| A-10 | Eqs. A.10, A.14–A.18, Eqs. 3–6 | Segregation, flow-regime instability criteria and pipe rotation | — | **Deliberately not implemented in Phase 1** | The target application is a vertical well (`β = 0`). Every buoyancy criterion degenerates there: `v_t = √(At·g·sin β·D) = 0`, so `Fr = u/v_t` is undefined and `Re_t = ρ v_t D / μ = 0`, which fails `Re_t > 1` (Eq. A.18) — segregation can never activate. Eqs. A.10/A.14/A.15 all carry `cos β = 1` on the left and a degenerate `Fr` on the right. Pipe rotation acts only through segregation and mixing, both inactive | n/a (structurally inactive at `β = 0`) |

## B. Additional decisions made during implementation

| ID | Paper ref | What the paper specifies | What is missing | Our choice | Justification | Sensitivity tested? |
|----|-----------|--------------------------|-----------------|------------|----------------|---------------------|
| A-07 | Eq. A.8/A.9 | Conservative finite-volume update of `f_i` | Nothing about what happens when the axial velocity profile varies with depth | **Transverse redistribution closure** (`NumericsConfig.transverse_closure = "redistribute"`, the default). The two alternatives, `"none"` (Eq. A.9 verbatim) and `"local"` (subtract `f·∇·u`, i.e. the advective form of Eq. 2), remain selectable | **This resolves a genuine inconsistency in the source model, not a transcription error.** With no transverse velocity, each `(layer, azimuth)` column carries its own axial velocity; when neighbouring stations hold different effective fluids their *profile shapes* differ, so a column has `∂u/∂z ≠ 0` even though every station passes the same `Q`. The discrete continuity condition `Σ_j u_j A_j = 0` then fails per cell, and the conservative update and the sum-to-one constraint become mutually incompatible. See the measured table below. The resolution: because `Σ_c (∇·u)_c A_c = 0` across the cross-section, the imbalance is a *redistribution*, not a source — columns losing volume axially shed it laterally carrying their own composition, and columns gaining volume receive the donor mixing-cup composition. Both invariants then hold to round-off. **Physical content, stated explicitly:** this assumes lateral redistribution is instantaneous and well-mixed across the section. That is a strong assumption, of the same character as the paper's own algorithmically-imposed segregation and instantaneous mixing — it is *not* a solved transverse velocity, and no momentum equation is involved. It is inert wherever `∇·u = 0`, so it changes nothing in any single-rheology case | yes — all three closures measured, see below |

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
