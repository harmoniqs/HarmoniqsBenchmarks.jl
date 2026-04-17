# Photonic Dual-Rail Benchmark Suites for Piccolissimo

**Date**: 2026-04-17
**Scope**: Add two new `@testitem` benchmark suites to Piccolissimo PR #50 using dual-rail photonic qubit systems from the rqpnn project.

## Motivation

The existing Piccolissimo benchmarks (testitems 3 & 4) use textbook Pauli Hamiltonians (single-qubit Z drift + X/Y drives, 2-qubit ZZ coupling). These don't represent the workloads the Harmoniqs stack is actually used for. The rqpnn project optimizes gates on dual-rail photonic systems with Kerr nonlinearity, beam-splitter drives, and photon-number-conserving subspaces. The benchmarks should reflect this.

## Physical System: Dual-Rail Photonic Qubits

### Encoding

Each logical qubit is encoded in 2 cavity modes (dual-rail):
- Logical |0> = |1,0> (one photon in first cavity, zero in second)
- Logical |1> = |0,1> (zero in first, one in second)

### Hamiltonian

**Drift** (self-Kerr nonlinearity on each cavity):
```
H_drift = sum_i (a_i^dag)^2 (a_i)^2
```

**Drives** (two types):
1. **Beam-splitter / hopping** between cavities i and j: `NonlinearDrive` with operator `a_j^dag a_i + a_i^dag a_j` and coupling function `u -> u[j] * u[i]` (quadratic in controls). One drive per unordered pair {i, j}.
2. **Single-cavity number operator**: `LinearDrive` with operator `a_i^dag a_i`, controlled by `u[N_cav + i]`.

### Subspace Projection

All operators and states are projected into the **total-photon-number-conserving subspace** where `sum_i n_i = N_phot`. This reduces the Hilbert space dimension significantly.

### Control Bounds

- First N_cav controls (cavity amplitudes): `(0.0, 10.0)`
- Next N_cav controls (single-cavity ops): `(-10.0, 10.0)`

## Gates

| Gate | N_cav | N_phot | # ket states | Subspace dim | Description |
|------|-------|--------|-------------|--------------|-------------|
| X    | 2     | 1      | 2           | 2            | Single-qubit NOT: swap photon between 2 cavities |
| CX   | 4     | 2      | 4           | 10           | Two-qubit CNOT: flip target qubit iff control is |1> |

State construction follows `create_n_toffoli_states` from rqpnn — all computational basis states are enumerated and paired with their target states under the gate action. Only the "hard" states (where initial != target) require actual optimization; the rest are identity mappings included for completeness.

No CCX/Toffoli gate — that is research-grade and too slow for CI.

## Trajectory Type

`MultiKetTrajectory` with explicit dual-rail basis states for both gates. This matches the rqpnn research workflow and exercises a different code path than `UnitaryTrajectory`.

Weights: 100.0 for "hard" states (where the gate actually flips something), 1.0 for identity states — matching rqpnn convention.

## New Testitems

Existing testitems 1-4 are preserved unchanged. Two new testitems are added:

### Testitem 5: Photonic dual-rail — HermitianExponential + SmoothPulse

**Integrator**: `HermitianExponentialIntegrator`
**Pulse**: Piecewise constant via `ZeroOrderPulse` / `SmoothPulseProblem`
**Problem template**: `SmoothPulseProblem` with `Q=100.0`, `R=1e-10`, `ddu_bound=1.0`, `Dt_bounds=(1e-3, 1.0)`

**Runs** (12 total):

| Gate | N  | Solvers |
|------|----|---------|
| X    | 11 | Ipopt, MadNLP, Altissimo |
| X    | 21 | Ipopt, MadNLP, Altissimo |
| CX   | 11 | Ipopt, MadNLP, Altissimo |
| CX   | 21 | Ipopt, MadNLP, Altissimo |

T = 10.0 for all runs. `Random.seed!(42)` for reproducibility.

### Testitem 6: Photonic dual-rail — SplineIntegrator + SplinePulse

**Integrator**: `SplineIntegrator` with `spline_order=1` (linear splines), `alg=Piccolissimo.MagnusAdapt4Alg(tol=1e-7)` — matching rqpnn's configuration.
**Pulse**: `LinearSplinePulse`
**Problem template**: `SplinePulseProblem` with `Q=100.0`, `R=1e-10`, `du_bound=Inf`, `Dt_bounds=(1e-3, 1.0)`

**Runs** (12 total): Same matrix as testitem 5.

## System Construction

Port the following functions from rqpnn's `src/helper_functions.jl` into a helper module within `benchmark/benchmarks.jl` (or a separate `benchmark/photonic_helpers.jl` included by it):

- `set_up_system(N_cav, N_phot)` — builds full-space H_drift and H_drive operators using QuantumOpticsBase
- `set_up_hamiltonian_drives(N_cav, N_phot)` — reduces to subspace, wraps in `NonlinearDrive` / `LinearDrive`
- `total_photon_subspace_transform(B, N)` — projection matrix for N-photon subspace
- `reduce_dim(psi_or_op, N)` — subspace reduction for kets and operators
- `create_n_toffoli_states(N_cav, N_phot)` — dual-rail state construction (used for both X and CX; for X gate with N_cav=2, only 2 states; for CX with N_cav=4, 4 states)

**New dependency**: `QuantumOpticsBase` added to `benchmark/Project.toml`. A native Piccolo reimplementation (using `annihilate`/`create`/`lift_operator`) has been prototyped and confirmed feasible as a future follow-up.

## Solver Configuration

### Ipopt
```
max_iter = 300
eval_hessian = false
hessian_approximation = "limited-memory"
print_level = 0
```

### MadNLP
```
max_iter = 300
print_level = 1
```

### Altissimo
```
search_direction = :LBFGS
lbfgs_memory = 50
line_search = :StrongWolfe
ls_max_evals = 100
max_outer_iter = 20
max_inner_iter = 500
inner_tol = 1e-8
rho_init = 100.0
rho_max = 1e8
polish = true
verbose = false
```

## Metrics Captured

Per run, via `benchmark_solve!` from HarmoniqsBenchmarks.jl:
- Wall time (seconds)
- Iterations
- Objective value
- Constraint violation
- Solver status
- Total allocations (bytes)
- GC time, GC count
- Peak RSS delta, live heap delta

Post-solve: per-state fidelities stored in `solver_options[:per_state_fidelities]`, total fidelity in `solver_options[:fidelity]`.

## Results Storage

All results saved as JLD2 via `save_results` / `save_micro_results` to `benchmark/results/` (gitignored). CI uploads as artifacts.

Naming convention:
- `photonic_X_N{N}_hermexp_{solver}_{sha}.jld2`
- `photonic_CX_N{N}_spline_{solver}_{sha}.jld2`
- etc.

## CI Impact

24 new benchmark runs. With Altissimo being the slowest (~7x Ipopt currently, expected to improve), estimated total CI time for new testitems: ~30-60 minutes. Acceptable per user — up to 1 hour CI budget confirmed.

## Out of Scope

- CCX/Toffoli gate benchmarks (research-grade, too slow)
- Curriculum / warm-start optimization strategies
- Native Piccolo operator port (confirmed feasible, separate follow-up)
- Global parameter optimization (coupling matrix coefficients)
- Open system / dissipation benchmarks
