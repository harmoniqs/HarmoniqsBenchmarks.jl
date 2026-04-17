# Photonic Dual-Rail Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new `@testitem` benchmark suites to Piccolissimo's `benchmark/benchmarks.jl` using dual-rail photonic qubit systems (X and CX gates) with both HermitianExponential and SplineIntegrator paths, benchmarked across Ipopt/MadNLP/Altissimo at N=11 and N=21.

**Architecture:** A new `benchmark/photonic_helpers.jl` file contains the system construction functions ported from rqpnn. Two new `@testitem` blocks in `benchmark/benchmarks.jl` loop over gate/N/solver combinations using builder closures (matching testitem 4's pattern). Each solver gets a fresh problem instance for fair comparison.

**Tech Stack:** Piccolissimo, Piccolo, DirectTrajOpt, HarmoniqsBenchmarks, QuantumOpticsBase, MadNLP, Altissimo

**Spec:** `docs/superpowers/specs/2026-04-17-photonic-dual-rail-benchmarks-design.md`

**Target repo:** `/home/jack/repos/harmoniqs/Piccolissimo.jl` on branch `benchmarks/initial-suites`

---

### Task 1: Add QuantumOpticsBase dependency to benchmark/Project.toml

**Files:**
- Modify: `/home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/Project.toml`

- [ ] **Step 1: Add QuantumOpticsBase to [deps]**

Add `QuantumOpticsBase` to the `[deps]` section of `benchmark/Project.toml`:

```toml
QuantumOpticsBase = "4f57444f-1401-5e15-980d-4471b28d5678"
```

Insert it alphabetically between `OrdinaryDiffEq` and `Piccolissimo`.

- [ ] **Step 2: Verify the environment resolves**

Run:
```bash
cd /home/jack/repos/harmoniqs/Piccolissimo.jl
JULIA_PKG_USE_CLI_GIT=true julia --project=benchmark -e 'using Pkg; Pkg.instantiate(); using QuantumOpticsBase; println("OK: ", pkgversion(QuantumOpticsBase))'
```
Expected: Prints `OK: <version>` without errors.

- [ ] **Step 3: Commit**

```bash
git add benchmark/Project.toml
git commit -m "benchmark: add QuantumOpticsBase dependency for photonic system construction"
```

---

### Task 2: Create benchmark/photonic_helpers.jl with system construction functions

**Files:**
- Create: `/home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/photonic_helpers.jl`

These functions are ported from `/home/jack/repos/harmoniqs/rqpnn/src/helper_functions.jl`. Read that file for reference. The key differences from the rqpnn original:

1. No `includet` (not interactive here)
2. No global state — pure functions only
3. Only port what's needed: `multi_index`, `total_photon_subspace_transform`, `reduce_dim` (Ket and Operator), `set_up_system`, `set_up_hamiltonian_drives`, `create_n_toffoli_states`
4. No `create_n_cz_states`, `reduce_dim_open`, `set_up_hamiltonian_open`, `set_up_hamiltonian`, or global-parameter functions — not needed for X/CX benchmarks

- [ ] **Step 1: Write photonic_helpers.jl**

Create `/home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/photonic_helpers.jl` with these contents:

```julia
# Dual-rail photonic qubit system construction
# Ported from rqpnn/src/helper_functions.jl (Matias Bundgaard-Nielsen)
#
# Each logical qubit is encoded in 2 cavity modes:
#   |0⟩ = |1,0⟩  (one photon in first cavity)
#   |1⟩ = |0,1⟩  (one photon in second cavity)
#
# Drift: self-Kerr (a†a†aa) on each cavity
# Drives: beam-splitter hopping (NonlinearDrive) + number operators (LinearDrive)
# All projected into photon-number-conserving subspace

using QuantumOpticsBase
using SparseArrays

function multi_index(i::Int, shape)
    i0 = i - 1
    N = length(shape)
    arr = Vector{Int}(undef, N)
    for n in 1:N
        arr[n] = i0 % shape[n]
        i0 ÷= shape[n]
    end
    return arr
end

function total_photon_subspace_transform(B, N)
    D = prod(B.shape)
    idxs = Int[]
    for i in 1:D
        if sum(multi_index(i, B.shape)) == N
            push!(idxs, i)
        end
    end
    M = length(idxs)
    T = spzeros(ComplexF64, D, M)
    for (col, i_state) in enumerate(idxs)
        T[i_state, col] = 1.0
    end
    return T, idxs
end

function reduce_dim(psi::Ket, N)
    T, idxs = total_photon_subspace_transform(psi.basis, N)
    return Ket(GenericBasis(length(idxs)), T' * psi.data)
end

function reduce_dim(O::Operator, N)
    T, idxs = total_photon_subspace_transform(O.basis_r, N)
    return Operator(GenericBasis(length(idxs)), GenericBasis(length(idxs)), T' * O.data * T)
end

function set_up_system(N_cav, N_phot)
    b = FockBasis(N_phot)
    a = QuantumOpticsBase.destroy(b)
    ad = QuantumOpticsBase.create(b)

    b_total = b^N_cav
    H_drift = 0 * identityoperator(b_total)
    for i in 1:N_cav
        a_i = QuantumOpticsBase.embed(b_total, b_total, i, a)
        ad_i = QuantumOpticsBase.embed(b_total, b_total, i, ad)
        H_drift += ad_i * ad_i * a_i * a_i
    end

    H_drive = []
    for i in 1:N_cav
        for j in 1:N_cav
            a_i = QuantumOpticsBase.embed(b_total, b_total, i, a)
            ad_j = QuantumOpticsBase.embed(b_total, b_total, j, ad)
            push!(H_drive, ad_j * a_i)
        end
    end
    for i in 1:N_cav
        a_i = QuantumOpticsBase.embed(b_total, b_total, i, a)
        ad_i = QuantumOpticsBase.embed(b_total, b_total, i, ad)
        push!(H_drive, ad_i * a_i)
    end
    return H_drift, H_drive
end

function set_up_hamiltonian_drives(N_cav, N_phot)
    H_drift, H_drive = set_up_system(N_cav, N_phot)
    H_drift_reduced = reduce_dim(H_drift, N_phot).data
    H_drive_reduced = reduce_dim.(H_drive, N_phot)
    H_drive_reduced = [H.data for H in H_drive_reduced]

    drives = AbstractDrive[]
    for i in 1:N_cav
        for j in i+1:N_cav
            H_hermitian = H_drive_reduced[(i-1)*N_cav+j] + H_drive_reduced[(j-1)*N_cav+i]
            push!(drives, NonlinearDrive(H_hermitian, u -> u[j] * u[i]; active_controls=[i, j]))
        end
    end
    for i in 1:N_cav
        push!(drives, LinearDrive(H_drive_reduced[N_cav^2+i], N_cav + i))
    end

    return H_drift_reduced, drives
end

function create_n_toffoli_states(N_cav, N_phot)
    n_qubits = N_cav ÷ 2
    ba = FockBasis(N_phot)
    initial_states = []
    target_states = []

    for state_idx in 0:(2^n_qubits-1)
        bit_string = digits(state_idx, base=2, pad=n_qubits)

        psi_init_components = []
        for bit in bit_string
            if bit == 0
                push!(psi_init_components, fockstate(ba, 1))
                push!(psi_init_components, fockstate(ba, 0))
            else
                push!(psi_init_components, fockstate(ba, 0))
                push!(psi_init_components, fockstate(ba, 1))
            end
        end
        psi_init = reduce(⊗, psi_init_components)
        psi_init = reduce_dim(psi_init, N_phot)

        target_bit_string = copy(bit_string)
        control_bits = bit_string[1:end-1]
        if all(control_bits .== 1)
            target_bit_string[end] = 1 - bit_string[end]
        end

        psi_target_components = []
        for bit in target_bit_string
            if bit == 0
                push!(psi_target_components, fockstate(ba, 1))
                push!(psi_target_components, fockstate(ba, 0))
            else
                push!(psi_target_components, fockstate(ba, 0))
                push!(psi_target_components, fockstate(ba, 1))
            end
        end
        psi_target = reduce(⊗, psi_target_components)
        psi_target = reduce_dim(psi_target, N_phot)

        push!(initial_states, psi_init)
        push!(target_states, psi_target)
    end

    return initial_states, target_states
end
```

- [ ] **Step 2: Smoke-test that it loads and builds systems**

Run:
```bash
cd /home/jack/repos/harmoniqs/Piccolissimo.jl
JULIA_PKG_USE_CLI_GIT=true julia --project=benchmark -e '
    include("benchmark/photonic_helpers.jl")
    using Piccolo

    # X gate: 2 cavities, 1 photon
    H_drift, drives = set_up_hamiltonian_drives(2, 1)
    println("X gate: H_drift size=$(size(H_drift)), $(length(drives)) drives")

    # CX gate: 4 cavities, 2 photons
    H_drift, drives = set_up_hamiltonian_drives(4, 2)
    println("CX gate: H_drift size=$(size(H_drift)), $(length(drives)) drives")

    # States
    init_x, target_x = create_n_toffoli_states(2, 1)
    println("X states: $(length(init_x)) pairs, dim=$(length(init_x[1].data))")

    init_cx, target_cx = create_n_toffoli_states(4, 2)
    println("CX states: $(length(init_cx)) pairs, dim=$(length(init_cx[1].data))")

    # Verify subspace dims
    @assert size(H_drift, 1) == 10 "CX subspace should be 10-dim"
    println("All checks passed")
'
```
Expected: Prints sizes and "All checks passed". X gate: 2×2 H_drift, CX gate: 10×10 H_drift. X: 2 state pairs dim 2. CX: 4 state pairs dim 10.

- [ ] **Step 3: Commit**

```bash
git add benchmark/photonic_helpers.jl
git commit -m "benchmark: add photonic dual-rail system construction helpers"
```

---

### Task 3: Write testitem 5 — Photonic HermitianExponential + SmoothPulse

**Files:**
- Modify: `/home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/benchmarks.jl` (append after testitem 4, before EOF)

This testitem follows the same pattern as testitem 4 ("Hilbert dim scaling"). Key differences:
- Uses `MultiKetTrajectory` instead of `UnitaryTrajectory`
- Constructs photonic systems via `include("photonic_helpers.jl")`
- Loops over 2 gates × 2 N values × 3 solvers = 12 runs
- Uses `time_dependent=true` on `QuantumSystem` because the drives are nonlinear

- [ ] **Step 1: Append testitem 5 to benchmarks.jl**

Add the following at the end of `benchmark/benchmarks.jl` (after the closing `end` of testitem 4):

```julia
# =============================================================================
# @testitem 5: Photonic dual-rail — HermitianExponential + SmoothPulse
# =============================================================================

@testitem "Photonic dual-rail: HermitianExponential + SmoothPulse" begin
    using Piccolissimo
    using Piccolo
    using DirectTrajOpt
    using HarmoniqsBenchmarks
    using MathOptInterface
    const MOI = MathOptInterface
    using Printf
    using Random
    using Dates
    import MadNLP

    include(joinpath(@__DIR__, "photonic_helpers.jl"))

    const MadNLPSolverExt = [
        mod for mod in reverse(Base.loaded_modules_order)
        if Symbol(mod) == :MadNLPSolverExt
    ][1]

    mkpath(joinpath(@__DIR__, "results"))
    commit = try String(strip(read(`git -C $(joinpath(@__DIR__, "..")) rev-parse --short HEAD`, String))) catch; "unknown" end

    T = 10.0

    # ── Gate builders ─────────────────────────────────────────────────────
    # Each returns a fresh (qcp, sys, psi_init, psi_target) so solvers start
    # from identical initial conditions.

    function make_photonic_smooth_builder(N_cav, N_phot, N; seed=42)
        return () -> begin
            Random.seed!(seed)
            H_drift, drives = set_up_hamiltonian_drives(N_cav, N_phot)
            control_bounds = vcat(
                [(0.0, 10.0) for _ in 1:N_cav],
                [(-10.0, 10.0) for _ in 1:N_cav],
            )
            sys = Piccolo.QuantumSystem(H_drift, drives, control_bounds; time_dependent=true)

            initial_states, target_states = create_n_toffoli_states(N_cav, N_phot)
            psi_init   = [s.data for s in initial_states]
            psi_target = [s.data for s in target_states]

            # Identify "hard" states (where initial != target)
            idx_interesting = [i for i in 1:length(psi_init) if psi_init[i] != psi_target[i]]
            weights = [i ∈ idx_interesting ? 100.0 : 1.0 for i in 1:length(psi_init)]

            qtraj = MultiKetTrajectory(sys, psi_init, psi_target, T; weights=weights)
            integrator = HermitianExponentialIntegrator(qtraj, N)
            qcp = SmoothPulseProblem(
                qtraj, N;
                Q          = 100.0,
                R          = 1e-10,
                ddu_bound  = 1.0,
                Δt_bounds  = (1e-3, 1.0),
                integrator = integrator,
            )
            (qcp, psi_init, psi_target)
        end
    end

    # ── Solver configs ────────────────────────────────────────────────────
    function ipopt_opts()
        opts = IpoptOptions(max_iter=300, print_level=0)
        opts.eval_hessian = false
        opts.hessian_approximation = "limited-memory"
        return opts
    end
    madnlp_opts()   = MadNLPSolverExt.MadNLPOptions(max_iter=300, print_level=1)
    altissimo_opts() = AltissimoOptions(
        search_direction       = :LBFGS,
        lbfgs_memory           = 50,
        line_search            = :StrongWolfe,
        ls_max_evals           = 100,
        max_outer_iter         = 20,
        max_inner_iter         = 500,
        inner_tol              = 1e-8,
        ρ_init                 = 100.0,
        ρ_max                  = 1e8,
        polish                 = true,
        polish_stall_min_iters = 10,
        polish_δ_w             = 1e-6,
        polish_δ_c             = 1e-8,
        verbose                = false,
    )

    # ── Benchmark matrix ──────────────────────────────────────────────────
    gate_configs = [
        ("X",  2, 1),   # N_cav=2, N_phot=1
        ("CX", 4, 2),   # N_cav=4, N_phot=2
    ]
    N_values = [11, 21]

    results = BenchmarkResult[]

    println("\n=== Photonic dual-rail: HermitianExponential + SmoothPulse ===")
    @printf("  %-8s | %4s | %-10s | %8s | %10s | %8s | %s\n",
        "Gate", "N", "Solver", "Wall(s)", "Alloc(MB)", "AllocCnt", "Fidelity")
    println("  " * "-"^8 * "-+-" * "-"^4 * "-+-" * "-"^10 * "-+-" * "-"^8 * "-+-" * "-"^10 * "-+-" * "-"^8 * "-+-" * "-"^8)

    for (gate_name, N_cav, N_phot) in gate_configs
        for N in N_values
            builder = make_photonic_smooth_builder(N_cav, N_phot, N)

            # ── Ipopt ─────────────────────────────────────────────────
            (qcp, psi_init, psi_target) = builder()
            r = benchmark_solve!(
                qcp.prob, ipopt_opts();
                benchmark_name = "photonic_$(gate_name)_N$(N)_hermexp_ipopt",
                runner = "local",
                verbose = false,
            )
            sync_trajectory!(qcp)
            r.solver_options[:fidelity] = fidelity(qcp)
            push!(results, r)
            @printf("  %-8s | %4d | %-10s | %8.2f | %10d | %8d | %.4f\n",
                gate_name, N, "Ipopt", r.wall_time_s,
                r.total_allocations_bytes ÷ (1024*1024),
                r.total_allocs_count,
                r.solver_options[:fidelity])
            flush(stdout)

            # ── MadNLP ────────────────────────────────────────────────
            (qcp, psi_init, psi_target) = builder()
            r = benchmark_solve!(
                qcp.prob, madnlp_opts();
                benchmark_name = "photonic_$(gate_name)_N$(N)_hermexp_madnlp",
                runner = "local",
                verbose = false,
            )
            sync_trajectory!(qcp)
            r.solver_options[:fidelity] = fidelity(qcp)
            push!(results, r)
            @printf("  %-8s | %4d | %-10s | %8.2f | %10d | %8d | %.4f\n",
                gate_name, N, "MadNLP", r.wall_time_s,
                r.total_allocations_bytes ÷ (1024*1024),
                r.total_allocs_count,
                r.solver_options[:fidelity])
            flush(stdout)

            # ── Altissimo ─────────────────────────────────────────────
            (qcp, psi_init, psi_target) = builder()
            alt_opts = altissimo_opts()
            traj_alt = qcp.prob.trajectory
            evaluator_pre = DirectTrajOpt.Solvers.Evaluator(qcp.prob; eval_hessian=false, verbose=false)
            n_constraints_alt = evaluator_pre.n_constraints
            n_variables_alt   = traj_alt.dim * traj_alt.N + traj_alt.global_dim
            control_dim_alt   = sum(traj_alt.dims[cn] for cn in traj_alt.control_names if cn != traj_alt.timestep; init=0)
            solver_opts_alt = Dict{Symbol,Any}(
                name => getfield(alt_opts, name) for name in fieldnames(typeof(alt_opts))
            )

            r = benchmark_solve!(
                package         = "Piccolissimo",
                package_version = "0.2.0",
                commit          = commit,
                solver          = "Altissimo",
                benchmark_name  = "photonic_$(gate_name)_N$(N)_hermexp_altissimo",
                N               = traj_alt.N,
                state_dim       = size(qcp.prob.trajectory.data[first(keys(qcp.prob.trajectory.data))], 1),
                control_dim     = control_dim_alt,
                n_constraints   = n_constraints_alt,
                n_variables     = n_variables_alt,
                solver_options  = solver_opts_alt,
                runner          = "local",
                post_solve      = function(_)
                    sync_trajectory!(qcp)
                    fid = fidelity(qcp)
                    evaluator_post = DirectTrajOpt.Solvers.Evaluator(qcp.prob; eval_hessian=false, verbose=false)
                    Z_vec = vcat(collect(traj_alt.datavec), collect(traj_alt.global_data))
                    obj = MOI.eval_objective(evaluator_post, Z_vec)
                    g = zeros(evaluator_post.n_constraints)
                    MOI.eval_constraint(evaluator_post, g, Z_vec)
                    cv = isempty(g) ? 0.0 : maximum(abs, g)
                    solver_opts_alt[:fidelity] = fid
                    return (
                        objective_value      = obj,
                        constraint_violation = cv,
                        solver_status        = cv < 1e-4 ? :Optimal : :Suboptimal,
                    )
                end,
            ) do
                DirectTrajOpt.Solvers.solve!(qcp.prob, alt_opts)
            end
            push!(results, r)
            @printf("  %-8s | %4d | %-10s | %8.2f | %10d | %8d | %.4f\n",
                gate_name, N, "Altissimo", r.wall_time_s,
                r.total_allocations_bytes ÷ (1024*1024),
                r.total_allocs_count,
                r.solver_options[:fidelity])
            flush(stdout)
        end
    end

    save_results(joinpath(@__DIR__, "results"), "photonic_hermexp_smooth", results)

    println("\nPhotonic HermExp+Smooth benchmark complete. $(length(results)) results saved.")
    @test length(results) == 12  # 2 gates × 2 N values × 3 solvers
end
```

- [ ] **Step 2: Verify the testitem parses (syntax check)**

Run:
```bash
cd /home/jack/repos/harmoniqs/Piccolissimo.jl
julia --project=benchmark -e '
    using TestItems
    # Just parse — dont run. If this errors, there is a syntax issue.
    include("benchmark/benchmarks.jl")
    println("Parse OK")
'
```
Expected: "Parse OK" (or TestItems might warn about duplicate names, but no syntax errors).

- [ ] **Step 3: Commit**

```bash
git add benchmark/benchmarks.jl
git commit -m "benchmark: add photonic dual-rail HermitianExponential + SmoothPulse testitem"
```

---

### Task 4: Write testitem 6 — Photonic SplineIntegrator + SplinePulse

**Files:**
- Modify: `/home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/benchmarks.jl` (append after testitem 5)

This testitem uses the same gate/solver loop structure as testitem 5, but with:
- `LinearSplinePulse` instead of `ZeroOrderPulse`
- `SplinePulseProblem` instead of `SmoothPulseProblem`
- `SplineIntegrator` with `spline_order=1` and `MagnusAdapt4Alg(tol=1e-7)`

Key API difference: `MultiKetTrajectory` needs an explicit `LinearSplinePulse` passed as the pulse argument (not created implicitly via `ZeroOrderPulse`). And `SplinePulseProblem` takes `du_bound` instead of `ddu_bound`.

- [ ] **Step 1: Append testitem 6 to benchmarks.jl**

Add the following at the end of `benchmark/benchmarks.jl`:

```julia
# =============================================================================
# @testitem 6: Photonic dual-rail — SplineIntegrator + SplinePulse
# =============================================================================

@testitem "Photonic dual-rail: SplineIntegrator + SplinePulse" begin
    using Piccolissimo
    using Piccolo
    using DirectTrajOpt
    using HarmoniqsBenchmarks
    using MathOptInterface
    const MOI = MathOptInterface
    using Printf
    using Random
    using Dates
    import MadNLP

    include(joinpath(@__DIR__, "photonic_helpers.jl"))

    const MadNLPSolverExt = [
        mod for mod in reverse(Base.loaded_modules_order)
        if Symbol(mod) == :MadNLPSolverExt
    ][1]

    mkpath(joinpath(@__DIR__, "results"))
    commit = try String(strip(read(`git -C $(joinpath(@__DIR__, "..")) rev-parse --short HEAD`, String))) catch; "unknown" end

    T = 10.0

    # ── Gate builders ─────────────────────────────────────────────────────
    function make_photonic_spline_builder(N_cav, N_phot, N; seed=42)
        return () -> begin
            Random.seed!(seed)
            H_drift, drives = set_up_hamiltonian_drives(N_cav, N_phot)
            control_bounds = vcat(
                [(0.0, 10.0) for _ in 1:N_cav],
                [(-10.0, 10.0) for _ in 1:N_cav],
            )
            sys = Piccolo.QuantumSystem(H_drift, drives, control_bounds; time_dependent=true)

            initial_states, target_states = create_n_toffoli_states(N_cav, N_phot)
            psi_init   = [s.data for s in initial_states]
            psi_target = [s.data for s in target_states]

            idx_interesting = [i for i in 1:length(psi_init) if psi_init[i] != psi_target[i]]
            weights = [i ∈ idx_interesting ? 100.0 : 1.0 for i in 1:length(psi_init)]

            times = collect(range(0, T, N))
            n_drives = 2 * N_cav
            pulse_init = zeros(n_drives, N)
            for k in 1:n_drives
                lower, upper = control_bounds[min(k, length(control_bounds))]
                pulse_init[k, :] .= (rand(N) .* (upper - lower) .+ lower) / 5
            end
            pulse = LinearSplinePulse(pulse_init, times)

            qtraj = MultiKetTrajectory(sys, pulse, psi_init, psi_target; weights=weights)
            integrator = SplineIntegrator(qtraj, N; spline_order=1, alg=Piccolissimo.MagnusAdapt4Alg(tol=1e-7))
            qcp = SplinePulseProblem(
                qtraj, N;
                Q         = 100.0,
                R         = 1e-10,
                du_bound  = Inf,
                Δt_bounds = (1e-3, 1.0),
                integrator = integrator,
            )
            (qcp, psi_init, psi_target)
        end
    end

    # ── Solver configs ────────────────────────────────────────────────────
    function ipopt_opts()
        opts = IpoptOptions(max_iter=300, print_level=0)
        opts.eval_hessian = false
        opts.hessian_approximation = "limited-memory"
        return opts
    end
    madnlp_opts()   = MadNLPSolverExt.MadNLPOptions(max_iter=300, print_level=1)
    altissimo_opts() = AltissimoOptions(
        search_direction       = :LBFGS,
        lbfgs_memory           = 50,
        line_search            = :StrongWolfe,
        ls_max_evals           = 100,
        max_outer_iter         = 20,
        max_inner_iter         = 500,
        inner_tol              = 1e-8,
        ρ_init                 = 100.0,
        ρ_max                  = 1e8,
        polish                 = true,
        polish_stall_min_iters = 10,
        polish_δ_w             = 1e-6,
        polish_δ_c             = 1e-8,
        verbose                = false,
    )

    # ── Benchmark matrix ──────────────────────────────────────────────────
    gate_configs = [
        ("X",  2, 1),
        ("CX", 4, 2),
    ]
    N_values = [11, 21]

    results = BenchmarkResult[]

    println("\n=== Photonic dual-rail: SplineIntegrator + SplinePulse ===")
    @printf("  %-8s | %4s | %-10s | %8s | %10s | %8s | %s\n",
        "Gate", "N", "Solver", "Wall(s)", "Alloc(MB)", "AllocCnt", "Fidelity")
    println("  " * "-"^8 * "-+-" * "-"^4 * "-+-" * "-"^10 * "-+-" * "-"^8 * "-+-" * "-"^10 * "-+-" * "-"^8 * "-+-" * "-"^8)

    for (gate_name, N_cav, N_phot) in gate_configs
        for N in N_values
            builder = make_photonic_spline_builder(N_cav, N_phot, N)

            # ── Ipopt ─────────────────────────────────────────────────
            (qcp, psi_init, psi_target) = builder()
            r = benchmark_solve!(
                qcp.prob, ipopt_opts();
                benchmark_name = "photonic_$(gate_name)_N$(N)_spline_ipopt",
                runner = "local",
                verbose = false,
            )
            sync_trajectory!(qcp)
            r.solver_options[:fidelity] = fidelity(qcp)
            push!(results, r)
            @printf("  %-8s | %4d | %-10s | %8.2f | %10d | %8d | %.4f\n",
                gate_name, N, "Ipopt", r.wall_time_s,
                r.total_allocations_bytes ÷ (1024*1024),
                r.total_allocs_count,
                r.solver_options[:fidelity])
            flush(stdout)

            # ── MadNLP ────────────────────────────────────────────────
            (qcp, psi_init, psi_target) = builder()
            r = benchmark_solve!(
                qcp.prob, madnlp_opts();
                benchmark_name = "photonic_$(gate_name)_N$(N)_spline_madnlp",
                runner = "local",
                verbose = false,
            )
            sync_trajectory!(qcp)
            r.solver_options[:fidelity] = fidelity(qcp)
            push!(results, r)
            @printf("  %-8s | %4d | %-10s | %8.2f | %10d | %8d | %.4f\n",
                gate_name, N, "MadNLP", r.wall_time_s,
                r.total_allocations_bytes ÷ (1024*1024),
                r.total_allocs_count,
                r.solver_options[:fidelity])
            flush(stdout)

            # ── Altissimo ─────────────────────────────────────────────
            (qcp, psi_init, psi_target) = builder()
            alt_opts = altissimo_opts()
            traj_alt = qcp.prob.trajectory
            evaluator_pre = DirectTrajOpt.Solvers.Evaluator(qcp.prob; eval_hessian=false, verbose=false)
            n_constraints_alt = evaluator_pre.n_constraints
            n_variables_alt   = traj_alt.dim * traj_alt.N + traj_alt.global_dim
            control_dim_alt   = sum(traj_alt.dims[cn] for cn in traj_alt.control_names if cn != traj_alt.timestep; init=0)
            solver_opts_alt = Dict{Symbol,Any}(
                name => getfield(alt_opts, name) for name in fieldnames(typeof(alt_opts))
            )

            r = benchmark_solve!(
                package         = "Piccolissimo",
                package_version = "0.2.0",
                commit          = commit,
                solver          = "Altissimo",
                benchmark_name  = "photonic_$(gate_name)_N$(N)_spline_altissimo",
                N               = traj_alt.N,
                state_dim       = size(qcp.prob.trajectory.data[first(keys(qcp.prob.trajectory.data))], 1),
                control_dim     = control_dim_alt,
                n_constraints   = n_constraints_alt,
                n_variables     = n_variables_alt,
                solver_options  = solver_opts_alt,
                runner          = "local",
                post_solve      = function(_)
                    sync_trajectory!(qcp)
                    fid = fidelity(qcp)
                    evaluator_post = DirectTrajOpt.Solvers.Evaluator(qcp.prob; eval_hessian=false, verbose=false)
                    Z_vec = vcat(collect(traj_alt.datavec), collect(traj_alt.global_data))
                    obj = MOI.eval_objective(evaluator_post, Z_vec)
                    g = zeros(evaluator_post.n_constraints)
                    MOI.eval_constraint(evaluator_post, g, Z_vec)
                    cv = isempty(g) ? 0.0 : maximum(abs, g)
                    solver_opts_alt[:fidelity] = fid
                    return (
                        objective_value      = obj,
                        constraint_violation = cv,
                        solver_status        = cv < 1e-4 ? :Optimal : :Suboptimal,
                    )
                end,
            ) do
                DirectTrajOpt.Solvers.solve!(qcp.prob, alt_opts)
            end
            push!(results, r)
            @printf("  %-8s | %4d | %-10s | %8.2f | %10d | %8d | %.4f\n",
                gate_name, N, "Altissimo", r.wall_time_s,
                r.total_allocations_bytes ÷ (1024*1024),
                r.total_allocs_count,
                r.solver_options[:fidelity])
            flush(stdout)
        end
    end

    save_results(joinpath(@__DIR__, "results"), "photonic_spline", results)

    println("\nPhotonic Spline benchmark complete. $(length(results)) results saved.")
    @test length(results) == 12  # 2 gates × 2 N values × 3 solvers
end
```

- [ ] **Step 2: Verify parse**

Run:
```bash
cd /home/jack/repos/harmoniqs/Piccolissimo.jl
julia --project=benchmark -e '
    using TestItems
    include("benchmark/benchmarks.jl")
    println("Parse OK — 6 testitems expected")
'
```
Expected: "Parse OK — 6 testitems expected"

- [ ] **Step 3: Commit**

```bash
git add benchmark/benchmarks.jl
git commit -m "benchmark: add photonic dual-rail SplineIntegrator + SplinePulse testitem"
```

---

### Task 5: Run testitem 5 (HermExp) and fix issues

**Files:**
- May modify: `/home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/benchmarks.jl`
- May modify: `/home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/photonic_helpers.jl`

- [ ] **Step 1: Run just the photonic HermExp testitem**

Run:
```bash
cd /home/jack/repos/harmoniqs/Piccolissimo.jl
JULIA_PKG_USE_CLI_GIT=true julia --project=benchmark -t auto -e '
    using TestItemRunner
    @run_package_tests filter=ti->occursin("HermitianExponential", ti.name) verbose=true
' 2>&1
```

This will take several minutes. Watch for:
- Package precompilation errors (missing deps, version conflicts)
- QuantumSystem construction errors (Hermiticity checks, drive dimension mismatches)
- MultiKetTrajectory errors (state dimension vs system levels)
- benchmark_solve! dispatch errors (IpoptOptions/MadNLPOptions/AltissimoOptions)
- Fidelity computation errors (sync_trajectory! on MultiKetTrajectory)

- [ ] **Step 2: Fix any errors found**

Debug and fix issues. Common things to watch for:
- `QuantumSystem` may not accept `time_dependent=true` with typed drives — check the constructor signature
- `MultiKetTrajectory` expects `system.levels` to match the state vector dimension — verify after subspace reduction
- `fidelity(qcp)` may need the qcp to be a specific type — check what `sync_trajectory!` and `fidelity` expect for MultiKetTrajectory-based problems
- `IpoptOptions` constructor may not accept `eval_hessian` and `hessian_approximation` as kwargs — check the existing testitem 3 pattern (it sets fields after construction)

- [ ] **Step 3: Verify testitem 5 completes with results**

Confirm that:
- All 12 runs complete (check stdout table)
- JLD2 file is saved in `benchmark/results/`
- `@test length(results) == 12` passes

Run:
```bash
ls -la /home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/results/photonic_hermexp_smooth*
```
Expected: A `.jld2` file exists.

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve issues in photonic HermExp benchmark testitem"
```

---

### Task 6: Run testitem 6 (Spline) and fix issues

**Files:**
- May modify: `/home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/benchmarks.jl`
- May modify: `/home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/photonic_helpers.jl`

- [ ] **Step 1: Run just the photonic Spline testitem**

Run:
```bash
cd /home/jack/repos/harmoniqs/Piccolissimo.jl
JULIA_PKG_USE_CLI_GIT=true julia --project=benchmark -t auto -e '
    using TestItemRunner
    @run_package_tests filter=ti->occursin("SplineIntegrator", ti.name) verbose=true
' 2>&1
```

Additional things to watch for beyond Task 5's list:
- `LinearSplinePulse` constructor signature — may need `(controls, times)` or `(controls, derivatives, times)`
- `SplineIntegrator` with `alg=Piccolissimo.MagnusAdapt4Alg(tol=1e-7)` — verify the algorithm type exists and the kwarg is `alg` not `algorithm`
- `SplinePulseProblem` kwarg differences from `SmoothPulseProblem` — `du_bound` instead of `ddu_bound`

- [ ] **Step 2: Fix any errors found**

Debug and fix. Pay special attention to:
- `LinearSplinePulse` may need derivative initialization as well as control values
- `SplineIntegrator` may need `global_names` passed when the system has global params
- The spline path may have different `fidelity`/`sync_trajectory!` behavior

- [ ] **Step 3: Verify testitem 6 completes with results**

Confirm 12 runs complete and JLD2 saved:
```bash
ls -la /home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/results/photonic_spline*
```

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve issues in photonic Spline benchmark testitem"
```

---

### Task 7: Run full benchmark suite and verify no regressions

**Files:** None (validation only)

- [ ] **Step 1: Run all 6 testitems together**

Run:
```bash
cd /home/jack/repos/harmoniqs/Piccolissimo.jl
JULIA_PKG_USE_CLI_GIT=true julia --project=benchmark -t auto -e '
    using TestItemRunner
    TestItemRunner.run_tests("benchmark/")
' 2>&1
```

This will take a while (potentially up to an hour with Altissimo). Verify:
- All 6 testitems pass
- Testitems 1-4 (existing) are not broken by the new code
- All JLD2 artifacts are in `benchmark/results/`

- [ ] **Step 2: List all result files**

```bash
ls -la /home/jack/repos/harmoniqs/Piccolissimo.jl/benchmark/results/
```

Expected: Files for all 6 testitems' results.

- [ ] **Step 3: Final commit if any adjustments were needed**

```bash
git add -u
git commit -m "benchmark: verify full suite passes with photonic benchmarks"
```
