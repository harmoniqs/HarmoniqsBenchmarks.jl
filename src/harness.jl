import Pkg
import BenchmarkTools
import Dates
import DirectTrajOpt
import MathOptInterface
import Profile
const MOI = MathOptInterface

# ============================================================================ #
# Task 4: build_evaluator and evaluator_dims
# ============================================================================ #

"""
    build_evaluator(prob; eval_hessian=true) -> (evaluator, Z_vec)

Construct a `DirectTrajOpt.Solvers.Evaluator` from a `DirectTrajOptProblem` and
return it together with the flattened decision vector `Z_vec`.
"""
function build_evaluator(
    prob::DirectTrajOpt.Problems.DirectTrajOptProblem;
    eval_hessian::Bool = true,
)
    evaluator = DirectTrajOpt.Solvers.Evaluator(prob; eval_hessian = eval_hessian, verbose = false)
    traj = prob.trajectory
    Z_vec = vcat(collect(traj.datavec), collect(traj.global_data))
    return (evaluator, Z_vec)
end

"""
    evaluator_dims(evaluator) -> NamedTuple

Return problem-dimension information extracted from a `DirectTrajOpt.Solvers.Evaluator`.

Fields: `n_constraints`, `n_variables`, `n_jacobian_entries`, `n_hessian_entries`.
"""
function evaluator_dims(evaluator::DirectTrajOpt.Solvers.Evaluator)
    traj = evaluator.trajectory
    n_variables = traj.dim * traj.N + traj.global_dim
    return (
        n_constraints      = evaluator.n_constraints,
        n_variables        = n_variables,
        n_jacobian_entries = length(evaluator.jacobian_structure),
        n_hessian_entries  = length(evaluator.hessian_structure),
    )
end

# ============================================================================ #
# Public helpers — extract repeated boilerplate from benchmark scripts so
# the do-block form of benchmark_solve! stays clean.
# ============================================================================ #

const DirectTrajOptProblem = DirectTrajOpt.Problems.DirectTrajOptProblem
const AbstractSolverOptions = DirectTrajOpt.Solvers.AbstractSolverOptions
const Evaluator = DirectTrajOpt.Solvers.Evaluator

"""
    problem_dims(prob::DirectTrajOptProblem) -> NamedTuple

Extract all benchmark-relevant dimensions from a `DirectTrajOptProblem`:
`N`, `state_dim`, `control_dim`, `n_constraints`, `n_variables`.

State dim is inferred by looking for common trajectory component names
(`:x`, `:ψ̃`, `:Ũ⃗`, `:ρ̃`), falling back to the first component's dimension.
Control dim is the sum of all non-timestep control component dimensions.
"""
function problem_dims(prob::DirectTrajOptProblem)
    traj = prob.trajectory
    return (
        N              = traj.N,
        state_dim      = _infer_state_dim(prob),
        control_dim    = _infer_control_dim(prob),
        n_constraints  = _count_constraints(prob),
        n_variables    = traj.dim * traj.N + traj.global_dim,
    )
end

"""
    evaluate_post_solve(prob::DirectTrajOptProblem) -> NamedTuple

Build a lightweight evaluator (no Hessian) and compute post-solve
`objective_value`, `constraint_violation`, and inferred `solver_status`.

Useful in do-block `benchmark_solve!` for solvers that don't expose these
directly (e.g. Altissimo).
"""
function evaluate_post_solve(prob::DirectTrajOptProblem)
    evaluator = Evaluator(prob; eval_hessian = false, verbose = false)
    traj = prob.trajectory
    Z_vec = vcat(collect(traj.datavec), collect(traj.global_data))

    objective_value = MOI.eval_objective(evaluator, Z_vec)

    g = zeros(evaluator.n_constraints)
    MOI.eval_constraint(evaluator, g, Z_vec)
    constraint_violation = isempty(g) ? 0.0 : maximum(abs, g)

    solver_status = constraint_violation < 1e-4 ? :Optimal : :Suboptimal

    return (
        objective_value      = objective_value,
        constraint_violation = constraint_violation,
        solver_status        = solver_status,
    )
end

"""
    snapshot_options(options) -> Dict{Symbol,Any}

Snapshot all fields of a solver options struct into a `Dict{Symbol,Any}`.
Works with any struct type — `IpoptOptions`, `MadNLPOptions`,
`AltissimoOptions`, or custom types.
"""
function snapshot_options(options)::Dict{Symbol,Any}
    return Dict{Symbol,Any}(
        name => getfield(options, name) for name in fieldnames(typeof(options))
    )
end

# ============================================================================ #
# Private helpers used by benchmark_solve! and the public functions above.
# ============================================================================ #

function _solver_name(options)
    tname = string(typeof(options).name.name)
    if occursin("Ipopt", tname)
        return "Ipopt"
    elseif occursin("MadNLP", tname)
        return "MadNLP"
    else
        return tname
    end
end

function _infer_state_dim(prob::DirectTrajOptProblem)
    dims = prob.trajectory.dims
    for name in (:x, :ψ̃, :Ũ⃗, :ρ̃)
        if haskey(dims, name)
            return dims[name]
        end
    end
    first_name = first(prob.trajectory.names)
    return dims[first_name]
end

function _infer_control_dim(prob::DirectTrajOptProblem)
    traj = prob.trajectory
    timestep_name = traj.timestep
    total = 0
    for cname in traj.control_names
        if cname != timestep_name
            total += traj.dims[cname]
        end
    end
    return total
end

function _count_constraints(prob::DirectTrajOptProblem)
    dynamics_dim = sum(integrator.dim for integrator in prob.integrators; init = 0)
    n_nonlinear = sum(
        c.dim for c in prob.constraints
        if c isa DirectTrajOpt.Constraints.AbstractNonlinearConstraint;
        init = 0,
    )
    return dynamics_dim + n_nonlinear
end

"""
    _get_package_version(pkg_name) -> String

Look up the version of a loaded package by name using `Pkg.dependencies()`.
Returns `"unknown"` if not found.
"""
function _get_package_version(pkg_name::String)::String
    try
        for (_, info) in Pkg.dependencies()
            if info.name == pkg_name
                v = info.version
                return isnothing(v) ? "unknown" : string(v)
            end
        end
    catch
    end
    return "unknown"
end

"""
    _get_git_commit() -> String

Return the short git commit hash of the working directory, or `"unknown"`.
"""
function _get_git_commit()::String
    try
        return strip(read(`git rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

# ============================================================================ #
# _capture_solve_metrics: shared primitive — runs the solve closure with
# @timed + GC.gc_num() tracking and returns a NamedTuple of timing/alloc/GC
# fields. Both public `benchmark_solve!` methods call this so the capture path
# is guaranteed identical.
# ============================================================================ #

function _capture_solve_metrics(solve_fn::Function)
    # Clean up garbage so peak/heap deltas reflect this solve's work, not
    # leftover state from earlier code in the same process.
    GC.gc(true)
    gc_before      = Base.gc_num()
    rss_before     = Sys.maxrss()
    live_before    = Base.gc_live_bytes()

    timed = @timed solve_fn()

    rss_after      = Sys.maxrss()
    gc_after_solve = Base.gc_num()

    # Full GC to measure heap retained by the solve (persistent state / leaks).
    GC.gc(true)
    live_after = Base.gc_live_bytes()

    gc_diff = Base.GC_Diff(gc_after_solve, gc_before)

    # OOM headroom at end of solve: how many bytes of host RAM remained
    # between RSS and total system memory. Signed so callers can distinguish
    # "plenty of room" (> 0) from "about to OOM" (≈ 0) from "already past
    # limit" (< 0, e.g. cgroup-constrained environments).
    oom_margin_bytes = Int(Sys.total_memory()) - Int(rss_after)

    return (
        result                  = timed.value,
        wall_time_s             = timed.time,
        total_allocations_bytes = Int(timed.bytes),
        total_allocs_count      = Int(timed.gcstats.malloc + timed.gcstats.poolalloc + timed.gcstats.bigalloc),
        gc_time_ns              = Int(round(timed.gctime * 1e9)),
        gc_count                = Int(gc_diff.pause),
        gc_full_count           = Int(gc_diff.full_sweep),
        # Sys.maxrss() is monotonic over the process lifetime, so this delta
        # is a LOWER BOUND on the peak added by this solve. First solve in a
        # process gives the true peak; later solves give 0 unless they exceed
        # prior peaks. See schema.jl docstring.
        peak_rss_delta_bytes    = Int(max(rss_after - rss_before, 0)),
        # Retained Julia heap after a full post-solve GC. Signed — negative
        # means the solve freed more of the pre-existing heap than it kept.
        live_heap_delta_bytes   = Int(live_after - live_before),
        oom_margin_bytes        = oom_margin_bytes,
    )
end

# ============================================================================ #
# benchmark_solve! — two public methods sharing _capture_solve_metrics:
#
#   1. (prob::DirectTrajOptProblem, options) — DTO convenience: introspects
#      the problem for dimensions, runs `DirectTrajOpt.solve!`, and does a
#      post-solve evaluator pass for objective_value + constraint_violation.
#
#   2. (fn::Function; kwargs...) — generic do-block form: caller supplies all
#      metadata; closure is any solve procedure (Altissimo, PulseTuningProblem,
#      etc.). Both return a fully populated `BenchmarkResult`.
# ============================================================================ #

"""
    benchmark_solve!(prob::DirectTrajOptProblem, options; benchmark_name, runner, verbose, kwargs...) -> BenchmarkResult

DTO-specific convenience: introspects the problem, runs
`DirectTrajOpt.solve!`, and populates all `BenchmarkResult` fields including
objective_value and constraint_violation from a post-solve evaluator pass.

The problem trajectory is updated in-place by `solve!` as usual.
"""
function benchmark_solve!(
    prob::DirectTrajOptProblem,
    options::AbstractSolverOptions;
    benchmark_name::String = "unnamed",
    runner::String = "local",
    verbose::Bool = false,
    kwargs...,
)
    opts_snapshot = snapshot_options(options)
    dims = problem_dims(prob)

    metrics = _capture_solve_metrics() do
        DirectTrajOpt.solve!(prob; options = options, verbose = verbose, kwargs...)
    end

    post = evaluate_post_solve(prob)

    return BenchmarkResult(
        package                = "DirectTrajOpt",
        package_version        = _get_package_version("DirectTrajOpt"),
        commit                 = _get_git_commit(),
        benchmark_name         = benchmark_name,
        N                      = dims.N,
        state_dim              = dims.state_dim,
        control_dim            = dims.control_dim,
        n_constraints          = dims.n_constraints,
        n_variables            = dims.n_variables,
        wall_time_s            = metrics.wall_time_s,
        iterations             = -1,  # solve! returns nothing; sentinel
        objective_value        = post.objective_value,
        constraint_violation   = post.constraint_violation,
        solver_status          = post.solver_status,
        solver                 = _solver_name(options),
        total_allocations_bytes = metrics.total_allocations_bytes,
        total_allocs_count     = metrics.total_allocs_count,
        gc_time_ns             = metrics.gc_time_ns,
        gc_count               = metrics.gc_count,
        gc_full_count          = metrics.gc_full_count,
        peak_rss_delta_bytes   = metrics.peak_rss_delta_bytes,
        live_heap_delta_bytes  = metrics.live_heap_delta_bytes,
        oom_margin_bytes       = metrics.oom_margin_bytes,
        solver_options         = opts_snapshot,
        julia_version          = string(VERSION),
        timestamp              = Dates.now(),
        runner                 = runner,
        n_threads              = Threads.nthreads(),
    )
end

"""
    benchmark_solve!(solve_fn; package, solver, benchmark_name, N, state_dim, control_dim,
                     n_constraints, n_variables, ...) -> BenchmarkResult

Generic do-block form: caller supplies all metadata, closure is any solve
procedure. Useful for Altissimo (private solver) and PulseTuningProblem
(outer QILC loop) where the DirectTrajOpt `benchmark_solve!` signature doesn't
apply.

```julia
result = benchmark_solve!(
    package="Piccolissimo", solver="Altissimo", benchmark_name="xgate_altissimo",
    N=100, state_dim=8, control_dim=2, n_constraints=800, n_variables=1008,
    solver_options=Dict(:fidelity => fid),
) do
    DirectTrajOpt.Solvers.solve!(qcp.prob, altissimo_opts)
end
```

# Required keyword arguments
- `package::String`, `solver::String`, `benchmark_name::String`
- `N::Int`, `state_dim::Int`, `control_dim::Int`,
  `n_constraints::Int`, `n_variables::Int`

# Optional keyword arguments
- `package_version::String` — looked up from `Pkg.dependencies()` if omitted
- `commit::String` — derived from `git rev-parse --short HEAD` if omitted
- `iterations::Int = -1`
- `objective_value::Float64 = NaN`
- `constraint_violation::Float64 = NaN`
- `solver_status::Symbol = :Unknown`
- `solver_options::Dict{Symbol,Any} = Dict{Symbol,Any}()` — merges in any
  solver config *and* run-time metrics (fidelity, J_before, etc.)
- `runner::String = "local"`
"""
function benchmark_solve!(
    solve_fn::Function;
    package::String,
    solver::String,
    benchmark_name::String,
    N::Int,
    state_dim::Int,
    control_dim::Int,
    n_constraints::Int,
    n_variables::Int,
    package_version::String = _get_package_version(package),
    commit::String = _get_git_commit(),
    iterations::Int = -1,
    objective_value::Float64 = NaN,
    constraint_violation::Float64 = NaN,
    solver_status::Symbol = :Unknown,
    solver_options::Dict{Symbol,Any} = Dict{Symbol,Any}(),
    runner::String = "local",
    post_solve::Union{Nothing,Function} = nothing,
)
    metrics = _capture_solve_metrics(solve_fn)

    # If a post_solve closure is provided, call it with the solve return value
    # and let it override post-solve fields (iterations, objective_value,
    # constraint_violation, solver_status). It may return a NamedTuple with
    # any subset of those keys.
    if post_solve !== nothing
        override = post_solve(metrics.result)
        if override !== nothing
            iterations           = get(override, :iterations, iterations)
            objective_value      = get(override, :objective_value, objective_value)
            constraint_violation = get(override, :constraint_violation, constraint_violation)
            solver_status        = get(override, :solver_status, solver_status)
        end
    end

    return BenchmarkResult(
        package                 = package,
        package_version         = package_version,
        commit                  = commit,
        benchmark_name          = benchmark_name,
        N                       = N,
        state_dim               = state_dim,
        control_dim             = control_dim,
        n_constraints           = n_constraints,
        n_variables             = n_variables,
        wall_time_s             = metrics.wall_time_s,
        iterations              = iterations,
        objective_value         = objective_value,
        constraint_violation    = constraint_violation,
        solver_status           = solver_status,
        solver                  = solver,
        total_allocations_bytes = metrics.total_allocations_bytes,
        total_allocs_count      = metrics.total_allocs_count,
        gc_time_ns              = metrics.gc_time_ns,
        gc_count                = metrics.gc_count,
        gc_full_count           = metrics.gc_full_count,
        peak_rss_delta_bytes    = metrics.peak_rss_delta_bytes,
        live_heap_delta_bytes   = metrics.live_heap_delta_bytes,
        oom_margin_bytes        = metrics.oom_margin_bytes,
        solver_options          = solver_options,
        julia_version           = string(VERSION),
        timestamp               = Dates.now(),
        runner                  = runner,
        n_threads               = Threads.nthreads(),
    )
end

# ============================================================================ #
# Task 6: trial_to_eval_benchmark
# ============================================================================ #

"""
    trial_to_eval_benchmark(trial::BenchmarkTools.Trial) -> EvalBenchmark

Convert a `BenchmarkTools.Trial` into an `EvalBenchmark`.
"""
function trial_to_eval_benchmark(trial::BenchmarkTools.Trial)::EvalBenchmark
    return EvalBenchmark(
        times_ns    = Float64.(trial.times),
        gctimes_ns  = Float64.(trial.gctimes),
        memory_bytes = Int(trial.memory),
        allocs       = Int(trial.allocs),
    )
end

# ============================================================================ #
# Per-op micro-benchmarks — run BenchmarkTools on each MOI callback that a
# solver exercises so we can tell where solve time is actually going (obj vs
# grad vs Jacobian vs Hessian vs constraint). Useful alongside `benchmark_solve!`
# when triaging GPU-vs-CPU discrepancies or Altissimo-vs-MadNLP shape.
# ============================================================================ #

"""
    per_op_benchmark(evaluator, Z_vec; seconds=0.5, evals=1, samples=Inf) -> Dict{Symbol, EvalBenchmark}

Run `BenchmarkTools.@benchmarkable` against each MOI callback that a typical
NLP solver exercises, returning a `Dict` keyed by operation name:
`:eval_objective`, `:eval_objective_gradient`, `:eval_constraint`,
`:eval_constraint_jacobian`, and (when `evaluator.eval_hessian`)
`:eval_hessian_lagrangian`.

Each value is an `EvalBenchmark` summarizing `BenchmarkTools.Trial` timing and
allocation distributions. Pass the returned dict into
`MicroBenchmarkResult(..., eval_benchmarks=...)` to persist alongside a
`BenchmarkResult` from the same solve.
"""
function per_op_benchmark(
    evaluator::DirectTrajOpt.Solvers.Evaluator,
    Z_vec::AbstractVector;
    seconds::Real = 0.5,
    evals::Int = 1,
    samples::Int = 1_000,
)::Dict{Symbol,EvalBenchmark}
    n_vars = length(Z_vec)
    n_con = evaluator.n_constraints
    g = zeros(n_vars)
    cons = zeros(n_con)

    function _set_params!(b)
        b.params.seconds = seconds
        b.params.evals   = evals
        b.params.samples = samples
        return b
    end

    results = Dict{Symbol,EvalBenchmark}()

    # Objective
    bench_obj = BenchmarkTools.@benchmarkable MOI.eval_objective($evaluator, $Z_vec)
    _set_params!(bench_obj)
    results[:eval_objective] = trial_to_eval_benchmark(BenchmarkTools.run(bench_obj))

    # Objective gradient
    bench_grad = BenchmarkTools.@benchmarkable MOI.eval_objective_gradient($evaluator, $g, $Z_vec)
    _set_params!(bench_grad)
    results[:eval_objective_gradient] =
        trial_to_eval_benchmark(BenchmarkTools.run(bench_grad))

    # Constraint values
    bench_con = BenchmarkTools.@benchmarkable MOI.eval_constraint($evaluator, $cons, $Z_vec)
    _set_params!(bench_con)
    results[:eval_constraint] = trial_to_eval_benchmark(BenchmarkTools.run(bench_con))

    # Constraint Jacobian
    n_jac = length(evaluator.jacobian_structure)
    jac_buf = zeros(n_jac)
    bench_jac = BenchmarkTools.@benchmarkable MOI.eval_constraint_jacobian($evaluator, $jac_buf, $Z_vec)
    _set_params!(bench_jac)
    results[:eval_constraint_jacobian] =
        trial_to_eval_benchmark(BenchmarkTools.run(bench_jac))

    # Lagrangian Hessian (only if the evaluator computes it)
    if evaluator.eval_hessian
        n_hess = length(evaluator.hessian_structure)
        hess_buf = zeros(n_hess)
        σ = 1.0
        μ = zeros(n_con)
        bench_hess = BenchmarkTools.@benchmarkable MOI.eval_hessian_lagrangian(
            $evaluator, $hess_buf, $Z_vec, $σ, $μ
        )
        _set_params!(bench_hess)
        results[:eval_hessian_lagrangian] =
            trial_to_eval_benchmark(BenchmarkTools.run(bench_hess))
    end

    return results
end

# ============================================================================ #
# benchmark_memory! — do-block sibling of benchmark_solve! that runs the
# closure under `Profile.Allocs.@profile` and returns an `AllocProfileResult`
# for line-level triage of allocation hotspots. Output is saved to its own
# JLD2 via `save_alloc_profile` so alloc artifacts stay separate from the
# main `BenchmarkResult` vector.
# ============================================================================ #

"""
    benchmark_memory!(solve_fn; package, solver, benchmark_name, N, state_dim,
                      control_dim, sample_rate=1.0, warmup=true, ...) -> AllocProfileResult

Run `solve_fn` under `Profile.Allocs.@profile` and return an
`AllocProfileResult` with the flattened sample list, totals, and identity
matching a sibling `BenchmarkResult` run of the same `(benchmark_name, commit)`.

A warmup pass is run first (off the profiler) to burn off JIT/compile-time
allocations so the recorded samples reflect steady-state solve behavior. Pass
`warmup=false` to skip.

```julia
profile = benchmark_memory!(
    package = "Piccolissimo",
    solver  = "Altissimo",
    benchmark_name = "xgate_altissimo_N100",
    N = 100, state_dim = 8, control_dim = 2,
    sample_rate = 1.0,
) do
    DirectTrajOpt.Solvers.solve!(qcp.prob, altissimo_opts)
end
save_alloc_profile("benchmark/results/allocs", profile.benchmark_name, profile)
```

# Required keyword arguments
- `package::String`, `solver::String`, `benchmark_name::String`
- `N::Int`, `state_dim::Int`, `control_dim::Int`

# Optional keyword arguments
- `sample_rate::Float64 = 1.0` — fraction of allocations sampled. Drop to
  `0.01`–`0.1` for long solves where 1.0 would itself OOM.
- `warmup::Bool = true` — run one unprofiled pass before the measured pass.
- `commit::String = _get_git_commit()`
- `runner::String = "local"`
"""
function benchmark_memory!(
    solve_fn::Function;
    package::String,
    solver::String,
    benchmark_name::String,
    N::Int,
    state_dim::Int,
    control_dim::Int,
    sample_rate::Float64 = 1.0,
    warmup::Bool = true,
    commit::String = _get_git_commit(),
    runner::String = "local",
)
    if warmup
        solve_fn()
    end

    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = sample_rate solve_fn()
    fetched = Profile.Allocs.fetch()

    samples = Vector{AllocSample}(undef, length(fetched.allocs))
    @inbounds for (i, a) in enumerate(fetched.allocs)
        samples[i] = AllocSample(
            Int(a.size),
            string(a.type),
            String[string(f) for f in a.stacktrace],
        )
    end

    total_bytes = isempty(samples) ? 0 : sum(s.size_bytes for s in samples)
    total_count = length(samples)

    return AllocProfileResult(
        package,
        solver,
        benchmark_name,
        commit,
        N,
        state_dim,
        control_dim,
        sample_rate,
        samples,
        total_bytes,
        total_count,
        string(VERSION),
        Dates.now(),
        runner,
    )
end
