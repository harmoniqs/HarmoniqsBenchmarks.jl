using Dates
using Statistics

"""
    ConvergenceCriterion

Abstract supertype for per-benchmark convergence criteria. Concrete subtypes
declare a problem-specific success bar (e.g. infidelity ≤ target) and are
attached to a `BenchmarkResult` via its `convergence` field.

The single-method API is `converged(::ConvergenceCriterion)::Bool`.
"""
abstract type ConvergenceCriterion end

"""
    InfidelityConvergence

Convergence criterion for quantum gate/state synthesis: the solve is
considered converged iff `final_infidelity ≤ target_infidelity` AND
`primal_infeasibility ≤ feas_tol` (i.e. the dynamics constraints are
actually satisfied — we never accept "good infidelity but infeasible").
"""
struct InfidelityConvergence <: ConvergenceCriterion
    target_infidelity::Float64
    final_infidelity::Float64
    primal_infeasibility::Float64
    feas_tol::Float64
end

function InfidelityConvergence(;
    target_infidelity::Float64,
    final_infidelity::Float64,
    primal_infeasibility::Float64,
    feas_tol::Float64,
)
    return InfidelityConvergence(
        target_infidelity,
        final_infidelity,
        primal_infeasibility,
        feas_tol,
    )
end

"""
    ObjectiveConvergence

Convergence criterion for non-quantum DTO problems where success is gated on
the solver driving the objective below a problem-specific bar:
`final_objective ≤ target_objective` AND `primal_infeasibility ≤ feas_tol`.
"""
struct ObjectiveConvergence <: ConvergenceCriterion
    target_objective::Float64
    final_objective::Float64
    primal_infeasibility::Float64
    feas_tol::Float64
end

function ObjectiveConvergence(;
    target_objective::Float64,
    final_objective::Float64,
    primal_infeasibility::Float64,
    feas_tol::Float64,
)
    return ObjectiveConvergence(
        target_objective,
        final_objective,
        primal_infeasibility,
        feas_tol,
    )
end

"""
    converged(c::ConvergenceCriterion) -> Bool

Did this benchmark meet its success bar AND satisfy the dynamics tolerance?
"""
function converged end

converged(c::InfidelityConvergence) =
    c.final_infidelity ≤ c.target_infidelity && c.primal_infeasibility ≤ c.feas_tol

converged(c::ObjectiveConvergence) =
    c.final_objective ≤ c.target_objective && c.primal_infeasibility ≤ c.feas_tol

"""
    EvalBenchmark

Stores per-eval-function timing data from a BenchmarkTools trial or similar measurement.
Derived statistics (median, min, mean) are computed at construction time.
"""
struct EvalBenchmark
    times_ns::Vector{Float64}
    gctimes_ns::Vector{Float64}
    memory_bytes::Int
    allocs::Int
    # Derived stats
    median_ns::Float64
    min_ns::Float64
    mean_ns::Float64
end

function EvalBenchmark(;
    times_ns::Vector{Float64},
    gctimes_ns::Vector{Float64},
    memory_bytes::Int,
    allocs::Int,
)
    median_ns = median(times_ns)
    min_ns = minimum(times_ns)
    mean_ns = mean(times_ns)
    return EvalBenchmark(
        times_ns,
        gctimes_ns,
        memory_bytes,
        allocs,
        median_ns,
        min_ns,
        mean_ns,
    )
end

"""
    BenchmarkResult

Stores full-solve benchmark data including identity, problem dimensions, solve metrics,
memory usage, solver options, and execution metadata.
"""
struct BenchmarkResult
    # Identity
    package::String
    package_version::String
    commit::String
    benchmark_name::String
    # Problem dims
    N::Int
    state_dim::Int
    control_dim::Int
    n_constraints::Int
    n_variables::Int
    # Solve metrics
    wall_time_s::Float64
    iterations::Int
    objective_value::Float64
    constraint_violation::Float64
    solver_status::Symbol
    solver::String
    # Memory
    total_allocations_bytes::Int
    total_allocs_count::Int
    gc_time_ns::Int
    gc_count::Int
    gc_full_count::Int
    # Peak memory deltas across the solve.
    # peak_rss_delta_bytes: change in `Sys.maxrss()` before/after the solve.
    #   Monotonic caveat: Sys.maxrss() only goes UP over process lifetime, so
    #   this delta is a LOWER BOUND on the peak added by this solve. For the
    #   first solve in a process it's the true peak; for later solves it's 0
    #   if they didn't exceed previous peaks.
    # live_heap_delta_bytes: change in `Base.gc_live_bytes()` after a full
    #   `GC.gc(true)` on each side. Signed — negative means the solve freed
    #   more than it kept. Captures retained/leaked heap, not transient peak.
    peak_rss_delta_bytes::Int
    live_heap_delta_bytes::Int
    # OOM headroom: Sys.total_memory() - Sys.maxrss() at end of solve.
    # Positive means the process had that many bytes of host RAM to spare when
    # the solve completed. Zero or negative indicates the solve ran up against
    # (or got killed by) the host's memory limit. Use in conjunction with
    # peak_rss_delta_bytes when triaging GPU-bound benchmarks that exhibit
    # unexpected host-side pressure.
    oom_margin_bytes::Int
    # Options
    solver_options::Dict{Symbol,Any}
    # Per-backend iteration breakdown. Solver iteration structures are NOT
    # apples-to-apples, so each backend records its own keys rather than forcing
    # a shared meaning onto the scalar `iterations` above:
    #   - Ipopt / MadNLP (interior-point): Dict(:iterations => n)
    #   - Altissimo (augmented-Lagrangian): Dict(:outer => n_outer, :inner => n_inner)
    # The scalar `iterations` stays as the single comparable "headline" count
    # (outer iterations for AL methods). Empty Dict when the backend exposes no
    # counts. Consumers should render each key as its own series — never compare
    # an :outer count against an interior-point :iterations count.
    iteration_counts::Dict{Symbol,Int}
    # Convergence — optional; nothing for timing-only benchmarks.
    # When set, identifies whether the solve met its problem-specific success
    # bar (target_infidelity / target_objective) AND satisfied the dynamics
    # tolerance (primal_infeasibility ≤ feas_tol).
    convergence::Union{Nothing,ConvergenceCriterion}
    # Metadata
    julia_version::String
    timestamp::DateTime
    runner::String
    n_threads::Int
end

function BenchmarkResult(;
    package::String,
    package_version::String,
    commit::String,
    benchmark_name::String,
    N::Int,
    state_dim::Int,
    control_dim::Int,
    n_constraints::Int,
    n_variables::Int,
    wall_time_s::Float64,
    iterations::Int,
    objective_value::Float64,
    constraint_violation::Float64,
    solver_status::Symbol,
    solver::String,
    total_allocations_bytes::Int,
    total_allocs_count::Int,
    gc_time_ns::Int,
    gc_count::Int,
    gc_full_count::Int,
    peak_rss_delta_bytes::Int = 0,
    live_heap_delta_bytes::Int = 0,
    oom_margin_bytes::Int = 0,
    solver_options::Dict{Symbol,Any},
    iteration_counts::Dict{Symbol,Int} = Dict{Symbol,Int}(),
    convergence::Union{Nothing,ConvergenceCriterion} = nothing,
    julia_version::String,
    timestamp::DateTime,
    runner::String,
    n_threads::Int,
)
    return BenchmarkResult(
        package,
        package_version,
        commit,
        benchmark_name,
        N,
        state_dim,
        control_dim,
        n_constraints,
        n_variables,
        wall_time_s,
        iterations,
        objective_value,
        constraint_violation,
        solver_status,
        solver,
        total_allocations_bytes,
        total_allocs_count,
        gc_time_ns,
        gc_count,
        gc_full_count,
        peak_rss_delta_bytes,
        live_heap_delta_bytes,
        oom_margin_bytes,
        solver_options,
        iteration_counts,
        convergence,
        julia_version,
        timestamp,
        runner,
        n_threads,
    )
end

"""
    AllocSample

One sampled allocation from `Profile.Allocs.fetch()`, flattened to JLD2-safe
primitive types so the profile survives serialization across Julia sessions.
"""
struct AllocSample
    size_bytes::Int
    type_name::String
    stacktrace::Vector{String}
end

"""
    AllocProfileResult

Stores a `Profile.Allocs` run for one solve closure. Identity + problem dims
match `BenchmarkResult` so profiles join back to solve results by
`(benchmark_name, commit)`. The sample vector is kept raw — aggregations
(by type, by module, by file/line) are cheap to compute at query time and not
worth baking into the on-disk schema.
"""
struct AllocProfileResult
    # Identity
    package::String
    solver::String
    benchmark_name::String
    commit::String
    # Dims
    N::Int
    state_dim::Int
    control_dim::Int
    # Profile config + data
    sample_rate::Float64
    samples::Vector{AllocSample}
    total_bytes::Int
    total_count::Int
    # Metadata
    julia_version::String
    timestamp::DateTime
    runner::String
end

"""
    MicroBenchmarkResult

Stores per-function micro-benchmark data, keyed by function symbol.
"""
struct MicroBenchmarkResult
    # Identity
    package::String
    package_version::String
    commit::String
    benchmark_name::String
    # Dims
    N::Int
    state_dim::Int
    control_dim::Int
    # Data
    eval_benchmarks::Dict{Symbol,EvalBenchmark}
    # Metadata
    julia_version::String
    timestamp::DateTime
    runner::String
    n_threads::Int
end

function MicroBenchmarkResult(;
    package::String,
    package_version::String,
    commit::String,
    benchmark_name::String,
    N::Int,
    state_dim::Int,
    control_dim::Int,
    eval_benchmarks::Dict{Symbol,EvalBenchmark},
    julia_version::String,
    timestamp::DateTime,
    runner::String,
    n_threads::Int,
)
    return MicroBenchmarkResult(
        package,
        package_version,
        commit,
        benchmark_name,
        N,
        state_dim,
        control_dim,
        eval_benchmarks,
        julia_version,
        timestamp,
        runner,
        n_threads,
    )
end
