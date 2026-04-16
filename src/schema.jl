using Dates
using Statistics

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
    return EvalBenchmark(times_ns, gctimes_ns, memory_bytes, allocs, median_ns, min_ns, mean_ns)
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
    # Options
    solver_options::Dict{Symbol,Any}
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
    solver_options::Dict{Symbol,Any},
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
        solver_options,
        julia_version,
        timestamp,
        runner,
        n_threads,
    )
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
