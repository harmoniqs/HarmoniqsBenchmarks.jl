using JLD2

# ------------------------------------------------------------------ #
# Backward-compatibility upgrade shim for BenchmarkResult
# ------------------------------------------------------------------ #
# When the in-memory `BenchmarkResult` struct gains fields that a committed
# JLD2 blob was written without, JLD2 cannot map the on-disk layout onto the
# current type and hands back a `JLD2.ReconstructedMutable{:BenchmarkResult}`
# (a field-name/-value bag) instead of a real `BenchmarkResult`. `load_results`
# then tries to `convert` that bag into `Vector{BenchmarkResult}` and fails.
#
# This `convert` method is that upgrade path: it reads each field the old blob
# does carry and defaults any field the blob lacks. Concretely it lets any
# result serialized before the GPU device-memory fields existed
# (`gpu_allocations_bytes`, `gpu_live_bytes`) load with those fields defaulted
# to `nothing` ("device axis not measured"). It is written generically over the
# reconstructed field set so it also tolerates future additive schema changes.

# Read property `name` off a reconstructed bag if present, else `default`.
_rc_get(rc, name::Symbol, default) =
    name in propertynames(rc) ? getproperty(rc, name) : default

function Base.convert(
    ::Type{BenchmarkResult},
    rc::JLD2.ReconstructedMutable{:BenchmarkResult},
)
    return BenchmarkResult(;
        package = rc.package,
        package_version = rc.package_version,
        commit = rc.commit,
        benchmark_name = rc.benchmark_name,
        N = rc.N,
        state_dim = rc.state_dim,
        control_dim = rc.control_dim,
        n_constraints = rc.n_constraints,
        n_variables = rc.n_variables,
        wall_time_s = rc.wall_time_s,
        iterations = rc.iterations,
        objective_value = rc.objective_value,
        constraint_violation = rc.constraint_violation,
        solver_status = rc.solver_status,
        solver = rc.solver,
        total_allocations_bytes = rc.total_allocations_bytes,
        total_allocs_count = rc.total_allocs_count,
        gc_time_ns = rc.gc_time_ns,
        gc_count = rc.gc_count,
        gc_full_count = rc.gc_full_count,
        # Optional/defaulted fields — tolerate blobs predating each of them.
        peak_rss_delta_bytes = _rc_get(rc, :peak_rss_delta_bytes, 0),
        live_heap_delta_bytes = _rc_get(rc, :live_heap_delta_bytes, 0),
        oom_margin_bytes = _rc_get(rc, :oom_margin_bytes, 0),
        gpu_allocations_bytes = _rc_get(rc, :gpu_allocations_bytes, nothing),
        gpu_live_bytes = _rc_get(rc, :gpu_live_bytes, nothing),
        solver_options = rc.solver_options,
        iteration_counts = _rc_get(rc, :iteration_counts, Dict{Symbol,Int}()),
        convergence = _rc_get(rc, :convergence, nothing),
        julia_version = rc.julia_version,
        timestamp = rc.timestamp,
        runner = rc.runner,
        n_threads = rc.n_threads,
    )
end

"""
    save_results(dir, name, results::Vector{BenchmarkResult}) -> String

Save a vector of `BenchmarkResult` to `dir/name_commit.jld2`.
Returns the path of the saved file.
"""
function save_results(
    dir::AbstractString,
    name::AbstractString,
    results::Vector{BenchmarkResult},
)::String
    mkpath(dir)
    # Use commit from first result (all should share the same commit in a benchmark run)
    commit = isempty(results) ? "unknown" : results[1].commit
    filename = "$(name)_$(commit).jld2"
    path = joinpath(dir, filename)
    jldsave(path; results)
    return path
end

"""
    load_results(path) -> Vector{BenchmarkResult}

Load a vector of `BenchmarkResult` from a JLD2 file.
"""
function load_results(path::AbstractString)::Vector{BenchmarkResult}
    return jldopen(path, "r") do f
        f["results"]
    end
end

"""
    save_micro_results(dir, name, result::MicroBenchmarkResult) -> String

Save a `MicroBenchmarkResult` to `dir/name_commit.jld2`.
Returns the path of the saved file.
"""
function save_micro_results(
    dir::AbstractString,
    name::AbstractString,
    result::MicroBenchmarkResult,
)::String
    mkpath(dir)
    filename = "$(name)_$(result.commit).jld2"
    path = joinpath(dir, filename)
    jldsave(path; result)
    return path
end

"""
    load_micro_results(path) -> MicroBenchmarkResult

Load a `MicroBenchmarkResult` from a JLD2 file.
"""
function load_micro_results(path::AbstractString)::MicroBenchmarkResult
    return jldopen(path, "r") do f
        f["result"]
    end
end

"""
    save_alloc_profile(dir, name, profile::AllocProfileResult) -> String

Save an `AllocProfileResult` to `dir/name_commit_allocs.jld2` (a file distinct
from `save_results`/`save_micro_results` so allocation artifacts do not bloat
the main benchmark JLD2). Returns the path of the saved file.
"""
function save_alloc_profile(
    dir::AbstractString,
    name::AbstractString,
    profile::AllocProfileResult,
)::String
    mkpath(dir)
    filename = "$(name)_$(profile.commit)_allocs.jld2"
    path = joinpath(dir, filename)
    jldsave(path; profile)
    return path
end

"""
    load_alloc_profile(path) -> AllocProfileResult

Load an `AllocProfileResult` from a JLD2 file.
"""
function load_alloc_profile(path::AbstractString)::AllocProfileResult
    return jldopen(path, "r") do f
        f["profile"]
    end
end
