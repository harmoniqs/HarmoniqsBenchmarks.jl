"""
    ComparisonRow

Represents a comparison between baseline and current benchmark results for a single
benchmark run. Tracks wall time and allocation changes, and flags regressions.
"""
struct ComparisonRow
    benchmark_name::String
    solver::String
    N::Int
    state_dim::Int
    baseline_wall_s::Float64
    current_wall_s::Float64
    wall_time_pct_change::Float64
    baseline_alloc_bytes::Int
    current_alloc_bytes::Int
    alloc_bytes_pct_change::Float64
    has_regression::Bool
end

"""
    ConvergenceRow

One row of a convergence comparison: did `(benchmark_name, solver)` meet its
problem-specific success bar, in how many iters and wall-clock seconds, with
what criterion. Produced by [`compare_convergence`](@ref).
"""
struct ConvergenceRow
    benchmark_name::String
    solver::String
    converged::Bool
    iterations::Int
    # Per-backend iteration breakdown (e.g. Ipopt `:iterations`, Altissimo
    # `:outer`/`:inner`). Empty when the backend exposes no counts. `iterations`
    # above remains the single comparable headline count.
    iteration_counts::Dict{Symbol,Int}
    wall_time_s::Float64
    criterion::ConvergenceCriterion
end

"""
    compare_convergence(results::Vector{BenchmarkResult}) -> Vector{ConvergenceRow}

Distill convergence info from a vector of benchmark results into a flat,
report-friendly shape. Results whose `convergence` field is `nothing`
(timing-only benchmarks) are skipped. Input order is preserved for the
results that remain.
"""
function compare_convergence(results::Vector{BenchmarkResult})::Vector{ConvergenceRow}
    rows = ConvergenceRow[]
    for r in results
        crit = r.convergence
        crit === nothing && continue
        push!(
            rows,
            ConvergenceRow(
                r.benchmark_name,
                r.solver,
                converged(crit),
                r.iterations,
                r.iteration_counts,
                r.wall_time_s,
                crit,
            ),
        )
    end
    return rows
end

"""
    compare_results(baseline::Vector{BenchmarkResult}, current::Vector{BenchmarkResult}; regression_threshold=10.0) -> Vector{ComparisonRow}

Compare baseline and current benchmark results, matching by benchmark_name.

Computes percent changes for wall time and allocations using the formula:
`(new - old) / abs(old) * 100`

Handles zero baselines gracefully:
- 0 → 0 results in 0% change
- 0 → nonzero results in 100% change

Flags a regression when either wall_time OR allocations increase by more than `regression_threshold` percent.

# Arguments
- `baseline::Vector{BenchmarkResult}`: Baseline benchmark results
- `current::Vector{BenchmarkResult}`: Current benchmark results
- `regression_threshold::Float64=10.0`: Threshold (%) for flagging regression

# Returns
Vector of `ComparisonRow` structs, one per matched benchmark.
"""
function compare_results(
    baseline::Vector{BenchmarkResult},
    current::Vector{BenchmarkResult};
    regression_threshold = 10.0,
)::Vector{ComparisonRow}

    rows = ComparisonRow[]

    # Create a dict of baseline results keyed by benchmark_name
    baseline_dict = Dict(r.benchmark_name => r for r in baseline)

    # For each current result, try to find a matching baseline
    for curr in current
        base = get(baseline_dict, curr.benchmark_name, nothing)

        if isnothing(base)
            # No matching baseline; skip or could log warning
            continue
        end

        # Compute wall time percent change
        wall_baseline = base.wall_time_s
        wall_current = curr.wall_time_s
        wall_time_pct_change = if wall_baseline == 0.0
            if wall_current == 0.0
                0.0
            else
                100.0
            end
        else
            (wall_current - wall_baseline) / abs(wall_baseline) * 100.0
        end

        # Compute allocation percent change
        alloc_baseline = base.total_allocations_bytes
        alloc_current = curr.total_allocations_bytes
        alloc_bytes_pct_change = if alloc_baseline == 0
            if alloc_current == 0
                0.0
            else
                100.0
            end
        else
            (alloc_current - alloc_baseline) / abs(alloc_baseline) * 100.0
        end

        # Flag regression if either metric increased beyond threshold
        has_regression =
            (wall_time_pct_change > regression_threshold) ||
            (alloc_bytes_pct_change > regression_threshold)

        row = ComparisonRow(
            curr.benchmark_name,
            curr.solver,
            curr.N,
            curr.state_dim,
            wall_baseline,
            wall_current,
            wall_time_pct_change,
            alloc_baseline,
            alloc_current,
            alloc_bytes_pct_change,
            has_regression,
        )

        push!(rows, row)
    end

    return rows
end
