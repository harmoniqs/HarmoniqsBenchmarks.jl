import Pkg
import BenchmarkTools
import Dates
import DirectTrajOpt
import MathOptInterface
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
# Task 5: benchmark_solve! helpers
# ============================================================================ #

"""
    _solver_name(options) -> String

Infer the solver name ("Ipopt", "MadNLP", or the raw type name) from the options type.
"""
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

"""
    _infer_state_dim(prob) -> Int

Heuristically infer the state dimension from a `DirectTrajOptProblem` by looking
for component names `:x`, `:ψ̃`, `:Ũ⃗`, `:ρ̃` in `traj.dims`.  Falls back to the
first component dimension if none of those are found.
"""
function _infer_state_dim(prob::DirectTrajOpt.Problems.DirectTrajOptProblem)
    dims = prob.trajectory.dims
    for name in (:x, :ψ̃, :Ũ⃗, :ρ̃)
        if haskey(dims, name)
            return dims[name]
        end
    end
    # Fall back to the first component's dimension
    first_name = first(prob.trajectory.names)
    return dims[first_name]
end

"""
    _infer_control_dim(prob) -> Int

Infer the total control dimension by summing the dimensions of all non-timestep
control components in `traj.control_names`.
"""
function _infer_control_dim(prob::DirectTrajOpt.Problems.DirectTrajOptProblem)
    traj = prob.trajectory
    timestep_name = traj.timestep  # Symbol or nothing
    total = 0
    for cname in traj.control_names
        if cname != timestep_name
            total += traj.dims[cname]
        end
    end
    return total
end

"""
    _count_constraints(prob) -> Int

Count total nonlinear constraints: dynamics × (N−1) + nonlinear constraint dims.
"""
function _count_constraints(prob::DirectTrajOpt.Problems.DirectTrajOptProblem)
    traj = prob.trajectory
    # Dynamics dimension comes from integrators
    dynamics_dim = sum(integrator.dim for integrator in prob.integrators; init = 0)
    n_dynamics = dynamics_dim  # each integrator already accounts for N-1 internally

    # Nonlinear constraint dimensions
    n_nonlinear = sum(
        c.dim for c in prob.constraints
        if c isa DirectTrajOpt.Constraints.AbstractNonlinearConstraint;
        init = 0,
    )

    return n_dynamics + n_nonlinear
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

"""
    benchmark_solve!(prob, options; benchmark_name, runner, verbose, kwargs...) -> BenchmarkResult

Run `solve!` on `prob` using `options`, measure wall time and GC statistics, and
return a fully populated `BenchmarkResult`.

The problem trajectory is updated in-place by `solve!` as usual.
"""
function benchmark_solve!(
    prob::DirectTrajOpt.Problems.DirectTrajOptProblem,
    options::DirectTrajOpt.Solvers.AbstractSolverOptions;
    benchmark_name::String = "unnamed",
    runner::String = "local",
    verbose::Bool = false,
    kwargs...,
)
    # Snapshot solver options before the solve (solve! may mutate them)
    solver_options = Dict{Symbol,Any}(
        name => getfield(options, name) for name in fieldnames(typeof(options))
    )

    # Collect GC stats and run the solve
    GC.gc()
    gc_before = Base.gc_num()
    timed = @timed DirectTrajOpt.solve!(prob; options = options, verbose = verbose, kwargs...)
    gc_after = Base.gc_num()

    gc_diff = Base.GC_Diff(gc_after, gc_before)

    # Post-solve: extract objective value and constraint violation
    evaluator = DirectTrajOpt.Solvers.Evaluator(prob; eval_hessian = false, verbose = false)
    traj = prob.trajectory
    Z_vec = vcat(collect(traj.datavec), collect(traj.global_data))

    objective_value = MOI.eval_objective(evaluator, Z_vec)

    g = zeros(evaluator.n_constraints)
    MOI.eval_constraint(evaluator, g, Z_vec)
    constraint_violation = isempty(g) ? 0.0 : maximum(abs, g)

    # We do not have iteration count from the solve! interface (returns nothing),
    # so record -1 as a sentinel.
    iterations = -1

    # Infer solver status heuristically from constraint violation
    solver_status = constraint_violation < 1e-4 ? :Optimal : :Suboptimal

    # Problem dimensions
    n_constraints = evaluator.n_constraints
    n_variables   = traj.dim * traj.N + traj.global_dim
    state_dim     = _infer_state_dim(prob)
    control_dim   = _infer_control_dim(prob)

    return BenchmarkResult(
        package                = "DirectTrajOpt",
        package_version        = _get_package_version("DirectTrajOpt"),
        commit                 = _get_git_commit(),
        benchmark_name         = benchmark_name,
        N                      = traj.N,
        state_dim              = state_dim,
        control_dim            = control_dim,
        n_constraints          = n_constraints,
        n_variables            = n_variables,
        wall_time_s            = timed.time,
        iterations             = iterations,
        objective_value        = objective_value,
        constraint_violation   = constraint_violation,
        solver_status          = solver_status,
        solver                 = _solver_name(options),
        total_allocations_bytes = Int(timed.bytes),
        total_allocs_count     = Int(timed.gcstats.malloc + timed.gcstats.poolalloc + timed.gcstats.bigalloc),
        gc_time_ns             = Int(round(timed.gctime * 1e9)),
        gc_count               = Int(gc_diff.pause),
        gc_full_count          = Int(gc_diff.full_sweep),
        solver_options         = solver_options,
        julia_version          = string(VERSION),
        timestamp              = Dates.now(),
        runner                 = runner,
        n_threads              = Threads.nthreads(),
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
