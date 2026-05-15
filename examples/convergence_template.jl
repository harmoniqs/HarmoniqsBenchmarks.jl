# Convergence-benchmark template.
#
# Drop a copy of this file in your `benchmark/` directory, swap the placeholder
# problem builder for your real one, and you have a convergence benchmark that
# answers "did this solver meet the problem's success bar?" — distinct from a
# timing/allocation benchmark.
#
# Two flavors are demonstrated:
#   1. Ipopt via DirectTrajOpt + the `ipopt_capture` callback helper.
#   2. An arbitrary external solver via the do-block form of `benchmark_solve!`.
#
# Run with:
#   julia --project=benchmark benchmark/your_copy.jl

using HarmoniqsBenchmarks
using DirectTrajOpt

# ---------------------------------------------------------------------------- #
# Replace this with your real problem builder.
function build_my_problem()
    error("Replace `build_my_problem()` with the builder for the problem you ",
          "want to benchmark (e.g. a Piccolo `UnitarySmoothPulseProblem`).")
end

post_solve_fidelity(prob) = error("Return `1 - F(prob)` after the solve.")

# ---------------------------------------------------------------------------- #
# 1. Ipopt via DirectTrajOpt
# ---------------------------------------------------------------------------- #

function bench_ipopt()
    prob = build_my_problem()
    opts = IpoptOptions(max_iter = 200, tol = 1e-8, print_level = 0)

    state, cb = ipopt_capture()

    result = benchmark_solve!(prob, opts;
        benchmark_name = "x_gate_ipopt",
        runner         = "self_hosted",
        callback       = cb,
        convergence    = InfidelityConvergence(
            target_infidelity    = 1e-4,
            final_infidelity     = post_solve_fidelity(prob),
            primal_infeasibility = ipopt_primal_infeasibility(state),
            feas_tol             = 1e-6,
        ),
    )

    @info "Ipopt" iters=ipopt_iterations(state) converged=converged(result.convergence)
    return result
end

# ---------------------------------------------------------------------------- #
# 2. Arbitrary external solver via the do-block form (e.g. Altissimo, MadNLP).
# Whoever owns the benchmark script also owns the dep on the solver — HBJ does
# not depend on private/optional solvers.
# ---------------------------------------------------------------------------- #

function bench_external()
    prob = build_my_problem()

    # Pretend `MySolver` lives in the consumer repo; HBJ doesn't know about it.
    # opts = MySolver.Options(...)

    result = benchmark_solve!(
        package        = "MyConsumerRepo",
        solver         = "MySolver",
        benchmark_name = "x_gate_mysolver",
        N              = prob.trajectory.N,
        state_dim      = prob.trajectory.dims[:x],
        control_dim    = prob.trajectory.dims[:u],
        n_constraints  = 0,                  # fill in from your problem
        n_variables    = 0,
        runner         = "self_hosted",
    ) do
        # Replace with: MySolver.solve!(prob, opts) — must return whatever the
        # solver returns; the post_solve closure (below) reads it.
        nothing
    end

    # If your solver returns a result object with iterations/primal_infeasibility,
    # construct the criterion from it and re-emit the BenchmarkResult with
    # `convergence` populated. Alternatively, pass `post_solve = res -> (...)`
    # to `benchmark_solve!` above to override fields in-line.
    return result
end

# ---------------------------------------------------------------------------- #
# Save + report
# ---------------------------------------------------------------------------- #

results = BenchmarkResult[]
push!(results, bench_ipopt())
# push!(results, bench_external())

save_results("benchmark/results", "convergence_suite", results)

for row in compare_convergence(results)
    @info "convergence" row.benchmark_name row.solver row.converged row.iterations row.wall_time_s
end
