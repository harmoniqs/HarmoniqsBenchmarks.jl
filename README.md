# HarmoniqsBenchmarks.jl

Shared benchmarking harness for the [Harmoniqs](https://github.com/harmoniqs) Julia
ecosystem. Provides a common schema, solver-agnostic timing/memory capture, and
JLD2-backed result storage so benchmarks across
[DirectTrajOpt.jl](https://github.com/harmoniqs/DirectTrajOpt.jl),
[Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl),
[Piccolissimo.jl](https://github.com/harmoniqs/Piccolissimo.jl), and
[Intonato.jl](https://github.com/harmoniqs/Intonato.jl) record comparable results.

## Installation

```julia
using Pkg
Pkg.add("HarmoniqsBenchmarks")
```

## Usage

```julia
using HarmoniqsBenchmarks
using DirectTrajOpt

prob = build_my_problem()             # any DirectTrajOptProblem
opts = IpoptOptions(max_iter = 100)

result = benchmark_solve!(prob, opts; benchmark_name = "x_gate_ipopt")

save_results("results.jld2", [result])
```

Downstream packages call `benchmark_solve!` (or its `do`-block form for custom
solve loops) inside their own `benchmark/` test items, write JLD2 artifacts from
CI, and compare across commits via `compare_results`.

## Convergence benchmarks

Convergence benchmarks ask a different question than timing benchmarks: *did
this solver actually meet the problem's success bar?* Each benchmark declares
its own bar via a `ConvergenceCriterion`, attached to `BenchmarkResult` via the
optional `convergence` field. Two flavors ship:

- `InfidelityConvergence` — quantum gate / state synthesis. Converged iff
  `final_infidelity ≤ target_infidelity` AND `primal_infeasibility ≤ feas_tol`.
- `ObjectiveConvergence` — generic DTO problems. Converged iff
  `final_objective ≤ target_objective` AND the same feasibility check.

```julia
crit = InfidelityConvergence(
    target_infidelity = 1e-4,
    final_infidelity  = 5e-5,         # 1 - F(qcp) post-solve
    primal_infeasibility = 1e-7,       # from the solver
    feas_tol = 1e-6,
)

result = benchmark_solve!(prob, ipopt_opts;
    benchmark_name = "x_gate_ipopt",
    convergence    = crit,
)

@assert converged(result.convergence)
```

For Ipopt runs, `ipopt_capture()` returns a callback you can pass to
`DirectTrajOpt.solve!` to capture the final iteration count and primal
infeasibility directly from Ipopt:

```julia
state, cb = ipopt_capture()
DirectTrajOpt.solve!(prob; options = ipopt_opts, callback = cb)
ipopt_iterations(state)             # ::Int
ipopt_primal_infeasibility(state)   # ::Float64 — feed straight into the criterion
```

For non-Ipopt solvers (Altissimo, MadNLP, custom) use the do-block form of
`benchmark_solve!` and supply `convergence` yourself based on the solver's
native result. See [`examples/convergence_template.jl`](examples/convergence_template.jl)
for a runnable starting point.

`compare_convergence(results)` distills a vector of results into a flat table
of `(problem, solver, converged, iterations, wall_time, criterion)` rows for
reporting.

## License

MIT — see [LICENSE](LICENSE).
