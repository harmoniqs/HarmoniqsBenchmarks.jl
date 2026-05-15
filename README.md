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

## License

MIT — see [LICENSE](LICENSE).
