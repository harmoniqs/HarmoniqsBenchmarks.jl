module HarmoniqsBenchmarks

include("schema.jl")
include("storage.jl")
include("harness.jl")
include("report.jl")

export EvalBenchmark
export BenchmarkResult
export MicroBenchmarkResult
export save_results
export load_results
export save_micro_results
export load_micro_results

export build_evaluator
export evaluator_dims
export benchmark_solve!
export trial_to_eval_benchmark

export ComparisonRow
export compare_results

end
