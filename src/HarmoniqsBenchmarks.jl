module HarmoniqsBenchmarks

include("schema.jl")
include("storage.jl")
include("harness.jl")
include("extractors.jl")
include("report.jl")
include("analyze.jl")

export EvalBenchmark
export BenchmarkResult
export MicroBenchmarkResult
export AllocSample
export AllocProfileResult
export save_results
export load_results
export save_micro_results
export load_micro_results
export save_alloc_profile
export load_alloc_profile

export build_evaluator
export evaluator_dims
export problem_dims
export evaluate_post_solve
export snapshot_options
export benchmark_solve!
export benchmark_memory!
export trial_to_eval_benchmark
export per_op_benchmark

export ComparisonRow
export compare_results
export ConvergenceRow
export compare_convergence

export ConvergenceCriterion
export InfidelityConvergence
export ObjectiveConvergence
export converged

export IpoptCapture
export ipopt_capture
export ipopt_capture_callback
export ipopt_iterations
export ipopt_primal_infeasibility

export AllocFrameSummary
export top_alloc_types
export top_alloc_frames
export top_alloc_leaves
export report_alloc_profile

end
