using HarmoniqsBenchmarks
using Test
using Dates
using BenchmarkTools
using DirectTrajOpt
using NamedTrajectories
using SparseArrays
using LinearAlgebra

@testset "HarmoniqsBenchmarks" begin

    @testset "EvalBenchmark construction" begin
        times = [100.0, 200.0, 150.0, 120.0, 180.0]
        gctimes = [5.0, 10.0, 7.0, 6.0, 9.0]
        eb = EvalBenchmark(
            times_ns = times,
            gctimes_ns = gctimes,
            memory_bytes = 1024,
            allocs = 42,
        )

        @test eb.times_ns == times
        @test eb.gctimes_ns == gctimes
        @test eb.memory_bytes == 1024
        @test eb.allocs == 42

        # Derived stats
        @test eb.median_ns ≈ 150.0
        @test eb.min_ns ≈ 100.0
        @test eb.mean_ns ≈ sum(times) / length(times)
    end

    @testset "BenchmarkResult construction" begin
        br = BenchmarkResult(
            package = "DirectTrajOpt",
            package_version = "0.8.10",
            commit = "abc1234",
            benchmark_name = "test_problem",
            N = 100,
            state_dim = 4,
            control_dim = 2,
            n_constraints = 50,
            n_variables = 200,
            wall_time_s = 1.23,
            iterations = 30,
            objective_value = 0.001,
            constraint_violation = 1e-8,
            solver_status = :Optimal,
            solver = "Ipopt",
            total_allocations_bytes = 1_000_000,
            total_allocs_count = 5000,
            gc_time_ns = 100_000,
            gc_count = 3,
            gc_full_count = 1,
            solver_options = Dict{Symbol,Any}(:max_iter => 100),
            julia_version = "1.12.0",
            timestamp = DateTime(2026, 4, 14),
            runner = "local",
            n_threads = 1,
        )

        @test br.package == "DirectTrajOpt"
        @test br.package_version == "0.8.10"
        @test br.commit == "abc1234"
        @test br.benchmark_name == "test_problem"
        @test br.N == 100
        @test br.state_dim == 4
        @test br.control_dim == 2
        @test br.n_constraints == 50
        @test br.n_variables == 200
        @test br.wall_time_s ≈ 1.23
        @test br.iterations == 30
        @test br.objective_value ≈ 0.001
        @test br.constraint_violation ≈ 1e-8
        @test br.solver_status == :Optimal
        @test br.solver == "Ipopt"
        @test br.total_allocations_bytes == 1_000_000
        @test br.total_allocs_count == 5000
        @test br.gc_time_ns == 100_000
        @test br.gc_count == 3
        @test br.gc_full_count == 1
        @test br.solver_options[:max_iter] == 100
        @test br.julia_version == "1.12.0"
        @test br.timestamp == DateTime(2026, 4, 14)
        @test br.runner == "local"
        @test br.n_threads == 1
    end

    @testset "MicroBenchmarkResult construction" begin
        eb1 = EvalBenchmark(
            times_ns = [50.0, 60.0, 55.0],
            gctimes_ns = [1.0, 2.0, 1.5],
            memory_bytes = 256,
            allocs = 5,
        )
        eb2 = EvalBenchmark(
            times_ns = [200.0, 250.0, 220.0],
            gctimes_ns = [10.0, 12.0, 11.0],
            memory_bytes = 1024,
            allocs = 20,
        )

        mbr = MicroBenchmarkResult(
            package = "DirectTrajOpt",
            package_version = "0.8.10",
            commit = "abc1234",
            benchmark_name = "evaluator_micro",
            N = 50,
            state_dim = 4,
            control_dim = 2,
            eval_benchmarks = Dict{Symbol,EvalBenchmark}(
                :jacobian => eb1,
                :hessian => eb2,
            ),
            julia_version = "1.12.0",
            timestamp = DateTime(2026, 4, 14),
            runner = "local",
            n_threads = 1,
        )

        @test mbr.package == "DirectTrajOpt"
        @test mbr.benchmark_name == "evaluator_micro"
        @test mbr.N == 50
        @test mbr.state_dim == 4
        @test mbr.control_dim == 2
        @test haskey(mbr.eval_benchmarks, :jacobian)
        @test haskey(mbr.eval_benchmarks, :hessian)
        @test mbr.eval_benchmarks[:jacobian].memory_bytes == 256
        @test mbr.eval_benchmarks[:hessian].allocs == 20
        @test mbr.n_threads == 1
    end

    @testset "Storage round-trip: BenchmarkResult" begin
        br = BenchmarkResult(
            package = "DirectTrajOpt",
            package_version = "0.8.10",
            commit = "deadbeef",
            benchmark_name = "roundtrip_test",
            N = 75,
            state_dim = 3,
            control_dim = 1,
            n_constraints = 40,
            n_variables = 120,
            wall_time_s = 2.5,
            iterations = 50,
            objective_value = 1e-4,
            constraint_violation = 1e-9,
            solver_status = :Optimal,
            solver = "Ipopt",
            total_allocations_bytes = 500_000,
            total_allocs_count = 2500,
            gc_time_ns = 50_000,
            gc_count = 2,
            gc_full_count = 0,
            solver_options = Dict{Symbol,Any}(:tol => 1e-8, :max_iter => 200),
            julia_version = "1.12.0",
            timestamp = DateTime(2026, 4, 14, 10, 30, 0),
            runner = "ci",
            n_threads = 4,
        )

        mktempdir() do dir
            path = save_results(dir, "test_bench", [br])
            @test isfile(path)
            @test endswith(path, ".jld2")

            loaded = load_results(path)
            @test length(loaded) == 1
            r = loaded[1]

            @test r.package == br.package
            @test r.commit == br.commit
            @test r.benchmark_name == br.benchmark_name
            @test r.N == br.N
            @test r.wall_time_s ≈ br.wall_time_s
            @test r.iterations == br.iterations
            @test r.solver_status == br.solver_status
            @test r.solver_options[:tol] ≈ br.solver_options[:tol]
            @test r.timestamp == br.timestamp
            @test r.n_threads == br.n_threads
        end
    end

    # ------------------------------------------------------------------ #
    # Harness tests (Tasks 4–6)
    # ------------------------------------------------------------------ #

    # Helper: build the bilinear test problem used across harness tests
    function _make_bilinear_prob(; N=10, Δt=0.1, u_bound=0.1, ω=0.1)
        Gx = sparse(Float64[0 0 0 1; 0 0 1 0; 0 -1 0 0; -1 0 0 0])
        Gy = sparse(Float64[0 -1 0 0; 1 0 0 0; 0 0 0 -1; 0 0 1 0])
        Gz = sparse(Float64[0 0 1 0; 0 0 0 -1; -1 0 0 0; 0 1 0 0])
        G(u) = ω * Gz + u[1] * Gx + u[2] * Gy

        traj = NamedTrajectory(
            (x=2rand(4,N).-1, u=u_bound*(2rand(2,N).-1), du=randn(2,N), ddu=randn(2,N), Δt=fill(Δt,N));
            controls=(:ddu,:Δt), timestep=:Δt,
            bounds=(u=u_bound, Δt=(0.01,0.5)),
            initial=(x=[1.0,0.0,0.0,0.0], u=zeros(2)), final=(u=zeros(2),),
            goal=(x=[0.0,1.0,0.0,0.0],),
        )
        integrators = [
            BilinearIntegrator(G, :x, :u, traj),
            DerivativeIntegrator(:u, :du, traj),
            DerivativeIntegrator(:du, :ddu, traj),
        ]
        J = QuadraticRegularizer(:u, traj, 1.0)
        return DirectTrajOptProblem(traj, J, integrators)
    end

    @testset "build_evaluator" begin
        prob = _make_bilinear_prob()
        traj = prob.trajectory
        expected_len = traj.dim * traj.N + traj.global_dim

        (evaluator, Z_vec) = build_evaluator(prob)

        # Z_vec has the right length
        @test length(Z_vec) == expected_len

        # eval_objective is callable
        obj = DirectTrajOpt.Solvers.MOI.eval_objective(evaluator, Z_vec)
        @test obj isa Float64
        @test isfinite(obj)

        # eval_hessian=false variant also works
        (ev2, Z_vec2) = build_evaluator(prob; eval_hessian=false)
        @test length(Z_vec2) == expected_len
        @test !ev2.eval_hessian
    end

    @testset "evaluator_dims" begin
        prob = _make_bilinear_prob()
        traj = prob.trajectory

        (evaluator, _) = build_evaluator(prob)
        dims = evaluator_dims(evaluator)

        @test dims.n_constraints == evaluator.n_constraints
        @test dims.n_variables == traj.dim * traj.N + traj.global_dim
        @test dims.n_jacobian_entries == length(evaluator.jacobian_structure)
        @test dims.n_hessian_entries == length(evaluator.hessian_structure)

        # Sanity: all positive
        @test dims.n_constraints > 0
        @test dims.n_variables > 0
        @test dims.n_jacobian_entries > 0
        @test dims.n_hessian_entries > 0
    end

    @testset "problem_dims" begin
        prob = _make_bilinear_prob()
        dims = problem_dims(prob)

        @test dims.N == prob.trajectory.N
        @test dims.state_dim == 4              # :x component has dim 4
        @test dims.control_dim == 2            # :ddu (2); Δt is timestep, excluded
        @test dims.n_constraints > 0
        @test dims.n_variables == prob.trajectory.dim * prob.trajectory.N + prob.trajectory.global_dim
    end

    @testset "evaluate_post_solve" begin
        prob = _make_bilinear_prob()
        # Solve first so trajectory has meaningful values
        solve!(prob; options=IpoptOptions(max_iter=5, print_level=0), verbose=false)

        post = evaluate_post_solve(prob)
        @test isfinite(post.objective_value)
        @test post.constraint_violation >= 0.0
        @test post.solver_status isa Symbol
        @test post.solver_status in (:Optimal, :Suboptimal)
    end

    @testset "snapshot_options" begin
        opts = IpoptOptions(max_iter=42, tol=1e-6, print_level=0)
        snap = snapshot_options(opts)

        @test snap isa Dict{Symbol,Any}
        @test snap[:max_iter] == 42
        @test snap[:tol] == 1e-6
        @test snap[:print_level] == 0
        @test haskey(snap, :eval_hessian)  # all fields captured
        @test length(snap) == length(fieldnames(typeof(opts)))
    end

    @testset "benchmark_solve!" begin
        prob = _make_bilinear_prob()
        opts = IpoptOptions(max_iter=5, print_level=0)

        result = benchmark_solve!(prob, opts; benchmark_name="test_bilinear", runner="test")

        # Identity fields
        @test result.package == "DirectTrajOpt"
        @test result.benchmark_name == "test_bilinear"
        @test result.runner == "test"
        @test !isempty(result.commit)
        @test !isempty(result.package_version) || result.package_version == "unknown"

        # Problem dims
        @test result.N == prob.trajectory.N
        @test result.state_dim == 4   # :x has dim 4
        @test result.control_dim == 2  # :ddu (2); Δt is timestep, excluded
        @test result.n_constraints > 0
        @test result.n_variables > 0

        # Solve metrics
        @test result.wall_time_s > 0.0
        @test isfinite(result.objective_value)
        @test result.constraint_violation >= 0.0
        @test result.solver == "Ipopt"
        @test result.solver_status isa Symbol

        # Memory
        @test result.total_allocations_bytes >= 0
        @test result.gc_time_ns >= 0
        @test result.peak_rss_delta_bytes >= 0   # clamped to non-negative
        @test isa(result.live_heap_delta_bytes, Int)  # can be negative (GC freed state)

        # Options snapshot includes max_iter
        @test haskey(result.solver_options, :max_iter)
        @test result.solver_options[:max_iter] == 5

        # Metadata
        @test result.julia_version == string(VERSION)
        @test result.n_threads == Threads.nthreads()
        @test result.timestamp isa Dates.DateTime
    end

    @testset "benchmark_solve! do-block (generic)" begin
        # Run a fake "solve" that just allocates and sleeps
        result = benchmark_solve!(
            package         = "TestPkg",
            package_version = "0.0.0",
            commit          = "abc1234",
            solver          = "FakeSolver",
            benchmark_name  = "fake_solve",
            N               = 42,
            state_dim       = 7,
            control_dim     = 3,
            n_constraints   = 100,
            n_variables     = 200,
            iterations      = 5,
            objective_value = 0.001,
            constraint_violation = 1e-8,
            solver_status   = :Optimal,
            solver_options  = Dict{Symbol,Any}(:max_iter => 5, :fidelity => 0.999),
            runner          = "test",
        ) do
            # Simulate a solve that allocates
            v = zeros(10_000)
            for _ in 1:100
                v .+= rand(10_000)
            end
            return nothing
        end

        # Identity
        @test result.package == "TestPkg"
        @test result.solver == "FakeSolver"
        @test result.benchmark_name == "fake_solve"
        @test result.commit == "abc1234"

        # All caller-provided fields round-tripped
        @test result.N == 42
        @test result.state_dim == 7
        @test result.control_dim == 3
        @test result.n_constraints == 100
        @test result.n_variables == 200
        @test result.iterations == 5
        @test result.objective_value == 0.001
        @test result.constraint_violation == 1e-8
        @test result.solver_status == :Optimal

        # Timing/alloc captured from the closure
        @test result.wall_time_s > 0.0
        @test result.total_allocations_bytes > 0   # the closure allocated
        @test result.total_allocs_count > 0

        # Run-time metadata passed through solver_options
        @test result.solver_options[:fidelity] == 0.999
        @test result.solver_options[:max_iter] == 5

        # Julia + runner metadata populated
        @test result.julia_version == string(VERSION)
        @test result.runner == "test"
        @test result.n_threads == Threads.nthreads()
    end

    @testset "benchmark_solve! do-block with post_solve" begin
        # Simulate a solver that stores state; post_solve reads it
        solver_state = Ref(0)

        result = benchmark_solve!(
            package        = "TestPkg",
            solver         = "PostSolveTest",
            benchmark_name = "post_solve_test",
            N              = 10,
            state_dim      = 2,
            control_dim    = 1,
            n_constraints  = 20,
            n_variables    = 30,
            solver_options = Dict{Symbol,Any}(:max_iter => 42),
            commit         = "abc1234",
            post_solve     = function(_)
                # post_solve runs AFTER the closure; access state written there
                return (
                    iterations      = solver_state[],
                    objective_value = 1e-6,
                    solver_status   = :Optimal,
                )
            end,
        ) do
            solver_state[] = 7  # "solver did 7 iterations"
            return nothing
        end

        # Verify post_solve overrides took effect
        @test result.iterations == 7
        @test result.objective_value == 1e-6
        @test result.solver_status == :Optimal

        # Timing still captured from the closure
        @test result.wall_time_s >= 0.0
    end

    @testset "trial_to_eval_benchmark" begin
        # Create a small BenchmarkTools trial
        trial = @benchmark sum(1:100) seconds=0.5

        eb = trial_to_eval_benchmark(trial)

        @test eb isa EvalBenchmark
        @test eb.times_ns == Float64.(trial.times)
        @test eb.gctimes_ns == Float64.(trial.gctimes)
        @test eb.memory_bytes == Int(trial.memory)
        @test eb.allocs == Int(trial.allocs)
        @test length(eb.times_ns) > 0
        @test eb.median_ns > 0.0
        @test eb.min_ns > 0.0
        @test eb.mean_ns > 0.0
        # Derived stats are consistent
        @test eb.min_ns <= eb.median_ns
        @test eb.min_ns <= eb.mean_ns
    end

    @testset "Storage round-trip: MicroBenchmarkResult" begin
        eb = EvalBenchmark(
            times_ns = [100.0, 110.0, 105.0],
            gctimes_ns = [2.0, 3.0, 2.5],
            memory_bytes = 512,
            allocs = 10,
        )

        mbr = MicroBenchmarkResult(
            package = "DirectTrajOpt",
            package_version = "0.8.10",
            commit = "cafebabe",
            benchmark_name = "micro_roundtrip",
            N = 30,
            state_dim = 2,
            control_dim = 1,
            eval_benchmarks = Dict{Symbol,EvalBenchmark}(:jacobian => eb),
            julia_version = "1.12.0",
            timestamp = DateTime(2026, 4, 14, 12, 0, 0),
            runner = "local",
            n_threads = 2,
        )

        mktempdir() do dir
            path = save_micro_results(dir, "micro_bench", mbr)
            @test isfile(path)
            @test endswith(path, ".jld2")

            loaded = load_micro_results(path)

            @test loaded.package == mbr.package
            @test loaded.commit == mbr.commit
            @test loaded.benchmark_name == mbr.benchmark_name
            @test loaded.N == mbr.N
            @test haskey(loaded.eval_benchmarks, :jacobian)
            @test loaded.eval_benchmarks[:jacobian].memory_bytes == 512
            @test loaded.eval_benchmarks[:jacobian].median_ns ≈ eb.median_ns
            @test loaded.timestamp == mbr.timestamp
        end
    end

    @testset "compare_results" begin
        # Create baseline result
        baseline = BenchmarkResult(
            package = "DirectTrajOpt",
            package_version = "0.8.10",
            commit = "baseline123",
            benchmark_name = "regression_test",
            N = 50,
            state_dim = 4,
            control_dim = 2,
            n_constraints = 30,
            n_variables = 150,
            wall_time_s = 1.0,
            iterations = 25,
            objective_value = 0.001,
            constraint_violation = 1e-8,
            solver_status = :Optimal,
            solver = "Ipopt",
            total_allocations_bytes = 1_000_000,
            total_allocs_count = 5000,
            gc_time_ns = 100_000,
            gc_count = 2,
            gc_full_count = 0,
            solver_options = Dict{Symbol,Any}(:max_iter => 100),
            julia_version = "1.12.0",
            timestamp = DateTime(2026, 4, 14),
            runner = "local",
            n_threads = 1,
        )

        # Create current result with 20% wall time regression and 10% allocation improvement
        current = BenchmarkResult(
            package = "DirectTrajOpt",
            package_version = "0.8.10",
            commit = "current456",
            benchmark_name = "regression_test",
            N = 50,
            state_dim = 4,
            control_dim = 2,
            n_constraints = 30,
            n_variables = 150,
            wall_time_s = 1.2,  # 20% increase
            iterations = 25,
            objective_value = 0.001,
            constraint_violation = 1e-8,
            solver_status = :Optimal,
            solver = "Ipopt",
            total_allocations_bytes = 900_000,  # 10% decrease
            total_allocs_count = 4500,
            gc_time_ns = 95_000,
            gc_count = 2,
            gc_full_count = 0,
            solver_options = Dict{Symbol,Any}(:max_iter => 100),
            julia_version = "1.12.0",
            timestamp = DateTime(2026, 4, 14),
            runner = "local",
            n_threads = 1,
        )

        rows = compare_results([baseline], [current])

        @test length(rows) == 1
        row = rows[1]

        # Check identity fields
        @test row.benchmark_name == "regression_test"
        @test row.solver == "Ipopt"
        @test row.N == 50
        @test row.state_dim == 4

        # Check baseline and current values
        @test row.baseline_wall_s ≈ 1.0
        @test row.current_wall_s ≈ 1.2
        @test row.baseline_alloc_bytes == 1_000_000
        @test row.current_alloc_bytes == 900_000

        # Check percent changes
        @test row.wall_time_pct_change > 15.0  # Should be 20%
        @test row.alloc_bytes_pct_change < 0.0  # Should be -10%

        # Check regression flag
        @test row.has_regression == true  # Wall time regression exceeds default threshold
    end

end
