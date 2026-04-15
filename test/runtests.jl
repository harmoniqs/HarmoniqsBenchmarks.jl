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

        # Options snapshot includes max_iter
        @test haskey(result.solver_options, :max_iter)
        @test result.solver_options[:max_iter] == 5

        # Metadata
        @test result.julia_version == string(VERSION)
        @test result.n_threads == Threads.nthreads()
        @test result.timestamp isa Dates.DateTime
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

end
