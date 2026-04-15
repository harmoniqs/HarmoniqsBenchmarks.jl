using HarmoniqsBenchmarks
using Test
using Dates

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
