# TASKS

## Follow-ups

### Register in Julia General Registry
Once the package API stabilizes (~v0.2.0), register in the General registry so downstream packages can use standard `Pkg.add("HarmoniqsBenchmarks")` instead of `Pkg.add(url=...)`.

Steps:
1. Ensure Project.toml has proper `[compat]` bounds for all deps
2. Bump to a semver version (e.g. 0.2.0)
3. Tag the release
4. Register via Registrator.jl / JuliaRegistries

After registration, update downstream CI workflows:
- `harmoniqs/DirectTrajOpt.jl/.github/workflows/benchmark.yml` — remove `Pkg.add(url=...)` step
- `harmoniqs/Piccolissimo.jl/.github/workflows/benchmark.yml` — same
- `harmoniqs/Intonato.jl/.github/workflows/benchmark.yml` — same (when added)
- Any other packages using HarmoniqsBenchmarks

### Add Altissimo dispatch to benchmark_solve!
Currently `benchmark_solve!` dispatches on `DirectTrajOpt.Problems.DirectTrajOptProblem` + `AbstractSolverOptions`. Altissimo and PulseTuningProblem need their own dispatch:
- `benchmark_solve!(qcp::QuantumControlProblem, opts::AltissimoOptions; ...)`
- `benchmark_solve!(ptp::PulseTuningProblem; max_iter, ...)`

These currently require manual `@timed` + `BenchmarkResult` construction in each benchmark. A clean dispatch would make Piccolissimo/Intonato benchmarks simpler.

### Aggregator tool
A separate tool/script (in a central `harmoniqs-benchmarks` repo) that:
- Downloads JLD2 artifacts from each package's GitHub Actions
- Generates cross-package comparison tables
- Maintains a historical archive by package version

### Allocation profiling integration
Profile.Allocs is wired up via `benchmark_memory!` (sibling do-block to `benchmark_solve!`) returning an `AllocProfileResult` saved via `save_alloc_profile` to a distinct `*_allocs.jld2`. Still open: AllocCheck.jl regression guards once hotspots are fixed, and an optional PProf .pb.gz sidecar export.
