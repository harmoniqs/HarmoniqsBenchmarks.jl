using JLD2

"""
    save_results(dir, name, results::Vector{BenchmarkResult}) -> String

Save a vector of `BenchmarkResult` to `dir/name_commit.jld2`.
Returns the path of the saved file.
"""
function save_results(dir::AbstractString, name::AbstractString, results::Vector{BenchmarkResult})::String
    mkpath(dir)
    # Use commit from first result (all should share the same commit in a benchmark run)
    commit = isempty(results) ? "unknown" : results[1].commit
    filename = "$(name)_$(commit).jld2"
    path = joinpath(dir, filename)
    jldsave(path; results)
    return path
end

"""
    load_results(path) -> Vector{BenchmarkResult}

Load a vector of `BenchmarkResult` from a JLD2 file.
"""
function load_results(path::AbstractString)::Vector{BenchmarkResult}
    return jldopen(path, "r") do f
        f["results"]
    end
end

"""
    save_micro_results(dir, name, result::MicroBenchmarkResult) -> String

Save a `MicroBenchmarkResult` to `dir/name_commit.jld2`.
Returns the path of the saved file.
"""
function save_micro_results(dir::AbstractString, name::AbstractString, result::MicroBenchmarkResult)::String
    mkpath(dir)
    filename = "$(name)_$(result.commit).jld2"
    path = joinpath(dir, filename)
    jldsave(path; result)
    return path
end

"""
    load_micro_results(path) -> MicroBenchmarkResult

Load a `MicroBenchmarkResult` from a JLD2 file.
"""
function load_micro_results(path::AbstractString)::MicroBenchmarkResult
    return jldopen(path, "r") do f
        f["result"]
    end
end
