using Printf

"""
    AllocFrameSummary(name, count, bytes)

One row of an aggregated `AllocProfileResult` view — a frame, leaf call site,
or allocated type with its (optionally scaled) sample count and byte total.
"""
struct AllocFrameSummary
    name::String
    count::Int
    bytes::Int
end

# Frames that come from the Julia runtime, the allocation profiler itself,
# or the loader. Carry no information about user-code hotpaths.
const _ALLOC_NOISE_FRAME_PATTERNS = [
    "Profile.Allocs",
    "gc-alloc-profiler",
    "gc-stock.c",
    "gc.c:",
    "jl_apply",
    "jl_toplevel_",
    "ijl_toplevel_",
    "jl_interpret_toplevel_thunk",
    "jl_repl_entrypoint",
    "interpreter.c",
    "_include(",
    "include_string(",
    "loading.jl",
    "client.jl",
    "_start() at sys.so",
    "ip:0x",
    "_start at ",
    " at Base.jl:",
    "true_main at jlapi.c",
    "__libc_start_main",
    "loader_exe.c",
    "jl_system_image_data",
    "macro expansion at Allocs.jl",
    "boot.jl:",
    "jl_f__call_latest",
]

# Wrapper frames inside HarmoniqsBenchmarks itself — when the caller wants the
# first frame in *their* code, the harness call stack is noise.
const _ALLOC_HBJ_WRAPPER_FRAME_PATTERNS = ["benchmark_memory!", "HarmoniqsBenchmarks"]

const _ALLOC_NOISE_TYPE_PATTERNS = ["Profile.Allocs"]

_is_noise_frame(f, extra) =
    any(p -> occursin(p, f), _ALLOC_NOISE_FRAME_PATTERNS) || any(p -> occursin(p, f), extra)
_is_wrapper_frame(f, extra) =
    any(p -> occursin(p, f), _ALLOC_HBJ_WRAPPER_FRAME_PATTERNS) ||
    any(p -> occursin(p, f), extra)
_is_noise_type(t) = any(p -> occursin(p, t), _ALLOC_NOISE_TYPE_PATTERNS)

function _first_user_frame(stack, extra_noise, extra_wrappers)
    for f in stack
        _is_noise_frame(f, extra_noise) && continue
        _is_wrapper_frame(f, extra_wrappers) && continue
        return f
    end
    return isempty(stack) ? "<empty>" : stack[end]
end

_alloc_scale(profile::AllocProfileResult, scale::Bool) =
    scale ? max(1.0, 1 / profile.sample_rate) : 1.0

function _rank(by_key::Dict{String,Tuple{Int,Int}}, k::Int, scale::Real)
    rows = Vector{AllocFrameSummary}()
    sizehint!(rows, length(by_key))
    for (name, (cnt, bytes)) in by_key
        push!(
            rows,
            AllocFrameSummary(name, round(Int, cnt * scale), round(Int, bytes * scale)),
        )
    end
    sort!(rows; by = r -> -r.bytes)
    return rows[1:min(k, length(rows))]
end

"""
    top_alloc_types(profile; k=15, scale=true) -> Vector{AllocFrameSummary}

Aggregate samples in `profile` by allocated type, sorted by bytes descending.
With `scale=true` (default), counts and bytes are extrapolated by
`1 / profile.sample_rate` so totals match the actual run rather than the
recorded sample.
"""
function top_alloc_types(profile::AllocProfileResult; k::Int = 15, scale::Bool = true)
    by_type = Dict{String,Tuple{Int,Int}}()
    for s in profile.samples
        _is_noise_type(s.type_name) && continue
        cnt, bytes = get(by_type, s.type_name, (0, 0))
        by_type[s.type_name] = (cnt + 1, bytes + s.size_bytes)
    end
    return _rank(by_type, k, _alloc_scale(profile, scale))
end

"""
    top_alloc_frames(profile; k=25, scale=true, drop_wrappers=true, extra_noise=String[], extra_wrappers=String[])
        -> Vector{AllocFrameSummary}

Aggregate samples by every frame in their stacktraces (a single allocation
contributes once per frame on its stack). Useful for finding hot regions of
code that participate in many allocations even if they are not the leaf call.

Pass `extra_noise` / `extra_wrappers` to drop additional caller-specific
patterns alongside the built-in Julia runtime / HBJ wrapper filters.
"""
function top_alloc_frames(
    profile::AllocProfileResult;
    k::Int = 25,
    scale::Bool = true,
    drop_wrappers::Bool = true,
    extra_noise::Vector{String} = String[],
    extra_wrappers::Vector{String} = String[],
)
    by_frame = Dict{String,Tuple{Int,Int}}()
    for s in profile.samples
        _is_noise_type(s.type_name) && continue
        for frame in s.stacktrace
            _is_noise_frame(frame, extra_noise) && continue
            drop_wrappers && _is_wrapper_frame(frame, extra_wrappers) && continue
            cnt, bytes = get(by_frame, frame, (0, 0))
            by_frame[frame] = (cnt + 1, bytes + s.size_bytes)
        end
    end
    return _rank(by_frame, k, _alloc_scale(profile, scale))
end

"""
    top_alloc_leaves(profile; k=25, scale=true, extra_noise=String[], extra_wrappers=String[])
        -> Vector{AllocFrameSummary}

Aggregate samples by their *first non-noise, non-wrapper frame* — i.e. the
call site in user code that directly produced the allocation. Most useful
view for triaging allocation hotspots.
"""
function top_alloc_leaves(
    profile::AllocProfileResult;
    k::Int = 25,
    scale::Bool = true,
    extra_noise::Vector{String} = String[],
    extra_wrappers::Vector{String} = String[],
)
    by_leaf = Dict{String,Tuple{Int,Int}}()
    for s in profile.samples
        _is_noise_type(s.type_name) && continue
        leaf = _first_user_frame(s.stacktrace, extra_noise, extra_wrappers)
        cnt, bytes = get(by_leaf, leaf, (0, 0))
        by_leaf[leaf] = (cnt + 1, bytes + s.size_bytes)
    end
    return _rank(by_leaf, k, _alloc_scale(profile, scale))
end

function _fmt_alloc_bytes(b::Real)
    b >= 1 << 30 && return @sprintf("%.2f GB", b / (1 << 30))
    b >= 1 << 20 && return @sprintf("%.2f MB", b / (1 << 20))
    b >= 1 << 10 && return @sprintf("%.2f KB", b / (1 << 10))
    return @sprintf("%d B", Int(round(b)))
end

_truncate_name(s, n) = length(s) <= n ? s : string(first(s, n - 1), "…")

function _print_summary_table(
    io::IO,
    header::AbstractString,
    rows::Vector{AllocFrameSummary},
    name_width::Int,
)
    println(io, "\n", header)
    println(io, rpad("  bytes", 14), rpad("samples", 10), "name")
    for r in rows
        @printf(
            io,
            "  %-12s %-8d %s\n",
            _fmt_alloc_bytes(r.bytes),
            r.count,
            _truncate_name(r.name, name_width)
        )
    end
end

"""
    report_alloc_profile(profile; io=stdout, k_types=10, k_leaves=20, k_frames=20,
                         scale=true, extra_noise=String[], extra_wrappers=String[])

Pretty-print a three-section human-readable report of `profile`: top allocated
types, top leaf call sites, and top all-frames. Sections are sorted by
allocated bytes; counts and totals are scaled by `1 / sample_rate` unless
`scale=false`.
"""
function report_alloc_profile(
    profile::AllocProfileResult;
    io::IO = stdout,
    k_types::Int = 10,
    k_leaves::Int = 20,
    k_frames::Int = 20,
    scale::Bool = true,
    extra_noise::Vector{String} = String[],
    extra_wrappers::Vector{String} = String[],
)
    scale_factor = _alloc_scale(profile, scale)
    @printf(
        io,
        "=== Alloc profile: %s / %s (N=%d, sample_rate=%g, samples=%d, total≈%s) ===\n",
        profile.solver,
        profile.benchmark_name,
        profile.N,
        profile.sample_rate,
        profile.total_count,
        _fmt_alloc_bytes(profile.total_bytes * scale_factor),
    )
    _print_summary_table(
        io,
        "Top $k_types allocated types (scaled ×$(Int(round(scale_factor)))):",
        top_alloc_types(profile; k = k_types, scale = scale),
        120,
    )
    _print_summary_table(
        io,
        "Top $k_leaves leaf call sites (scaled ×$(Int(round(scale_factor)))):",
        top_alloc_leaves(
            profile;
            k = k_leaves,
            scale = scale,
            extra_noise = extra_noise,
            extra_wrappers = extra_wrappers,
        ),
        140,
    )
    _print_summary_table(
        io,
        "Top $k_frames frames (scaled ×$(Int(round(scale_factor)))):",
        top_alloc_frames(
            profile;
            k = k_frames,
            scale = scale,
            extra_noise = extra_noise,
            extra_wrappers = extra_wrappers,
        ),
        140,
    )
    return nothing
end
