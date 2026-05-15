using DirectTrajOpt

"""
    IpoptCapture

Mutable struct holding the final Ipopt iteration's state. Populated by
the callback returned by [`ipopt_capture`](@ref). Pass that callback to
`DirectTrajOpt.solve!(prob; callback=...)` and read fields after the
solve returns.

Fields mirror the `IpoptOptimizerState` NamedTuple fields exposed by
DirectTrajOpt's Ipopt callback infrastructure.
"""
mutable struct IpoptCapture
    iter_count::Int
    inf_pr::Float64
    inf_du::Float64
    objective::Float64
end

IpoptCapture() = IpoptCapture(0, Inf, Inf, NaN)

"""
    ipopt_capture_callback(state::IpoptCapture) -> Function

Bare callback closure (signature matches DirectTrajOpt's user-callback
contract: `(optimizer, optimizer_state::IpoptOptimizerState; kwargs...) -> Bool`)
that copies the per-iter Ipopt metrics into `state`. Use this if you need
to compose with other callbacks via
`DirectTrajOpt.Callbacks.callback_factory`. For the common one-shot case
use [`ipopt_capture`](@ref) instead.
"""
function ipopt_capture_callback(state::IpoptCapture)
    return function(_optimizer, optimizer_state; kwargs...)
        state.iter_count = Int(optimizer_state.iter_count)
        state.inf_pr     = Float64(optimizer_state.inf_pr)
        state.inf_du     = Float64(optimizer_state.inf_du)
        state.objective  = Float64(optimizer_state.obj_value)
        return true
    end
end

"""
    ipopt_capture() -> (state::IpoptCapture, callback::Function)

Builds an [`IpoptCapture`](@ref) and a matching callback in the form
DirectTrajOpt's `solve!(...; callback=...)` expects. After the solve
returns, `state` holds the final iteration's `iter_count`, primal/dual
infeasibility, and objective value.

```julia
state, cb = ipopt_capture()
DirectTrajOpt.solve!(prob; options = opts, callback = cb)
ipopt_iterations(state)             # ::Int — Ipopt iters used
ipopt_primal_infeasibility(state)   # ::Float64 — final inf_pr
```
"""
function ipopt_capture()
    state = IpoptCapture()
    user_cb = ipopt_capture_callback(state)
    factory = DirectTrajOpt.Callbacks.callback_factory(user_cb)
    return state, factory
end

"""
    ipopt_iterations(state::IpoptCapture) -> Int

Final Ipopt iteration count captured by [`ipopt_capture`](@ref).
"""
ipopt_iterations(state::IpoptCapture) = state.iter_count

"""
    ipopt_primal_infeasibility(state::IpoptCapture) -> Float64

Final Ipopt primal infeasibility (`inf_pr`) captured by [`ipopt_capture`](@ref).
Non-negative; values ≤ `feas_tol` indicate the dynamics constraints were
satisfied to the configured tolerance.
"""
ipopt_primal_infeasibility(state::IpoptCapture) = state.inf_pr
