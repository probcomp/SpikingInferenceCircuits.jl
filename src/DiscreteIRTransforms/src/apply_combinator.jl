module ApplyCombinator

using FunctionalCollections
using Gen

struct Apply{T, U} <: Gen.GenerativeFunction{PersistentVector{T}, Gen.VectorTrace{Gen.MapType}}
    kernels::Vector{GenerativeFunction{<:T, <:U}}
end
Apply(kernels::Vector{<:GenerativeFunction}) = Apply{Any, Gen.Trace}(kernels)

vec_trace(gen_fn::Apply{T, U}, traces::PersistentVector{U}, args) where {T, U} =
    Gen.VectorTrace{Gen.MapType, T, U}(gen_fn,  
        traces, PersistentVector{T}([Gen.get_retval(tr) for tr in traces]),
        args,
        sum((Gen.get_score(tr) for tr in traces)),
        sum(project(tr, EmptySelection()) for tr in traces),
        length(args[1]),
        sum(1 for tr in traces if !isempty(get_choices(tr)))
    )
function Gen.simulate(gen_fn::Apply{T, U}, args::Tuple) where {T, U}
    @assert all(length(vec) == length(gen_fn.kernels) for vec in args)
    return vec_trace(gen_fn, PersistentVector{U}([
        simulate(kernel, call_args) for (kernel, call_args...) in zip(gen_fn.kernels, args...)
    ]), args)
end

function Gen.generate(gen_fn::Apply{T, U}, args::Tuple, constraints::ChoiceMap) where {T, U}
    @assert all(length(vec) == length(gen_fn.kernels) for vec in args)
    traces_and_weights = [
        generate(kernel, call_args, get_submap(constraints, i))
        for (i, (kernel, call_args...)) in enumerate(zip(gen_fn.kernels, args...))
    ]
    traces = PersistentVector{U}([tr for (tr, _) in traces_and_weights])
    @assert traces isa PersistentVector{U}
    weight = sum(w for (_, w) in traces_and_weights)
    return (vec_trace(gen_fn, traces, args), weight)
end

end # module