module InferenceDSL

using SpikingInferenceCircuits
using SpikingInferenceCircuits.DiscreteIRTransforms: EnumeratedDomain 
using MacroTools
using MacroTools: unblock, rmlines, postwalk
using Gen

onehot(x, dom) =
    x < first(dom) ? onehot(first(dom), dom) :
    x > last(dom)  ? onehot(last(dom), dom)  :
                 [i == x ? 1. : 0. for i in dom]

maybe_one_off(idx, prob, dom) =
    (1 - prob) * onehot(idx, dom) +
    prob/2 * onehot(idx - 1, dom) +
    prob/2 * onehot(idx + 1, dom)

#####
##### (Model + proposal)
#####

XDOMAIN = 1:10
@gen (static) function object_motion_step(xₜ₋₁)
    xₜ   ~ categorical(maybe_one_off(xₜ₋₁, 0.8, XDOMAIN))
    obsₜ ~ categorical(maybe_one_off(xₜ, 0.5, XDOMAIN))
    return obsₜ
end

@gen (static) function step_proposal(xₜ₋₁, obsₜ)
    xₜ ~ categorical(maybe_one_off(obsₜ, 0.5, XDOMAIN))
end

#####
##### Inference circuits
#####

struct InferenceBundle
    genfn
    argtypes
    bijections
end

_lift(::Type{Bool}) = EnumeratedDomain([true, false])
_lift(r::UnitRange{Int64}) = EnumeratedDomain(r)

function striptypes(ex::Expr)
    ex.head == :(::) || return ex
    return ex.args[end]
end

macro compile(expr)
    @assert(@capture(expr, model_(args__)))
    argtypes = striptypes.(args)
    ex = quote
        argtypes = _lift.([$(argtypes...)])
        cpts, bijections = SpikingInferenceCircuits.DiscreteIRTransforms.to_indexed_cpts($model, argtypes)
        InferenceBundle(cpts, argtypes, bijections)
    end
    ex = postwalk(unblock ∘ rmlines, ex)
    esc(ex)
end

b = @compile object_motion_step(1:10)
display(b)

# Implies SMC -- loop and observe both specified.
#@infer function infprog(model, proposal)
#    loop(observe(:x)) do
#        is(model, proposal)
#        mh(model, [:y, :z])
#    end
#end
#
#circuit = infprog(m, sp)

end # module
