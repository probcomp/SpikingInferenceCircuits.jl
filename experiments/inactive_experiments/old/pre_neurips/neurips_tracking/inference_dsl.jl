using DiscreteIRTransforms: EnumeratedDomain 
using MacroTools
using MacroTools: unblock, rmlines, postwalk

struct InferenceBundle
    genfn
    argtypes
    bijections
end

_lift(v::Vector) = EnumeratedDomain(v)
_lift(::Type{Bool}) = EnumeratedDomain([true, false])
_lift(r::UnitRange{Int64}) = EnumeratedDomain(r)
# function _concrete(r::EnumeratedDomain{Vector{Bool}})
#     @assert(length(r.vals) == 2)
#     return FiniteDomain(2)
# end
# function _concrete(r::EnumeratedDomain{<:UnitRange})
#     v = r.vals
#     return FiniteDomain(v.stop - v.start)
# end
# function _concrete(r::EnumeratedDomain{<:Vector})
#     return FiniteDomain(length(r))
# end
_concrete(r) = FiniteDomain(length(r.vals))

function striptypes(ex::Expr)
    ex.head == :(::) || return ex
    return ex.args[end]
end

macro compile(expr)
    @assert(@capture(expr, model_(args__)))
    argtypes = striptypes.(args)
    ex = quote
        argtypes = _lift.([$(argtypes...)])
        cpts, bijections = DiscreteIRTransforms.to_indexed_cpts($model, argtypes)
        InferenceBundle(cpts, argtypes, bijections)
    end
    ex = postwalk(unblock ∘ rmlines, ex)
    esc(ex)
end

# m = @compile object_motion_step(1:10)
# sp = @compile step_proposal(1:10, 1:10)

function mh(m::InferenceBundle, p::InferenceBundle)
    return MHKernel(m.genfn, Tuple(_concrete.(m.argtypes)),
                    p.genfn, Tuple(_concrete.(p.argtypes)))
end

function mh(args::MHKernel...) 
    return MH(MHKernel[args...])
end

function is(m::InferenceBundle, p::InferenceBundle)
    return ISParticle(m.genfn, p.genfn,
                      Tuple(_concrete.(m.argtypes)),
                      Tuple(_concrete.(p.argtypes)))
end

# Not quite used yet.
struct ObservationSet
    v::Set{Symbol}
end
observe(args::Symbol...) = ObservationSet(Set{Symbol}([args...]))

function loop(kers::Tuple, obs::ObservationSet, n::Int)
    num_particles = n
    is_particle = kers[1]
    return SMC(n, is_particle)
end

macro infer(expr)
    @assert(@capture(expr, function prog_(model_, proposal_) 
                     body__ end))
    body = map(postwalk.(rmlines ∘ unblock, body)) do e
        @capture(e, loop(args__) do 
                     loopbody__ 
                 end) || return e
        return quote
            kers = ($(loopbody...), )
            loop(kers, $(args...))
        end
    end
    final = quote
        function $prog(model, proposal)
            $(body...)
        end
    end
    final = postwalk(rmlines ∘ unblock, final)
    esc(final)
end

# Implies SMC -- loop and observe both specified.
# @infer function infprog(model, proposal)
#     loop(observe(:x), 100) do
#         is(model, proposal)
#         mh(model, proposal)
#     end
# end

# circuit = infprog(m, sp)
