# usings, includes, ...

@gen (static) function iswet(in::Nothing)
    raining ~ bernoulli(in === nothing ? 0.3 : 0.3)
    sprinkler ~ bernoulli(in === nothing ? 0.3 : 0.3)
    grasswet ~ bernoulli(raining || sprinkler ? 0.9 : 0.1)
    return grasswet
end
@gen (static) function raining_proposal(raining, sprinkler, grasswet)
    raining ~ bernoulli(raining ? 0.5 : 0.5)
    return raining
end
@gen (static) function sprinkler_proposal(raining, sprinkler, grasswet)
    sprinkler ~ bernoulli(sprinkler ? 0.5 : 0.5)
    return sprinkler
end

### Inference DSL code

@compilable iswet(::Nothing)   # Could also use: `@compilable iswet(::Dom([nothing]))`
@compilable raining_proposal(::Bool, ::Bool, ::Bool)
@compilable sprinkler_proposal(::Bool, ::Bool, ::Bool)

mh_circuit = MH(is_wet, [raining_proposal, sprinkler_proposal])

### A possible compilation target:

binary = EnumeratedDomain([true, false])
iswet_cpts, _ = to_indexed_cpts(iswet, [EnumeratedDomain([nothing])])
raining_proposal_cpts, _ = to_indexed_cpts(raining_proposal, [binary, binary, binary])
sprinkler_proposal_cpts, _ = to_indexed_cpts(sprinkler_proposal, [binary, binary, binary])

raining_mh_kernel = MHKernel(
    iswet_cpts, (FiniteDomain(1),),
    raining_proposal_cpts, (FiniteDomain(2), FiniteDomain(2), FiniteDomain(2))
)
sprinkler_mh_kernel = MHKernel(
    iswet_cpts, (in=FiniteDomain(1),),
    sprinkler_proposal_cpts, (FiniteDomain(2), FiniteDomain(2), FiniteDomain(2))
)

mh_circuit = MH([raining_mh_kernel, sprinkler_mh_kernel])