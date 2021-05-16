using Gen

@gen (static) function iswet(in::Nothing)
    raining ~ bernoulli(in === nothing ? 0.3 : 0.3)
    sprinkler ~ bernoulli(in === nothing ? 0.3 : 0.3)
    grasswet ~ bernoulli(raining || sprinkler ? 0.9 : 0.1)
    return grasswet
end

@gen (static) function raining_proposal(trace)
    raining ~ bernoulli(trace[:raining] ? 0.5 : 0.5)
    return raining
end

@gen (static) function sprinkler_proposal(trace)
    sprinkler ~ bernoulli(trace[:sprinkler] ? 0.5 : 0.5)
    return sprinkler
end

@gen (static) function smart_block_proposal(trace)
    raining ~ bernoulli(trace[:grasswet] ? 0.55 : 0.2)
    sprinkler ~ bernoulli(trace[:grasswet] && !raining ? 0.9 : 0.3)
end

@load_generated_functions()

tr, _ = generate(iswet, (nothing,), choicemap((:grasswet, true), (:sprinkler, true), (:raining, true)))

function run_block_mh(initial_tr, iters)
    tr = initial_tr
    states = []
    for i=1:iters
        tr, _ = Gen.mh(tr, smart_block_proposal, ())
        push!(states, (tr[:raining], tr[:sprinkler]))
    end
    return states
end

counts(states) = Dict(
    (true, true) => length(filter(states) do (r, s); r && s; end),
    (false, true) => length(filter(states) do (r, s); !r && s; end),
    (true, false) => length(filter(states) do (r, s); r && !s; end),
    (false, false) => length(filter(states) do (r, s); !r && !s; end)
)

states = run_block_mh(tr, 100)