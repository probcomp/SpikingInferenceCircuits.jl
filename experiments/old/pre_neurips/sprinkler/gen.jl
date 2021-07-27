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
@load_generated_functions()

function run_unblocked_mh(initial_trace, iters)
    tr = initial_trace
    traces = []
    for _=1:iters
        tr, _ = Gen.mh(tr, raining_proposal, ())
        push!(traces, tr)
        tr, _ = Gen.mh(tr, sprinkler_proposal)
        push!(traces, tr)
    end
    return traces
end

initial_trace, _ = generate(iswet, (nothing,), choicemap((:grasswet, true), (:sprinkler, true), (:raining, true)))
# TODO: run unblocked MH on initial trace...


@gen (static) function smart_block_proposal(trace)
    raining ~ bernoulli(trace[:grasswet] ? 0.55 : 0.2)
    sprinkler ~ bernoulli(trace[:grasswet] && !raining ? 0.9 : 0.3)
end

@load_generated_functions()

function run_block_mh(initial_tr, iters)
    tr = initial_tr
    states = []
    for i=1:iters
        tr, _ = Gen.mh(tr, smart_block_proposal, ())
        push!(states, (tr[:raining], tr[:sprinkler]))
    end
    return states
end

# run block MH on the same initial trace from above.
traces = run_block_mh(tr, 100)




# util to count the number of times each assignment to `(raining, sprinkler)`  was visited,
# given `states` as a vector of `(raining, sprinkler)` pairs
counts(states) = Dict(
    (true, true) => length(filter(states) do (r, s); r && s; end),
    (false, true) => length(filter(states) do (r, s); !r && s; end),
    (true, false) => length(filter(states) do (r, s); r && !s; end),
    (false, false) => length(filter(states) do (r, s); !r && !s; end)
)
