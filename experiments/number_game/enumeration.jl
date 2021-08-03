membership_float_vec(tr::Trace) = membership_float_vec(tree(tr))
get_number_membership_probs(traces::Vector{<:Trace}, trace_probs::Vector{<:Real}) =
    sum(membership_float_vec(trace) * prob for (trace, prob) in zip(traces, trace_probs))
get_number_membership_probs(obs_choicemaps, nsteps::Int) =
    get_number_membership_probs(get_weighted_traces(obs_choicemaps, nsteps)...)

function get_weighted_traces(obs_choicemap, nsteps) # for multiple steps
    all_cms = get_all_trees(MAXDEPTH())
    println("generated all cms")
    logweighted_traces = [
        generate(
            model, (nsteps,),
            Gen.merge(DynamicModels.nest_at(:init => :latents, cm), obs_choicemap)
        )
        for cm in all_cms
    ]
    println("got all traces")
    filtered = [(tr, wt) for (tr, wt) in logweighted_traces if exp(wt) > 0.]
    trs = [tr for (tr, wt) in filtered]
    filtered_wts = [wt for (tr, wt) in filtered]
    normalized_weights = exp.(filtered_wts .- logsumexp(filtered_wts))
    return (trs, normalized_weights)
end

get_logweighted_traces(maxdepth) =
    [generate(sample_tree, (maxdepth,), cm) for cm in get_all_trees(maxdepth)]

get_all_trees(maxdepth) =
    if maxdepth == 1
        return nest_all(:terminal, get_all_terminals())
    else
        return union(
            Set(merge(cm, choicemap((:is_terminal, true))) for cm in nest_all(:terminal, get_all_terminals())),
            union(
                Set(
                    Gen.merge(nt_ch, choicemap((:is_terminal, false)))
                    for nt_ch in nest_all(:nt, get_all_nonterminals(maxdepth))
                )
            )
        )
    end
get_all_terminals() = union(
    Set([choicemap((:typ, :prime))]),
    Set(choicemap((:typ, :multiple_of), (:n1, n)) for n=1:100 if n1_pvec(:multiple_of)[n] > 0),
    Set(
        choicemap((:typ, :interval), (:n1, n1), (:n2, n2))
        for n1=1:100 for n2=1:100
            if n1_pvec(:interval)[n1] > 0 && n2_pvec(:interval, n1)[n2] > 0
    )
)
function get_all_nonterminals(maxdepth)
    subtrees = get_all_trees(maxdepth - 1)
    return reduce(union!, (
        Set([
            Gen.merge(Gen.merge(choicemap((:typ, :or)), left), right),
            Gen.merge(Gen.merge(choicemap((:typ, :and)), left), right)
        ])
        for left in nest_all(:left, subtrees)
            for right in nest_all(:right, subtrees)
    ))
end
nest_all(addr, cms::Set) = Set(DynamicModels.nest_at(addr, cm) for cm in cms)