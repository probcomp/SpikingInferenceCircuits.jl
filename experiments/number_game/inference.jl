# for now, from prior
@gen (static) function initial_proposal(obs)
    tree ~ sample_tree(MAXDEPTH())
end
@gen (static) function step_proposal(tree, obs) end

### Rejuvenation moves ###
nest_at(prefix, suffix) = prefix => suffix
nest_at(prefix::Pair, suffix) = prefix.first => nest_at(prefix.second, suffix)

subbranch_addr_maxdepth_pairs(maxdepth) = maxdepth == 1 ? [] : [
    (:nt => :left, maxdepth - 1), (:nt => :right, maxdepth - 1),
    ((nest_at(:nt => :left, a), d) for (a, d) in subbranch_addr_maxdepth_pairs(maxdepth - 1))...,
    ((nest_at(:nt => :right, a), d) for (a, d) in subbranch_addr_maxdepth_pairs(maxdepth - 1))...
]
subbranch_addr_maxdepth_pairs() = [
    (nothing, MAXDEPTH()),
    subbranch_addr_maxdepth_pairs(MAXDEPTH())...
]

@gen function resample_tree_branch(tr, subbranch_addr, maxdepth_remaining)
    if isnothing(subbranch_addr)
        {:init => :latents => :tree} ~ sample_tree(maxdepth_remaining)
    else
        @trace(sample_tree(maxdepth_remaining), nest_at(:init => :latents => :tree, subbranch_addr))
    end
end

function resimulate_branch(tr, subbranch_addr, max_depth_remaining)
    # proposed_choices, propweight, proposed_tree = propose(resample_tree_subbranch, (tr, subbranch_addr))
    # updated_tr, updateweight, _, _ = update(tr, proposed_choices)
    
    # for now, using MH.  TODO: particle Gibbs
    tr, acc = mh(tr, resample_tree_branch, (subbranch_addr, max_depth_remaining))
    println(acc)
    return tr
end
function resimulate_branch_cycle(tr)
    for (subbranch_addr, max_depth_remaining) in subbranch_addr_maxdepth_pairs()
        tr = resimulate_branch(tr, subbranch_addr, max_depth_remaining)
    end
    return tr
end
@load_generated_functions()