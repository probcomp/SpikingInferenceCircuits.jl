# for now, from prior
@gen (static) function _initial_proposal(obs)
    tree ~ sample_tree(MAXDEPTH())
end
@gen (static) function _step_proposal(tree, obs) end

initial_proposal = @compile_initial_proposal(_initial_proposal, 1)
step_proposal = @compile_step_proposal(_step_proposal, 1, 1)

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

choices_for_branch(tr, addr) = get_selected(get_choices(tr), select(nest_at(:init => :latents => :tree, addr)))
resimulate_branch(tr, subbranch_addr, max_depth_remaining, n_pgibbs_particles=2) =
    single_step_particle_gibbs(tr,
        resample_tree_branch, (subbranch_addr, max_depth_remaining), n_pgibbs_particles,
        (_, tr, subbranch_addr) -> choices_for_branch(tr, subbranch_addr)
    )
function resimulate_branch_cycle(tr, n_pgibbs_particles=2)
    for (subbranch_addr, max_depth_remaining) in subbranch_addr_maxdepth_pairs()
        tr = resimulate_branch(tr, subbranch_addr, max_depth_remaining, n_pgibbs_particles)
    end
    return tr
end
@load_generated_functions()