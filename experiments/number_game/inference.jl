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

rejuvenate_branch(tr, subbranch_addr, max_depth_remaining, n_pgibbs_particles=2; proposal=resample_tree_branch) =
    single_step_particle_gibbs(tr,
        proposal, (subbranch_addr, max_depth_remaining), n_pgibbs_particles,
        (_, tr, subbranch_addr) -> choices_for_branch(tr, subbranch_addr)
    )
function rejuvenate_branch_cycle(tr, n_pgibbs_particles=2; proposal=resample_tree_branch)
    for (subbranch_addr, max_depth_remaining) in subbranch_addr_maxdepth_pairs()
        tr = rejuvenate_branch(tr, subbranch_addr, max_depth_remaining, n_pgibbs_particles; proposal)
    end
    return tr
end

### Data-driven proposal ###
n1_pvec_smart(typ, num) =
    if typ == :prime
        # doesn't matter in this case, so have deterministic value to decrease weight estimate variance
        onehot(1, 1:100)
    elseif typ == :multiple_of
        normalize([
            (if 2 ≤ i ≤ 10 # only have multiples of small numbers
                if num % i == 0 # be much more likely to propose something that explains `num`
                    1.
                else
                    0.1
                end
            else
                0.
            end)
            for i=1:100
        ])
    elseif typ == :interval
        normalize([
            (if i ≤ 90 && i % 5 == 0
                if i ≤ num
                    1.
                else
                    0.1
                end
            else
                0.
            end)
            for i=1:100
        ])
    end
max_interval_size() = 35
n2_pvec_smart(typ, n1, num) =
    if typ == :prime || typ == :multiple_of
        # in either case, n2 is irrelevant
        onehot(1, 1:100)
    elseif typ == :interval
        # sample n2 from among those numbers greater than n1
        # (and try to make it ≤ num)
        normalize([
            (if n1 < i ≤ n1 + max_interval_size() && (i - n1) % 4 == 0
                if num ≤ i
                    1.
                else
                    0.1
                end
            else
                0.
            end)
            for i=1:100
        ])
    end

@gen (static) function _sample_terminal_likely_to_produce(num)
    typ ~ LCat([:prime, :multiple_of, :interval])([1/6, 1/2, 1/3])
    n1 ~ Cat(n1_pvec_smart(typ, num))
    n2 ~ Cat(n2_pvec_smart(typ, n1, num))
    return (:terminal, typ, n1, n2)
end
@gen function _sample_tree_likely_to_produce(num, maxdepth)
    terminal ~ _sample_terminal_likely_to_produce(num)
    if maxdepth > 1
        is_terminal ~ Bernoulli(bern_probs(1/4))
        nt ~ _sample_nt_likely_to_produce(num, maxdepth)
        return is_terminal ? terminal : nt
    else
        return terminal
    end
end
@gen (static) function _sample_nt_likely_to_produce(num, maxdepth)
    typ ~ LCat([:and, :or])([0.3, 0.7])
    left ~ _sample_tree_likely_to_produce(num, maxdepth - 1)
    right ~ _sample_tree_likely_to_produce(num, maxdepth - 1)
    return (:nonterminal, typ, left, right)
end

@gen function repropose_tree_branch_data_driven(tr, subbranch_addr, maxdepth_remaining)
    num = vals(tr)[end]
    if isnothing(subbranch_addr)
        {:init => :latents => :tree} ~ _sample_tree_likely_to_produce(num, maxdepth_remaining)
    else
        @trace(_sample_tree_likely_to_produce(num, maxdepth_remaining), nest_at(:init => :latents => :tree, subbranch_addr))
    end
end

@load_generated_functions()