# ProbEstimates can overwrite these if needed to enable the noise model
current_weighttype = use_propose_weights!() = nothing
done_using_propose_weights!(prev_weighttype) = nothing
check_weights_equal_if_perfect_weights(w1, w2) = @assert isapprox(w1, w2, atol=1e-6) "$w1 | $w2"

# TODO: pass through to make sure we handle noise correctly
function single_step_particle_gibbs(
    tr, proposal, proposal_args, nparticles,
    get_proposed_choices # get_proposed_choices(old_tr, new_tr, proposal_args) gives the choices `propose` needs to return
)
    @assert nparticles ≥ 2 "Using fewer than 2 particles means we could not possibly update the trace!  Our indexing scheme is such that 2 particles = current trace + one new particle."

    trs = []
    logweights = []
    for i=1:nparticles
        if i == 1
            choices = get_proposed_choices(tr, tr, proposal_args)

            typ = use_propose_weights!()
            propose_weight, _ = assess(proposal, (tr, proposal_args...), choices)
            done_using_propose_weights!(typ)
        else
            (choices, propose_weight, _) = propose(proposal, (tr, proposal_args...))
        end
        
        (new_tr, update_weight, _, _) = update(tr, choices)

        if i == 1
            @assert get_choices(new_tr) == get_choices(tr)
        end

        assess_weight, _ = assess(get_gen_fn(new_tr), get_args(new_tr), get_choices(new_tr))
       
        if i > 1
            check_weights_equal_if_perfect_weights(update_weight, assess_weight - get_score(tr))
        end

        push!(trs, new_tr)
        push!(logweights, assess_weight - propose_weight)
    end

    return resample(trs, logweights, 1) |> only
end
single_step_particle_gibbs_rejuv_kernel(args...) =
    tr -> single_step_particle_gibbs_rejuv_kernel(tr, args...)