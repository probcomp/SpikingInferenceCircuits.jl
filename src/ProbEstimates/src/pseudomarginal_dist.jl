struct PMDistTrace{R} <: Gen.Trace
    gf::Gen.GenerativeFunction
    args::Tuple
    retval::R
end
Gen.get_args(tr::PMDistTrace) = tr.args
Gen.get_retval(tr::PMDistTrace) = tr.retval
Gen.get_choices(tr::PMDistTrace) = StaticChoiceMap((val=get_retval(tr),), (;))

# This should be a "not implemented" error, but the Gen Static DSL calls `get_score` to accumulate trace model scores
# even when they are never accessed.  So to prevent errors in this case, we will return NaN.  If users try to access
# scores, hopefully they will realize something has gone wrong due to the NaN.
Gen.get_score(tr::PMDistTrace) = NaN
    # error("""
    # `get_score` not supported, since `get_score` is
    # used in `propose` mode, but propose-mode pseudomarginalization is currently not implemented.
    # """)
Gen.get_gen_fn(tr::PMDistTrace) = tr.gf

# Not totally sure this is right--
Gen.project(tr::PMDistTrace, ::EmptySelection) = 0.

struct PseudoMarginalizedDist{R} <: Gen.GenerativeFunction{R, PMDistTrace}
    latent_model         :: Gen.GenerativeFunction
    obs_model            :: Gen.GenerativeFunction{R}
    proposal             :: Gen.GenerativeFunction
    val_to_obs_choicemap
    n_particles          :: Int
    args_for_compilation
end
Base.:(==)(a::PseudoMarginalizedDist, b::PseudoMarginalizedDist) =
    a.latent_model == b.latent_model && a.obs_model == b.obs_model && a.proposal == b.proposal && a.n_particles == b.n_particles

latent_domains(      d :: PseudoMarginalizedDist) = d.args_for_compilation[1]
latent_addr_order(   d :: PseudoMarginalizedDist) = d.args_for_compilation[2]
obs_addr_order(      d :: PseudoMarginalizedDist) = d.args_for_compilation[3]
args_for_compilation(d :: PseudoMarginalizedDist) = d.args_for_compilation[4]
obs_arg_domains(d :: PseudoMarginalizedDist, arg_domains) = (arg_domains..., latent_domains(d)...)

(p::PseudoMarginalizedDist)(args...) = get_retval(simulate(p, args))

Gen.simulate(d::PseudoMarginalizedDist{R}, args::Tuple) where {R} =
    let latent_vals = get_retval(simulate(d.latent_model, args))
        PMDistTrace(d, args,
            simulate(d.obs_model, (args..., latent_vals...)) |> get_retval
        )
    end
function Gen.generate(d::PseudoMarginalizedDist{R}, args::Tuple, cm::Union{Gen.ChoiceMap, Gen.EmptyChoiceMap}) where {R}
    tr = if isempty(cm)
        simulate(d, args)
    else
        @assert has_value(cm, :val)
        @assert length(collect(get_submaps_shallow(cm))) == 0
        @assert length(collect(get_values_shallow(cm))) == 1
        PMDistTrace(d, args, cm[:val])
    end
    return (tr, log_pseudomarginal_prob_estimate(tr))
end
function Gen.update(tr::PseudoMarginalizedDist, args::Tuple, ::Tuple, cm::Gen.ChoiceMap)
    if isempty(cm) && args == get_args(tr)
        return (tr, 0., NoChange(), EmptyChoiceMap())
    else
        error("Not expecting nontrivial `update` to be called on `PseudoMarginalizedDist`. args $(args == get_args(tr) ? "did not" : "did") change; cm = $cm")
    end
end

log_pseudomarginal_prob_estimate(tr) =
    log_pseudomarginal_prob_estimate(
        get_gen_fn(tr), get_retval(tr), get_args(tr)
    )
function log_pseudomarginal_prob_estimate(d, val, args)
    weight_sum = 0.
    for _=1:d.n_particles
        proposed_choices, proposed_score = propose(d.proposal, (args..., val))

        latent_assessed_score, latent_vals = assess(d.latent_model, args, proposed_choices)
        obs_assessed_score,   v1 = assess(d.obs_model, (args..., latent_vals...), d.val_to_obs_choicemap(val))
        assessed_score = latent_assessed_score + obs_assessed_score
        @assert v1 == val "val = $val, v1 = $v1"

        weight_sum += exp(assessed_score - proposed_score)
    end

    return log(weight_sum) - log(d.n_particles)
end