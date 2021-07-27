@gen (static) function _pm_ischangepoint_aux_model(vₜ₋₁)
    vₜ₋₁ ~ LCat(Vels())(onehot(vₜ₋₁, Vels()))
    is_changepoint ~ LCat([true, false])( (vₜ₋₁ -> [SwitchProb(), 1 - SwitchProb()])(vₜ₋₁) )
    return (is_changepoint,)
end
@gen (static) function _pm_ischangepoint_vel_model(vₜ₋₁, is_changepoint)
    vₜ ~ LCat(Vels())(
        is_changepoint ? unif(Vels()) : discretized_gaussian(vₜ₋₁, VelStepStd(), Vels())
    )
    return vₜ
end
function is_changepoint_prob(vₜ₋₁, vₜ)
    likelihood_unif = unif(Vels())[vₜ - first(Vels()) + 1]
    likelihood_move = discretized_gaussian(vₜ₋₁, VelStepStd(), Vels())[vₜ - first(Vels()) + 1]
    return likelihood_unif/(likelihood_move + likelihood_unif)
end
bern_probs(p) = [p, 1 - p]
@gen (static) function _pm_vel_proposal(vₜ₋₁, vₜ)
    vₜ₋₁ ~ LCat(Vels())(onehot(vₜ₋₁, Vels()))
    is_changepoint ~ LCat([true, false])(bern_probs(is_changepoint_prob(vₜ₋₁, vₜ)))
    return is_changepoint
end
pm_vel_dist = PseudoMarginalizedDist(
    _pm_ischangepoint_aux_model,
    _pm_ischangepoint_vel_model,
    _pm_vel_proposal,
    ((vₜ,),) -> choicemap((:vₜ => :val, vₜ)),
    1,
    # args for compilation:
    (
        (SIC.DiscreteIRTransforms.EnumeratedDomain([true, false]),),
        [:vₜ₋₁, :is_changepoint], [:vₜ], :vₜ
    )
)

@gen (static) function pm_step_model(xₜ₋₁, vₜ₋₁)
    vₜ ~ pm_vel_dist(vₜ₋₁)
    xₜ ~ Cat(onehot(xₜ₋₁ + vₜ, Positions()))
    return (xₜ, vₜ)
end
@load_generated_functions()