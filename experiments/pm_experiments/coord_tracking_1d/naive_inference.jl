@gen (static) function naive_initial_proposal(obsx)
    xₜ ~ Cat(truncated_discretized_gaussian(obsx, 3.0, Positions()))
end
@gen (static) function naive_step_proposal(xₜ₋₁, vxₜ₋₁, obsx)
    xₜ ~ Cat(truncated_discretized_gaussian(obsx, 3.0, Positions()))
end