# Time 0 prior:
# TODO

onehot(x, dom) =
    x < first(dom) ? onehot(first(dom), dom) :
    x > last(dom)  ? onehot(last(dom), dom)  :
                 [i == x ? 1. : 0. for i in dom]

# prob vector to sample a value in `dom` which is 1 off
# from `idx` with probability `prob`, and `idx` otherwise
maybe_one_off(idx, prob, dom) =
    (1 - prob) * onehot(idx, dom) +
    prob/2 * onehot(idx - 1, dom) +
    prob/2 * onehot(idx + 1, dom)


@dist LabeledCategorical(labels, probs) = labels[categorical(probs)]

Xs = 1:40
HOME = 20
Vels = -4:4
Energies = 1:30

#                             energy, velocity, position
@gen (static) function step_model(eₜ₋₁, vₜ₋₁, xₜ₋₁)
    vₜ ~ vel_step_model(eₜ₋₁, vₜ₋₁, xₜ₋₁)
    xₜ ~ categorical(noisy(xₜ₋₁ + vₜ, Xs))

    expected_eₜ = eₜ₋₁ + (abs(vₜ) > 0 ? -abs(vₜ) : +2)
    eₜ ~ categorical(maybe_one_off(expected_eₜ, 0.5, Energies))
end

#=
P(eₜ, xₜ, vₜ ; eₜ₋₁, vₜ₋₁, xₜ₋₁)
P(vₜ ; eₜ₋₁, vₜ₋₁, xₜ₋₁) * P(xₜ ; vₜ), xₜ₋₁ * P()

p s.t. E[p] = P(vₜ ; eₜ₋₁, vₜ₋₁, xₜ₋₁)


Trace T = (stop_because_tired, stop_because_far_from_home, vₜ)
P(T ; eₜ₋₁, vₜ₋₁, xₜ₋₁)

IS:
N times:
Propose T_i  ~ Q( ; eₜ₋₁, vₜ₋₁, xₜ₋₁,    vₜ)
Score w_i = P(T ; eₜ₋₁, vₜ₋₁, xₜ₋₁) / Q(T ; ...)

1/N × ∑{w_i} ≈ P(vₜ ; eₜ₋₁, vₜ₋₁, xₜ₋₁)
as N --> infinity


P(T, vₜ; xₜ₋₁)
Q(T ; xₜ₋₁, vₜ)

E_{T~Q}[W_i]
∑_T{     w_i     Q(T ; xₜ₋₁, vₜ)  }

∑_T{     P(T, vₜ ;  xₜ₋₁) / Q(T ; xₜ₋₁, vₜ)     Q(T ; xₜ₋₁, vₜ)  }
∑_T{     P(T, vₜ ;  xₜ₋₁)  }
P(vₜ ;  xₜ₋₁)
=#


@gen (static) function vel_step_model(eₜ₋₁, vₜ₋₁, xₜ₋₁)
    stop_because_tired ~ bernoulli(some_decreasing_function_of(eₜ₋₁))
    
    dist_from_home = abs(xₜ₋₁ - HOME)
    moving_away_from_home = sign(vₜ₋₁) == sign(xₜ₋₁ - HOME)
    stop_because_far_from_home ~ bernoulli(
        moving_away_from_home ? 
            prob_stop_because_far(dist_from_home) :
            0.
    )

    stop = stop_because_tired || stop_because_far_from_home

    vₜ ~ LabeledCategorical(Vels, (
        stop ? onehot(0, Vels) :
               maybe_one_or_two_away(vₜ₋₁, 0.3, 0.2, Vels)
    ))

    return vₜ
end

# @gen function loop()
#     state1 ~ initial_state_model() # write this
#     for __
#         {t} ~ step_model()
#     end
# end


#=



=#




# smellsource at X = 10

# tumbling takes you outside xwin(nearsource)
# -> drawing velocities that depend on t-1
#

#  -> vt (drawn from dist centered on vt-1 + at-1)
#  -> vt (drawn from categorical distribution indep of vt-1)
#  a_t | v_t-1, a_t-1

# Step Latents Prior:
@gen (static) function step_model(aₜ₋₁, vₜ₋₁, xₜ₋₁)
    aₜ ~ # TODO: distribution here. will have acceleration depend upon aₜ₋₁, xₜ₋₁
    hit_wall = did_hit_wall(xₜ₋₁)

    vₜ ~ velocity_step_model(aₜ₋₁, vₜ₋₁, xₜ₋₁) # 40 * 40 * 40
    
    xₜ ~ maybe_a_bit_off(xₜ₋₁ + vₜ)
    return xₜ
end

@gen (static) function velocity_step_model(aₜ₋₁, vₜ₋₁, xₜ₋₁)
    # Noisily change the velocity according to the acceleration
    not_at_edge_vel ~ categorical(vel_model(vₜ₋₁ + aₜ₋₁)) # 40 * 40
    
    # whether the object ran into a wall
    hit_wall = at_edge_with_same_sign(xₜ₋₁, vₜ₋₁) # 40 * 40

    # If the object hit a wall, it either stopped or reflected off it.
    # Get the velocity in the case where it didn't run into wall, or didn't reflect:
    maybe_reflected_vel = hit_wall ? not_at_edge_vel : -not_at_edge_vel

    # Sample the velocity, depending on `hit_wall` and `maybe_reflected_vel`.
    # If it didn't hit a wall, just return the `maybe_reflected_vel` (which is _not_ reflected).
    # If it did hit a wall, w.p. 0.5, set velocity = 0 (the object stopped upon collision),
    # and w.p. 0.5, set velocity = reflected velocity (the object reflected)
    vₜ ~ categorical(                                                                           # 40 * 40
        hit_wall ? 0.5 * onehot(0, VEL_DOM) + 0.5 * onehot(maybe_reflected_vel, VEL_DOM) :
                       onehot(maybe_reflected_vel, VEL_DOM)
    )
    return vₜ
end

# Observation model:
@gen (static) function obs_model(aₜ, vₜ, xₜ)
    obsₜ ~ noise_model(xₜ)
    return obsₜ
end












# TODO: improve acceleration change model.  (Should depend on x and a.)
categorical(at_edge_with_same_sign(xₜ₋₁, aₜ₋₁) ?
                    onehot(0, ACCELERATION_DOMAIN) :
                    maybe_one_off(aₜ₋₁, 0.5, ACCELERATION_DOMAIN)
                )