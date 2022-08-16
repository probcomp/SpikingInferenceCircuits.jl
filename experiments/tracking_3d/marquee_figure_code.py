### Mental Physics Simulation ###
@model object_motion():
    state₁ ~ uniform(STATES)
    stateₜ ~ motion_prior(stateₜ₋₁)
    obsₜ ~ noisy_image_render(stateₜ₋₁)

@proposal simulate_or_detect_object(imageₜ, stateₜ₋₁):
    do_top_down_proposal ~ bernoulli(0.5)
    if do_top_down_proposal:
        stateₜ ~ simulate_from_prior(imageₜ)
    else:
        stateₜ ~ cnn_object_detector(imageₜ)

@on_first_observation(image₁):
    particles₁ ~ SMCInit(simulate_or_detect_object, args=(image₁, EMPTYSTATE), n_particles=10)
@on_subsequent_observation(imageₜ):
    particlesₜ ~ SMCStep(particlesₜ₋₁, imageₜ, simulate_or_detect_object)


### Recursive Concept Learning ###
@submodel generate_and_combine_random_subtrees(depth):
    set1 ~ generate_number_set(depth - 1)
    set2 ~ generate_number_set(depth - 1)
    Op   ~ uniform([Intersection, Union])
    return Op(set1, set2)
@submodel generate_number_set(depth):
    if depth > 1:
        return set ~ generate_and_combine_random_subtrees(depth)
    else:
        return set ~ generate_simple_ruleset()
@model generate_number_set_and_sequence(depth):
    set ~ generate_number_set(depth)
    numₜ ~ uniform(set)

@proposal change_nothing(previous_num_set, new_num):
    return previous_num_set
@proposal regenerate_treebranch(num, tree_branch):
    if depth(tree_branch) > 1:
        set ~ generate_and_combine_random_subtrees(depth)
    else:
        set ~ generate_simple_ruleset_containing(num)

particles₀ ~ SMCInit(generate_number_set, args=(), n_particles=10)
@on_any_observation(numₜ):
    particlesₜ ~ SMCStep(particlesₜ₋₁, numₜ, change_nothing)
    for particleₜ₋₁ in pre_particlesₜ:
        for i in tree_branch_indices(number_set_tree(pre_particlesₜ)):
            tree_branch = number_set_tree(pre_particlesₜ)[i]
            particlesₜ[i] ~ PGibbs(regenerate_treebranch, args=(num, tree_branch), n_particles=2, n_sweeps=2)

# This isn't all quite right.  But I'm worried that I'm operating in some way at the wrong level of abstraction
# so will need to redo a larger portion of it -- so I will stop iterating myself until I get input from Vikash.


### Tracking 3D Object from 2D Detections ###

@model object_motion():
    pos_xyz₁ ~ uniform(SPATIAL_GRID)
    vel_xyz₁ ~ uniform(VELOCITY_GRID)
    pos_xyzₜ, vel_xyzₜ ~ motion_prior_3d(pos_xyzₜ₋₁, vel_xyzₜ₋₁)
    observed_azimuth_altitudeₜ ~ project_to_retina(pos_xyzₜ)

@proposal update_pos_vel(pos_xyzₜ₋₁, vel_xyzₜ₋₁, observed_azimuth_altitudeₜ):
    pos_xyzₜ₋₁, vel_xyzₜ₋₁ ~ likely_update_given_az_alt(
        pos_xyzₜ₋₁, vel_xyzₜ₋₁, observed_azimuth_altitudeₜ
    )

@on_first_observation(observed_azimuth_altitude₁):
    particles₁ ~ SMCInit(update_pos_vel, args=(pos_xyz₀, vel_xyz₀, observed_azimuth_altitude₁), n_particles=10)
@on_subsequent_observation(observed_azimuth_altitudeₜ):
    particlesₜ ~ SMCStep(particlesₜ₋₁, observed_azimuth_altitudeₜ, update_pos_vel)











####### SCRATCH WORK ##################








@submodel generate_number_set(rule_depth):
    is_last_rule ~ bernoulli(rule_depth == 1 ? 0 : 0.5)
    if is_last_rule:
        SetConstructor ~ uniform([Interval, MultipleOf])
        n1 ~ uniform(1:100)
        n2 ~ uniform(n1:100)
        return SetConstructor(n1, n2)
    else:
        Op ~ uniform([Union, Intersection])
        set1 ~ generate_number_set(rule_depth - 1)
        set2 ~ generate_number_set(rule_depth - 1)
        return Op(set1, set2)









@model generate_number_set():
    rule ~ 
    all_numbers = filter(1:100, x -> satisfies_rule(rule, x))

    @submodel generate_rule(depth):
        is_last_rule ~ bernoulli(0.4)
        if is_last_rule
            n1 ~ uniform(1, 100)
            n2 ~ uniform(n2, 100)
            return IsInInterval(n1, n2)
        subrule1 ~ generate_rule(depth-1)
        subrule2 ~ generate_rule(depth-1)
        return AND(subrule1, subrule2)



