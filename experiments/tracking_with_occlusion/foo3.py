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
        particlesₜ[i] ~ PGibbs(regenerate_treebranch,
          args=(num, tree_branch), n_particles=2, n_sweeps=2)

                