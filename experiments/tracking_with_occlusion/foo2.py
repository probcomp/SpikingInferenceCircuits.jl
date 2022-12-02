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
  particles₁ ~ SMCInit(simulate_or_detect_object,
    args=(image₁, EMPTYSTATE), n_particles=10)
@on_subsequent_observation(imageₜ):
  particlesₜ ~ SMCStep(particlesₜ₋₁, imageₜ,
                    simulate_or_detect_object)
