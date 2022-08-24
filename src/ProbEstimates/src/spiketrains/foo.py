

@model object_motion():
    pos_xyz₁ ~ uniform(SPATIAL_GRID)
    vel_xyz₁ ~ uniform(VELOCITY_GRID)
    pos_xyzₜ, vel_xyzₜ ~ motion_prior_3d(
        pos_xyzₜ₋₁, vel_xyzₜ₋₁)
    observed_azimuth_altitudeₜ ~ project_to_retina(
        pos_xyzₜ)

@proposal update_pos_vel(pos_xyzₜ₋₁, vel_xyzₜ₋₁,
    observed_azimuth_altitudeₜ):
    pos_xyzₜ₋₁, vel_xyzₜ₋₁ ~ likely_update_given_az_alt(
        pos_xyzₜ₋₁, vel_xyzₜ₋₁, observed_azimuth_altitudeₜ
    )

@on_first_observation(observed_azimuth_altitude₁):
    particles₁ ~ SMCInit(update_pos_vel,
        args=(pos_xyz₀, vel_xyz₀, observed_azimuth_altitude₁),
            n_particles=10)
@on_subsequent_observation(observed_azimuth_altitudeₜ):
    particlesₜ ~ SMCStep(particlesₜ₋₁, observed_azimuth_altitudeₜ,
        update_pos_vel)
