function distance_velocity_prior(x, y, z)
    counts = [0 for _ in Rs()]
    for dx in Vels(), dy in Vels(), dz in Vels()
        r = norm_3d(x + dx, y + dy, z + dz)
        if minimum(Rs()) ≤ floor(r) ≤ maximum(Rs())
            counts[Int(floor(r))] += 1
        end
        if minimum(Rs()) ≤ ceil(r) ≤ maximum(Rs())
            counts[Int(ceil(r))] += 1
        end
    end
    return normalize(counts)
end

@gen function propose_first_two_timesteps_smart(ϕ₀, θ₀, ϕ₁, θ₁)
    ### Propose first timestep values other than velocity ###
    true_θ₀ = { :init => :latents => :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(θ₀, 0.05, θs()))
    true_ϕ₀ = { :init => :latents => :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(ϕ₀, 0.05, ϕs()))
    r_max₀ = max_distance_inside_grid(true_ϕ₀, true_θ₀)
    l₀ = length(Rs())
    r_probvec₀ = normalize(vcat(ones(Int64(r_max₀)), zeros(Int64(l₀-r_max₀))))
    r₀ = { :init => :latents => :r } ~ LCat(Rs())(r_probvec₀)
    x_prop₀ = r₀ * cos(true_ϕ₀) * cos(true_θ₀)
    y_prop₀ = r₀ * cos(true_ϕ₀) * sin(true_θ₀)
    z_prop₀ = r₀ * sin(true_ϕ₀)
    x₀ = { :init => :latents => :x } ~ LCat(Xs())(truncated_discretized_gaussian(round(x_prop₀), .1, Xs()))
    y₀ = { :init => :latents => :y } ~ LCat(Ys())(truncated_discretized_gaussian(round(y_prop₀), .1, Ys()))
    z₀ = { :init => :latents => :z } ~ LCat(Zs())(truncated_discretized_gaussian(round(z_prop₀), .1, Zs()))

    ### Propose second timestep values other than velocity ###
    true_θ₁ = { :steps => 1 => :latents => :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(θ₁, 0.05, θs()))
    true_ϕ₁ = { :steps => 1 => :latents => :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(ϕ₁, 0.05, ϕs()))
    r_max₁ = max_distance_inside_grid(true_ϕ₁, true_θ₁)
    l₁ = length(Rs())
    # propose the distance value, incorporating how likely the model is move each distance from the position at time 0
    r_probvec₁ = normalize(
        vcat(ones(Int64(r_max₁)), zeros(Int64(l₁-r_max₁))) .* distance_velocity_prior(x₀, y₀, z₀)
    )
    r₁ = { :steps => 1 => :latents => :r } ~ LCat(Rs())(r_probvec₁)
    x_prop₁ = r₁ * cos(true_ϕ₁) * cos(true_θ₁)
    y_prop₁ = r₁ * cos(true_ϕ₁) * sin(true_θ₁)
    z_prop₁ = r₁ * sin(true_ϕ₁)
    x₁ = { :steps => 1 => :latents => :x } ~ LCat(Xs())(truncated_discretized_gaussian(round(x_prop₁), .1, Xs()))
    y₁ = { :steps => 1 => :latents => :y } ~ LCat(Ys())(truncated_discretized_gaussian(round(y_prop₁), .1, Ys()))
    z₁ = { :steps => 1 => :latents => :z } ~ LCat(Zs())(truncated_discretized_gaussian(round(z_prop₁), .1, Zs()))
    
    ### Propose velocities ###
    expected_dx₁ = x₁ - x₀
    expected_dy₁ = y₁ - y₀
    expected_dz₁ = z₁ - z₀

    dx₁ = {:steps => 1 => :latents => :dx} ~ LCat(Vels())(truncated_discretized_gaussian(expected_dx₁, 0.2, Vels()))
    dy₁ = {:steps => 1 => :latents => :dy} ~ LCat(Vels())(truncated_discretized_gaussian(expected_dy₁, 0.2, Vels()))
    dz₁ = {:steps => 1 => :latents => :dz} ~ LCat(Vels())(truncated_discretized_gaussian(expected_dz₁, 0.2, Vels()))

    dx₀ = {:init => :latents => :dx} ~ LCat(Vels())(truncated_discretized_gaussian(dx₁, 0.2, Vels()))
    dy₀ = {:init => :latents => :dy} ~ LCat(Vels())(truncated_discretized_gaussian(dy₁, 0.2, Vels()))
    dz₀ = {:init => :latents => :dz} ~ LCat(Vels())(truncated_discretized_gaussian(dz₁, 0.2, Vels()))
end
