
function get_inputs(runtime, inter_obs_interval, initial_x, initial_v, initial_e, observations, nparticles)
    inputs = Tuple{Float64, Tuple}[(0., (Iterators.flatten(
            (
                :initial_latents => i => :xₜ => initial_x,
                :initial_latents => i => :vₜ => initial_v,
                :initial_latents => i => :eₜ => initial_e
            )
            for i=1:nparticles
        )...,
        :obs => :obsₜ => popfirst!(observations)...)
    )]
    t = 0
    while t < runtime
        t += inter_obs_interval
        newobs = popfirst!(observations)
        push!(inputs, (t, (:obs => :obsₜ => newobs,)))
    end

    return inputs
end
