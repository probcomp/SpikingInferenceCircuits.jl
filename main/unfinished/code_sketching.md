# some sketching of what the interface could look like -- 
```julia
@component function SMC(proposal_comp, observation, num_particles)
    bank = WeightBank(num_particles)
    particles = [SMCParticle(proposal_comp, observation, num_particles) for _=1:num_particles]
    
    c( # this should give us a component with `observation` as input, and the sampled values & weight bank samples as outputs
        (p1, observation) => p2
        for p1 in particles
            for p2 in particles,
        inputs=(observation,)
        outputs=(outputs(particles), outputs(bank))
    )
end
@component function SMCParticle(proposal_comp, observation, num_particles)
    mux = Multiplexer(num_particles, inputs(proposal_comp))
    c(sync(mux, observation) => proposal)
end
```