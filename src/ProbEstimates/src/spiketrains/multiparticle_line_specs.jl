### Line Specs for importance sampling multiple traces, and doing weight multiplication + auto-normalization ###

"""
A spec for a line in a spiketrain for inference with multiple particles
doing a single step of importance sampling.
"""
abstract type MultiParticleLineSpec end

"""
Line spec for importance sampling in one of the subsidiary particles.
"""
struct ParticleLineSpec <: MultiParticleLineSpec
    particle_idx::Int
    spec::LineSpec
end

"""
Spec for the spiketrain for the line normalizing the weights.
"""
struct LogNormalization <: MultiParticleLineSpec end

"""
Spec for the spiketrain for the normalized weight for a particle.
"""
struct NormalizedWeight <: MultiParticleLineSpec
    particle_idx::Int
end

"""
Get lines for a step of multi-particle importance sampling,
in the submap with name `nest_all_at` in `trs`,
where `trs` are the traces produced by NG-F for this step of importance sampling.

`other_factors_to_multiply_in`
is a vector of values, one per particle, to multiply each particle weight by during
auto-normalization.  (E.g. this can be used to pass through the weight from the previous
timestep of SMC.)
"""
function get_lines_for_particles(
    specs, # ::Vector{MultiParticleLineSpec}
    trs,
    expected_log_weight_updates, # this should be the log of the weight-update factors computed here
    spiketrain_data_args;
    nest_all_at=nothing,
    other_factors_to_multiply_in=[1. for _ in trs]
)
    needs_autonormalization = any(spec isa Union{LogNormalization, NormalizedWeight} for spec in specs)
    needs_is = needs_autonormalization || any(spec isa ParticleLineSpec && spec.spec isa SpiketrainSpec for spec in specs)

    is_spiketrain_data = 
        if needs_is
            [
                sample_is_spiketimes_for_trace(tr, spiketrain_data_args...; nest_all_at)
                for tr in trs
            ]
        else
            nothing
        end

    autonormalization_data =
        if needs_autonormalization
            get_autonormalization_data(
                is_spiketrain_data,
                other_factors_to_multiply_in;
                expected_log_weight_updates # used to check that the computed pre-normalization particle weights are correct
            )
        else
            nothing
        end

    return [
        get_line_in_multiparticle_spec(spec, trs, is_spiketrain_data, autonormalization_data, nest_all_at)
        for spec in specs
    ]
end
get_line_in_multiparticle_spec(spec::ParticleLineSpec, trs, is_spiketrain_data, _, nest_all_at) =
    get_line(spec.spec, trs[spec.particle_idx], is_spiketrain_data[spec.particle_idx]; nest_all_at)
get_line_in_multiparticle_spec(::LogNormalization, _, _, autonormalization_data, _) =
    autonormalization_data.log_normalization_line
get_line_in_multiparticle_spec(spec::NormalizedWeight, _, _, autonormalization_data, _) =
    autonormalization_data.normalized_weight_lines[spec.particle_idx]