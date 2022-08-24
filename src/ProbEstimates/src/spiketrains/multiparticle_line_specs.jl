### Line Specs for importance sampling multiple traces, and doing weight multiplication + auto-normalization ###

"""
A spec for a line in a spiketrain for inference with multiple particles
doing a single step of importance sampling.
"""
abstract type MultiParticleLineSpec end

"""
Line spec for importance sampling in one of the subsidiary particles.
"""
struct SubsidiarySingleParticleLineSpec <: MultiParticleLineSpec
    particle_idx::Int
    spec::LineSpec
end

"""
Spec for the spiketrain for the line normalizing the weights.
"""
struct LogNormalization <: MultiParticleLineSpec
    line_to_show::Union{CountAssembly, NeuronInCountAssembly}
end

"""
Spec for the spiketrain for the normalized weight for a particle.
"""
struct NormalizedWeight <: MultiParticleLineSpec
    particle_idx::Int
    line_to_show::Union{CountAssembly, NeuronInCountAssembly}
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
function get_lines_for_multiparticle_specs(
    specs, # ::Vector{MultiParticleLineSpec}
    trs,
    expected_log_weight_updates, # this should be the log of the weight-update factors computed here
    spiketrain_data_args;
    nest_all_at=nothing,
    other_factors_to_multiply_in=[1. for _ in trs],
    num_autonormalization_spikes=nothing,
    vars_disc_to_cont=Dict()
)
    needs_autonormalization = any(spec isa Union{LogNormalization, NormalizedWeight} for spec in specs)
    needs_is = needs_autonormalization || any(spec isa SubsidiarySingleParticleLineSpec && spec.spec isa SpiketrainSpec for spec in specs)

    is_spiketrain_data = 
        if needs_is
            [
                sample_is_spiketimes_for_trace(tr, spiketrain_data_args...; nest_all_at, vars_disc_to_cont)
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
                expected_log_weight_updates, # used to check that the computed pre-normalization particle weights are correct
                num_autonormalization_spikes
            )
        else
            nothing
        end

    return [
        get_line_in_multiparticle_spec(spec, trs, is_spiketrain_data, autonormalization_data, nest_all_at)
        for spec in specs
    ]
end
get_line_in_multiparticle_spec(spec::SubsidiarySingleParticleLineSpec, trs, is_spiketrain_data, _, nest_all_at) =
    get_line(spec.spec, trs[spec.particle_idx], is_spiketrain_data[spec.particle_idx]; nest_all_at)

function _get_neuron_or_assembly(vec_of_neuron_spiketrains, line_to_show)
    if line_to_show isa NeuronInCountAssembly
        vec_of_neuron_spiketrains[line_to_show.idx]
    else
        @assert line_to_show isa CountAssembly
        sort(reduce(vcat, vec_of_neuron_spiketrains))
    end
end
get_line_in_multiparticle_spec(s::LogNormalization, _, _, autonormalization_data, _) =
    _get_neuron_or_assembly(autonormalization_data.log_normalization_lines, s.line_to_show)
get_line_in_multiparticle_spec(spec::NormalizedWeight, _, _, autonormalization_data, _) =
    _get_neuron_or_assembly(autonormalization_data.normalized_weight_lines[spec.particle_idx], spec.line_to_show)

### Text for multi-particle line specs
abstract type MultiParticleText <: MultiParticleLineSpec end
struct SingleParticleTextWrapper <: MultiParticleText
    particle_idx::Int
    single_particle_text::SingleParticleText
    show_particle_idx::Bool
end
function get_line_in_multiparticle_spec(spec::MultiParticleText, trs, is_spiketrain_data, _, nest_all_at)
    text = get_line(spec.single_particle_text, trs[spec.particle_idx], is_spiketrain_data[spec.particle_idx]; nest_all_at)
    if spec.show_particle_idx
        return "P$(spec.particle_idx): " * text
    else
        return text
    end
end

struct FixedText <: MultiParticleText
    text::String
end
get_line_in_multiparticle_spec(t::FixedText, args...) = t.text

## TODO: handle MultiParticle text for auto-normalization and importance weight lines!!!