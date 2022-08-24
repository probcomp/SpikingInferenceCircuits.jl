### Grouped Line Specs ###
"""
A LabeledSingleParticleLineGroup describes a collection of lines from SingleParticleSpecs
in a spiketrain which should all share a label.  (E.g. this might be used when showing
multiple lines used to convey a single value, to label all those lines
with the name of that value.)
"""
struct LabeledSingleParticleLineGroup
    label_spec::SingleParticleText
    line_specs::Vector{SingleParticleLineSpec}
end
get_lines(groups::Vector{LabeledSingleParticleLineGroup}, tr, args; nest_all_at=nothing) =
    get_lines(reduce(vcat, g.line_specs for g in groups), tr, args; nest_all_at)
get_labels(groups::Vector{LabeledSingleParticleLineGroup}) = get_labels(reduce(vcat, g.line_specs for g in groups))

get_group_labels(groups::Vector{LabeledSingleParticleLineGroup}, tr; nest_all_at) =
    [ # list of (label, num lines in group) tuples for each group
        (get_line(g.label_spec, tr, nothing; nest_all_at), length(g.line_specs))
        for g in groups
    ]

get_group_label(group::LabeledSingleParticleLineGroup, tr) = get_line(group.label_spec, tr)

"""
A `LabeledMultiParticleLineGroup` is like a `LabeledSingleParticleLineGroup` but
uses `MultiParticleLineSpec`s instead of `SingleParticleLineSpec`s to describe lines.
"""
struct LabeledMultiParticleLineGroup
    label_spec::MultiParticleText
    line_specs::Vector{MultiParticleLineSpec}
end

# TODO: once we handle text for auto-normalization and weight lines, we may need to change this interface
get_lines_for_multiparticle_spec_groups(groups, args...; kwargs...) =
    get_lines_for_multiparticle_specs(reduce(vcat, g.line_specs for g in groups), args...; kwargs...)
get_group_labels_for_multiparticle_spec_groups(groups::Vector{LabeledMultiParticleLineGroup}, trs; nest_all_at) =
    [(get_line_in_multiparticle_spec(g.label_spec, trs, [nothing for _ in trs]; nest_all_at), length(g.line_specs)) for g in groups]

get_static_group_labels_for_multiparticle_spec_groups(groups::Vector{LabeledMultiParticleLineGroup}; kwargs...) =
    [(get_static_multi_textline(group.label_spec; kwargs...), length(group.line_specs)) for group in groups]
get_static_multi_textline(t::FixedText; kwargs...) = t.text
function get_static_multi_textline(t::SingleParticleTextWrapper; kwargs...)
    text = get_textline_sing_static(t.single_particle_text; kwargs...)
    if t.show_particle_idx
        return "P$(t.particle_idx): " * text
    else
        return text
    end
end
get_textline_sing_static(s::FwdScoreText; addr_to_name=identity) = "P[$(addr_to_name(s.addr))]"
get_textline_sing_static(s::RecipScoreText; addr_to_name=identity) = "1/Q[$(addr_to_name(s.addr))]"
get_textline_sing_static(s::SampledValue; addr_to_name=identity) = "$(addr_to_name(s.addr))"