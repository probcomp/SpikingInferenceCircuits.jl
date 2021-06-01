using JSON
includet("../../visualization/component_interface.jl")
# subcomp_graph = viz_graph(impl_deep[:particles => 1 => :propose => :sub_gen_fns => Symbol("##vₜ#687") => :component => :sample => :ss])
# open("visualization/frontend/renders/smc_energy_subcomp.json", "w") do f
#     JSON.print(f, subcomp_graph, 2)
# end
# println("Wrote component viz file.")

is_neuron(x) = x isa SpikingCircuits.InputFunctionPoisson
function replace_neurons_with_single_input(c::CompositeComponent)
    return CompositeComponent(
        inputs(c), outputs(c),
        map(replace_neurons_with_single_input, c.subcomponents),
        (
            update_edge(c, src) => update_edge(c, dst)
            for  (src, dst) in Circuits.get_edges(c)
        ),
        abstract(c)
    )
end
replace_neurons_with_single_input(::SpikingCircuits.InputFunctionPoisson) =
    PulseIR.PoissonNeuron(
        [identity, identity, identity], 1., identity
    )
update_edge(::CompositeComponent, v::Input) = v
update_edge(::CompositeComponent, v::Output) = v
update_edge(::CompositeComponent, v::CompOut) = v
update_edge(c::CompositeComponent, v::CompIn) =
    is_neuron(c[v.comp_name]) ? CompIn(v.comp_name, 2) : v

subcomp = implemented[:particles => 1 => :propose => :sub_gen_fns]# => Symbol("##eₜ#980") => :component]
simplified_impl = replace_neurons_with_single_input(subcomp)

subcomp_graph = viz_graph(simplified_impl)
open("visualization/frontend/renders/smc_energy_subcomp3.json", "w") do f
    JSON.print(f, subcomp_graph, 2)
end
println("Wrote component viz file.")