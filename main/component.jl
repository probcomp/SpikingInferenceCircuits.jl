# TODO: docstrings

using LightGraphs

abstract type Component end
abstract type GenericComponent <: Component end
abstract type PrimitiveComponent{Target} <: Component end
inputs(::Component)::CompositeValue = error("Not implemented.")
outputs(::Component)::CompositeValue = error("Not implemented.")

abstract type NodeName end
struct Input{ID} <: NodeName
    id::ID
end
struct Output{ID} <: NodeName
    id::ID
end
struct CompIn{I1, I2} <: NodeName
    comp_name::I1
    in_name::I2
end
struct CompOut{I1, I2} <: NodeName
    comp_name::I1
    out_name::I2
end
"""
    CompositeComponent

Has an input `CompositeValue`, an output `CompositeValue`, and some number of internal `Component`s.
Each input/output `Value` and each `Component` has a name.
There is a graph on the input/output sub-Values and all the input/output Values for the internal components.
A `NodeName` is thus one of:
1. An `Input` name
2. An `Output` name
3. A pair `(subcomponent name, subcomponent input name)`
4. A pair `(subcomponent name, subcomponent output name)`

We store a 2-way mapping between these node names and vertex indices in the graph.
We store the `Value` for each input/output name, and the `Component` for each component name.
"""
struct CompositeComponent{InID, OutID, SC} <: Component
    input::CompositeValue{InID}
    output::CompositeValue{OutID}
    subcomponents::SC
    node_to_idx::Dict{NodeName, UInt}
    idx_to_node::Vector{NodeName}
    graph::SimpleDiGraph
    
    # ensure `subcomponents` is either a Tuple or NamedTuple of Components
    CompositeComponent(
        input::CompositeValue{InID},
        output::CompositeValue{OutID},
        subcomponents::SC,
        node_to_idx::Dict{NodeName, UInt}
        idx_to_node::Vector{NodeName}
        graph::SimpleDiGraph
    ) where {InID, OutID, SC <: Union{
        Tuple{Vararg{<:Component}}, 
        NamedTuple{<:Any, <:Tuple{Vararg{<:Component}}}
    }} = CompositeComponent{InID, OutID, SC}(input, output, subcomponents, node_to_idx, idx_to_node, graph)
end

function CompositeComponent(
    input::CompositeValue,
    output::CompositeValue,
    subcomponents,
    edges
)
    idx_to_node = collect(Iterators.flatten((
        (Input(k) for k in keys(inputs)), (Output(k) for k in keys(outputs)),
        (CompIn(compname, inname) for compname in keys(subcomponents) for inname in keys(input(subcomponents))),
        (CompOut(compname, outname) for compname in keys(subcomponents) for outname in keys(output(subcomponents)))
    )))
    node_to_idx = Dict(name => idx for (idx, name) in idx_to_node)
    graph = SimpleDiGraph(length(idx_to_node), (node_to_idx[src_name] => node_to_idx[dst_name] for (src_name, dst_name) in edges))

    CompositeComponent(inputs, outputs, subcomponents, node_to_idx, idx_to_node, graph)
end
inputs(c::CompositeComponent) = c.input
outputs(c::CompositeComponent) = c.output

# TODO: figure out the details of Junction.  Do we want one `Junction` concrete type, or instead
# something like a `Junction` abstract type with concrete subtypes `ValSplit`, `ValMerge`, and some combiner?
"""
    Junction <: PrimitiveComponent{Target}

A Junction reroutes Values, for instance joining values together or splitting a Value into several.
It is a primitive component for every target.

Implementation & details of how this works: TODO
"""
struct Junction <: PrimitiveComponent{Target}

end

# TODO: what are the accessors we need?
# Base.getindex(c::CompositeComponent, i::Input) = input(c)[i.id]
# Base.getindex(c::CompositeComponent, o::Output) = output(c)[o.id]
# Base.getindex(c::CompositeComponent, i::CompIn) = 