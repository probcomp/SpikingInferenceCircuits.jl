struct PulseStep <: GenericComponent
    input::Value
end
Circuits.abstract(s::PulseStep) = Step(s.input)
Circuits.target(::PulseStep) = Spiking()
Circuits.inputs(s::PulseStep) = inputs(abstract(s))
Circuits.outputs(s::PulseStep) = outputs(abstract(s))
Circuits.implement(s::Step, ::Spiking) = PulseStep(s.input)

Circuits.implement(s::PulseStep, ::Spiking) =
    CompositeComponent(
        implement_deep(inputs(s), Spiking()),
        implement_deep(outputs(s), Spiking()),
        (sync=PulseIR.Sync(collect(cluster_sizes(s))),),
        (inedges(s)..., outedges(s)...),
        s
    )

### Utils to construct the edges
pairs_deep(val) = (k => val[k] for k in keys_deep(val))

is_nontrivial_cluster(::FiniteDomainValue) = true
is_nontrivial_cluster(::IndicatedSpikeCountReal) = true
is_nontrivial_cluster(::SpikeWire) = true
is_nontrivial_cluster(::Value) = false

implement_to_nontrivial_clusters(v::CompositeValue) = CompositeValue(map(implement_to_nontrivial_clusters, v.vals))
implement_to_nontrivial_clusters(v::Value) = is_nontrivial_cluster(v) ? v : implement_to_nontrivial_clusters(implement(v, Spiking()))

# List of iterators which implement `length`.  Each iterator lists all the deep keys within one cluster.
# If one Input goes to multiple outputs
clusterings(v::FiniteDomainValue) = [keys(implement_deep(v, Spiking()))]
clusterings(::IndicatedSpikeCountReal{UnbiasedSpikeCountReal}) = [(:ind,), (:ind, :count)]
clusterings(::SpikeWire) = [(:nothing,)]

#=
For IndicatedSpikeCountReal, we want one cluster with just `:ind`, and one with both `:ind` and `:count`.
The reason for this is that we want to make sure we output the count as soon as `:ind` arrives,
even if the count is 0 and there are no count spikes.
We also want to wait until `:ind` arrives to output `:ind`, even if count spikes arrive.

What we should end up with looks like the following, where `k` is the key at which a IndicatedSpikeCountReal lives,
and `i` and `i+1` are the cluster indices used for this SpikeCountReal.

Input(k => :ind) => CompIn(:sync, i => 1)

Input(k => :ind) => CompIn(:sync, i+1 => 1)
Input(k => :count) => CompIn(:sync, i+1 => 2)

CompOut(:sync, i => 1) => Output(:k => :ind)
CompOut(:sync, i+1 => 2) => Output(:k => :count)
=#

_inedges(val_implemented_to_clusters) = (
    if isnothing(key_extension) # this cluster isa SpikeWire
        Input(:in => keystart) => CompIn(:sync, i => 1)
    else
        Input(:in => Circuits.nest(keystart, key_extension)) => CompIn(:sync, i => j)
    end

    for (i, (keystart, cluster_key_list)) in enumerate(nested_clusterlists(val_implemented_to_clusters))
        for (j, key_extension) in enumerate(cluster_key_list)
)
nested_clusterlists(val_implemented_to_clusters) = (
    (keystart, cluster_key_list)
    for (keystart, val) in pairs_deep(val_implemented_to_clusters)
        for cluster_key_list in clusterings(val)    
)

inedges(s::PulseStep) =  _inedges(implement_to_nontrivial_clusters(s.input))

# Send one wire from the SYNC to each output.  If an input wire goes to multiple clusters,
# send the corresponding output from the first input listed in `clusterings`.
function outedges(s::PulseStep)
    edges = []
    reached_outputs = Set{Output}()
    for (input, compin) in inedges(s)
        compout = CompOut(:sync, Circuits.valname(compin))
        output = Output(:out => Circuits.valname(input).second)
        if !(output in reached_outputs)
            push!(edges, compout => output)
            push!(reached_outputs, output)
        end
    end
    return edges
end

_cluster_sizes(val_implemented_to_clusters) = (
    length(cluster_key_list)
    for (_, cluster_key_list) in nested_clusterlists(val_implemented_to_clusters)
)
cluster_sizes(s::PulseStep) = _cluster_sizes(implement_to_nontrivial_clusters(s.input))