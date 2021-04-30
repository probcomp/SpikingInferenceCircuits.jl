"""
    Sync(cluster_sizes)

A synchronization buffer with `length(cluster_sizes)` clusters.
Waits until it has received at least one spike in each cluster,
then "transitions" by rapidly outputting the number of spikes
received in each line since the previous transition.

Each element of `cluster_sizes` should give the number of lines in that cluster.
"""
struct Sync <: GenericComponent
    cluster_sizes::Vector{Int}
end
Circuits.target(::Sync) = Spiking()
Circuits.inputs(s::Sync) = IndexedValues(
    IndexedValues(SpikeWire() for _=1:size)
    for size in s.cluster_sizes
)
Circuits.outputs(s::Sync) = Circuits.inputs(s)