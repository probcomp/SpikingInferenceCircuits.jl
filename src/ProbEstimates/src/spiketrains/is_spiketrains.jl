"""
    DenseValueSpiketrain

Stores the information describing the transmission of a dense-value.
"""
struct DenseValueSpiketrain
    ready_time   :: Float64
    neuron_times :: Vector{Vector{Float64}}
    # neuron_times[i] = spiketrain for the `i`th neuron in the output gate assembly
end
"""
    ISSpiketrains

Stores the information describing the spiketrains from importance-sampling a single trace.
Includes the value spiketrains, the forward-probability-estimate spiketrains,
and the reciprocal-probability-estimate spiketrains.
"""
struct ISSpiketrains
    valtimes::Dict{<:Any, Float64}
    val_trains::Dict{<:Any, Vector{Float64}}
    recip_trains::Dict{<:Any, DenseValueSpiketrain}
    fwd_trains::Dict{<:Any, DenseValueSpiketrain}
end

"""
    SpikelineInScore

A line of spikes which is part of the information for conveying a continuous score.
Can be the "ready" indicator line, the assembly for the spike-count,
or a neuron within the assembly for spike-count.
"""
abstract type SpikelineInScore end
struct CountAssembly <: SpikelineInScore; end
struct NeuronInCountAssembly <: SpikelineInScore; idx; end
struct IndLine <: SpikelineInScore; end

"""
Sample an `ISSpiketrains` for a SNN simulation that produced the same results
as the NG-F importance-sampling simulation which produced `tr`.

`nest_all_at` is a Gen address specifying the submap in `tr` that was sampled by the NG-F
importance-sampling simulation.  (E.g. this might be the address for a timestep in a dynamic model
for which importance-sampling occurred during SMC.)
"""
function sample_is_spiketimes_for_trace(
    tr,
    inter_sample_time_dist,
    propose_sampling_tree, # as Dict(addr -> [list of parent addrs])
    assess_sampling_tree,
    propose_addr_topological_order,
    to_ready_spike_dist;
    nest_all_at=nothing, wta_memory=Latency()
)
    valtimes = sampled_value_times(inter_sample_time_dist, propose_sampling_tree, propose_addr_topological_order)
    val_trains = value_trains(valtimes, tr, MaxRate(), K_recip(), wta_memory; nest_all_at)
    recip_times = recip_spiketimes(valtimes, propose_addr_topological_order, tr, AssemblySize(), MaxRate(), K_recip(), to_ready_spike_dist; nest_all_at)
    fwd_times   = fwd_spiketimes(
        fwd_score_ready_times(valtimes, assess_sampling_tree),
        keys(assess_sampling_tree), tr, AssemblySize(), MaxRate(), K_fwd(), to_ready_spike_dist; nest_all_at
    )
    
    return ISSpiketrains(valtimes, val_trains, recip_times, fwd_times)
end
sample_is_spiketimes_for_trace(tr, propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order; nest_all_at=nothing) =
    sample_is_spiketimes_for_trace(
        tr, DefaultInterSampleTimeDist(), propose_sampling_tree,
        assess_sampling_tree, propose_addr_topological_order, DefaultToReadySpikeDist();
        nest_all_at
    )

fwd_score_ready_times(propose_ready_times, assess_sampling_tree) = Dict(
    # We can score an address once it's value has been proposed, and all it's parents' values have been proposed
    addr => reduce(max, (get(propose_ready_times, a, 0.) for a in assess_sampling_tree[addr]); init=get(propose_ready_times, addr, 0.))
    for addr in keys(assess_sampling_tree)
)

function sampled_value_times(
    inter_sample_time_dist,
    sampling_tree, # Dict(addr => [list of parent addrs])
    addr_topological_order
)
    value_times = Dict{Any, Float64}()
    for addr in addr_topological_order
        parents = sampling_tree[addr]
        parent_times = [value_times[parent] for parent in parents]
        value_times[addr] = reduce(max, parent_times; init=0.) + rand(inter_sample_time_dist)
    end
    return value_times
end

function value_trains(
    valtimes, # addr -> spike which samples the outcome
    tr, neuron_rate,
    count_threshold, memory;
    nest_all_at
)
    d = Dict{Any, Vector{Float64}}()
    ch = get_ch(tr, nest_all_at)
    for (addr, first_spike_time) in valtimes
        num_spikes = try
            get_recip_score(ch, addr) * count_threshold |> to_int
        catch e
            @error "count threshold = $count_threshold; addr = $addr; get_recip_score(ch, addr) = $(get_recip_score(ch, addr))" exception=(e, catch_backtrace())
            error()
        end
        p = with_weight_type(:perfect, () -> exp(Gen.project(tr, Gen.select(nest(nest_all_at, addr)))))
        rate = neuron_rate * p
        d[addr] = poisson_process_train_with_rate(first_spike_time, rate, memory)
    end
    return d
end

function poisson_process_train_with_rate(starttime, rate, memory)
    times = [starttime]
    while last(times) - starttime < memory
        push!(times, rand(last(times) + Exponential(1/rate)))
    end
    if last(times) - starttime > memory
        pop!(times)
    end
    return times
end

### TODO: refactor code to reduce repeat between `recip_spiketimes` and `fwd_spiketimes`
function recip_spiketimes(
    ready_times, # addr -> time at which it becomes possible to start scoring the addr
    addrs, tr,
    assembly_size,
    neuron_rate,
    count_threshold,
    dist_to_ready_spike;
    nest_all_at
)
    ch = get_ch(tr, nest_all_at)
    times = Dict{Any, DenseValueSpiketrain}() # addr => [ [ times at which this neuron spikes ] for i=1:assembly_size ]
    for addr in addrs
        num_spikes = try
            get_recip_score(ch, addr) * count_threshold |> to_int
        catch e
            @error "count threshold = $count_threshold; addr = $addr; get_recip_score(ch, addr) = $(get_recip_score(ch, addr))" exception=(e, catch_backtrace())
            error()
        end
        
        neuron_times = spiketrains_for_n_spikes(
            num_spikes, assembly_size, neuron_rate, ready_times[addr]
        )
        ready_time = last(sort(reduce(vcat, neuron_times; init=Float64[]))) + rand(dist_to_ready_spike)
        times[addr] = DenseValueSpiketrain(ready_time, neuron_times)
    end

    return times
end

function fwd_spiketimes(
    ready_times, # addr -> time at which it becomes possible to start scoring the addr
    addrs, tr,
    assembly_size,
    neuron_rate,
    count_threshold,
    dist_to_ready_spike;
    nest_all_at
)
    ch = get_ch(tr, nest_all_at)
    # get_score = do_recip_score ? get_recip_score : get_fwd_score
    times = Dict{Any, DenseValueSpiketrain}() # addr => [ [ times at which this neuron spikes ] for i=1:assembly_size ]
    for addr in addrs
        num_spikes = get_fwd_score(ch, addr) * count_threshold |> to_int
        p = with_weight_type(:perfect, () -> exp(Gen.project(tr, Gen.select(nest(nest_all_at, addr)))))
        
        neuron_times = spiketrains_for_n_spikes(
            num_spikes, assembly_size, neuron_rate * p, ready_times[addr]
        )

        # TODO: handle the case where we get 0 spikes

        ready_time = last(sort(reduce(vcat, neuron_times; init=Float64[]))) + rand(dist_to_ready_spike)
        times[addr] = DenseValueSpiketrain(ready_time, neuron_times)
    end

    return times
end

function spiketrains_for_n_spikes(num_spikes, num_neurons, neuron_rate, starttime)
    t = starttime
    times = [[] for _=1:num_neurons]
    for _=1:num_spikes
        t += rand(Exponential(1 / (neuron_rate * num_neurons)))
        idx = rand(DiscreteUniform(1, num_neurons))
        push!(times[idx], t)
    end
    return times
end

function to_int(v)
    @assert isapprox(v, floor(v)) || isapprox(v, ceil(v)) "v = $v"
    return Int(isapprox(v, floor(v)) ? floor(v) : ceil(v))
end

### Some default distributions for spiketrain production ###
DefaultInterSampleTimeDist() = Exponential(1 / MaxRate())
DefaultToReadySpikeDist() = Exponential(1 / MaxRate())
