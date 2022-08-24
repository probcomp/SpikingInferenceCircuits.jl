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
    p_dist_trains::Dict # {<:Any, Dict{Any, Vector{Vector{Float64}}}} # [addr][value][neuron_idx] = vector of spiketimes for this neuron
    q_dist_trains::Dict # {<:Any, Dict{Any, Vector{Vector{Float64}}}}
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
    addr_to_domain,
    to_ready_spike_dist;
    vars_disc_to_cont=Dict(),
    nest_all_at=nothing, wta_memory=Latency()
)
    valtimes = sampled_value_times(inter_sample_time_dist, propose_sampling_tree, propose_addr_topological_order)
    val_trains = value_trains(valtimes, tr, MaxRate(), K_recip(), wta_memory; nest_all_at)
    recip_times, all_q_times = recip_spiketimes(valtimes, propose_addr_topological_order, [addr_to_domain[a] for a in propose_addr_topological_order], tr, AssemblySize(), MaxRate(), K_recip(), to_ready_spike_dist; nest_all_at, vars_disc_to_cont)
    fwd_times, all_p_times  = fwd_spiketimes(
        fwd_score_ready_times(valtimes, assess_sampling_tree),
        keys(assess_sampling_tree), [addr_to_domain[a] for a in keys(assess_sampling_tree)], tr, AssemblySize(), MaxRate(), K_fwd(), to_ready_spike_dist; nest_all_at, vars_disc_to_cont
    )
            
    return ISSpiketrains(valtimes, val_trains, recip_times, fwd_times, all_p_times, all_q_times)
end
sample_is_spiketimes_for_trace(tr, propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order, addr_to_domain; nest_all_at=nothing, vars_disc_to_cont=Dict()) =
    sample_is_spiketimes_for_trace(
        tr, DefaultInterSampleTimeDist(), propose_sampling_tree,
        assess_sampling_tree, propose_addr_topological_order, addr_to_domain, DefaultToReadySpikeDist();
        nest_all_at, vars_disc_to_cont
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
    addrs, vals, tr,
    assembly_size,
    neuron_rate,
    count_threshold,
    dist_to_ready_spike;
    nest_all_at, vars_disc_to_cont
)
    ch = get_ch(tr, nest_all_at)
    selected_times = Dict{Any, DenseValueSpiketrain}() # addr => [ [ times at which this neuron spikes ] for i=1:assembly_size ]
    all_times = Dict()
    for (addr, values) in zip(addrs, vals)
        if addr in keys(vars_disc_to_cont)
            sumest_over_pcontgivendisc = get_recip_score(ch, addr)
            pcontgivendisc = exp(project(tr, select(vars_disc_to_cont[addr](nest_all_at))))
            sumest = sumest_over_pcontgivendisc * pcontgivendisc
            num_spikes = try
                (sumest * ContinuousToDiscreteScoreNumSpikes()) |> to_int
            catch
                @error("""
                    get_recip_score(ch, $addr) = $(get_recip_score(ch, addr))) ;
                    pcontgivendisc [exp(project(tr, select($(vars_disc_to_cont[addr](nest_all_at)))))] = $(exp(project(tr, select(vars_disc_to_cont[addr](nest_all_at)))));
                    sumest * ContinuousToDiscreteScoreNumSpikes() = $(sumest * ContinuousToDiscreteScoreNumSpikes()))
                """)
                error()
            end

            neuron_times = spiketrains_for_n_spikes_in_time(
                num_spikes, LatencyForContinuousToDiscreteScore(), ready_times[addr], assembly_size
            )
        else
            num_spikes = try
                get_recip_score(ch, addr) * count_threshold |> to_int
            catch e
                @error "count threshold = $count_threshold; addr = $addr; get_recip_score(ch, addr) = $(get_recip_score(ch, addr))" exception=(e, catch_backtrace())
                error()
            end
            
            neuron_times = spiketrains_for_n_spikes(
                num_spikes, assembly_size, neuron_rate, ready_times[addr]
            )
            flat_neuron_times = reduce(vcat, neuron_times)

            all_times[addr] = get_nonselected_spiketimes(tr, addr, values, num_spikes, ready_times, count_threshold, neuron_rate, assembly_size, neuron_times;
                recip=true, last_spike_time=maximum(flat_neuron_times), nest_all_at
            )
        end

        ready_time = last(sort(reduce(vcat, neuron_times; init=Float64[]))) + rand(dist_to_ready_spike)
        selected_times[addr] = DenseValueSpiketrain(ready_time, neuron_times)
    end

    return (selected_times, all_times)
end

function fwd_spiketimes(
    ready_times, # addr -> time at which it becomes possible to start scoring the addr
    addrs, vals, tr,
    assembly_size,
    neuron_rate,
    count_threshold,
    dist_to_ready_spike;
    nest_all_at, vars_disc_to_cont
)
    ch = get_ch(tr, nest_all_at)
    # get_score = do_recip_score ? get_recip_score : get_fwd_score
    selected_times = Dict{Any, DenseValueSpiketrain}() # addr => [ [ times at which this neuron spikes ] for i=1:assembly_size ]
    all_times = Dict()
    for (addr, values) in zip(addrs, vals)
        # if addr in keys(vars_disc_to_cont)
        #     # There is no model-score for discrete -> continuous conversions; hence there is no fwd-spiketrain.
        #     continue;
        # end
        num_spikes = get_fwd_score(ch, addr) * count_threshold |> to_int
        p = with_weight_type(:perfect, () -> exp(Gen.project(tr, Gen.select(nest(nest_all_at, addr)))))
        
        neuron_times = spiketrains_for_n_spikes(
            num_spikes, assembly_size, neuron_rate * p, ready_times[addr]
        )

        flat_neuron_times = reduce(vcat, neuron_times; init=Float64[])
        if isempty(flat_neuron_times)
            selected_times[addr] = DenseValueSpiketrain(Inf, neuron_times)
        else
            ready_time = last(sort(flat_neuron_times)) + rand(dist_to_ready_spike)
            selected_times[addr] = DenseValueSpiketrain(ready_time, neuron_times)
        end

        all_times[addr] = get_nonselected_spiketimes(tr, addr, values, num_spikes, ready_times, count_threshold, neuron_rate, assembly_size, neuron_times; nest_all_at)
    end

    return (selected_times, all_times)
end

function get_nonselected_spiketimes(tr, addr, values, num_spikes, ready_times, count_threshold, neuron_rate, assembly_size, score_neuron_times; recip=false, last_spike_time=nothing, nest_all_at)
    num_spikes_for_selected = recip ? count_threshold : num_spikes
    num_spikes_from_assembly = recip ? num_spikes : count_threshold

    if !recip
        selected_neuron_times = score_neuron_times
    else
        selected_neuron_times = select_n_times_including_last(num_spikes_for_selected, score_neuron_times)
    end
    
    # Spiketimes for the non-selected assemblies
    ### CORRECTNESS TODO: if this is a recip score, the non-selected assembly times should be fixed
    ### by the given `score_neuron_times` array, not generated randomly.
    all_spiketimes = Dict()
    selected_value = get_choices(tr)[nest_add_val(nest_all_at, addr)]
    ordered_values = collect(values)
    n_spikes_for_others = num_spikes_from_assembly - num_spikes_for_selected
    probs = [
        with_weight_type(:perfect,
            () -> exp(Gen.project(
                Gen.update(tr, choicemap((nest_add_val(nest_all_at, addr), val)))[1],
                Gen.select(nest_add_val(nest_all_at, addr))
            )
        )
        ) for val in ordered_values
    ]
    selected_idx = 
        try
            only(findall(ordered_values .== selected_value))
        catch e
            @error "ordered_values = $ordered_values ; selected_value = $selected_value"
            throw(e)
        end

    probs_no_selected = copy(probs)
    probs_no_selected[selected_idx] = 0
    normalized_probs_no_selected = probs_no_selected ./ sum(probs_no_selected)
    counts = [0 for _ in ordered_values]
    counts[selected_idx] = num_spikes_for_selected
    for _=1:n_spikes_for_others
        counts[rand(Categorical(normalized_probs_no_selected))] += 1
    end
    
    if isnothing(last_spike_time)
        @assert !recip
        get_spiketrains(num_spikes, num_neurons, neuron_rate, starttime, is_selected_val) =
            is_selected_val ? selected_neuron_times : spiketrains_for_n_spikes(num_spikes, num_neurons, neuron_rate, starttime)
    else
        @assert recip
        get_spiketrains = (
            (num_spikes, num_neurons, neuron_rate, starttime, is_selected_val) ->
                ### CORRECTNESS TODO: For the first period before Q scoring is complete, we should output spikes in a way constrained
                ### by the Q scoring behavior.  There currently appear to be bugs in the code that is supposed to do this.
                random_q_spiking(starttime, Latency(), num_neurons, neuron_rate)
                # append_neuron_times_arrays(
                #     is_selected_val ? selected_neuron_times : spiketrains_for_n_spikes_in_time(num_spikes, last_spike_time - starttime, starttime, num_neurons),
                #     random_q_spiking(last_spike_time, Latency(), num_neurons, neuron_rate)
                # )
        )
    end
    
    for (count, prob, val) in zip(counts, probs, values)
        all_spiketimes[val] = get_spiketrains(count, assembly_size, neuron_rate * prob, ready_times[addr], val == selected_value)
        # if val != selected_value
        #     all_spiketimes[val] = get_spiketrains(count, assembly_size, neuron_rate * prob, ready_times[addr])
        # else
        #     all_spiketimes[val] = append_neuron_times_arrays(selected_neuron_times, random_q_spiking(last_spike_time, Latency(), neuron_rate * prob, neuron_rate))
        # end
    end
   
    return all_spiketimes
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
function spiketrains_for_n_spikes_in_time(num_spikes, time_window_length, starttime, num_neurons)
    # Poisson-process spikes are uniformly distributed given the interval they fell within
    times = [[] for _=1:num_neurons]
    for _=1:num_spikes
        t = uniform_continuous(starttime, starttime + time_window_length)
        idx = rand(DiscreteUniform(1, num_neurons))
        push!(times[idx], t)
    end
    for t in times
        sort!(t)
    end
    return times
end

function append_neuron_times_arrays(a1, a2)
    return [
        vcat(x1, x2) for (x1, x2) in zip(a1, a2)
    ]
end
function random_q_spiking(starttime, endtime, num_neurons, neuron_rate)
    n_spikes = poisson(num_neurons * neuron_rate * (endtime - starttime))
    return spiketrains_for_n_spikes_in_time(n_spikes, endtime-starttime, starttime, num_neurons)
end

function select_n_times_including_last(n, times)
    lengths = [length(tms) for tms in times]
    num_per_array = [0 for _ in times]
    for _=1:n
        neuron_idx = categorical(lengths ./ sum(lengths))
        num_per_array[neuron_idx] += 1
        lengths[neuron_idx] -= 1
    end
    return [
        sample_n_items_from_list_in_order(num, times_for_this_neuron)
        for (num, times_for_this_neuron) in zip(num_per_array, times)
    ]
end
function sample_n_items_from_list_in_order(n, vals)
    vals2 = copy(vals)
    sampled = []
    for _=1:n
        idx = uniform_discrete(1, length(vals2))
        push!(sampled, vals2[idx])
        deleteat!(vals2, idx)
    end
    return sort(sampled)
end

function to_int(v)
    @assert isapprox(v, floor(v)) || isapprox(v, ceil(v)) "v = $v"
    return Int(isapprox(v, floor(v)) ? floor(v) : ceil(v))
end

### Some default distributions for spiketrain production ###
DefaultInterSampleTimeDist() = Exponential(1 / (MaxRate() * AssemblySize()))
DefaultToReadySpikeDist() = Exponential(1 / (AssemblySize() * MaxRate()))
