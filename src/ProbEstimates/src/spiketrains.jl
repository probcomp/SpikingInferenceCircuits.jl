#=
General workflow:
Call some global function, which gives you an empty mutable object.
Run inference in Gen.
Now the mutable object is populated with data.  You can use the mutable object
to produce spiketrains (and visualizations).


--- What is recorded during inference run? ---



--- What is the interface to produce spiketrains / visualizations? ---
User provides:
- the inference-run data (the mutable object)
- the traces produced during visualization (so it can cross-reference to figure out the addresses)
- a specification of the visualization to produce
- Some additional parameters explaining how to view the inference algorithm as being situated in time
  - (e.g. how far apart each observation was)

Global values for latency, assembly sizes, etc., are used to produce the visualizations.

First draft:
- Visualize spikes for single trace produced via a Propose+Assess.

Second iteration:
- Do the above, sequenced in time somehow


-- What the user can specify to show --
User specifies a sequence of lines, which will be vertically arranged on an outputted "spiketrain" image.

Each row can either be:

- ("sampled value", trace address)          : write the sampled value in English
- ("fwd score in English", trace address)   : write the sampled forward score in English
- ("recip score in English", trace address) : write the sampled recip score in English

- ("spiketrain samples for variable", trace address, value line) : adds line in spiketrain for "onehot" value representation - line for value `value`
- ("spiketrain for recip value" | "spiketrain for value", trace address, 'assembly' | 'neuron k' | 'count ready')


We may also eventually want utils to automatically show all the lines for a onehot value, etc.

--- How to associate addresses to values? ---
When making a spiketrain for a run of IS that produced a trace, we pass in the produced trace.
This trace must provide a way to get the information about what spike counts occurred during
inference.

When calling `update` or `generate` on `LCat`, we should store the counts from assess score estimation
in the produced trace.
When calling `propose`, we need to return a choicemap with some meta-data, such that when generating a `LCat`
trace from that metadata, the LCat stores the count in the trace.


***
Time estimates:
- [x] [40 mins] Tracking sampled probability estimates in NG-F traces.
- [x] [40 mins] Interface to produce some non-rendered output with strings, based on inference run.
- [x] [1.5 hr]    Interface + first pass of math to produce non-rendered spiketrains.
- [1 hr]    Above --> Visualizations
=#

#=
Note to self on progress:

I think I have the first TODO (getting the scores into the trace so we can reference
them later) working.  But I'm worried there could be edge cases where this doesn't work properly
(I'm going through pretty fast and not taking too much time to think about this)!

In particular I think in places where we try to get a reciprocal probability estimate using `assess`
by calling `use_only_recip_weights!`, we could run into trouble (since I put any call to `recip_score`
into the trace's recip column, regardless of the weight mode).
=#
module Spiketrains
using Distributions: Exponential, DiscreteUniform
using ProbEstimates: MaxRate, AssemblySize, K_fwd, K_recip, with_weight_type
using Gen

nest(a, b) = a => b
nest(a::Pair, b) = a.first => nest(a.second, b)

abstract type LineSpec end
function get_lines(specs, tr, spiketrain_data_args)
    spiketrain_data =
        if any(spec isa SpiketrainSpec for spec in specs)
            sample_spiketimes_for_trace(tr, spiketrain_data_args...)
        else
            nothing
        end
    return [get_line(spec, tr, spiketrain_data) for spec in specs]
end
get_labels(lines) = map(get_label, lines)

get_line(spec::LineSpec, tr) = get_line(spec, tr, nothing)

### Text ###
abstract type Text <: LineSpec; end
struct SampledValue <: Text; addr; end
struct FwdScoreText <: Text; addr; end
struct RecipScoreText <: Text; addr; end

get_line(spec::SampledValue, tr, _) = "$(tr[spec.addr])" #"$(spec.addr)=$(tr[spec.addr])"
get_line(spec::FwdScoreText, tr, _) = "$(get_fwd_score(tr, spec.addr))" # "P[$(spec.addr) ; Pa($(spec.addr))] ≈ $(get_fwd_score(tr, spec.addr))"
get_line(spec::RecipScoreText, tr, _) = "$(get_recip_score(tr, spec.addr))" #"Q[$(spec.addr) ; Pa($(spec.addr))] ≈ $(get_recip_score(tr, spec.addr))"

get_label(spec::SampledValue) = "$(spec.addr) = "
get_label(spec::FwdScoreText) = "P[$(spec.addr) ; Pa($(spec.addr))] ≈"
get_label(spec::RecipScoreText) = "1/Q[$(spec.addr) ; Pa($(spec.addr))] ≈"

get_fwd_score(tr, addr) = tr[nest(addr, :fwd_score)]
get_recip_score(tr, addr) = tr[nest(addr, :recip_score)]

### Spikes ###
struct DenseValueSpiketrain
    ready_time   :: Float64
    neuron_times :: Vector{Vector{Float64}}
    # neuron_times[i] = spiketrain for the `i`th neuron in the output gate assembly
end
struct ISSpiketrains
    valtimes::Dict{<:Any, Float64}
    recip_trains::Dict{<:Any, DenseValueSpiketrain}
    fwd_trains::Dict{<:Any, DenseValueSpiketrain}
end

abstract type SpikelineInScore end
struct CountAssembly <: SpikelineInScore; end
struct NeuronInCountAssembly <: SpikelineInScore; idx; end
struct IndLine <: SpikelineInScore; end

abstract type SpiketrainSpec <: LineSpec; end
struct VarValLine <: SpiketrainSpec; addr; value; end
struct ScoreLine <: SpiketrainSpec
    do_recip_score::Bool # alternatively do fwd
    addr
    line_to_show::SpikelineInScore
end
RecipScoreLine(addr, line_to_show) = ScoreLine(true, addr, line_to_show)
FwdScoreLine(addr, line_to_show) = ScoreLine(false, addr, line_to_show)

get_line(spec::VarValLine, tr, trains) = tr[spec.addr] == spec.value ? [trains.valtimes[spec.addr]] : []
get_line(spec::ScoreLine, tr, trains) = get_score_line(spec.line_to_show, (spec.do_recip_score ? trains.recip_trains : trains.fwd_trains)[spec.addr])
get_score_line(::IndLine, trains::DenseValueSpiketrain) = [trains.ready_time]
get_score_line(::CountAssembly, trains::DenseValueSpiketrain) = sort(reduce(vcat, trains.neuron_times))
get_score_line(n::NeuronInCountAssembly, trains::DenseValueSpiketrain) = trains.neuron_times[n.idx]

get_label(spec::VarValLine) = "$(spec.addr)=$(spec.value)"
function get_label(spec::ScoreLine)
    val_label = spec.do_recip_score ? "1/Q[$(spec.addr) ; Pa($(spec.addr))]" : "P[$(spec.addr) ; Pa($(spec.addr))]"
    if spec.line_to_show isa IndLine
        return "$val_label ready"
    elseif spec.line_to_show isa CountAssembly
        return "$val_label count"
    elseif spec.line_to_show isa NeuronInCountAssembly
        return "$val_label count - neuron $(spec.line_to_show.idx)"
    end
end

function sample_spiketimes_for_trace(
    tr,
    inter_sample_time_dist,
    propose_sampling_tree, # as Dict(addr -> [list of parent addrs])
    assess_sampling_tree,
    propose_addr_topological_order,
    to_ready_spike_dist
)
    valtimes = sampled_value_times(inter_sample_time_dist, propose_sampling_tree, propose_addr_topological_order)
    recip_times = recip_spiketimes(valtimes, propose_addr_topological_order, tr, AssemblySize(), MaxRate(), K_recip(), to_ready_spike_dist)
    fwd_times   = fwd_spiketimes(
        fwd_score_ready_times(valtimes, assess_sampling_tree),
        keys(assess_sampling_tree), tr, AssemblySize(), MaxRate(), K_fwd(), to_ready_spike_dist
    )

    return ISSpiketrains(valtimes, recip_times, fwd_times)
end
sample_spiketimes_for_trace(tr, propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order) =
    sample_spiketimes_for_trace(
        tr, DefaultInterSampleTimeDist(), propose_sampling_tree,
        assess_sampling_tree, propose_addr_topological_order, DefaultToReadySpikeDist()
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

### TODO: refactor code to reduce repeat between `recip_spiketimes` and `fwd_spiketimes`
function recip_spiketimes(
    ready_times, # addr -> time at which it becomes possible to start scoring the addr
    addrs, tr,
    assembly_size,
    neuron_rate,
    count_threshold,
    dist_to_ready_spike
)
    times = Dict{Any, DenseValueSpiketrain}() # addr => [ [ times at which this neuron spikes ] for i=1:assembly_size ]
    for addr in addrs
        num_spikes = get_recip_score(tr, addr) * count_threshold |> to_int
        
        neuron_times = spiketrains_for_n_spikes(
            num_spikes, assembly_size, neuron_rate, ready_times[addr]
        )
        ready_time = last(sort(reduce(vcat, neuron_times))) + rand(dist_to_ready_spike)
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
    dist_to_ready_spike
)
    # get_score = do_recip_score ? get_recip_score : get_fwd_score
    times = Dict{Any, DenseValueSpiketrain}() # addr => [ [ times at which this neuron spikes ] for i=1:assembly_size ]
    for addr in addrs
        num_spikes = get_fwd_score(tr, addr) * count_threshold |> to_int
        p = with_weight_type(:perfect, () -> exp(Gen.project(tr, Gen.select(addr))))
        
        neuron_times = spiketrains_for_n_spikes(
            num_spikes, assembly_size, neuron_rate * p, ready_times[addr]
        )
        ready_time = last(sort(reduce(vcat, neuron_times))) + rand(dist_to_ready_spike)
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
    @assert isapprox(v, floor(v))
    return Int(floor(v))
end

### Some default distributions for spiketrain production ###
DefaultInterSampleTimeDist() = Exponential(1 / MaxRate())
DefaultToReadySpikeDist() = Exponential(1 / MaxRate())

export LineSpec, get_line, get_lines, get_label, get_labels
export SampledValue, FwdScoreText, RecipScoreText
export VarValLine, ScoreLine, RecipScoreLine, FwdScoreLine
export CountAssembly, NeuronInCountAssembly, IndLine

include("spiketrain_visualization.jl")
export SpiketrainViz

end # module