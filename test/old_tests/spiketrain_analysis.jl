### spiketrain analysis utils ###

"""
Given a vector of events from running the spiking simulator,
return a dict s.t. `dict[output_name] = spiketimes`.
Will have an entry for each output of the simulated circuit.
Each `spiketimes` will be a vector of the times at which this output spiked.
"""
function output_spiketrain_dict(event_vector)
    event_vector = filter(((t,args...),) -> is_primary_output(args...), event_vector)
    spiketrains = Dict()
    for (time, _, outspike) in event_vector
        if haskey(spiketrains, outspike.name)
            push!(spiketrains[outspike.name], time)
        else
            spiketrains[outspike.name] = [time]
        end
    end
    return spiketrains
end
is_primary_output(compname, event) = (isnothing(compname) && event isa SpikingSimulator.OutputSpike)

maybe_events_to_dict(v::Vector) = output_spiketrain_dict(v)
maybe_events_to_dict(d::Dict) = d

"""
Given a dictionary of the output spikes from running a Gen Fn circuit in
the spiking simulator, this constructs a Gen choicemap containing the choices made during the run.
(It does this by inspecting the `:trace` outputs of the circuit.)
"""
function gen_fn_circuit_choicemap(dict)
    c = choicemap()
    for (key, spiketimes) in dict
        if key isa Pair && key.first == :trace
            @assert length(spiketimes) == 1 "We expect to get 1 spike for each traced value"
            addr, value = unpack_traced_value(key)
            @assert !has_value(c, addr) "Got spikes for multiple entries of $addr"
            c[addr] = value
        end
    end
    return c
end
function unpack_traced_value(p::Pair)
    @assert p.first == :trace
    @assert p.second isa Pair
    addr = p.second.first
    remaining = p.second.second

    while remaining isa Pair
        addr = addr => remaining.first
        remaining = remaining.second
    end

    return (invert_nesting(addr), remaining)
end

# convert ((:a => :b) => :c) => :d  -->  :a => (:b => (:c => :d))
_invert_nesting(a, nested) = a => nested
_invert_nesting(p::Pair, nested) = _invert_nesting(p.first, p.second => nested)
invert_nesting(a) = a
invert_nesting(p::Pair) = _invert_nesting(p.first, p.second)

"""
Given a vector of event-vectors from running a Gen Fn circuit for `model` in Propose mode
with the given `args`, this calculates the emperical KL from the distribution which was sampled
from to the model distribution.

Ie. E_emp[log(emp / true)].

(We do the KL in this direction so we don't penalize infinitely for having no samples of an unlikely outcome.)
"""
function propose_emperical_kl_to_true(model, args, run_event_vecs)
    choicemaps = map(gen_fn_circuit_choicemap ∘ maybe_events_to_dict, run_event_vecs)
    model_logprob(cm) = Gen.assess(model, args, cm)[1]

    # we have to do `to_array` since Gen currently doesn't implement hashing for choicemaps!
    emp_counts = Dict()
    for cm in choicemaps
        emp_counts[cm] = get(emp_counts, cm, 0) + 1
    end
    emp_probs = Dict(k => v / length(choicemaps) for (k, v) in emp_counts)

    true_logprobs = Dict(cm => model_logprob(cm) for cm in keys(emp_probs))

    println("True vs emp probs:")
    for cm in keys(emp_probs)
        println("  emp: $(emp_probs[cm]) | true: $(exp(true_logprobs[cm])) | diff of logs: $(emp_probs[cm] - true_logprobs[cm]) | KL contribution: $(emp_probs[cm] * (emp_probs[cm] - true_logprobs[cm]))")
    end

    display(emp_probs)

    return sum(
        emp_prob * (log(emp_prob) - model_logprob(cm))
        for (cm, emp_prob) in emp_probs
    )
end
Base.hash(c::Gen.ChoiceMap, h::UInt) = hash(
    collect(get_values_shallow(c)), hash(
        collect(get_submaps_shallow(c)),
        h
    )
)

"""
Average rate of the `:prob` output in a spiketrain dict of for a run of length `run_time`.
Average is taken over the second half of the run.
"""
probrate(dict, run_time) =
    if haskey(dict, :prob)
        length([x for x in dict[:prob] if x > 1/2 * run_time]) / (1/2 * run_time)
    else
        0.
    end

"""
Return a tuple `(measured_weight, true_weight)` giving the weight read out from the circuit in the
last 1/2 of the run compared to the exact weight value.
"""
function compare_spike_rate_weight(model, args, op, spiketrain_dict, run_length; obs=nothing)
    choices = gen_fn_circuit_choicemap(maybe_events_to_dict(spiketrain_dict))
    measured_weight = probrate(spiketrain_dict, run_length) / REF_RATE()

    if op == Propose()
        @assert obs === nothing
        true_logweight, _ = Gen.assess(model, args, choices)
    else
        @assert op isa Generate
        @assert obs isa ChoiceMap "`obs` must be a choicemap when `op` isa Generate"
        tr, _ = Gen.generate(model, args, merge(choices, obs))
        true_logweight = Gen.project(tr, op.observed_addrs)
    end

    return (measured_weight, exp(true_logweight))
end

function do_run_and_check_spike_rate(run_length=100.0, input_val=2)
    events = SpikingSimulator.simulate_for_time_and_get_events(implemented, run_length;
        initial_inputs=(:inputs => :input => input_val,),
    )
    dict = spiketrain_dict(filter(((t,args...),) -> is_primary_output(args...), events))

    y1 = haskey(dict, :trace => :y1 => 1) ? 1 : 2
    y2 = haskey(dict, :trace => :y2 => 1) ? 1 : 2
    x1 = haskey(dict, :trace => :output => 1 => :x => 1) ? 1 : 2
    x2 = haskey(dict, :trace => :output => 2 => :x => 1) ? 1 : 2

    true_prob = exp(assess(outside, (input_val,), choicemap(
        (:y1, y1), (:y2, y2),
        (:output => 1 => :x, x1),
        (:output => 1 => :x, x2),
    ))[1])

    expected_rate = true_prob * REF_RATE()
    actual_rate   = probrate(dict, run_length)

    return (expected_rate, actual_rate)
end