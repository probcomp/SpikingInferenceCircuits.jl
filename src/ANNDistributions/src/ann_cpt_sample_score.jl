"""
Use a compiled ANN to implement the CPT Sample / CPT Score interface.
"""

# Abstract ANNCPTSample type (without implementation parameters)
struct ANNCPTSample <: GenericComponent
    layers::Vector{<:Flux.Dense}
    input_ncategories::Tuple
end
ANNCPTSample(chain::Flux.Chain, input_ncategories) =
    ANNCPTSample(chain.layers |> collect, input_ncategories)
Circuits.inputs(c::ANNCPTSample) = NamedValues(
    :in_vals => IndexedValues(
        SIC.FiniteDomainValue(n)
        for n in c.input_ncategories
    )
)
Circuits.outputs(c::ANNCPTSample) = NamedValues(
    :value => SIC.FiniteDomainValue(length(last(c.layers).bias)),
    :inverse_prob => SIC.ReciprocalProbEstimate()
)

# Concrete ANNCPTSample which can be fully implemented
struct ConcreteANNCPTSample <: GenericComponent
    ann::FullyConnectedANNWithDelay
    input_ncategories::Tuple
end
Circuits.abstract(c::ConcreteANNCPTSample) = ANNCPTSample(c.ann.layers, c.input_ncategories)
function ConcreteANNCPTSample(
    a::ANNCPTSample;
    neuron_memory::Real,
    network_memory_per_layer=missing::Union{Real, Missing},
    network_memory=(length(a.layers) * network_memory_per_layer)::Union{Real, Missing},
    timer_params,
    timer_memory_mult=2. # how many times the expected timer time should the timer remember?
)
    if ismissing(network_memory)
        error("Kwarg `network_memory` or `network_memory_per_layer` must be provided.")
    end
    
    return ConcreteANNCPTSample(
        FullyConnectedANNWithDelay(
            FullyConnectedANN(a.layers, neuron_memory), network_memory, timer_params, timer_memory_mult),
            a.input_ncategories
        )
end
out_domain_size(c::ConcreteANNCPTSample) = n_output_lines(c.ann.ann)

Circuits.inputs(c::ConcreteANNCPTSample) = NamedValues(
    :in_vals => IndexedValues(
        SIC.FiniteDomainValue(n)
        for n in c.input_ncategories
    )
)
Circuits.outputs(c::ConcreteANNCPTSample) = NamedValues(
    :value => SIC.FiniteDomainValue(length(outputs(c.ann))),
    :inverse_prob => SIC.ReciprocalProbEstimate()
)

# TODO: this is sloppy!
# But for now to get the `wta` and `counter` implementations,
# we just get what the implementation is for a simple ConditionalSample
# with a simple CPT with the right output size
# This will use whatever implementaiton rule is defined for ConditionalSample
example_pulse_cond_sample(c) = implement(
    SIC.SDCs.ConditionalSample([i == 1 ? 1. : 0. for i=1:out_domain_size(c)] |> x->reshape(x, (1, :))),
    Spiking()
)
wta(c::ConcreteANNCPTSample) = SIC.SDCs.wta(
    example_pulse_cond_sample(c)
)
probcounter(c::ConcreteANNCPTSample) = SIC.SDCs.probcounter(
    example_pulse_cond_sample(c)
)

Circuits.implement(c::ConcreteANNCPTSample, ::Spiking) =
    CompositeComponent(
        implement_deep(inputs(c), Spiking()), 
        implement_deep(outputs(c), Spiking(), :value), # only implement `value`
        (
            ann = c.ann,
            # TODO; methods to get WTA and ProbCounter
            wta = wta(c),
            counter = probcounter(c)
        ),
        (
            # Inputs -> ANN
            onehots_to_ann_input(c.input_ncategories)...,

            # The remaining edges are pretty much copied exactly from `ConditionalSample`
            (
                CompOut(:ann, i) => CompIn(:wta, i)
                for i=1:out_domain_size(c)
            )...,
            Iterators.flatten(
                (
                    CompOut(:wta, i) => CompIn(:counter, :sel => i),
                    CompOut(:ann, i) => CompIn(:counter, :samples => i)
                )
                for i=1:out_domain_size(c)
            )...,
            (
                CompOut(:wta, i) => Output(:value => i)
                for i=1:out_domain_size(c)
            )...,
            CompOut(:counter, :count) => Output(:inverse_prob)
        ), c
    )
function onehots_to_ann_input(input_ncategories)
    edges = []
    for (varidx, n) in enumerate(input_ncategories)
        start = length(edges)
        for i=1:n
            val = start + i
            push!(edges,
                Input(:in_vals => varidx => i) => CompIn(:ann, val)
            )
        end
    end
    return edges
end