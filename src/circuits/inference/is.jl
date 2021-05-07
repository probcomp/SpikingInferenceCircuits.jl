abstract type ImportanceSamplingParticle <: Circuits.GenericComponent end
struct PriorISParticle <: ImportanceSamplingParticle
    model::GenFn
end
struct ProposalISParticle <: ImportanceSamplingParticle
    model::GenFn
    proposal::GenFn
    observed_addrs
end
function PriorISParticle(model::Gen.GenerativeFunction, model_arg_domains::Tuple, observed_addrs::Selection)
    circuit = gen_fn_circuit(model, model_arg_domains, Generate(observed_addrs))
    @assert haskey(outputs(circuit), :prob) && haskey(outputs(circuit), :trace) "There is no trace or no weight outputted when doing importance sampling with this spec!  (So why do IS?  Just use a gen fn circuit in Propose or Assess mode.)"
    return PriorISParticle(circuit)
end

function ProposalISParticle(model::Gen.GenerativeFunction, model_arg_domains, observed_addrs::Selection, proposal::Gen.GenerativeFunction, proposal_arg_domains)
    prop_circuit = gen_fn_circuit(proposal, proposal_arg_domains, Propose())
    model_circuit = gen_fn_circuit(
        model, model_arg_domains, Generate(
            merge_selection(observed_addrs, select((addr for addr in keys_deep(outputs(prop_circuit)[:trace]))))
        ))

    @assert haskey(outputs(prop_circuit), :trace) "No trace output from proposal!  (So why are you using a custom proposal?)"
    @assert haskey(outputs(prop_circuit), :prob) "If the proposal has a trace, it should also have a probability output..."
    @assert haskey(outputs(model_circuit), :prob) "No prob output from the model!  (So why are you doing IS or using the model at all?)"

    return ProposalISParticle(model_circuit, prop_circuit, observed_addrs)
end

# TODO: something like this should be part of Gen
merge_selection(::AllSelection, _) = AllSelection()
merge_selection(_, ::AllSelection) = AllSelection()
merge_selection(::EmptySelection, x) = x
merge_selection(x, ::EmptySelection) = x
merge_selection(a::Union{DynamicSelection, StaticSelection}, b::Union{DynamicSelection, StaticSelection}) =
    DynamicSelection(Dict(
        k => (
            if haskey(a.subselection, k) && haskey(b.subselection, k)
                merge(a.subselection[k], b.subselection[k])
            elseif haskey(a.subselection, k)
                a.subselection[k]
            else
                @assert haskey(b.subselection, k)
                b.subselection[k]
            end
        )
        for k in unique(Iterators.flatten((keys(a.subselection), keys(b.subselection))))
    ))

Circuits.inputs(p::PriorISParticle) = inputs(p.model)
Circuits.outputs(p::PriorISParticle) = outputs(p.model)
# IS from the prior is just `generate`, where the prob output is labeled as the `:weight`
# TODO: change gen_fn_circuits so their output is called `:weight`
Circuits.implement(p::PriorISParticle, ::Target) = RelabeledIOComponent(p.model, (), (:prob => :weight,), p)

Circuits.inputs(p::ProposalISParticle) = NamedValues(
    :proposal_args => inputs(p.proposal)[:inputs],
    :model_args => inputs(p.model)[:inputs],
    :obs => inputs(p.model)[:obs],
)
Circuits.outputs(p::ProposalISParticle) = NamedValues(
    :weight => NonnegativeReal(),
    :trace => trace_output(p)
)
trace_output(p::ProposalISParticle) =
    if !haskey(outputs(p.model), :trace)
        outputs(p.proposal)[:trace]
    else
        merge_composite_value(outputs(p.proposal)[:trace], outputs(p.model)[:trace])
    end

Circuits.implement(p::ProposalISParticle) = CompositeComponent(
    inputs(p), outputs(p),
    (model=p.model, proposal=p.proposal, divider=NonnegativeRealDivider()),
    Iterators.flatten((
        ( # inputs: observation, model args, proposal args
            Input(:obs) => CompIn(:model, :obs),
            Input(:model_args) => CompIn(:model, :inputs),
            Input(:proposal_args) => CompIn(:proposal, :inputs)
        ),
        ( # proposal samples to model obs
            CompOut(:proposal, :trace => addr) => CompIn(:model, :obs => addr)
            for addr in keys(outputs(p.proposal)[:trace])
        ),
        ( # proposal trace to output
            CompOut(:proposal, :trace => addr) => Output(:trace => addr)
            for addr in keys(outputs(p.proposal)[:trace])
        ),
        haskey(outputs(p.model), :trace) ? () : ( # model trace to output
            CompOut(:model, :trace => addr) => Output(:trace => addr)
            for addr in keys(outputs(p.model)[:trace])
        ),
        ( # importance weight division and output
            CompOut(:model, :prob) => CompIn(:divider, :numerator),
            CompOut(:proposal, :prob) => CompIn(:divider, :denominator),
            CompOut(:divider, :out) => Output(:weight)
        )
    )),
    p
)