### Kernel ###

struct MHKernel <: GenericComponent
    model::GenFn{Generate} # Assess
    propose::GenFn{Propose}
    assess_proposal::GenFn{Generate} # Assess
end
MHKernel(model::Gen.GenerativeFunction, model_arg_domains, proposal::Gen.GenerativeFunction, proposal_arg_domains) =
    MHKernel(
        gen_fn_circuit(model, model_arg_domains, Assess()),
        gen_fn_circuit(proposal, proposal_arg_domains, Propose()),
        gen_fn_circuit(proposal, proposal_arg_domains, Assess())
    )

# TODO: constructor to create these circuits!!

Circuits.inputs(mh::MHKernel) = NamedValues(
    :prev_trace => traceable_value(mh.model),
    :model_args => inputs(mh.model)[:inputs]
)
Circuits.outputs(mh::MHKernel) = NamedValues(
    :next_trace => traceable_value(mh.model)
)

Circuits.implement(mh::MHKernel, ::Target) =
    CompositeComponent(
        inputs(mh), outputs(mh),
        (
            propose=mh.propose,
            assess_bwd_proposal=mh.assess_proposal,
            assess_old_trace=mh.model,
            assess_new_trace=mh.model,
            new_trace_mult=NonnegativeRealMultiplier((
                outputs(mh.model)[:score],
                outputs(mh.propose)[:score],
                outputs(mh.assess_proposal)[:score]
            )),
            old_trace_mult=NonnegativeRealMultiplier((outputs(mh.model)[:score],)),
            theta=Theta(2),
            trace_mux=Mux(2, traceable_value(mh.model))
        ),
        (
            Input(:prev_trace) => CompIn(:propose, :inputs),

            Input(:model_args) => CompIn(:assess_new_trace, :inputs),
            edges_from_new_trace_to(mh, CompIn(:assess_new_trace, :obs))...,

            Input(:prev_trace) => CompIn(:assess_old_trace, :obs),
            Input(:model_args) => CompIn(:assess_old_trace, :inputs),

            (
                Input(:prev_trace => key) => CompIn(:assess_bwd_proposal, :obs => key)
                for key in keys(inputs(mh.assess_proposal)[:obs])
            )...,

            edges_from_new_trace_to(mh, CompIn(:assess_bwd_proposal, :inputs))...,

            CompOut(:assess_new_trace, :score) => CompIn(:new_trace_mult, 1),
            CompOut(:propose, :score) => CompIn(:new_trace_mult, 2),
            CompOut(:assess_bwd_proposal, :score) => CompIn(:new_trace_mult, 3),

            CompOut(:assess_old_trace, :score) => CompIn(:old_trace_mult, 1),

            CompOut(:old_trace_mult, :out) => CompIn(:theta, 1),
            CompOut(:new_trace_mult, :out) => CompIn(:theta, 2),

            CompOut(:theta, :val) => CompIn(:trace_mux, :sel),
            Input(:prev_trace) => CompIn(:trace_mux, :values => 1),
            edges_from_new_trace_to(mh, CompIn(:trace_mux, :values => 2))...,

            CompOut(:trace_mux, :out) => Output(:next_trace)
        ),
        mh
    )

edges_from_new_trace_to(mh::MHKernel, receiver::Circuits.NodeName) =( 
        if haskey(outputs(mh.propose)[:trace], key)
            CompOut(:propose, :trace => key) => Circuits.append_to_valname(receiver, key)
        else
            Input(:prev_trace => key) => Circuits.append_to_valname(receiver, key)
        end

        for key in keys(traceable_value(mh.model))
    )

### MH Inference alg ###
struct MH <: GenericComponent
    kernels::Vector{MHKernel}
    function MH(kernels::Vector{MHKernel})
        # All kernels should be for the same model!
        # @assert length(Set(kernel.model for kernel in kernels)) == 1
        # TODO: add in this check.  It currently doesn't work since
        # equality for GenFnCircuits is not defined properly.

        return new(kernels)
    end
end
Circuits.inputs(mh::MH) = NamedValues(
    :model_args => inputs(first(mh.kernels).model)[:inputs],
    :initial_trace => traceable_value(first(mh.kernels).model)
)
Circuits.outputs(mh::MH) = NamedValues(:updated_traces => IndexedValues(
    outputs(kernel)[:next_trace] for kernel in mh.kernels
))
Circuits.implement(mh::MH, ::Target) = CompositeComponent(
    inputs(mh), outputs(mh), (
        kernels=IndexedComponentGroup(mh.kernels),
        steps=IndexedComponentGroup(
            Step(NamedValues(
                :trace => inputs(mh)[:initial_trace],
                :args => inputs(mh)[:model_args]
            ))
            for _ in mh.kernels
        )
    ), (
        Input(:model_args) => CompIn(:kernels => 1, :model_args),
        Input(:initial_trace) => CompIn(:kernels => 1, :prev_trace),

        # advance traces forward 1 --> N
        (
            CompOut(:kernels => i, :next_trace) => CompIn(:steps => i, :in => :trace)
            for i=1:(length(mh.kernels) - 1)
        )...,

        # advance model_args forward 1-->n
        Input(:model_args) => CompIn(:steps => 1, :in => :args),
        (
            CompOut(:steps => i, :out => :args) => CompIn(:steps => i + 1, :in => :args)
            for i=2:(length(mh.kernels) - 1)
        )...,

        # Cycle trace and model args around
        CompOut(:steps => length(mh.kernels), :out => :args) => CompIn(:kernels => 1, :model_args),
        CompOut(:steps => length(mh.kernels), :out => :trace) => CompIn(:kernels => 1, :prev_trace),
        CompOut(:steps => length(mh.kernels), :out => :args) => CompIn(:steps => 1, :in => :args),

        # Output each next_trace
        (
            CompOut(:kernels => i, :next_trace) => Output(:updated_traces => i)
            for i=1:length(mh.kernels)
        )...
    )
)