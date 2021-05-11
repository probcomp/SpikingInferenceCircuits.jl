struct PoissonSync <: ConcretePulseIRPrimitive
    cluster_sizes::Vector{Int}
    gate_params::NTuple{2, Float64} # (M, R)
    ti_params::NTuple{3, Float64} # (max_delay, M, R)
    timer_params::Tuple{Float64, Int, NTuple{3, Float64}, Float64, Float64}
    # PoissonTimer_params= (ΔT_PoissonTimer, n_spikes, PoissonTimer_ti_params, offrate, memory)
end
Circuits.abstract(s::PoissonSync) = Sync(s.cluster_sizes)
Circuits.target(::PoissonSync) = Spiking()
Circuits.inputs(s::PoissonSync) = inputs(abstract(s))
Circuits.outputs(s::PoissonSync) = outputs(abstract(s))

cluster(s, i) = PoissonSyncCluster(s.cluster_sizes[i], (Inf, s.gate_params...))
Circuits.implement(s::PoissonSync, ::Spiking) = CompositeComponent(
    inputs(s), outputs(s),
    (
        clusters = IndexedComponentGroup(
            cluster(s, i) for i=1:length(s.cluster_sizes)
        ),
        ti = PoissonThresholdedIndicator(
            length(s.cluster_sizes), Inf, s.ti_params...
        ),
        timer = PoissonTimer(s.timer_params...),
    ),
    (
        (
            Input(i) => CompIn(:clusters => i, :values)
            for i=1:length(s.cluster_sizes)
        )...,
        (
            CompOut(:clusters => i, :values) => Output(i)
            for i=1:length(s.cluster_sizes)
        )...,
        (
            CompOut(:clusters => i, :activated) => CompIn(:ti, :in)
            for i=1:length(s.cluster_sizes)
        )...,
        (
            CompOut(:ti, :out) => CompIn(:clusters => i, :step)
            for i=1:length(s.cluster_sizes)
        )...,
        CompOut(:ti, :out) => CompIn(:timer, :start),
        (
            CompOut(:timer, :out) => CompIn(:clusters => i, :reset)
            for i=1:length(s.cluster_sizes)
        )...
    ), s
)

### Cluster ###
"""
    PoissonSyncCluster(nlines, gate_params) # gate_params = (ΔT, M, R)

Sub-component of a PoissonSync which handles a ``cluster'' of lines.
Outputs a spike in `:activated` once at least one spike has been received
in one of the input `:values` lines.
Upon receiving a `:step` spike, outputs the number of spikes received in each
`:values` input to the corresponding `:values` output.
Upon receiving a `:reset` spike, resets so it can be used again.
"""
struct PoissonSyncCluster <: GenericComponent
    nlines::Int
    gate_params::Tuple{Float64, Float64, Float64}
end
Circuits.target(::PoissonSyncCluster) = Spiking()
Circuits.inputs(c::PoissonSyncCluster) = NamedValues(
    :values => IndexedValues(SpikeWire() for _=1:c.nlines),
    :step => SpikeWire(), :reset => SpikeWire()
)
Circuits.outputs(c::PoissonSyncCluster) = NamedValues(
    :values => inputs(c)[:values],
    :activated => SpikeWire()
)

Circuits.implement(c::PoissonSyncCluster, ::Spiking) =
    CompositeComponent(
        inputs(c), outputs(c),
        (
            oneout=PoissonOnOffGate(:on, c.gate_params...),
            gates=IndexedComponentGroup(
                PoissonOnOffGate(:off, c.gate_params...) for _=1:c.nlines
            )
        ),
        (
            # edges for oneout
            ( # inspikes can activate oneout
                Input(:values => i) => CompIn(:oneout, :in)
                for i=1:c.nlines
            )...,
            CompOut(:oneout, :out) => Output(:activated),
            CompOut(:oneout, :out) => CompIn(:oneout, :off), # turn off after emitting 1 spike
            Input(:reset) => CompIn(:oneout, :on), # reenable oneout after a reset

            (
                Input(:values => i) => CompIn(:gates => i, :in)
                for i=1:c.nlines
            )...,
            Iterators.flatten(
                (
                    Input(:step) => CompIn(:gates => i, :on),
                    Input(:reset) => CompIn(:gates => i, :off)
                )
                for i=1:c.nlines
            )...,
            (
                CompOut(:gates => i, :out) => Output(:values => i)
                for i=1:c.nlines
            )...
        ),
        c
    )

### OnOffGate ###
# TODO: make this a PulseIRPrimitive?
# Or, even better, make it a primitive which `OnGate` and `OffGate`
# are implemented in terms of?
struct PoissonOnOffGate <: GenericComponent
    starts_on::Bool
    ΔT::Float64
    M::Float64
    R::Float64
end
PoissonOnOffGate(s::Symbol, params...) = PoissonOnOffGate(
    s == :on ? true : (s == :off ? false : error("unrecognized start state (should be :on or :off)")),
    params...
)
Circuits.inputs(::PoissonOnOffGate) = NamedValues(:in => SpikeWire(), :on => SpikeWire(), :off => SpikeWire())
Circuits.outputs(::PoissonOnOffGate) = NamedValues(:out => SpikeWire())
Circuits.implement(g::PoissonOnOffGate, ::Spiking) = 
    CompositeComponent(
        inputs(g), outputs(g),
        (
            neuron=PoissonNeuron([
                x -> x, x -> -x, x -> g.M*min(x, 1), x -> -g.M*min(x, 1)
            ], g.ΔT, u -> exp(g.R * (u - 1/2 - (g.starts_on ? 0. : g.M)))),
        ),
        (
            Input(:in) => CompIn(:neuron, 1),
            CompOut(:neuron, :out) => CompIn(:neuron, 2),
            Input(:on) => CompIn(:neuron, 3),
            Input(:off) => CompIn(:neuron, 4),
            CompOut(:neuron, :out) => Output(:out)
        ),
        g
    )