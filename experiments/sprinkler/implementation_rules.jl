Circuits.implement(ta::SIC.SDCs.ToAssmts, ::Spiking) =
    SDCs.PulseToAssmts(
        ta, PulseIR.PoissonThresholdedIndicator,
        # ΔT, max_delay, M, R
        (500, 0.5, 50, 40)
        # Note that the R needs to be high since getting spikes while off is catastrophic.
        # TODO: Design things so this is not catastrophic (or can't happen at
        # realistic rates)!
    )

K = 10
ONRATE = 0.2
Circuits.implement(cs::SIC.SDCs.ConditionalSample, ::Spiking) =
    SDCs.PoissonPulseConditionalSample(
        (cs, K, ONRATE,
            500, # ΔT
            0.2, # max_delay
            1000, # M (num spikes to override offs/ons)
            50, # max delay before sample is emitted
            0.1 # intersample hold
        ),
        10^(-10), 12
    )
Circuits.implement(cs::SIC.SDCs.ConditionalScore, ::Spiking) =
    SDCs.PoissonPulseConditionalScore((cs, K, ONRATE, 500, 0.2, 1000), 10^(-10), 12)

Circuits.implement(lt::SIC.SDCs.LookupTable, ::Spiking) =
    SIC.SDCs.OneHotLookupTable(lt)

Circuits.implement(::Binary, ::Spiking) = SpikeWire()

Circuits.implement(m::SDCs.NonnegativeRealMultiplier, ::Spiking) = 
    SDCs.PulseNonnegativeRealMultiplier(
        map(to_spiking_real, m.inputs),
        (indenoms, outdenom) -> PulseIR.PoissonSpikeCountMultiplier(
            indenoms, outdenom,
            10, 50., 300., 0.5,
            ((500, 12), 0.), #(ti_params, offrate)
            (500, 12)
        ),
        K,
        threshold -> begin
            println("thresh: $threshold")
            PulseIR.PoissonThresholdedIndicator(threshold, 500, 0.5, 50, 40)
        end
    )

to_spiking_real(::SDCs.SingleNonnegativeReal) = 
    SDCs.IndicatedSpikeCountReal(SDCs.UnbiasedSpikeCountReal(K))
to_spiking_real(v::SDCs.ProductNonnegativeReal) =
    SDCs.ProductNonnegativeReal(map(to_spiking_real, v.factors))
to_spiking_real(v::SDCs.NonnegativeReal) = to_spiking_real(implement(v, Spiking()))

Circuits.implement(::SDCs.Theta, ::Spiking) =
    SDCs.PulseTheta(
        10,2, PulseIR.PoissonTheta, 1000, 400., 100.,
        PulseIR.PoissonOffGate(
            PulseIR.ConcreteOffGate(400. #= ΔT =#, 0.2, 500), 20
        ),
        PulseIR.PoissonThresholdedIndicator,
        (400., 0.2, 500, 20)
    )

Circuits.implement(m::SDCs.Mux, ::Spiking) =
    SDCs.PulseMux(m,
        PulseIR.PoissonAsyncOnGate(
            PulseIR.ConcreteAsyncOnGate(500, 0.1, 50),
            30
        )
    )