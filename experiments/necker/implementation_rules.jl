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