Circuits.implement(ta::SIC.SDCs.ToAssmts, ::Spiking) =
    SDCs.PulseToAssmts(
        ta, PulseIR.PoissonThresholdedIndicator,
        # ΔT, max_delay, M, R
        (175, 0.5, 1000, 40)
        # Note that the R needs to be high since getting spikes while off is catastrophic.
        # TODO: Design things so this is not catastrophic (or can't happen at
        # realistic rates)!
    )

K() = 20
SAMPLE_ONRATE() = 1.0
SCORE_ONRATE() = 1.0
Circuits.implement(cs::SIC.SDCs.ConditionalSample, ::Spiking) =
    SDCs.PoissonPulseConditionalSample(
        (cs, K(), SAMPLE_ONRATE(),
            175, # ΔT        // Note that it can take 2x as long as this to reset!
            0.2, # max_delay
            1000, # M (num spikes to override offs/ons)
            50, # max delay before sample is emitted
            0.1 # intersample hold
        ),
        10^(-10), 24  # max_delay | lower_R
    )
Circuits.implement(cs::SIC.SDCs.ConditionalScore, ::Spiking) =
                                                            #  ΔT, max_delay, M | offrate, lower_R 
    SDCs.PoissonPulseConditionalScore((cs, K(), SCORE_ONRATE(), 175, 0.2, 1000), 10^(-10), 24)

Circuits.implement(lt::SIC.SDCs.LookupTable, ::Spiking) =
    SIC.SDCs.OneHotLookupTable(lt)

Circuits.implement(::Binary, ::Spiking) = SpikeWire()

Circuits.implement(m::SDCs.NonnegativeRealMultiplier, ::Spiking) = 
    SDCs.PulseNonnegativeRealMultiplier(
        map(to_spiking_real, m.inputs),
        (indenoms, outdenom) -> PulseIR.PoissonSpikeCountMultiplier(
            indenoms, outdenom,
            30, 50., 245., 0.5, # erlang_shape/num_timer_spikes | expected_output_time | max_input_memory | max_delay
            ((500, 30), 0.), #(ti_params = (M, R), offrate)
            (500, 30) # (M, R) offgate params
        ), 
        K(),
        threshold -> begin
            # println("thresh: $threshold")
            PulseIR.PoissonThresholdedIndicator(threshold, 245, 0.5, 1000, 40) # threshold, ΔT, max_delay, M, R 
        end
    )

to_spiking_real(::SDCs.SingleNonnegativeReal) = 
    SDCs.IndicatedSpikeCountReal(SDCs.UnbiasedSpikeCountReal(K()))
to_spiking_real(v::SDCs.ProductNonnegativeReal) =
    SDCs.ProductNonnegativeReal(map(to_spiking_real, v.factors))
to_spiking_real(v::SDCs.NonnegativeReal) = to_spiking_real(implement(v, Spiking()))

# TODO: improve where we set the `rate_rescaling_factor`...
# Could we set things up so the rate is self-normalizing?
Circuits.implement(theta::SDCs.Theta, ::Spiking) =
    SDCs.PulseTheta(                                    # M,     L,  ΔT,    rate
        K(), theta.n_possibilities, PulseIR.PoissonTheta, 1000, -20, 175., 1.,
        PulseIR.PoissonOffGate(
            PulseIR.ConcreteOffGate(175. #= ΔT =#, 0.2 #=max_delay=#, 500 #=M=#), 30 #=R=#
        ),
        PulseIR.PoissonThresholdedIndicator,
        (175., 0.2, 500, 30) # ΔT, Maxdelay, M, R
    )

Circuits.implement(m::SDCs.Mux, ::Spiking) =
    SDCs.PulseMux(m,
        PulseIR.PoissonAsyncOnGate(
            PulseIR.ConcreteAsyncOnGate(245., 0.1, 1000), # ΔT, max_delay, M
            30 # R
        )
    )

Circuits.implement(s::PulseIR.Sync, ::Spiking) =
    PulseIR.PoissonSync(
        s.cluster_sizes,
        (1000, 30.), # M R
        (0.1, 30.), # max_delay, R
        (25., 20, (0.1, 1000, 30.), 0., 50.) # ΔT_timer, N_spikes_timer | timer TI params (maxdelay M R) | offrate | timer memory
    )