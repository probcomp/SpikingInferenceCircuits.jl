include("impl_hyperparameters.jl")

Circuits.implement(cs::SIC.SDCs.ConditionalScore, ::Spiking) =      # max_delay
    SDCs.PoissonPulseConditionalScore((cs, K(), SCORE_ONRATE(), ΔT(),    2,     M()), OFFRATE(), (GATE_OFFRATE(), LOWER_GATE_ONRATE()))

Circuits.implement(cs::SIC.SDCs.ConditionalSample, ::Spiking) =
    SDCs.PoissonPulseConditionalSample(
        (cs, K(), SAMPLE_ONRATE(),
            ΔT_SAMPLE(), # ΔT      --  Note that it can take 2x as long as this to reset!
            2, # max_delay
            M(), # M (num spikes to override offs/ons)
            500, # max delay before sample is emitted
            1 # intersample hold
        ),
        OFFRATE(), (GATE_OFFRATE(), LOWER_GATE_ONRATE())
    )

Circuits.implement(theta::SDCs.Theta, ::Spiking) =
    SDCs.PulseTheta(                                    # M,     L,  ΔT,         rate
        K(), theta.n_possibilities, PulseIR.PoissonTheta, M(), -20, ΔT_THETA(), THETA_RATE(),
        PulseIR.PoissonOffGate(
            PulseIR.ConcreteOffGate(ΔT_THETA() #= ΔT =#, 2. #=max_delay=#, M() #=M=#), GATE_RATES()...
        ),
        PulseIR.PoissonThresholdedIndicator,
        (ΔT_THETA(), 2., M(), GATE_RATES()...) # 2. is maxdelay
    )

Circuits.implement(m::SDCs.NonnegativeRealMultiplier, ::Spiking) = 
    SDCs.PulseNonnegativeRealMultiplier(
        map(to_spiking_real, m.inputs),
        (indenoms, outdenom) -> PulseIR.PoissonSpikeCountMultiplier(
            indenoms, outdenom,
            TIMER_N_SPIKES(),
            MULT_EXPECTED_OUT_TIME(), # expected_output_time
            MULT_MAX_INPUT_MEMORY(), # | max_input_memory
            5, # max_delay
            ((M(), GATE_RATES()...), 0.), #(ti_params = (M, R), offrate)
            (M(), GATE_RATES()...) # (M, R) offgate params
        ), 
        K(),
        threshold -> begin
            # println("thresh: $threshold")
            #                                   threshold, ΔT,             max_delay,  M,    
            PulseIR.PoissonThresholdedIndicator(threshold, MULT_MAX_INPUT_MEMORY(), 5, M(), GATE_RATES()...)
        end
    )

to_spiking_real(::SDCs.SingleNonnegativeReal) = 
    SDCs.IndicatedSpikeCountReal(SDCs.UnbiasedSpikeCountReal(K()))
to_spiking_real(v::SDCs.ProductNonnegativeReal) =
    SDCs.ProductNonnegativeReal(map(to_spiking_real, v.factors))
to_spiking_real(v::SDCs.NonnegativeReal) = to_spiking_real(implement(v, Spiking()))

Circuits.implement(s::PulseIR.Sync, ::Spiking) =
    PulseIR.PoissonSync(
        s.cluster_sizes,
        (M(), GATE_RATES()...),
        (1, GATE_RATES()...), # 1 = max_delay
        (SYNC_ΔT_TIMER(), NSPIKES_SYNC_TIMER(),  # ΔT_timer, N_spikes_timer
            (1, M(), GATE_RATES()...), 0., # timer TI params (maxdelay M gaterates...) | offrate
            SYNC_TIMER_MEMORY() # timer memory
        )
    )

# I think these next few are pretty straightforward:

Circuits.implement(ta::SIC.SDCs.ToAssmts, ::Spiking) =
    SDCs.PulseToAssmts(
        ta, PulseIR.PoissonThresholdedIndicator,
        (ΔT(), 5, M(), GATE_RATES()...)
    )

Circuits.implement(m::SDCs.Mux, ::Spiking) =
    SDCs.PulseMux(m,
        PulseIR.PoissonAsyncOnGate(
            PulseIR.ConcreteAsyncOnGate(ΔT_MUX(), 1, M()), # ΔT, max_delay, M
            GATE_RATES()...
        )
    )

### Not dependent upon hyperparameters:

Circuits.implement(lt::SIC.SDCs.LookupTable, ::Spiking) =
    SIC.SDCs.OneHotLookupTable(lt)

Circuits.implement(::Binary, ::Spiking) = SpikeWire()