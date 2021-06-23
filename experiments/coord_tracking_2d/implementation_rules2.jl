Circuits.implement(cs::SIC.SDCs.ConditionalScore, ::Spiking) =      # max_delay
    SDCs.PoissonPulseConditionalScore((cs, K(), SCORE_ONRATE(), ΔT(),    2,     M()), OFFRATE(), LOWER_R())

Circuits.implement(cs::SIC.SDCs.ConditionalSample, ::Spiking) =
    SDCs.PoissonPulseConditionalSample(
        (cs, K(), SAMPLE_ONRATE(),
            ΔT_SAMPLE(), # ΔT      --  Note that it can take 2x as long as this to reset!
            2, # max_delay
            M(), # M (num spikes to override offs/ons)
            500, # max delay before sample is emitted
            1 # intersample hold
        ),
        OFFTATE(), LOWER_R()
    )

Circuits.implement(theta::SDCs.Theta, ::Spiking) =
    SDCs.PulseTheta(                                    # M,     L,  ΔT,         rate
        K(), theta.n_possibilities, PulseIR.PoissonTheta, M(), -20, ΔT_THETA(), THETA_RATE(),
        PulseIR.PoissonOffGate(
            PulseIR.ConcreteOffGate(ΔT_THETA() #= ΔT =#, 2. #=max_delay=#, M() #=M=#), R_MID() #=R=#
        ),
        PulseIR.PoissonThresholdedIndicator,
        (ΔT_THETA(), 2., M(), R_MID()) # ΔT, Maxdelay, M, R
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
            ((M(), MID_R()), 0.), #(ti_params = (M, R), offrate)
            (M(), MID_R()) # (M, R) offgate params
        ), 
        K(),
        threshold -> begin
            # println("thresh: $threshold")
            #                                   threshold, ΔT,             max_delay,  M,    R 
            PulseIR.PoissonThresholdedIndicator(threshold, MULT_MAX_INPUT_MEMORY(), 5, M(), R_MID())
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
        (M(), MID_R()), # M R
        (1, MID_R()), # max_delay, R
        (SYNC_ΔT_TIMER(), NSPIKES_SYNC_TIMER(),  # ΔT_timer, N_spikes_timer
            (1, M(), MID_R()), 0., # timer TI params (maxdelay M R) | offrate
            SYNC_TIMER_MEMORY() # timer memory
        )
    )

# I think these next few are pretty straightforward:

Circuits.implement(ta::SIC.SDCs.ToAssmts, ::Spiking) =
    SDCs.PulseToAssmts(
        ta, PulseIR.PoissonThresholdedIndicator,
        (ΔT(), 5, M(), MID_R())
    )

Circuits.implement(m::SDCs.Mux, ::Spiking) =
    SDCs.PulseMux(m,
        PulseIR.PoissonAsyncOnGate(
            PulseIR.ConcreteAsyncOnGate(ΔT_MUX(), 1, M()), # ΔT, max_delay, M
            MID_R() # R
        )
    )

### Not dependent upon hyperparameters:

Circuits.implement(lt::SIC.SDCs.LookupTable, ::Spiking) =
    SIC.SDCs.OneHotLookupTable(lt)

Circuits.implement(::Binary, ::Spiking) = SpikeWire()