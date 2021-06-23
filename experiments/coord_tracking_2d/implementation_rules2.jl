SAMPLE_ONRATE() = 0.2
SCORE_ONRATE() = 0.1
K() = 20 # count denominator for scores

ΔT() = 500 # ms  -- memory time for scoring unit (also used by many other units)
SCORE_FAIL_PROB() = 10e-10
SAMPLE_FAIL_PROB() = 10e-10

OFFRATE() = 10e-20

LOWER_R() = 36
MID_R() = 40

M() = 1000 # number of spikes to override off/on gate

# We need K() spikes to occur from an assembly with rate SCORE_ONRATE() when sampling.
# So we need P[spikes from P.P. with rate SCORE_ONRATE() in ΔT > K()] > SCORER_FAILUREPROB
# I.e. P[Poisson(SCORE_ONRATE() * ΔT) > K()] > SCORER_FAILUREPROB
@assert cdf(Poisson(SCORE_ONRATE() * ΔT), K()) < SCORE_FAIL_PROB() "Probability of not getting enough spikes from scoring unit before memory runs out is too high."

MinProb() = 0.1
ΔT_SAMPLE() = ΔT() * SCORE_ONRATE() / SAMPLE_ONRATE()
# We may need K() spikes from a spiker of rate MinProb() * SAMPLE_ONRATE()
@assert cdf(Poisson(SAMPLE_ONRATE() * MinProb() * ΔT_SAMPLE()), K()) < SAMPLE_FAIL_PROB() "Probability of not getting enough spikes from reciprical scoring unit before memory runs out is too high (at least for the lowest probability outcome)"

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

# Note: the below parameters are not set very carefully; there may be more sensible settings
ΔT_THETA() = 2.5 * ΔT()
THETA_RATE() = SAMPLE_ONRATE()
MULT_EXPECTED_OUT_TIME() = ΔT()/2
MULT_MAX_INPUT_MEMORY() = ΔT_THETA()

MULT_FAIL_PROB = 1e-5
P_FAILURE_MULT_INTO_THETA = 1e-5

# Main point is: theta needs to remember everything that comes in from the multiplier.
# Rate of theta shouldn't matter much, I think, so long as P[get 0 spikes] remains low.
# Main constraint should be whether ΔT_THETA() is long enough to receive all the input
# from the multiplier.
@assert p_mult_output_longer_than(ΔT_THETA()) < P_FAILURE_MULT_INTO_THETA
p_mult_output_longer_than(T) = cdf(Erlang(TIMER_N_SPIKES(), 1/timer_rate_for_time(MULT_EXPECTED_OUT_TIME())), T)
timer_rate_for_time(T) = T/TIMER_N_SPIKES()

Circuits.implement(theta::SDCs.Theta, ::Spiking) =
    SDCs.PulseTheta(                                    # M,     L,  ΔT,         rate
        K(), theta.n_possibilities, PulseIR.PoissonTheta, M(), -20, ΔT_THETA(), THETA_RATE(),
        PulseIR.PoissonOffGate(
            PulseIR.ConcreteOffGate(ΔT_THETA() #= ΔT =#, 2. #=max_delay=#, M() #=M=#), R_MID() #=R=#
        ),
        PulseIR.PoissonThresholdedIndicator,
        (ΔT_THETA(), 2., M(), R_MID()) # ΔT, Maxdelay, M, R
    )


# Want: P[doesn't finish timer before forgetting inputs] ≈ 0
# ie    P[Erlang(...) > MULT_MAX_INPUT_MEMORY()] ≈ 0
@assert cdf(Erlang(TIMER_N_SPIKES(), 1/timer_rate_for_time(MULT_EXPECTED_OUT_TIME())), MULT_MAX_INPUT_MEMORY()) < MULT_FAIL_PROB

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


SYNC_ΔT_TIMER() = ΔT()/4; NSPIKES_SYNC_TIMER() = N_SPIKES_TIMER()
SYNC_TIMER_MEMORY() = 3 * SYNC_ΔT_TIMER()
SYNC_FORGET_FAIL_PROB = 1e-5
@assert cdf(Erlang(NSPIKES_SYNC_TIMER(), NSPIKES_SYNC_TIMER()/SYNC_ΔT_TIMER()), SYNC_TIMER_MEMORY()) > SYNC_FORGET_FAIL_PROB
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