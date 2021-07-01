INTER_OBS_INTERVAL() = 1000. # ms

SAMPLE_ONRATE() = 1.25
SCORE_ONRATE() = 0.25
PEstDenom() = 20 # count denominator for ProbEstimates
RecipPEstDenom() = 10 # count denominator for ReciprocalProbEstimates
MultOutDenom() = 200 # count denominator for the output of a multiplier

ΔT() = 240 # ms  -- memory time for scoring unit (also used by many other units)

MinProb() = 0.1
ΔT_SAMPLE() = ΔT() * SCORE_ONRATE() / SAMPLE_ONRATE() / MinProb()

OFFRATE() = 10e-20

# rates for neurons in logic gates like TI, OFFGATE, ASYNC_ON_GATE
GATE_OFFRATE() = 0.
GATE_ONRATE() = 200. # KHz
LOWER_GATE_ONRATE() = 50_000. # KHz -- not realistic at all.  making this realistic: TODO
GATE_RATES() = (GATE_OFFRATE(), GATE_ONRATE())

M() = 1000 # number of spikes to override off/on gate

# Note: the below parameters are not set very carefully; there may be more sensible settings
ΔT_THETA() = 2.5 * ΔT()
THETA_RATE() = SAMPLE_ONRATE()
MULT_EXPECTED_OUT_TIME() = ΔT()/2
MULT_MAX_INPUT_MEMORY() = ΔT_THETA()
TIMER_N_SPIKES() = 30

SYNC_ΔT_TIMER() = ΔT()/4; NSPIKES_SYNC_TIMER() = TIMER_N_SPIKES()
SYNC_TIMER_MEMORY() = 3 * SYNC_ΔT_TIMER()

ΔT_MUX() = ΔT()

### Check probabilities of certain types of failure

SCORE_FAIL_PROB() = 1e-8
SAMPLE_FAIL_PROB() = 1e-8
MULT_FAIL_PROB() = 5e-5
P_FAILURE_MULT_INTO_THETA() = 5e-5
SYNC_FORGET_FAIL_PROB() = 5e-5
P_WTA_TOO_SLOW() = 5e-5

bound_on_overall_failure_prob(n_steps, n_vars, n_particles) = 1 - (
  (    ((1 - SCORE_FAIL_PROB()) * (1 - SAMPLE_FAIL_PROB()) * (1 - P_WTA_TOO_SLOW()))^n_vars # prob failure due to sampling/scoring
    * (1 - MULT_FAIL_PROB()) * (1 - P_FAILURE_MULT_INTO_THETA())    )^n_particles           # prob we have a failure on the multiplier or theta
    * (1 - SYNC_FORGET_FAIL_PROB())                                                         # prob failure in sync
)^n_steps

function run_hyperparameter_checks()
    # check that none of the memory times are too log_interval
    ΔT_SAMPLE_RESET_TIME() = 2 * ΔT_SAMPLE() 
    for time_constant in (:ΔT, :ΔT_SAMPLE_RESET_TIME, :ΔT_THETA, :SYNC_TIMER_MEMORY, :ΔT_MUX)
        val = eval(:($time_constant()))
        @assert val < INTER_OBS_INTERVAL() "$time_constant() must be smaller than INTER_OBS_INTERVAL() so all circuitry resets before the next timestep"
    end

    # Low probability of getting an eroneous spike from a logic gate while it's supposed to be off:
    @assert GATE_OFFRATE() == 0 || error("TODO: implement a check for logic gate being unlikely to spike while off!")

    # Low probability of multiple spikes into a WTA before it can output a spike
    # P[exponential(LOWER_GATE_ONRATE()) > exponential(SAMPLE_ONRATE())]
    # = 1 - LOWER_GATE_ONRATE()/(SAMPLE_ONRATE() + LOWER_GATE_ONRATE())
    @assert 1 - (LOWER_GATE_ONRATE()/(SAMPLE_ONRATE() + LOWER_GATE_ONRATE())) < P_WTA_TOO_SLOW()  "Probability of WTA receving multiple spikes before output is too high!"

    # We need PEstDenom() spikes to occur from an assembly with rate SCORE_ONRATE() when sampling.
    # So we need P[spikes from P.P. with rate SCORE_ONRATE() in ΔT > PEstDenom()] > SCORER_FAILUREPROB
    # I.e. P[Poisson(SCORE_ONRATE() * ΔT) > PEstDenom()] > SCORER_FAILUREPROB
    @assert cdf(Poisson(SCORE_ONRATE() * ΔT()), PEstDenom()) < SCORE_FAIL_PROB() "Probability of not getting enough spikes from scoring unit before memory runs out is too high."

    # We may need RecipPEstDenom() spikes from a spiker of rate SAMPLE_ONRATE()*MinProb()
    @assert cdf(Poisson(SAMPLE_ONRATE()*MinProb()* ΔT_SAMPLE()), RecipPEstDenom()) < SAMPLE_FAIL_PROB() "Probability of not getting enough spikes from reciprical scoring unit before memory runs out is too high (at least from lowest prob)"

    p_mult_output_longer_than(T) = cdf(Erlang(TIMER_N_SPIKES(), 1/timer_rate_for_time(MULT_EXPECTED_OUT_TIME())), T)
    timer_rate_for_time(T) = T/TIMER_N_SPIKES()
    # Want: P[doesn't finish timer before forgetting inputs] ≈ 0
    # ie    P[Erlang(...) > MULT_MAX_INPUT_MEMORY()] ≈ 0
    @assert cdf(Erlang(TIMER_N_SPIKES(), 1/timer_rate_for_time(MULT_EXPECTED_OUT_TIME())), MULT_MAX_INPUT_MEMORY()) > 1 - MULT_FAIL_PROB()

    # similar check to above, but for reset timer in sync units
    @assert cdf(Erlang(NSPIKES_SYNC_TIMER(), NSPIKES_SYNC_TIMER()/SYNC_ΔT_TIMER()), SYNC_TIMER_MEMORY()) > 1 - SYNC_FORGET_FAIL_PROB()
end
run_hyperparameter_checks()