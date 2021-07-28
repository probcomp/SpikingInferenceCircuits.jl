using Distributions: Erlang, Poisson, cdf
ExpectedLatency()    = 5
SampleAssemblySize() = 100
ScoreAssemblySize()  = 100
MaxNeuronRate()      = 0.2 # KHz
MinProb()            = 0.1

INTER_OBS_INTERVAL() = 200. # ms

SAMPLE_ONRATE() = SampleAssemblySize() * MaxNeuronRate()
SCORE_ONRATE()  = ScoreAssemblySize()  * MaxNeuronRate()
# count denominator for ProbEstimates
PEstDenom() = ExpectedLatency() * SCORE_ONRATE() |> Int
# count denominator for ReciprocalProbEstimates
RecipPEstDenom() = ExpectedLatency() *  SAMPLE_ONRATE() * MinProb() |> Int
MultOutDenom() = 200 # count denominator for the output of a multiplier

ΔT() = 24 # ms  -- memory time for scoring unit (also used by many other units)

MinProb() = 0.1
ΔT_SAMPLE() = ΔT() * 2

OFFRATE() = 10e-20

# rates for neurons in logic gates like TI, OFFGATE, ASYNC_ON_GATE
GATE_OFFRATE() = 0.
GATE_ONRATE() = 500_000. # KHz
LOWER_GATE_ONRATE() = 500_000. # KHz -- not realistic at all.  making this realistic: TODO
GATE_RATES() = (GATE_OFFRATE(), GATE_ONRATE())

M() = 1000 # number of spikes to override off/on gate

# Note: the below parameters are not set very carefully; there may be more sensible settings
ΔT_THETA() = 4. * ΔT()
THETA_RATE() = SAMPLE_ONRATE()
MULT_EXPECTED_OUT_TIME() = ΔT()/2
MULT_OUT_TIME_BOUND() = 2 * MULT_EXPECTED_OUT_TIME()
MULT_MAX_INPUT_MEMORY() = ΔT_THETA()

# TODO: understand how the timer & mult n spikes, rate, denominator, etc., relate to assembly size
TIMER_N_SPIKES() = 30

SYNC_ΔT_TIMER() = ΔT()/4; NSPIKES_SYNC_TIMER() = TIMER_N_SPIKES()
SYNC_TIMER_MEMORY() = 3 * SYNC_ΔT_TIMER()

ΔT_GATES() = ΔT_THETA()

### Check probabilities of certain types of failure

SCORE_FAIL_PROB() = 1e-8
SAMPLE_FAIL_PROB() = 1e-8
MULT_FAIL_PROB() = 5e-5
P_FAILURE_MULT_INTO_THETA() = 5e-5
SYNC_FORGET_FAIL_PROB() = 5e-5
P_SAMPLE_WTA_TOO_SLOW() = 5e-5
P_THETA_WTA_TOO_SLOW() = 5e-5

prob_get_est_of_0(n_vars) = 1 - (1 - (1 - MinProb())^PEstDenom())^n_vars
bound_on_overall_failure_prob_due_to_mistake(n_steps, n_vars, n_particles) = 1 - (
    ( # prob failure
        (
            ((1 - SCORE_FAIL_PROB()) * (1 - SAMPLE_FAIL_PROB()) * (1 - P_SAMPLE_WTA_TOO_SLOW()))^n_vars # prob failure due to sampling/scoring
        * (1 - MULT_FAIL_PROB()) * (1 - P_FAILURE_MULT_INTO_THETA())                             # prob we have a failure on the multiplier or theta
        * (1 - P_THETA_WTA_TOO_SLOW())
        )^n_particles                     
        * (1 - SYNC_FORGET_FAIL_PROB())                                                          # prob failure in sync
    )^n_steps
)
bound_on_overall_failure_prob(n_steps, n_vars, n_particles) = 1 - (
    # P we don't have a component failure, on any step
    (1 - bound_on_overall_failure_prob_due_to_mistake(n_steps, n_vars, n_particles)) * 
    ( # P we don't have all failures, at every step
            ( # P we don't have all failures, on a given step
                1 -
                prob_get_est_of_0(n_vars)^n_particles # P we do get all failures, on a given step
            )
    )^n_steps
)

ΔT_SAMPLE_RESET_TIME() = 2 * ΔT_SAMPLE() 
# TODO: is it possible to have a tighter bound on this to check with?
ΔT_SAMPLE_SCORE_RESAMPLE() = max(ΔT_SAMPLE(), ΔT()) + MULT_OUT_TIME_BOUND() + ΔT_THETA()
function run_hyperparameter_checks()
    # check that none of the memory times are too log_interval
    for time_constant in (:ΔT, :ΔT_SAMPLE_SCORE_RESAMPLE, :ΔT_SAMPLE_RESET_TIME, :ΔT_THETA, :SYNC_TIMER_MEMORY, :ΔT_GATES)
        val = eval(:($time_constant()))
        @assert val < INTER_OBS_INTERVAL() "$time_constant() must be smaller than INTER_OBS_INTERVAL() so all circuitry resets before the next timestep"
    end

    # Low probability of getting an eroneous spike from a logic gate while it's supposed to be off:
    @assert GATE_OFFRATE() == 0 || error("TODO: implement a check for logic gate being unlikely to spike while off!")

    # Low probability of multiple spikes into a WTA before it can output a spike
    # P[exponential(LOWER_GATE_ONRATE()) > exponential(SAMPLE_ONRATE())]
    # = 1 - LOWER_GATE_ONRATE()/(SAMPLE_ONRATE() + LOWER_GATE_ONRATE())
    @assert 1 - (LOWER_GATE_ONRATE()/(SAMPLE_ONRATE() + LOWER_GATE_ONRATE())) < P_SAMPLE_WTA_TOO_SLOW()  "Probability of Sampling WTA receving multiple spikes before output is too high!"

    @assert 1 - (GATE_ONRATE()/(THETA_RATE() + GATE_ONRATE())) < P_SAMPLE_WTA_TOO_SLOW()  "Probability of Theta Gate WTA receving multiple spikes before output is too high!"

    # We need PEstDenom() spikes to occur from an assembly with rate SCORE_ONRATE() when sampling.
    # So we need P[spikes from P.P. with rate SCORE_ONRATE() in ΔT > PEstDenom()] > SCORER_FAILUREPROB
    # I.e. P[Poisson(SCORE_ONRATE() * ΔT) > PEstDenom()] > SCORER_FAILUREPROB
    @assert cdf(Poisson(SCORE_ONRATE() * ΔT()), PEstDenom()) < SCORE_FAIL_PROB() "Probability of not getting enough spikes from scoring unit before memory runs out is too high."

    # We may need RecipPEstDenom() spikes from a spiker of rate SAMPLE_ONRATE()*MinProb()
    @assert cdf(Poisson(SAMPLE_ONRATE()*MinProb()* ΔT_SAMPLE()), RecipPEstDenom()) < SAMPLE_FAIL_PROB() "Probability of not getting enough spikes from reciprical scoring unit before memory runs out is too high (at least from lowest prob)"

    p_mult_output_longer_than(T) = cdf(Erlang(TIMER_N_SPIKES(), timer_rate_for_time(MULT_EXPECTED_OUT_TIME())), T)
    timer_rate_for_time(T) = TIMER_N_SPIKES()/T
    # Want: P[doesn't finish timer before forgetting inputs] ≈ 0
    # ie    P[Erlang(...) > MULT_MAX_INPUT_MEMORY()] ≈ 0
    @assert MULT_OUT_TIME_BOUND() ≤ MULT_MAX_INPUT_MEMORY()
    @assert cdf(Erlang(TIMER_N_SPIKES(), 1/timer_rate_for_time(MULT_EXPECTED_OUT_TIME())), MULT_OUT_TIME_BOUND()) > 1 - MULT_FAIL_PROB()

    # similar check to above, but for reset timer in sync units
    @assert cdf(Erlang(NSPIKES_SYNC_TIMER(), 1/(NSPIKES_SYNC_TIMER()/SYNC_ΔT_TIMER())), SYNC_TIMER_MEMORY()) > 1 - SYNC_FORGET_FAIL_PROB()

    # Time to sample & score + multiply values < memory of Theta and Muxes in resample unit
    @assert max(ΔT(), ΔT_SAMPLE()) + MULT_OUT_TIME_BOUND() < min(ΔT_THETA(), ΔT_GATES())
end
run_hyperparameter_checks()