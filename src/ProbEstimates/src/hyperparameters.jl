# Users of this library who want different constants
# should change these values via lines like `ProbEstimate.MaxRate() = custom_rate`
DefaultLatency() = 50 # ms
MaxRate() = 0.2 # KHz
DefaultAssemblySize() = 10 # neurons
DefaultMinProb() = 0.1
DefaultUseAutonormalization() = false

# Do we use ``single-line compression'' multiplication, which results in low precision?
# If not, we use ``neural-floating-point''-style multiplication.
UseLowPrecisionMultiply() = !is_using_autonormalization()

# assembly sizes used for multiplication in either auto-normalized or low-precision multiply implementation
MultAssemblySize() = AssemblySize()

# Hyperparameters for multiplication to a single line during single-line compression:
TimerExpectedT() = 50 # ms
TimerAssemblySize() = 10

# The scale of the importance weights being multiplied, which controls (in part)
# how many total spikes we will get.  Change this to tune this.
MultOutScale() = 1.

TimerNSpikes() = TimerExpectedT() * TimerAssemblySize() * MaxRate() |> Int
MaxMultRate() = MultAssemblySize() * MaxRate()
MultOutDenominator() = TimerExpectedT() * MultAssemblySize() * MaxRate() / min(10 * MultOutScale(), 1.)

# Hyperparameters for Neural-Floating-Point style multiplication
AutonormalizeCountThreshold() = 2
AutonormalizeSpeedupFactor() = 2
AutonormalizeRepeaterRate() = 5 * MaxRate()
WeightAutonormalizationParams() = (AutonormalizeCountThreshold(), AutonormalizeSpeedupFactor(), AutonormalizeRepeaterRate())

LatencyForContinuousToDiscreteScore() = Latency() / 2
ContinuousToDiscreteScoreNumSpikes() = LatencyForContinuousToDiscreteScore() * MaxRate() * AssemblySize()

# Error check hyperparams:
AutonormalizationLatency() = Latency()
AutonormalizationMinResultingRate() = 1/Latency()

# Should we check that the probability ranges make it possible to actually compile
# the model, and that the probabilities are actually greater than the MinProb value
DoRecipPECheck() = true

# By default, don't truncate fwd dists, and truncate recip dists if we are not using perfect weights
TruncateFwdDists()   = false
TruncateRecipDists() = weight_type() !== :perfect


