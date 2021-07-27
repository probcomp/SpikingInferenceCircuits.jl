# Users of this library who want different constants
# should change these values via lines like `ProbEstimate.MaxRate() = custom_rate`
DefaultLatency() = 50 # ms
MaxRate() = 0.2 # KHz
DefaultAssemblySize() = 10 # neurons
MinProb() = 0.1

# Hyperparameters for multiplication to a single line:

TimerExpectedT() = 50 # ms
TimerAssemblySize() = 10
MultAssemblySize() = 20

TimerNSpikes() = TimerExpectedT() * TimerAssemblySize() * MaxRate() |> Int
MultOutDenominator() = TimerExpectedT() * MultAssemblySize() * MaxRate()

# Should we check that the probability ranges make it possible to actually compile
# the model, and that the probabilities are actually greater than the MinProb value
DoRecipPECheck() = true

# By default, don't truncate fwd dists, and truncate recip dists if we are not using perfect weights
TruncateFwdDists()   = false
TruncateRecipDists() = weight_type() !== :perfect