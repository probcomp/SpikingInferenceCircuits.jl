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