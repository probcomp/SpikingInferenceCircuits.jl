ImageSideLength() = 10
OccluderLength() = 3
SquareSideLength() = 1
Vels() = -2:2
MinProb() = ProbEstimates.MinProb()
p_flip() = 0.1

OccOneOffProb() = 0.3
VelOneOffProb() = 0.2

Positions() = 1:ImageSideLength()
SqPos()     = 1:(ImageSideLength() - SquareSideLength() + 1)
OccPos()    = 1:(ImageSideLength() - OccluderLength() + 1)