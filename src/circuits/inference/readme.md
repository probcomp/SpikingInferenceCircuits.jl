- IS Particle
- Resample
  - 2-particle particle Gibbs   (similar to MH)  (accepts 2 (trace, weight) pairs; outputs a trace)
- (Gibbs?)
- LongRunningInference
  - SMC
  - MCMC

Previous: 
> Compilation from test (a static generative func equivalent to a BN) to a circuit.
> circuit = gen_fn_circuit(test, (input=2,), Propose()

Particle:
p_circ = particle(test, proposal)

function particle(test, proposal)
    # Ignore constructors for nowa
    pro_circ = propose(proposal, (all_addrs = ...))
end

- addresses we propose to
- obs = addresses in the model not proposed to

(obs, arguments to the proposal, arguments to the model)



min(1, pq'/qp')
p/(p + q)     = (p/q)/(1 + p/q)

theta(p/q, 1)

theta(p/q, p'/q')