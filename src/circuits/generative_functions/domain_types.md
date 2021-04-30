Values are either in a:
- `FiniteDomain(n)` (ie. the value is in the set `{1, ... n}`)
- `IndexedProductDomain(subdomains)` (ie. the value is a Vector `[x_1, ..., x_k]` where each `x_i \in subdomains[i]`.)

The compiler implementation as of April 8, 2021 supports the following types of of nodes in the circuit,
which can accept/output values from the following types of domains:

Distribution:
-- Each arg is a FiniteDomainValue
-- Outputs FiniteDomainValue

Generic Function:
- Either:
-- Each arg is FiniteDomainValue; outputs FiniteDomainValue
-- Each arg is FiniteDomainValue; outputs ProductDomainValue [code to implement this in SNN isn't there yet]

Static generative function:
-- Args and output can be of any type


_Not yet implemented:_

Map Generative Function:
-- Each arg is a ProductDomainValue of the same length; outputs ProductDomainValue

[ This just falls out of Map generative function:
Mapped Function:
-- Each arg is a ProductDomainValue of the same length; outputs ProductDomainValue
]

Note that this means we can _only_ widen from finite values to vectors;
we cannot return from vector-values to Finite Domain Values.
(In the future we could add support for generic functions from ProductDomain -->  FiniteDomain).
