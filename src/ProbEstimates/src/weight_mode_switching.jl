global weighttype = :noisy
global assemblysize = DefaultAssemblySize()

function AssemblySize()
    return assemblysize
end
function set_assembly_size!(size)
    global assemblysize = size
end

"""
Estimate of `p`
"""
fwd_prob_estimate(p) =
    if weighttype in (:noisy, :fwd)
        fwd_pe(p)
    elseif weighttype == :recip
        1/recip_pe(p)
    elseif weighttype == :constant
        1.
    else
        @assert weighttype == :perfect	
        p
    end
"""
Estimate of `1/p`
"""
recip_prob_estimate(p) =
    if weighttype in (:noisy, :recip)
        recip_pe(p)
    elseif weighttype == :fwd
        1/fwd_pe(p)
    elseif weighttype == :constant
        1.
    else
        @assert weighttype == :perfect
        1/p
    end

fwd_pe(p) = rand(Binomial(K_fwd(), p))/K_fwd()
function recip_pe(p)
    if p < 0.000001
        Inf
    else
        @assert p > zero(p)
        @assert K_recip() > zero(K_recip())
        rand(NegativeBinomial(K_recip(), p))/K_recip() + 1
    end
end

function use_noisy_weights!()
    global weighttype = :noisy
end
function use_only_fwd_weights!()
    global weighttype = :fwd
end
function use_only_recip_weights!()
    global weighttype = :recip
end
function use_perfect_weights!()
    global weighttype = :perfect
end
function use_constant_weights!() # always end up with score = 1
    global weighttype = :constant
end
function reset_weights_to!(typ)
    global weighttype = typ
end
function weight_type()
    return weighttype
end

use_noisy_weights!()