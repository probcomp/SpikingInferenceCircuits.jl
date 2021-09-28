global weighttype = :noisy
global assemblysize = DefaultAssemblySize()
global latency      = DefaultLatency()
global use_autonormalization = DefaultUseAutonormalization()

function AssemblySize()
    return assemblysize
end
function set_assembly_size!(size)
    global assemblysize = size
end

function Latency()
    return latency
end
function set_latency!(l)
    global latency = l
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
    elseif p == 1.
        1.
    else
        if DoRecipPECheck()
            @assert p ≥ MinProb() * (1 - 1e-6)
            @assert K_recip() > zero(K_recip())
        end
        try
            rand(NegativeBinomial(K_recip(), p))/K_recip() + 1
        catch e
            @error "$(K_recip()), $p" exception=(e, catch_backtrace())
        end
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
function set_weighttype_to!(typ)
    global weighttype = typ
end
reset_weights_to!(typ) = set_weighttype_to!(typ)
function weight_type()
    return weighttype
end

with_weight_type(typ::Symbol, f) = with_weight_type(f, typ)
function with_weight_type(f, typ::Symbol)
    current_type = weight_type()
    set_weighttype_to!(typ)
    v = f()
    set_weighttype_to!(current_type)
    return v
end

function set_autonormalization!(use_autonorm)
    global use_autonormalization = use_autonorm
end
function is_using_autonormalization()
    return global use_autonormalization
end