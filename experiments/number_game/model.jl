include("../utils/modeling_utils.jl")
const Bernoulli = LCat([true, false])
bern_probs(p) = [p, 1 - p]

divides(x, divisor) = x % divisor == 0
const primes_before_10 = (2, 3, 5, 7)
isprime(x) = x in primes_before_10 || !any(divides(x, p) for p in primes_before_10) # works for any x ≤ 100

### Latent tree sampling model ###
@gen function sample_tree(maxdepth)
    terminal ~ sample_terminal()
    if maxdepth > 1
        is_terminal ~ Bernoulli(bern_probs(1/4))
        nt ~ sample_nt(maxdepth)
        return is_terminal ? terminal : nt
    else
        return terminal
    end
end
@gen (static) function sample_nt(maxdepth)
    typ ~ LCat([:and, :or])([0.3, 0.7])
    left ~ sample_tree(maxdepth - 1)
    right ~ sample_tree(maxdepth - 1)
    return (:nonterminal, typ, left, right)
end
@gen (static) function sample_terminal()
    typ ~ LCat([:prime, :multiple_of, :interval])([1/6, 1/2, 1/3])
    n1 ~ Cat(n1_pvec(typ))
    n2 ~ Cat(n2_pvec(typ, n1))
    return (:terminal, typ, n1, n2)
end

n1_pvec(typ) =
    if typ == :prime
        # doesn't matter in this case, so have deterministic value to decrease weight estimate variance
        onehot(1, 1:100)
    elseif typ == :multiple_of
        # only have multiples of small numbers
        [2 ≤ i ≤ 10 ? 1/9 : 0. for i=1:100]
    elseif typ == :interval
        [i ≤ 90 && i % 5 == 0 ? 5/90 : 0. for i=1:100]
    end
max_interval_size() = 35
n2_pvec(typ, n1) =
    if typ == :prime || typ == :multiple_of
        # in either case, n2 is irrelevant
        onehot(1, 1:100)
    elseif typ == :interval
        # sample n2 from among those numbers greater than n1
        normalize([n1 < i ≤ n1 + max_interval_size() && (i - n1) % 4 == 0 ? 1. : 0. for i=1:100])
    end

### Obs model ###

# Sample number uniformly from the set
membership_bool_vec(tree) = [is_in_set(tree, x) for x=1:100]
membership_float_vec(tree) = map(in_set -> in_set ? 1. : 0., membership_bool_vec(tree))
@gen function sample_number_direct(tree)
    unnormalized_probs = membership_float_vec(tree)

    up2 = [unnormalized_probs..., (sum(unnormalized_probs) == 0. ? 1. : 0.)]
    number ~ Cat(normalize(up2))

    # number ~ Cat(normalize(unnormalized_probs))
    return number
end
function is_in_set(tree, x)
    (isterminal, typ, arg1, arg2) = tree
    if isterminal == :terminal
        return in_terminal_set((typ, arg1, arg2), x)
    else
        in_left = is_in_set(arg1, x)
        in_right = is_in_set(arg2, x)
        return (typ == :and) ? in_left && in_right : in_left || in_right
    end
end
function in_terminal_set((typ, n1, n2), x)
    membership_from_prime = typ == :prime && isprime(x)
    membership_from_mult = typ == :multiple_of && (x % n1) == 0
    membership_from_int = typ == :interval && n1 ≤ x ≤ n2
    return membership_from_prime || membership_from_mult || membership_from_int
end

### Trace through OR nodes before sampling from set
# TODO
# @gen function sample_number_via_trace(tree)
#     (isterminal, typ, arg1, arg2) = tree
#     if !is_terminal
#         from_left_tree ~ Bernoulli(bern_probs(0.5))
#     end
    
#     is_or_node = (isterminal == :nonterminal) && typ == :or
    
#     if is_or_node
#         sample ~ 

#     return result
# end

# @gen function sample_from_or(tree)
#     (_, _, arg1, arg2) = tree
#     go_left ~ Bernoulli(bern_probs(0.5))
#     number ~ sample_number_via_trace(go_left ? arg1 : arg2)
#     return number
# end


### Dynamic model
MAXDEPTH() = 2
@gen (static) function initial_model()
    tree ~ sample_tree(MAXDEPTH())
    return (tree,)
end
@gen (static) function step_model(tree)
    return (tree,)
end
@gen (static) function obs_model(tree)
    number ~ sample_number_direct(tree)
    return (number,)
end

model = @DynamicModel(initial_model, step_model, obs_model, 1)

@load_generated_functions()