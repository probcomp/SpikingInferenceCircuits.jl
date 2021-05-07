const MAX_IDX = 10

# these specific probabilities are not carefully chosen;
# this is just a random function which will put more weight close
# to the given `true_idx`
relative_prob(i, true_idx) = 1/(1 + abs(i - true_idx))^3
noisy_val_probs(true_idx) = [relative_prob(i, true_idx) for i=1:10] |> normalize

@gen (static) function add_noise(val)
    noisy ~ categorical(noisy_val_probs(val))
    return noisy
end

@gen (static) function one_hot_add_noise()
    idx1 ~ uniform_discrete(1, MAX_IDX)
    idx2 ~ uniform_discrete(1, MAX_IDX)

    one_hot1 = [i == idx1 ? 1 : 0 for i=1:MAX_IDX]
    one_hot2 = [i == idx2 ? 1 : 0 for i=1:MAX_IDX]
    sum = one_hot1 + one_hot2

    with_noise ~ Map(add_noise)(sum)
    
    return with_noise
end

lcpts = to_labeled_cpts(one_hot_add_noise, [])
(cpts, bijs) = to_indexed_cpts(one_hot_add_noise, [])
@load_generated_functions()
from_l = lcpts()
from_i = cpts()