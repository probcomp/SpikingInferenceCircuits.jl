@gen (static) function one_hot_add_noise()
    idx1 ~ uniform_discrete(1, 10)
    idx2 ~ uniform_discrete(1, 10)

    one_hot1 = [i == idx1 ? 1 : 0 for _=1:10]
    one_hot2 = [i == idx2 ? 1 : 0 for _=1:10]
    sum = one_hot1 + one_hot2

    with_noise ~ Map(normal)(sum)
    
    return with_noise
end

@load_generated_functions()
lcpts = to_labeled_cpts(one_hot_add_noise, [])