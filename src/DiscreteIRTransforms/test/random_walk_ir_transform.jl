up_down_stay_probs(max_idx, current_idx) =
    if current_idx == max_idx
        vcat(zeros(max_idx - 2), [.5, .5])
    elseif current_idx == 1
        vcat([0.5, 0.5], zeros(max_idx - 2))
    else
        let prob(i) = abs(i - current_idx) > 1 ? 0.0 : 1/3
            [prob(i) for i in 1:max_idx]
        end
    end

@gen (static) function get_rand_walk_positions(pos1)
    pos2 ~ categorical(up_down_stay_probs(10, pos1))
    pos3 ~ categorical(up_down_stay_probs(10, pos2))
    pos4 ~ categorical(up_down_stay_probs(10, pos3))
    return pos4
end

probably_up_probs(max_idx, current_idx) =
    if current_idx == max_idx
        vcat(zeros(max_idx - 2), [0.2, 0.8])
    elseif current_idx == 1
        vcat([0.2, 0.8], zeros(max_idx - 2))
    else
        [
            if i == current_idx
                0.3
            elseif i == current_idx + 1
                0.65
            elseif i == current_idx - 1
                0.05
            else
                0.0
            end
            for i=1:max_idx
        ]
    end

@gen (static) function get_up_walk_positions(pos1)
    pos2 ~ categorical(probably_up_probs(10, pos1))
    pos3 ~ categorical(probably_up_probs(10, pos2))
    pos4 ~ categorical(probably_up_probs(10, pos3))
    return pos4
end

@gen (static) function take_a_walk()
    pos1 ~ uniform_discrete(1, 10)
    do_up ~ bernoulli(0.5)
    last_pos ~ Switch(get_up_walk_positions, get_rand_walk_positions)(do_up ? 1 : 2, pos1)
    return last_pos
end
@load_generated_functions()

# with_lcpts = to_labeled_cpts(take_a_walk, [])
(with_cpts, bijections) = to_indexed_cpts(take_a_walk, [])

@load_generated_functions()
[with_cpts() for _=1:20]