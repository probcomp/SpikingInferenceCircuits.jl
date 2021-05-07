using Gen: categorical

# reciprocal
function recip_get_output_count(pvec, sample, K)
    total = 0
    sample_count = 0
    while sample_count < K
        val = categorical(pvec)
        if val == sample
            sample_count += 1
        end
        total += 1
    end
    return total
end

# standard
function std_get_output_count(pvec, sample, K)
    total = 0
    sample_count = 0
    while total < K
        val = categorical(pvec)
        if val == sample
            sample_count += 1
        end
        total += 1
    end
    return sample_count
end

cnts = [std_get_output_count([.9, .1], 1, 5) for _=1:100000]
avg = sum(cnts)/100000