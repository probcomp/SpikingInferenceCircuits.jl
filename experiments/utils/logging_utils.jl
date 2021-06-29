using Printf: @sprintf
function get_log_filter(log_interval)
    cnt = 0
    function log_filter(time, compname, event)
        cnt += 1
        return (cnt - 1) % log_interval == 0
    end
    return log_filter
end
function time_log_str(time, compname, event)
    @sprintf("%.4f", time)
end
