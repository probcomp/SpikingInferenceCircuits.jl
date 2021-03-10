animationTimeout = undefined;

function setup_animation(viz) {
    d3.select("#play_animation").on("click", () => {
        d3.json("renders/conc_samp_animation.json", animation => {
            if (animationTimeout) { // end old animation if one is running
                clearTimeout(animationTimeout);
                end_animation(viz);
            }

            slowdown = d3.select("#slowdown").node().value;
            console.log("slowdown is " + slowdown)
            run_animation(animation, viz, slowdown)
        });
    });
}

const spike_length = 0.15; // seconds

/**
 * Utils.
 */

function size_should_be_one_will_use_first(list, listname) {
    if (list.length !== 1) {
        console.warn("`" + listname + "` did not have one element!: ");
        console.warn(list)
        if (list.length > 1) {
            console.log("---will proceed with first element...")
        }
    }

}

function get_group(graph, group_name) {
    group = graph.groups[0];
    for (name of group_name) {
        next_groups = group.groups.filter(g => g.name == name);
        size_should_be_one_will_use_first(next_groups, "next_groups");
        group = next_groups[0];
    }
    return group
}

function get_outnode(group, outname) {
    outnodes = group.leaves.filter(n => n.name === outname);
    size_should_be_one_will_use_first(outnodes, "outnodes");
    return outnodes[0];
}

/**
 * Setup / begin / end animation.
 */

/*
Animation looks like
{
    "initial_states": [s1, s2, ...],
    "events": [e1, e2, ...]
}
For details on these, see `../animation_interface.jl`.
*/
function run_animation(animation, viz, slowdown) {
    console.log(animation);
    begin_animation(animation, viz);
    if (animation.events.length !== 0) {
        first_action_time = animation.events[0].time;
        sleep_or_do(
            first_action_time,
            slowdown,
            () => handle_frame(viz, first_action_time, animation.events, [], slowdown)
        )
    }
}

function begin_animation(animation, viz) {
    setup_initial_states(animation.initial_states, viz.graph, viz.groups)
}

// called in the last `setup_next_action` call
function end_animation(viz) {
    unset_on_off_states(viz.groups);
}

function setup_initial_states(initial_states, graph, groups) {
    for (state of initial_states) {
        groups.filter(g => g == get_group(graph, state.group))
            .classed(state.val ? "onState" : "offState", true)
            .classed(state.val ? "offState" : "OnState", false);
    }
}

function unset_on_off_states(groups) {
    groups.filter(is_poisson)
        .classed("onState", false)
        .classed("offState", false);
}

/**
 * Primary animation running functions.
 */

function handle_frame(viz, current_time, remaining_frames, currently_spiking, slowdown) {
    next_frame = remaining_frames.splice(0, 1)[0]
    if (Math.abs(next_frame.time - current_time) > 0.0001) {
        console.warn("Reached a frame at the wrong time!  next_frame.time = " + next_frame.time + " but current_time = " + current_time)
    }

    if (next_frame.type === "spike") {
        currently_spiking = handle_spike(viz, next_frame, current_time, currently_spiking)
    } else if (next_frame.type === "statechange") {
        handle_state_change(viz, next_frame)
    }

    setup_next_action(viz, current_time, remaining_frames, currently_spiking, slowdown);
}

/**
 * Show the next node to be spiking!
 */
function handle_spike(viz, next_frame, current_time, currently_spiking) {
    // TODO: give SVG elements IDs so we can more efficiently access them!

    group = get_group(viz.graph, next_frame.group);
    outnode = get_outnode(group, next_frame.out);

    on_groups = viz.groups.filter(g => g === group)
        .attr("class", "group spikingGroup");
    on_nodes = viz.nodes.filter(n => n === outnode)
        .attr("class", "node spikingNode");

    on_links = links.filter(d => d.source === outnode)
        .attr("class", "link spikingLink");

    // TODO: change style of nodes receiving spikes

    currently_spiking.splice(0, 0, {
        time: current_time,
        links: on_links,
        groups: on_groups,
        nodes: on_nodes
    });

    return currently_spiking;
}

function handle_state_change(viz, frame) {
    if (frame.statetype === "OnOffState") {
        viz.groups.filter(g => g == get_group(viz.graph, frame.group))
            .classed(frame.val ? "onState" : "offState", true)
            .classed(frame.val ? "offState" : "onState", false);
    } else {
        console.warn("Unrecognized state type: " + frame.statetype);
        console.warn(frame)
    }
}

/**
 * Turn off the node we set to spiking longest ago.
 */
function remove_spiking(viz, current_time, remaining_frames, currently_spiking, slowdown) {
    obj = currently_spiking.splice(0, 1)[0];
    if (obj.time + spike_length !== current_time) {
        console.warn("`removed_spiking` called to remove a frame at the wrong time!  current_time = " + current_time + "; obj.time = " + obj.time + "; spike_length = " + spike_length)
    }

    obj.groups.attr("class", "poissonGroup");
    obj.nodes.attr("class", "node");
    obj.links.attr("class", "link");

    setup_next_action(viz, current_time, remaining_frames, currently_spiking, slowdown);
}

/**
 * Queue up the next action in the animation to be performed after the appropriate delay.
 */
function setup_next_action(viz, current_time, remaining_frames, currently_spiking, slowdown) {
    time_to_frame = remaining_frames.length === 0 ? Infinity : (remaining_frames[0].time - current_time)
    time_to_removal = currently_spiking.length === 0 ? Infinity : (currently_spiking[0].time + spike_length) - current_time

    if (time_to_frame === Infinity && time_to_removal === Infinity) {
        end_animation(viz); // animation is completed
    } else if (time_to_frame <= time_to_removal) {
        sleep_or_do(time_to_frame, slowdown, () => handle_frame(viz, current_time + time_to_frame, remaining_frames, currently_spiking, slowdown))
    } else {
        sleep_or_do(time_to_removal, slowdown, () => remove_spiking(viz, current_time + time_to_removal, remaining_frames, currently_spiking, slowdown))
    }
}

function sleep_or_do(time, slowdown, fn) {
    console.log("slowdown is " + slowdown)
    if (time === 0) {
        fn()
    } else {
        animationTimeout = setTimeout(fn, time * 1000 * slowdown)
    }
}