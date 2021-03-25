animationTimeout = undefined;

// set anim_filename when `comp_filename` is changed
function set_anim_filename() {
    var comp_filename = d3.select("#component_filename").property("value")
    var anim_filename = d3.select("#animation_filename").property("value")

    var cpieces = comp_filename.split(".")
    if (cpieces[cpieces.length - 1] === "json") {
        cbeginning = cpieces.slice(0, cpieces.length - 1).join(".")
        if (anim_filename.length == 0) {
            new_animname = cbeginning + "_anim.json"
            d3.select("#animation_filename").property("value", new_animname)
        }
    }
}

function setup_animation(viz) {
    d3.select("#play_animation").on("click", () => {
        d3.json("renders/" + d3.select("#animation_filename").property("value"), animation => {
            if (animationTimeout) { // end old animation if one is running
                clearTimeout(animationTimeout);
                end_animation(viz);
            }

            var slowdown = d3.select("#slowdown").node().value;
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
    var group = graph.groups[0];
    for (name of group_name) {
        next_groups = group.groups.filter(g => g.name == name);
        size_should_be_one_will_use_first(next_groups, "next_groups");
        group = next_groups[0];
    }
    return group
}

// TODO: what if we have an input and output with the same name?!
function get_node(group, name, is_output_node) {
    var nodes = group.leaves.filter(n => n.name === name && n.is_output == is_output_node);
    size_should_be_one_will_use_first(nodes, "nodes");
    return nodes[0];
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
        var first_action_time = animation.events[0].time;
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
    // unset_on_off_states(viz.groups);
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
    var next_frame = remaining_frames.splice(0, 1)[0]
    if (Math.abs(next_frame.time - current_time) > 0.0001) {
        console.warn("Reached a frame at the wrong time!  next_frame.time = " + next_frame.time + " but current_time = " + current_time)
    }

    if (next_frame.type === "spike") {
        var currently_spiking = handle_spike(viz, next_frame, current_time, currently_spiking)
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
    var isoutput = next_frame.spiketype === "output";

    var group = get_group(viz.graph, next_frame.group);
    var node = get_node(group, next_frame.name, isoutput);

    if (isoutput) {
        on_groups = viz.groups.filter(g => g === group)
            .classed("spikingGroup", true);
    } else {
        on_groups = undefined;
    }
    var on_nodes = viz.nodes.filter(n => n === node)
        .classed("spikingNode", true);

    var on_links = viz.links.filter(d => d.source === node)
        .classed("spikingLink", true);

    // TODO: change style of nodes receiving spikes

    currently_spiking.push({
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
    var obj = currently_spiking.splice(0, 1)[0];
    if (obj.time + spike_length - current_time > 0.00001) {
        console.warn("`removed_spiking` called to remove a frame at the wrong time!  current_time = " + current_time + "; obj.time = " + obj.time + "; spike_length = " + spike_length)
    }

    if (obj.groups) {
        obj.groups.classed("spikingGroup", false);
    }
    obj.nodes.classed("spikingNode", false);
    obj.links.classed("spikingLink", false);

    setup_next_action(viz, current_time, remaining_frames, currently_spiking, slowdown);
}

/**
 * Queue up the next action in the animation to be performed after the appropriate delay.
 */
function setup_next_action(viz, current_time, remaining_frames, currently_spiking, slowdown) {
    var time_to_frame = remaining_frames.length === 0 ? Infinity : (remaining_frames[0].time - current_time)
    var time_to_removal = currently_spiking.length === 0 ? Infinity : (currently_spiking[0].time + spike_length) - current_time

    if (time_to_frame === Infinity && time_to_removal === Infinity) {
        end_animation(viz); // animation is completed
    } else if (time_to_frame <= time_to_removal) {
        sleep_or_do(time_to_frame, slowdown, () => handle_frame(viz, current_time + time_to_frame, remaining_frames, currently_spiking, slowdown))
    } else {
        sleep_or_do(time_to_removal, slowdown, () => remove_spiking(viz, current_time + time_to_removal, remaining_frames, currently_spiking, slowdown))
    }
}

function sleep_or_do(time, slowdown, fn) {
    if (time === 0) {
        fn()
    } else {
        animationTimeout = setTimeout(fn, time * 1000 * slowdown)
    }
}