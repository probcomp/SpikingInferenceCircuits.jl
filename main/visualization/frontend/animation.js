function setup_animation(viz) {
    d3.json("animation.json", animation => run_animation(animation, viz));
    d3.select("#play_animation").on("click", () => {
        d3.json("animation.json", animation => run_animation(animation, viz));
    });
}

const spike_length = 0.15; // seconds

// animation looks like
// { frames: [f1, f2, ...] }
// where each frame `f_i` looks like
// { time: time, group: group_name, out: out_name }
function run_animation(animation, viz) {
    if (animation.frames.length !== 0) {
        first_action_time = animation.frames[0].time;
        sleep_or_do(
            first_action_time,
            handle_frame(viz, first_action_time, animation.frames, [])
        )
    }
}

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
 * Show the next node to be spiking!
 */
function handle_frame(viz, current_time, remaining_frames, currently_spiking) {
    // TODO: give SVG elements IDs so we can more efficiently access them!

    next_frame = remaining_frames.splice(0, 1)[0]
    if (next_frame.time !== current_time) {
        console.warn("Reached a frame at the wrong time!  next_frame.time = " + next_frame.time + " but current_time = " + current_time)
    }

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
    })

    setup_next_action(viz, current_time, remaining_frames, currently_spiking);
}

/**
 * Turn off the node we set to spiking longest ago.
 */
function remove_spiking(viz, current_time, remaining_frames, currently_spiking) {
    obj = currently_spiking.splice(0, 1)[0];
    if (obj.time + spike_length !== current_time) {
        console.warn("`removed_spiking` called to remove a frame at the wrong time!  current_time = " + current_time + "; obj.time = " + obj.time + "; spike_length = " + spike_length)
    }

    obj.groups.attr("class", "poissonGroup");
    obj.nodes.attr("class", "node");
    obj.links.attr("class", "link");

    setup_next_action(viz, current_time, remaining_frames, currently_spiking);
}

/**
 * Queue up the next action in the animation to be performed after the appropriate delay.
 */
function setup_next_action(viz, current_time, remaining_frames, currently_spiking) {
    time_to_frame = remaining_frames.length === 0 ? Infinity : (remaining_frames[0].time - current_time)
    time_to_removal = currently_spiking.length === 0 ? Infinity : spike_length + current_time - currently_spiking[0].time

    if (time_to_frame === Infinity && time_to_removal === Infinity) {
        return; // animation is completed
    } else if (time_to_frame <= time_to_removal) {
        sleep_or_do(time_to_frame, () => handle_frame(viz, current_time + time_to_frame, remaining_frames, currently_spiking))
    } else {
        sleep_or_do(time_to_removal, () => remove_spiking(viz, current_time + time_to_removal, remaining_frames, currently_spiking))
    }
}

function sleep_or_do(time, fn) {
    if (time === 0) {
        fn()
    } else {
        setTimeout(fn, time * 1000)
    }
}