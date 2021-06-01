// get computed canvas width/height
var graph = d3.select("#graph");
var graphbox = graph.node().getBoundingClientRect();
var width = graphbox.width,
    height = graphbox.height;

var cola = cola.d3adaptor(d3)
    .avoidOverlaps(true)
    .handleDisconnected(false)
    .size([width, height]);

var svg = d3.select("#graph").append("svg")
    .attr("width", width)
    .attr("height", height);

d3.select("#load_component").on("click", function() {
    filename = d3.select("#component_filename").property("value");
    set_anim_filename();
    d3.json("renders/" + filename, function(graph) {
        make_initial_graph_modifications(graph)
        console.log(graph);

        // initialize cola
        x = cola.nodes(graph.nodes)
            .links(graph.links)
            .groups(graph.groups)
            .constraints(graph.constraints)
            // .flowLayout("x")  // this doesn't seem to work.  But would it be better?
            .symmetricDiffLinkLengths(20)
            .start(10, 15, 10, 10);

        // clear svg so we can put in new circuit
        svg.selectAll("*").remove();

        var groups = add_groups(svg, graph)
        var nodes = add_nodes(svg, graph)
        var links = add_links(svg, graph)
        var group_labels = add_group_labels(svg, graph)
        var node_labels = add_node_labels(svg, graph)

        cola.on("tick", function() {
            update_links(links)
            update_nodes(nodes)
            update_groups(groups)
            update_node_labels(node_labels)
            update_group_labels(group_labels)
        })

        setup_animation({
            nodes: nodes,
            links: links,
            groups: groups.neuron,
            graph: graph
        })
    })
})

/**
 * Initial modifications to the graph to add needed visual constraints and properties
 */

const GroupPadding = 1;
const NodeH = 0.2;
const NodeW = 0.2;
const PoissonH = 25;
const PoissonW = 25;

function make_initial_graph_modifications(graph) {
    graph.nodes.forEach(v => {
        v.width = NodeW;
        v.height = NodeH;
    });
    graph.groups.forEach(g => {
        g.padding = is_composite(g) || is_generic(g) ? GroupPadding : 0.001;
    })

    graph.constraints.filter(c => c.type === "separation" && c.axis === "x")
        .forEach(c => { c.gap = 2 });
    graph.constraints.filter(c => c.type === "separation" && c.axis === "y")
        .forEach(c => { c.gap = 0.4 });

    // add constraints to ensure the neurons triangles
    graph.groups.filter(is_neuron).forEach(function(g) {
        if (g.leaves.length > 2) {
            graph.constraints.push({ // top and bottom y separation
                type: "separation",
                axis: "y",
                left: g.leaves[0],
                right: g.leaves[2],
                gap: PoissonH,
                equality: true
            })
            graph.constraints.push({ // left and right x separation
                type: "separation",
                axis: "x",
                left: g.leaves[0],
                right: g.leaves[3],
                gap: PoissonW,
                equality: true
            })
            graph.constraints.push({ // y position of output
                    type: "separation",
                    axis: "y",
                    left: g.leaves[0],
                    right: g.leaves[3],
                    gap: PoissonH / 2,
                    equality: true
                }),
                graph.constraints.push({ // y position of center dot
                    type: "separation",
                    axis: "y",
                    left: g.leaves[0],
                    right: g.leaves[1],
                    gap: PoissonH / 2,
                    equality: true
                })
        }
    })
}

/**
 * Initial drawing
 */

function add_groups(svg, graph) {
    var composite_groups = svg.selectAll(".compGroup")
        .data(graph.groups.filter(is_composite))
        .enter().append("rect")
        .attr("rx", 8).attr("ry", 8)
        .attr("class", "compGroup")
        .call(cola.drag)
        .on("mouseup", d => { d.fixed = false });

    var generic_groups = svg.selectAll(".genericGroup")
        .data(graph.groups.filter(is_generic))
        .enter().append("rect")
        .attr("rx", 0.8).attr("ry", 0.8)
        .attr("class", "genericGroup")
        .call(cola.drag)
        .on("mouseup", d => { d.fixed = false });

    var neuron_groups = svg.selectAll(".neuronGroup")
        .data(graph.groups.filter(is_neuron))
        .enter().append("polygon")
        .attr("class", "neuronGroup")
        .attr("class", d => "neuronGroup_" + d.comptype)
        .call(cola.drag);

    return {
        composite: composite_groups,
        neuron: neuron_groups,
        generic: generic_groups
    }
}

function add_nodes(svg, graph) {
    const NodeR = 0.2;
    return svg.selectAll(".node")
        .data(graph.nodes)
        .enter().append("circle")
        .attr("class", "node")
        .attr("r", NodeR)
        .call(cola.drag);
}

function add_links(svg, graph) {
    return svg.selectAll(".link")
        .data(graph.links)
        .enter().append("line")
        .attr("class", "link");
}

function add_group_labels(svg, graph) {
    return svg.selectAll(".nodeLabel")
        .data(graph.groups.filter(is_generic))
        .enter().append("text")
        .attr("class", "groupLabel")
        .attr("alignment-baseline", "top")
        .text(g => g.comptype)
        .call(cola.drag);
    // TODO: add `title` with node name?
}

function add_node_labels(svg, graph) {
    return svg.selectAll(".nodeLabel")
        .data(graph.nodes)
        .enter().append("text")
        .attr("class", "nodeLabel")
        .text(d => d.name)
        .call(cola.drag);
}

/**
 * Drawing Updates
 */

function update_links(links) {
    links.attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
}

function update_nodes(nodes) {
    nodes.attr("cx", d => d.x - d.width / 2 + NodeW / 2)
        .attr("cy", d => d.y - d.height / 2 + NodeH / 2);
}

function update_groups(groups) {
    groups.composite
        .attr("x", g => g.bounds.x + GroupPadding)
        .attr("y", g => g.bounds.y)
        .attr("width", g => g.bounds.width() - 2 * GroupPadding)
        .attr("height", g => g.bounds.height());

    groups.generic
        .attr("x", g => g.bounds.x + GroupPadding)
        .attr("y", g => g.bounds.y)
        .attr("width", g => g.bounds.width() - 2 * GroupPadding)
        .attr("height", g => g.bounds.height());

    groups.neuron
        .attr("points", triangle_points);
}

function update_node_labels(node_labels) {
    node_labels.attr("x", d => d.x).attr("y", d => d.y);
}

function update_group_labels(group_labels) {
    group_labels
        .attr("x", g => g.bounds.x + g.bounds.width() / 2)
        .attr("y", g => g.bounds.y + GroupPadding);
}

/**
 * Helper Functions
 */

function is_composite(g) {
    return g.comptype === "CompositeComponent"
}

function is_neuron(g) {
    return g.comptype === "OnOffPoissonNeuron" || g.comptype === "IntegratingPoisson" || g.comptype == "InputFunctionPoisson"
}

function is_generic(g) {
    return !is_composite(g) && !is_neuron(g);
}

function triangle_points(g) {
    // top left
    var x1 = g.bounds.x + NodeW / 2
    var y1 = g.bounds.y + NodeH / 2
        // bottom left
    var x2 = g.bounds.x + NodeW / 2
    var y2 = g.bounds.y + g.bounds.height() - NodeH / 2
        // center right
    var x3 = g.bounds.x + g.bounds.width()
    var y3 = g.bounds.y + g.bounds.height() / 2 + NodeH / 2

    return "" + x1 + "," + y1 + " " + x2 + "," + y2 + " " + x3 + "," + y3
}