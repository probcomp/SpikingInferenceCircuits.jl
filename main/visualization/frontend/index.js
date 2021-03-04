// get computed canvas width/height
var graph = d3.select("#graph");
var graphbox = graph.node().getBoundingClientRect();
var width = graphbox.width,
    height = graphbox.height;

var cola = cola.d3adaptor(d3)
    .jaccardLinkLengths(40, 0.7) // TODO: be smarter with this
    .avoidOverlaps(true)
    .handleDisconnected(false)
    .size([width, height]);

var svg = d3.select("#graph").append("svg")
    .attr("width", width)
    .attr("height", height);

d3.json("testgraph2.json", function(graph) {
    make_initial_graph_modifications(graph)
    console.log(graph);

    // initialize cola
    x = cola.nodes(graph.nodes)
        .links(graph.links)
        .groups(graph.groups)
        .constraints(graph.constraints)
        .start(50, 100, 50, 50);

    groups = add_groups(svg, graph)
    nodes = add_nodes(svg, graph)
    links = add_links(svg, graph)
    group_labels = add_group_labels(svg, graph)
    node_labels = add_node_labels(svg, graph)

    cola.on("tick", function() {
        update_links(links)
        update_nodes(nodes)
        update_groups(groups)
        update_node_labels(node_labels)
        update_group_labels(group_labels)
    })
})

/**
 * Initial modifications to the graph to add needed visual constraints and properties
 */

const GroupPadding = 10;
const NodeH = 2;
const NodeW = 2;
const PoissonH = 25;
const PoissonW = 25;

function make_initial_graph_modifications(graph) {
    graph.nodes.forEach(v => {
        v.width = NodeW;
        v.height = NodeH;
    });
    graph.groups.forEach(g => {
        g.padding = is_composite(g) ? GroupPadding : 0.01;
    })

    graph.constraints.filter(c => c.type === "separation")
        .forEach(c => { c.gap = 20 });

    // add constraints to ensure the poisson neurons are triangles
    graph.groups.filter(is_poisson).forEach(function(g) {
        graph.constraints.push({ // top and bottom y separation
            type: "separation",
            axis: "y",
            left: g.leaves[0],
            right: g.leaves[1],
            gap: PoissonH,
            equality: true
        })
        graph.constraints.push({ // left and right x separation
            type: "separation",
            axis: "x",
            left: g.leaves[0],
            right: g.leaves[2],
            gap: PoissonW,
            equality: true
        })
        graph.constraints.push({ // y position of output
            type: "separation",
            axis: "y",
            left: g.leaves[0],
            right: g.leaves[2],
            gap: PoissonH / 2,
            equality: true
        })
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

    var poisson_groups = svg.selectAll(".poissonGroup")
        .data(graph.groups.filter(is_poisson))
        .enter().append("polygon")
        .attr("class", "poissonGroup")
        .call(cola.drag);

    return {
        composite: composite_groups,
        poisson: poisson_groups
    }
}

function add_nodes(svg, graph) {
    const NodeR = 2;
    return svg.selectAll(".node")
        .data(graph.nodes)
        .enter().append("circle")
        .attr("class", "node")
        .attr("r", NodeR)
        .style("fill", d => "black")
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
        .data(graph.groups.filter(is_composite))
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

    groups.poisson
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

function is_poisson(g) {
    return g.comptype === "PoissonNeuron"
}

function triangle_points(g) {
    // top left
    x1 = g.bounds.x + NodeW / 2
    y1 = g.bounds.y + NodeH / 2
        // bottom left
    x2 = g.bounds.x + NodeW / 2
    y2 = g.bounds.y + g.bounds.height() - NodeH / 2
        // center right
    x3 = g.bounds.x + g.bounds.width()
    y3 = g.bounds.y + g.bounds.height() / 2 + NodeH / 2

    return "" + x1 + "," + y1 + " " + x2 + "," + y2 + " " + x3 + "," + y3
}