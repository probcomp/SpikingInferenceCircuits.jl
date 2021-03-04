// color scheme for groups
var color = d3.scaleOrdinal(d3.schemeCategory10);

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
    graph.nodes.forEach(v => { v.width = v.height = 8; });
    graph.groups.forEach(g => { g.padding = 0.1; });
    graph.constraints.filter(c => c.type === "separation")
        .forEach(c => { c.gap = 20 });
    console.log(graph);

    // give cola the graph and begin running the layout alg!
    x = cola.nodes(graph.nodes)
        .links(graph.links)
        .groups(graph.groups)
        .constraints(graph.constraints)
        .start(50, 100, 50, 50);

    // a `group` is a component
    var group = svg.selectAll(".group")
        .data(graph.groups)
        .enter().append("rect")
        .attr("rx", 8).attr("ry", 8)
        .attr("class", "group")
        .style("fill", (d, i) => color(i))
        .call(cola.drag)
        .on("mouseup", d => { d.fixed = false });

    // a `node` is an input to/output from a component
    const NodeR = 2;
    var node = svg.selectAll(".node")
        .data(graph.nodes)
        .enter().append("circle")
        .attr("class", "node")
        .attr("r", NodeR)
        .style("fill", d => "black")
        .call(cola.drag);

    var link = svg.selectAll(".link")
        .data(graph.links)
        .enter().append("line")
        .attr("class", "link");

    // TODO: display node name?
    var label = svg.selectAll(".label")
        .data(graph.nodes)
        .enter().append("text")
        .attr("class", "label")
        .text(d => d.name)
        .call(cola.drag);
    // TODO: add `title` with node name?

    cola.on("tick", () => {
        link.attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node.attr("cx", d => d.x - NodeR / 2)
            .attr("cy", d => d.y - NodeR / 2);

        group.attr("x", d => d.bounds.x)
            .attr("y", d => d.bounds.y)
            .attr("width", d => d.bounds.width())
            .attr("height", d => d.bounds.height());

        label.attr("x", d => d.x).attr("y", d => d.y);
    })
})