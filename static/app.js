/* concept-atlas explorer: force-directed causal concept graph. */
"use strict";

const GROUP_COLORS = ["#9fb6c9", "#b9a97f", "#8fae8a", "#a98fae", "#c9a67f"];
const EDGE_COLOR = { excitatory: "#7fa6c9", inhibitory: "#c98f7f" };

const svg = d3.select("#graph");
const tooltip = document.getElementById("tooltip");
const stage = document.getElementById("stage");

fetch("graph.json")
  .then((r) => r.json())
  .then(render)
  .catch((err) => {
    document.getElementById("model-name").textContent = `failed to load graph.json: ${err}`;
  });

function render(data) {
  document.getElementById("model-name").textContent = `model: ${data.meta.model}`;

  const groups = [...new Set(data.nodes.map((n) => n.group))];
  const color = d3.scaleOrdinal(groups, GROUP_COLORS);
  const radius = d3.scaleLinear([0.5, 1], [5, 14]).clamp(true);
  const maxWeight = d3.max(data.links, (d) => Math.abs(d.weight)) || 1;
  const edgeWidth = d3.scaleLinear([0, maxWeight], [0.6, 5]);
  const width = stage.clientWidth;
  const height = stage.clientHeight;

  /* state driven by the controls */
  let minWeight = 0;
  const activeGroups = new Set(groups);

  /* arrowheads, one per edge kind so color matches */
  const defs = svg.append("defs");
  for (const kind of Object.keys(EDGE_COLOR)) {
    defs
      .append("marker")
      .attr("id", `arrow-${kind}`)
      .attr("viewBox", "0 -4 8 8")
      .attr("refX", 20)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L8,0L0,4")
      .attr("fill", EDGE_COLOR[kind]);
  }

  const zoomLayer = svg.append("g");
  svg.call(
    d3.zoom().scaleExtent([0.3, 4]).on("zoom", (ev) => zoomLayer.attr("transform", ev.transform))
  );

  const link = zoomLayer
    .append("g")
    .selectAll("line")
    .data(data.links)
    .join("line")
    .attr("stroke", (d) => EDGE_COLOR[d.kind])
    .attr("stroke-width", (d) => edgeWidth(Math.abs(d.weight)))
    .attr("stroke-opacity", 0.75)
    .attr("marker-end", (d) => `url(#arrow-${d.kind})`);

  const node = zoomLayer
    .append("g")
    .selectAll("g")
    .data(data.nodes)
    .join("g")
    .attr("class", "node")
    .call(drag());

  node
    .append("circle")
    .attr("r", (d) => radius(d.accuracy))
    .attr("fill", (d) => color(d.group))
    .on("mousemove", showTooltip)
    .on("mouseleave", () => (tooltip.hidden = true));

  node
    .append("text")
    .attr("dx", (d) => radius(d.accuracy) + 4)
    .attr("dy", "0.32em")
    .text((d) => d.name);

  const sim = d3
    .forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id((d) => d.id).distance(90))
    .force("charge", d3.forceManyBody().strength(-260))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("x", d3.forceX(width / 2).strength(0.07))
    .force("y", d3.forceY(height / 2).strength(0.09))
    .force("collide", d3.forceCollide().radius((d) => radius(d.accuracy) + 6))
    .on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

  /* -- filtering ---------------------------------------------------------- */

  function applyFilters() {
    const nodeVisible = new Map(
      data.nodes.map((n) => [n.id, activeGroups.has(n.group)])
    );
    node.style("display", (d) => (nodeVisible.get(d.id) ? null : "none"));
    link.style("display", (d) =>
      Math.abs(d.weight) >= minWeight &&
      nodeVisible.get(d.source.id) &&
      nodeVisible.get(d.target.id)
        ? null
        : "none"
    );
  }

  const slider = document.getElementById("weight-filter");
  slider.max = maxWeight.toFixed(2);
  slider.step = (maxWeight / 100).toFixed(3);
  slider.addEventListener("input", () => {
    minWeight = +slider.value;
    document.getElementById("weight-value").textContent = minWeight.toFixed(2);
    applyFilters();
  });

  const toggles = document.getElementById("group-toggles");
  for (const g of groups) {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.addEventListener("change", () => {
      box.checked ? activeGroups.add(g) : activeGroups.delete(g);
      applyFilters();
    });
    const dot = document.createElement("span");
    dot.className = "swatch";
    dot.style.background = color(g);
    dot.style.height = "8px";
    dot.style.width = "8px";
    label.append(box, dot, g);
    toggles.append(label);
  }

  /* -- helpers ------------------------------------------------------------ */

  function showTooltip(ev, d) {
    const out = data.links.filter((l) => l.source.id === d.id).length;
    const inn = data.links.filter((l) => l.target.id === d.id).length;
    tooltip.innerHTML =
      `<strong>${d.name}</strong><br>` +
      `<span class="k">set</span> ${d.group}<br>` +
      `<span class="k">home layer</span> ${d.layer}<br>` +
      `<span class="k">probe acc</span> ${d.accuracy.toFixed(2)}<br>` +
      `<span class="k">edges</span> ${out} out / ${inn} in`;
    tooltip.style.left = `${ev.offsetX + 14}px`;
    tooltip.style.top = `${ev.offsetY + 14}px`;
    tooltip.hidden = false;
  }

  function drag() {
    return d3
      .drag()
      .on("start", (ev, d) => {
        if (!ev.active) sim.alphaTarget(0.25).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (ev, d) => {
        d.fx = ev.x;
        d.fy = ev.y;
      })
      .on("end", (ev, d) => {
        if (!ev.active) sim.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
  }
}
