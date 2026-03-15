// SAE Feature Dashboard — UMedPT
// Adapted from the SAELens/SAE-Visualiser reference implementation.
// Components used: featureTables, actsHistogram, topPatches, universalityProfile.

// ─── Component: featureTables ────────────────────────────────────────────────

function setupFeatureTables(featureIdx, componentData, containerId) {
    const tablesMeta = [
        {
            title: "CORRELATED FEATURES",
            columns: ["Index", "Co-activation", "Dec. Cos-Sim."],
            dataKey: "correlatedFeatures",
        },
    ];

    tablesMeta.forEach(meta => {
        const rows = (componentData[meta.dataKey] || []);
        if (rows.length === 0) return;

        const wrapper = d3.select(`#${containerId}`)
            .append("div")
            .attr("id", `${meta.dataKey}-${containerId}`);

        wrapper.append("h4").html(meta.title);

        const table = wrapper.append("table").attr("class", "table-left");
        const thead  = table.append("thead");
        const tbody  = table.append("tbody");

        const headerRow = thead.append("tr");
        meta.columns.forEach(col => {
            headerRow.append("td")
                .attr("class", col === "Index" ? "left-aligned" : "right-aligned")
                .html(col);
        });

        const dataRows = tbody.selectAll("tr")
            .data(rows)
            .enter()
            .append("tr");

        dataRows.selectAll("td")
            .data(row => Object.values(row))
            .enter()
            .append("td")
            .attr("class", (d, i) => i === 0 ? "left-aligned" : "right-aligned")
            .html(d => d);
    });
}

// ─── Component: actsHistogram ────────────────────────────────────────────────

function setupActsHistogram(featureIdx, componentData, containerId) {
    const container = d3.select(`#${containerId}`).attr("class", "plotly-hist");

    if (!componentData.x || componentData.x.length === 0) {
        container.append("div")
            .style("color", "#aaa")
            .style("font-size", "0.85em")
            .style("padding", "12px 0")
            .text("DEAD FEATURE — never activates");
        return;
    }

    const layout = {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor:  "rgba(0,0,0,0)",
        xaxis: {
            gridcolor: "#eee",
            zerolinecolor: "#eee",
            tickvals: componentData.ticks,
            range: [0, 1.15 * Math.max(...componentData.x)],
        },
        yaxis: { gridcolor: "#eee", zerolinecolor: "#eee" },
        barmode: "relative",
        bargap: 0.01,
        showlegend: false,
        margin: { l: 50, r: 25, b: 25, t: 25, pad: 4 },
        responsive: true,
        shapes: [createHistLine("black", 2, 1.0), createHistLine("black", 1, 0.9)],
        annotations: [createHistAnnotation(), createHistAnnotation()],
    };

    const maxX = Math.max(...componentData.x);
    const traces = [{
        x: componentData.x,
        y: componentData.y,
        type: "bar",
        marker: {
            color: componentData.x.map(v => actColor(0.4 + 0.6 * v / maxX, 1)),
        },
    }];

    Plotly.newPlot(containerId, traces, layout, { responsive: true, displayModeBar: false });

    if (componentData.title) {
        const h = document.createElement("h4");
        h.innerHTML = componentData.title;
        container.node().parentNode.insertBefore(h, container.node());
    }
}

// ─── Component registration map ──────────────────────────────────────────────
// topPatches and universalityProfile are added by html_dashboard.py extra_js.

const componentMap = {
    featureTables:      setupFeatureTables,
    actsHistogram:      setupActsHistogram,
};

// ─── Utility functions ───────────────────────────────────────────────────────

function actColor(value, maxValue, isDfa = false) {
    value = Math.min(Math.max(value, 0), maxValue);
    return d3.interpolate(
        "rgb(255,255,255)",
        isDfa ? "rgb(0,150,50)" : "rgb(255,140,0)"
    )(value / maxValue);
}

function lossColor(loss, maxLoss, opacity = 0.5) {
    loss = Math.min(Math.max(loss, -maxLoss), maxLoss);
    return loss < 0
        ? d3.interpolate(`rgba(255,255,255,${opacity})`, `rgba(0,0,255,${opacity})`)(-loss / maxLoss)
        : d3.interpolate(`rgba(255,255,255,${opacity})`, `rgba(255,0,0,${opacity})`)(loss / maxLoss);
}

function createHistLine(color, width, height) {
    return {
        type: "line", line: { color, width },
        x0: 0, x1: 0, xref: "x",
        y0: 0, y1: height, yref: "paper",
        visible: false,
    };
}

function createHistAnnotation() {
    return {
        text: "", showarrow: false,
        x: 0, xref: "x",
        y: 0.9, yref: "paper",
        xshift: 3, align: "left", xanchor: "left",
        visible: false,
    };
}

function measureTextWidth(text, font) {
    const span = document.createElement("span");
    span.style.cssText = "visibility:hidden;position:absolute;white-space:nowrap;";
    span.style.font = font;
    span.textContent = text;
    document.body.appendChild(span);
    const w = span.offsetWidth;
    document.body.removeChild(span);
    return w;
}

// ─── Page rendering ──────────────────────────────────────────────────────────

function updateDropdowns(options) {
    const currentSelections = options.map((_, i) =>
        d3.select(`#select-${i}`).property("value")
    );
    setupPage(currentSelections.join("|"));
}

function setupPage(key) {
    const gridContainer = d3.select(".grid-container")
        .attr("style", `height: ${METADATA.height}px;`);
    gridContainer.selectAll("*").remove();

    METADATA.layout.forEach((columnComponents, columnIdx) => {
        const columnWidth = METADATA.columnWidths[columnIdx];
        const column = gridContainer.append("div")
            .attr("id", `column-${columnIdx}`)
            .attr("class", "grid-column")
            .style("width", columnWidth ? `${columnWidth}px` : "auto");

        columnComponents.forEach(componentName => {
            const fn = componentMap[componentName];
            if (!fn) return;
            const containerId = `${componentName}-${key}`;
            column.append("div").attr("id", containerId).attr("class", componentName);
            fn(key, DATA[key][componentName], containerId);
        });
    });
}

// ─── Bootstrap ───────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", function () {
    const allKeys = Object.keys(DATA);

    if (!allKeys.includes(String(START_KEY))) {
        console.error(`START_KEY ${START_KEY} not found in DATA`);
    }

    if (allKeys.length > 1) {
        const options = [new Set(allKeys)];

        const selectDiv = d3.select("#dropdown-container")
            .append("div").attr("class", "select");
        const selectElem = selectDiv.append("select").attr("id", "select-0");

        let maxWidth = 0;
        allKeys.forEach(k => {
            selectElem.append("option").text(`Feature ${k}`).attr("value", k);
            const w = measureTextWidth(`Feature ${k}`, "1em Consolas");
            if (w > maxWidth) maxWidth = w;
        });
        selectElem.property("value", String(START_KEY));
        selectDiv.style("width", `${maxWidth + 50}px`);
        selectElem.on("change", function () {
            setupPage(d3.select(this).property("value"));
        });
    }

    setupPage(String(START_KEY));
});
