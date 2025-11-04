(function () {
  const STYLE_ID = "site-insights-style";
  const PANEL_CLASS = "insights-panel";
  const INSIGHTS_ATTR = "data-insights-ready";
  const GLOBAL_RECORDS = new Map();
  let lastConsoleSnapshotSize = 0;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", runInsightScanner);
  } else {
    runInsightScanner();
  }

  function runInsightScanner() {
    ensureStyles();
    waitForAva()
      .then(() => {
        const delays = [400, 1500, 3000];
        delays.forEach((delay) => setTimeout(scanCharts, delay));
      })
      .catch((err) => {
        console.warn("AVA insight engine not available", err);
      });
  }

  function waitForAva(maxAttempts = 20, interval = 150) {
    return new Promise((resolve, reject) => {
      const check = (attempt = 0) => {
    if (
      window.AVA &&
      (typeof window.AVA.getInsights === "function" || typeof window.AVA.insight === "function")
    ) {
      resolve(window.AVA);
      return;
    }
        if (attempt >= maxAttempts) {
          reject(new Error("window.AVA.insight missing"));
          return;
        }
        setTimeout(() => check(attempt + 1), interval);
      };
      check();
    });
  }

  function ensureStyles() {
    if (document.getElementById(STYLE_ID)) {
      return;
    }
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .${PANEL_CLASS} { font-family: 'Josefin Sans', 'Helvetica Neue', Arial, sans-serif; background:#f8f9fa; border:1px solid #d8e1dc; padding:10px 14px; margin-top:8px; border-radius:8px; box-shadow:0 1px 2px rgba(13,59,41,0.06); }
      .${PANEL_CLASS} h4 { margin:0 0 6px 0; font-size:14px; color:#0d3b29; letter-spacing:0.02em; text-transform:uppercase; }
      .${PANEL_CLASS} ul { margin:0; padding-left:18px; font-size:13px; color:#174c38; line-height:1.45; }
      .${PANEL_CLASS} li + li { margin-top:4px; }
      .${PANEL_CLASS} button { margin-top:8px; font-size:12px; padding:4px 10px; border-radius:4px; border:1px solid #0d3b29; background:#134f38; color:#f2fbf5; cursor:pointer; }
      .${PANEL_CLASS} button:hover { background:#0d3b29; }
      .${PANEL_CLASS} button:active { transform:translateY(1px); }
      .${PANEL_CLASS} .insights-empty { font-size:12px; color:#4f6f62; margin:0; }
    `;
    document.head.appendChild(style);
  }

  function scanCharts() {
    const charts = [
      ...collectPlotlyCharts(),
      ...collectVegaCharts(),
      ...collectEChartsCharts(),
    ];

    let newRecords = 0;

    charts.forEach((chart) => {
      if (!chart || !chart.el || chart.el.getAttribute(INSIGHTS_ATTR) === "true") {
        return;
      }

      try {
        if (!chart.rows || chart.rows.length === 0) {
          return;
        }

        const insights = runInsightEngine(chart.rows);
        renderInsights(chart.el, insights);
        chart.el.setAttribute(INSIGHTS_ATTR, "true");

        if (insights && insights.length) {
          insights.forEach((item, idx) => {
            const recordKey = `${chart.id}::${idx}::${item.description || item.title || item.type || idx}`;
            if (GLOBAL_RECORDS.has(recordKey)) {
              return;
            }
            GLOBAL_RECORDS.set(recordKey, {
              chart: chart.id,
              library: chart.type,
              type: item.type || item.subtype || "unknown",
              score: typeof item.score === "number" ? item.score : null,
              description: formatInsightDescription(item),
            });
            newRecords += 1;
          });
        }
      } catch (err) {
        console.warn(`Insight scan failed for chart ${chart.id}`, err);
      }
    });

    if (GLOBAL_RECORDS.size && newRecords) {
      logInsightsToConsole();
    }
  }

  function logInsightsToConsole() {
    if (!GLOBAL_RECORDS.size) {
      return;
    }
    if (GLOBAL_RECORDS.size === lastConsoleSnapshotSize) {
      return;
    }
    const table = Array.from(GLOBAL_RECORDS.values()).map((item) => ({
      chart: item.chart,
      library: item.library,
      type: item.type,
      score: item.score != null ? item.score.toFixed(2) : "",
      desc: item.description,
    }));
    if (console && typeof console.table === "function") {
      console.table(table);
    } else {
      console.log("Chart insights", table);
    }
    lastConsoleSnapshotSize = GLOBAL_RECORDS.size;
  }

  function collectPlotlyCharts() {
    if (!(window.Plotly || window.PlotlyConfig)) {
      return [];
    }
    return Array.from(document.querySelectorAll(".js-plotly-plot, .plotly-graph-div"))
      .map((el, idx) => {
        const gd = el;
        const traces = extractPlotlyTraces(gd);
        if (!traces.length) {
          return null;
        }
        const chartId = el.id || `plotly-${idx + 1}`;
        const rows = [];
        traces.forEach((trace, traceIndex) => {
          if (!trace) {
            return;
          }
          const seriesName = trace.name || trace.legendgroup || `Series ${traceIndex + 1}`;
          const xValues = normaliseArray(trace.x || trace.labels || trace.theta);
          const yValues = normaliseArray(trace.y || trace.values || trace.r);
          const len = Math.min(xValues.length, yValues.length);
          for (let i = 0; i < len; i += 1) {
            const y = safeNumber(yValues[i]);
            if (Number.isNaN(y)) {
              continue;
            }
            rows.push({
              x: toDisplayValue(xValues[i], i),
              y,
              series: seriesName,
              chart_id: chartId,
            });
          }
        });
        return {
          el,
          rows,
          id: chartId,
          type: "plotly",
        };
      })
      .filter(Boolean);
  }

  function extractPlotlyTraces(gd) {
    if (!gd) {
      return [];
    }
    if (Array.isArray(gd.data) && gd.data.length) {
      return gd.data;
    }
    if (Array.isArray(gd._fullData) && gd._fullData.length) {
      return gd._fullData.map((trace) => ({
        ...trace,
        x: trace.x || trace.locations || [],
        y: trace.y || trace.z || [],
      }));
    }
    if (gd.__data__) {
      const data = gd.__data__;
      if (Array.isArray(data)) {
        return data;
      }
    }
    return [];
  }

  function collectVegaCharts() {
    const vegaCandidates = Array.from(
      document.querySelectorAll(".vega-embed, [data-vega-view]")
    );
    const charts = [];
    vegaCandidates.forEach((el, idx) => {
      const chartId = el.id || `vega-${idx + 1}`;
      const view = resolveVegaView(el);
      let rows = [];
      if (view) {
        rows = extractVegaViewData(view);
      } else {
        const spec = resolveVegaSpec(el);
        if (spec) {
          rows = extractVegaSpecData(spec);
        }
      }
      charts.push({
        el,
        rows: rows.map((row) => ({ ...row, chart_id: chartId })),
        id: chartId,
        type: "vega",
      });
    });
    return charts;
  }

  function resolveVegaView(el) {
    if (el && el.__vega__) {
      if (el.__vega__.view) {
        return el.__vega__.view;
      }
      if (el.__vega__.viewPromise && typeof el.__vega__.viewPromise.then === "function") {
        el.__vega__.viewPromise.then(() => {
          setTimeout(scanCharts, 60);
        });
      }
    }
    return null;
  }

  function resolveVegaSpec(el) {
    if (!el) {
      return null;
    }
    const script = el.querySelector("script[type='application/json']");
    if (!script) {
      return null;
    }
    try {
      return JSON.parse(script.textContent || "");
    } catch (err) {
      return null;
    }
  }

  function extractVegaViewData(view) {
    try {
      const state = view.getState({ data: true });
      const dataSets = state?.data || {};
      const key = Object.keys(dataSets).find((name) => Array.isArray(dataSets[name]));
      if (!key) {
        return [];
      }
      return flattenGenericRows(dataSets[key]);
    } catch (err) {
      console.warn("Unable to extract Vega view data", err);
      return [];
    }
  }

  function extractVegaSpecData(spec) {
    const values = spec?.data?.values;
    if (Array.isArray(values)) {
      return flattenGenericRows(values);
    }
    return [];
  }

  function flattenGenericRows(rows) {
    if (!Array.isArray(rows)) {
      return [];
    }
    const result = [];
    rows.forEach((row, index) => {
      if (row == null || typeof row !== "object") {
        return;
      }
      const keys = Object.keys(row);
      if (!keys.length) {
        return;
      }
      const [xKey, yKey] = inferXYKeys(keys);
      const xVal = row[xKey];
      const yVal = safeNumber(row[yKey]);
      if (Number.isNaN(yVal)) {
        return;
      }
      const seriesKey = keys.find((k) => k !== xKey && k !== yKey) || "series";
      result.push({
        x: toDisplayValue(xVal, index),
        y: yVal,
        series: String(row[seriesKey] ?? "series"),
      });
    });
    return result;
  }

  function inferXYKeys(keys) {
    if (keys.length <= 2) {
      return [keys[0] || "x", keys[1] || "y"];
    }
    const xCandidates = ["x", "category", "date", "time", "period", "label", "name"];
    const yCandidates = ["y", "value", "amount", "total", "count", "metric"];
    const lowerKeys = keys.map((k) => k.toLowerCase());
    const xIndex = lowerKeys.findIndex((k) => xCandidates.includes(k));
    const yIndex = lowerKeys.findIndex((k) => yCandidates.includes(k));
    const xKey = keys[xIndex >= 0 ? xIndex : 0];
    const yKey = keys[yIndex >= 0 ? yIndex : Math.min(keys.length - 1, 1)];
    return [xKey, yKey];
  }

  function collectEChartsCharts() {
    if (!(window.echarts && typeof window.echarts.getInstanceByDom === "function")) {
      return [];
    }
    return Array.from(document.querySelectorAll("[_echarts_instance_]"))
      .map((el, idx) => {
        let instance;
        try {
          instance = window.echarts.getInstanceByDom(el);
        } catch (err) {
          instance = null;
        }
        if (!instance) {
          return null;
        }
        const option = instance.getOption ? instance.getOption() : {};
        const chartId = el.id || `echarts-${idx + 1}`;
        const rows = extractEChartsData(option).map((row) => ({
          ...row,
          chart_id: chartId,
        }));
        return {
          el,
          rows,
          id: chartId,
          type: "echarts",
        };
      })
      .filter(Boolean);
  }

  function extractEChartsData(option) {
    if (!option || typeof option !== "object") {
      return [];
    }
    const seriesList = Array.isArray(option.series) ? option.series : [];
    const xAxis = Array.isArray(option.xAxis) ? option.xAxis[0] : option.xAxis || {};
    const xAxisData = normaliseArray(xAxis?.data);
    const result = [];

    seriesList.forEach((series, seriesIndex) => {
      if (!series) {
        return;
      }
      const seriesName = series.name || `Serie ${seriesIndex + 1}`;
      const points = normaliseArray(series.data);
      points.forEach((point, idx) => {
        const parsed = normaliseEChartsPoint(point, idx, xAxisData);
        if (!parsed) {
          return;
        }
        const y = safeNumber(parsed.y);
        if (Number.isNaN(y)) {
          return;
        }
        result.push({
          x: toDisplayValue(parsed.x, idx),
          y,
          series: seriesName,
        });
      });
    });

    return result;
  }

  function normaliseEChartsPoint(point, index, xAxisData) {
    if (Array.isArray(point)) {
      if (point.length >= 2) {
        return { x: point[0], y: point[1] };
      }
      return { x: xAxisData[index] ?? index, y: point[0] };
    }
    if (point && typeof point === "object") {
      if (Array.isArray(point.value)) {
        const value = point.value;
        if (value.length >= 2) {
          return { x: value[0], y: value[1] };
        }
        return { x: xAxisData[index] ?? index, y: value[0] };
      }
      if (Object.prototype.hasOwnProperty.call(point, "value")) {
        return { x: xAxisData[index] ?? index, y: point.value };
      }
      if (Object.prototype.hasOwnProperty.call(point, "y")) {
        return { x: point.x ?? xAxisData[index] ?? index, y: point.y };
      }
    }
    return { x: xAxisData[index] ?? index, y: point };
  }

  function normaliseArray(value) {
    if (Array.isArray(value)) {
      return value;
    }
    if (
      value &&
      typeof value === "object" &&
      typeof value.length === "number" &&
      value.length >= 0 &&
      typeof value !== "function"
    ) {
      try {
        return Array.from(value);
      } catch (err) {
        // fall through
      }
    }
    if (value == null) {
      return [];
    }
    return [value];
  }

  function toDisplayValue(value, fallbackIndex) {
    if (value == null || value === "") {
      return typeof fallbackIndex === "number" ? fallbackIndex : "";
    }
    return value;
  }

  function safeNumber(value) {
    if (typeof value === "number") {
      return value;
    }
    if (typeof value === "string") {
      const normalised = value.replace(/[^0-9+\-\.eE]/g, "");
      const parsed = Number(normalised);
      return Number.isNaN(parsed) ? NaN : parsed;
    }
    if (value instanceof Date) {
      return value.getTime();
    }
    if (value && typeof value.valueOf === "function") {
      const primitive = value.valueOf();
      if (typeof primitive === "number") {
        return primitive;
      }
      if (typeof primitive === "string") {
        return safeNumber(primitive);
      }
    }
    return NaN;
  }

  function runInsightEngine(rows) {
    if (!rows || !rows.length) {
      return [];
    }
    const { getInsights, insight } = window.AVA || {};
    const extractor = typeof getInsights === "function" ? getInsights : insight;
    if (typeof extractor !== "function") {
      return [];
    }
    try {
      const options = {
        measures: ["y"],
        dimensions: ["x", "series"],
        limit: 6,
      };
      const result = extractor(rows, options) || {};
      const insightArray = Array.isArray(result) ? result : result?.insights;
      if (!Array.isArray(insightArray)) {
        return [];
      }
      return insightArray
        .map((item) => ({
          ...item,
          description: formatInsightDescription(item),
        }))
        .filter((item) => item.description);
    } catch (err) {
      console.warn("AVA insight processing failed", err);
      return [];
    }
  }

  function formatInsightDescription(item) {
    if (!item || typeof item !== "object") {
      return "";
    }
    const textCandidates = [
      item.description,
      item.summary,
      item.statement,
      item.title,
      item.message,
    ];
    const text = textCandidates.find((candidate) => typeof candidate === "string" && candidate.trim());
    if (text) {
      return text.trim();
    }
    if (item?.insights && Array.isArray(item.insights)) {
      const nested = item.insights
        .map((nestedItem) => formatInsightDescription(nestedItem))
        .filter(Boolean)
        .join("; ");
      if (nested) {
        return nested;
      }
    }
    try {
      return JSON.stringify(item);
    } catch (err) {
      return "";
    }
  }

  function renderInsights(chartEl, insights) {
    removeExistingPanel(chartEl);
    const panel = document.createElement("div");
    panel.className = PANEL_CLASS;
    const heading = document.createElement("h4");
    heading.textContent = "Insights";
    panel.appendChild(heading);

    if (!insights || !insights.length) {
      const empty = document.createElement("p");
      empty.className = "insights-empty";
      empty.textContent = "No notable patterns detected.";
      panel.appendChild(empty);
      chartEl.parentNode?.insertBefore(panel, chartEl.nextSibling);
      return;
    }

    const list = document.createElement("ul");
    insights.slice(0, 6).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = formatInsightDescription(item);
      list.appendChild(li);
    });
    panel.appendChild(list);

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Copy to Markdown";
    button.addEventListener("click", () => {
      const lines = Array.from(panel.querySelectorAll("li")).map((li) => `- ${li.textContent}`);
      copyToClipboard(lines.join("\n"));
    });
    panel.appendChild(button);

    chartEl.parentNode?.insertBefore(panel, chartEl.nextSibling);
  }

  function removeExistingPanel(chartEl) {
    const sibling = chartEl.nextSibling;
    if (sibling && sibling.classList && sibling.classList.contains(PANEL_CLASS)) {
      sibling.remove();
    }
  }

  function copyToClipboard(text) {
    if (!text) {
      return;
    }
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      navigator.clipboard.writeText(text).catch((err) => {
        console.warn("Clipboard write failed", err);
      });
      return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    try {
      document.execCommand("copy");
    } catch (err) {
      console.warn("Fallback clipboard copy failed", err);
    }
    document.body.removeChild(textarea);
  }
})();
