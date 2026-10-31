/* Local-first album: no fetch, external fonts, frameworks, or telemetry. */
(() => {
  "use strict";
  const data = window.ALBUM_DATA;
  if (!data) return;
  const $ = (id) => document.getElementById(id);
  const create = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  document.querySelectorAll("[data-count]").forEach(node => { node.textContent = data.counts[node.dataset.count]; });
  data.timeline.forEach(item => {
    const article = create("article", "timeline-item");
    article.append(create("div", "timeline-year", item.year), create("h3", "", item.title), create("p", "", item.description));
    const link = create("a", "commit", `${item.date} / ${item.commit} ↗`);
    link.href = `https://github.com/cloud2010/dl-models/commit/${item.commit}`;
    link.title = `${item.source}: ${item.message}`;
    article.append(link);
    $("timeline").append(article);
  });
  document.querySelectorAll("[data-filter]").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll("[data-filter]").forEach(other => {
      const active = other === button;
      other.classList.toggle("active", active);
      other.setAttribute("aria-pressed", String(active));
    });
    let count = 0;
    document.querySelectorAll("[data-category]").forEach(card => {
      card.hidden = button.dataset.filter !== "all" && card.dataset.category !== button.dataset.filter;
      if (!card.hidden) count += 1;
    });
    $("filter-status").textContent = `显示 ${count} 张研究手记`;
  }));

  let photoIndex = 0;
  let photoTrigger = null;
  const dialog = $("photo-dialog");
  function setPhoto(index) {
    photoIndex = (index + data.photos.length) % data.photos.length;
    const photo = data.photos[photoIndex];
    $("large-photo").src = photo.image;
    $("large-photo").alt = `${photo.title}：${photo.caption}`;
    $("photo-title").textContent = photo.title;
    $("photo-caption").textContent = photo.caption;
    $("photo-source").href = photo.source;
    $("photo-source").textContent = `原始 Notebook · 第 ${photo.cell} 个单元格 ↗`;
    $("photo-position").textContent = `CONTACT SHEET / ${String(photoIndex + 1).padStart(2, "0")} OF ${String(data.photos.length).padStart(2, "0")}`;
  }
  data.photos.forEach((photo, index) => {
    const figure = create("figure", "photo-card");
    const button = create("button");
    button.type = "button";
    button.setAttribute("aria-label", `放大：${photo.title}`);
    const img = create("img");
    img.src = photo.image; img.alt = photo.caption; img.loading = "lazy"; img.width = 600; img.height = 450;
    button.append(img);
    button.addEventListener("click", () => {
      setPhoto(index);
      photoTrigger = button;
      dialog.showModal();
    });
    const caption = create("figcaption", "", photo.title);
    caption.append(create("span", "photo-source", `NOTEBOOK / CELL ${photo.cell}`));
    figure.append(button, caption, create("span", "photo-number", `FIG. 0${index + 1}`));
    $("photo-grid").append(figure);
  });
  $("close-photo").addEventListener("click", () => dialog.close());
  $("previous-photo").addEventListener("click", () => setPhoto(photoIndex - 1));
  $("next-photo").addEventListener("click", () => setPhoto(photoIndex + 1));
  dialog.addEventListener("close", () => photoTrigger?.focus());
  dialog.addEventListener("click", event => { if (event.target === dialog) {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }});
  dialog.addEventListener("keydown", event => {
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault(); setPhoto(photoIndex + (event.key === "ArrowRight" ? 1 : -1));
    }
  });

  let metric = "mcc";
  const select = $("configuration");
  data.configurations.forEach((config, index) => {
    const option = create("option", "", config.label); option.value = String(index); select.append(option);
  });
  const ns = "http://www.w3.org/2000/svg";
  function svgNode(name, attributes = {}, text) {
    const node = document.createElementNS(ns, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  const chart = svgNode("svg", {viewBox: "0 0 1000 365", class: "chart-svg", role: "img", "aria-labelledby": "curve-svg-title curve-svg-desc"});
  chart.append(svgNode("title", {id: "curve-svg-title"}, "SNN 历史增量特征指标曲线"), svgNode("desc", {id: "curve-svg-desc"}, "横轴为特征数量。可通过下方滑块逐点读取原始值。"));
  const grid = svgNode("g", {fill: "#697568", "font-family": "sans-serif", "font-size": "12"});
  const line = svgNode("path", {fill: "none", stroke: "#476b58", "stroke-width": "1.5", "stroke-linejoin": "round"});
  const cursor = svgNode("line", {y1: "20", y2: "300", stroke: "#a65d41", "stroke-dasharray": "4 4"});
  const dot = svgNode("circle", {r: "5", fill: "#a65d41", stroke: "#faf8f0", "stroke-width": "2"});
  chart.append(grid, line, cursor, dot);
  $("interactive-chart").replaceChildren(chart);
  const slider = $("feature-slider");
  let downloadUrl;
  $("save-chart").addEventListener("click", () => {
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    const exported = chart.cloneNode(true);
    exported.setAttribute("xmlns", ns);
    exported.setAttribute("width", "1000"); exported.setAttribute("height", "405");
    exported.setAttribute("viewBox", "0 0 1000 405");
    exported.prepend(svgNode("rect", {width: 1000, height: 405, fill: "#faf8f0"}));
    const config = data.configurations[Number(select.value)];
    exported.append(svgNode("text", {x: 60, y: 390, fill: "#697568", "font-size": 10, "font-family": "sans-serif"},
      `${config.source[metric]} | Historical log, not revalidated.`));
    downloadUrl = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(exported)], {type: "image/svg+xml;charset=utf-8"}));
    $("save-chart").href = downloadUrl;
    $("save-chart").download = `snn-${data.configurations[Number(select.value)].id}-${metric}.svg`;
  });
  let scaleY = value => value;
  const scaleX = (index, total) => 60 + index * 910 / (total - 1);
  function updatePoint() {
    const values = data.configurations[Number(select.value)][metric];
    const index = Number(slider.value) - 1;
    const x = scaleX(index, values.length), y = scaleY(values[index]);
    cursor.setAttribute("x1", x); cursor.setAttribute("x2", x); dot.setAttribute("cx", x); dot.setAttribute("cy", y);
    $("feature-readout").textContent = index + 1;
    $("value-readout").textContent = values[index].toFixed(6);
    slider.setAttribute("aria-valuetext", `${index + 1} 个特征，${metric.toUpperCase()} ${values[index].toFixed(6)}`);
  }
  function updateChart() {
    const config = data.configurations[Number(select.value)];
    const values = config[metric];
    const lower = metric === "mcc" && Math.min(...values) < 0 ? -1 : 0;
    scaleY = value => 300 - (value - lower) * 280 / (1 - lower);
    grid.replaceChildren();
    for (let i = 0; i <= 5; i += 1) {
      const value = lower + i * (1 - lower) / 5, y = scaleY(value);
      grid.append(svgNode("line", {x1: 60, x2: 970, y1: y, y2: y, stroke: "#deded4"}), svgNode("text", {x: 45, y: y + 4, "text-anchor": "end"}, value.toFixed(1)));
    }
    [1, 200, 400, 600, 800, values.length].forEach(value => grid.append(svgNode("text", {x: scaleX(value - 1, values.length), y: 328, "text-anchor": "middle"}, value)));
    grid.append(svgNode("text", {x: 60, y: 12}, metric.toUpperCase()), svgNode("text", {x: 970, y: 355, "text-anchor": "end"}, "特征数量 / NUMBER OF FEATURES"));
    line.setAttribute("d", values.map((value, i) => `${i === 0 ? "M" : "L"}${scaleX(i, values.length).toFixed(2)} ${scaleY(value).toFixed(2)}`).join(" "));
    const peak = Math.max(...values), index = values.indexOf(peak);
    $("peak-readout").textContent = `历史曲线最高点 F${index + 1} / ${peak.toFixed(6)}`;
    $("metric-name").textContent = metric.toUpperCase();
    $("chart-label").textContent = `SNN / ${config.layers} 层 / ${config.units} 单元 / LR ${config.rate}`;
    $("curve-svg-title").textContent = `SNN / ${config.label} / ${metric.toUpperCase()}`;
    $("log-source").href = config.source[metric];
    slider.max = values.length;
    if (Number(slider.value) > values.length) slider.value = String(values.length);
    updatePoint();
  }
  select.addEventListener("change", updateChart);
  slider.addEventListener("input", updatePoint);
  chart.addEventListener("pointermove", event => {
    if (event.pointerType === "touch") return;
    const rect = chart.getBoundingClientRect();
    const x = (event.clientX - rect.left) * 1000 / rect.width;
    slider.value = String(Math.max(1, Math.min(Number(slider.max), Math.round((x - 60) / 910 * (Number(slider.max) - 1)) + 1)));
    updatePoint();
  });
  document.querySelectorAll("[data-metric]").forEach(button => button.addEventListener("click", () => {
    metric = button.dataset.metric;
    document.querySelectorAll("[data-metric]").forEach(other => { const active = other === button; other.classList.toggle("active", active); other.setAttribute("aria-pressed", String(active)); });
    updateChart();
  }));
  updateChart();
})();
