"use strict";

const $ = (id) => document.getElementById(id);
const state = { report: null, selected: null, camera: "agentview", filtered: [], request: 0 };
const number = (n) => n === null || n === undefined ? "—" : Number(n).toLocaleString("en-US");
const seconds = (n) => n === null || n === undefined ? "—" : Number(n).toLocaleString("en-US", { maximumFractionDigits: 2 });
const clock = (ms) => ms === null || ms === undefined ? "未记录时间" : new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Shanghai", hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3, hour12: false }).format(ms);
const json = (value) => typeof value === "string" ? value : JSON.stringify(value, null, 2);

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = text;
  return element;
}

function inlineText(parent, text) {
  for (const part of text.split(/(`[^`\n]+`|\*\*[^*]+\*\*)/g)) {
    if (part.startsWith("`") && part.endsWith("`")) parent.append(node("code", "", part.slice(1, -1)));
    else if (part.startsWith("**") && part.endsWith("**")) parent.append(node("strong", "", part.slice(2, -2)));
    else parent.append(document.createTextNode(part));
  }
}

function publicCopy(text) {
  const container = node("div", "public-copy");
  for (const block of text.trim().split(/\n\s*\n/)) {
    const lines = block.split("\n");
    if (lines.length >= 2 && lines[0].startsWith("|") && /^\|[\s:|\-]+\|$/.test(lines[1])) {
      const table = node("table", "public-table");
      lines.filter((_, index) => index !== 1).forEach((line, index) => {
        const row = node("tr");
        line.split("|").slice(1, -1).forEach((cell) => { const td = node(index ? "td" : "th"); inlineText(td, cell.trim()); row.append(td); });
        table.append(row);
      });
      const scroll = node("div", "table-scroll"); scroll.append(table); container.append(scroll);
    } else {
      const paragraph = node("p"); inlineText(paragraph, block); container.append(paragraph);
    }
  }
  return container;
}

function disclosure(label, value, open = false) {
  const details = node("details"); details.open = open;
  details.append(node("summary", "", label), node("pre", "code-block", json(value)));
  return details;
}

function asset(path) {
  return "/api/asset?" + new URLSearchParams({ run: state.report.run_id, path });
}

function metric(id, value, unit = "") {
  $(id).replaceChildren(document.createTextNode(value));
  if (unit && value !== "—") $(id).append(node("small", "", unit));
}

async function openPath(path, selectFirst = true) {
  const request = ++state.request;
  $("open-button").disabled = true;
  $("open-button").textContent = "读取中…";
  $("error").hidden = true;
  try {
    const response = await fetch("/api/open", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }) });
    const data = await response.json();
    if (request !== state.request) return;
    if (!response.ok) throw new Error(data.error || "读取失败，请检查目录。");
    if (data.runs) {
      $("runs").hidden = false;
      $("run-select").replaceChildren(...data.runs.map((run) => {
        const option = node("option", "", `${run.name}${run.success === true ? " · 成功" : ""}`);
        option.value = run.path; return option;
      }));
      localStorage.setItem("run-viewer:path", data.path);
      if (selectFirst) await openPath(data.runs[0].path, false);
    } else {
      state.report = data.report;
      localStorage.setItem("run-viewer:path", $("path-input").value);
      showReport();
    }
  } catch (error) {
    if (request === state.request) { $("error").textContent = error.message; $("error").hidden = false; }
  } finally {
    if (request === state.request) {
      $("open-button").disabled = false;
      $("open-button").replaceChildren(document.createTextNode("打开目录 "), node("span", "", "↗"));
    }
  }
}

function showReport() {
  const report = state.report;
  $("report").hidden = false; $("empty").hidden = true; $("reload").hidden = false;
  $("run-title").textContent = report.name;
  $("run-subtitle").textContent = [report.task, report.model, report.backend === "openpi" ? "π0.5" : report.backend.toUpperCase()].filter(Boolean).join("  /  ");
  $("run-status").textContent = ({ success: "环境确认成功", running: "运行中", finished: "已结束" })[report.status] || "已读取";
  $("run-status").classList.toggle("neutral", report.status !== "success");
  const m = report.metrics, t = m.tokens;
  metric("elapsed", seconds(m.elapsed_s), "s");
  $("elapsed-note").textContent = m.wall_elapsed_s !== null ? `RPent 计时 · 含启动约 ${seconds(m.wall_elapsed_s)} s` : "RPent 原始计时";
  metric("action-time", seconds(m.action_video_s), "s");
  $("action-note").textContent = m.action_frames !== null ? `${number(m.action_frames)} 帧仿真 · 工具实耗 ${seconds(m.action_execution_s)} s` : `视频未保存 · 工具实耗 ${seconds(m.action_execution_s)} s`;
  metric("tool-count", number(m.tool_calls), "次");
  $("tool-note").textContent = `${number(m.turns)} 个 Turn · 本地命令与 MCP 均计入`;
  metric("token-count", number(t.input_tokens + t.output_tokens));
  $("token-note").textContent = `输入 ${number(t.input_tokens)} · 输出 ${number(t.output_tokens)}`;
  $("token-count").title = `缓存输入 ${number(t.cached_input_tokens)}（已包含在输入中）\nReasoning ${number(t.reasoning_output_tokens)}（已包含在输出中）`;
  $("warnings-box").hidden = !report.warnings.length;
  $("warnings").replaceChildren(...report.warnings.map((text) => node("li", "", text)));
  const fullVideo = $("full-video");
  fullVideo.pause();
  fullVideo.removeAttribute("src");
  $("full-recording").hidden = !report.video;
  if (report.video) {
    fullVideo.src = asset(report.video.path);
    $("full-recording-note").textContent = `${seconds(report.video.duration_s)} s · ${number(report.video.frames)} 帧`;
  }
  fullVideo.load();
  $("search").value = "";
  state.selected = null;
  renderList();
  const hash = new URLSearchParams(location.search).get("path") === report.path ? location.hash.slice(1) : "";
  const first = report.events.find((event) => event.id === hash) || report.events[0];
  if (first) selectEvent(first.id, false);
  else $("detail").replaceChildren(node("p", "no-matches", "这轮没有保存可显示的事件。"));
}

function preview(event) {
  if (event.text) return event.title === "提交初始提示词" ? `${number(event.text.length)} 字符原始输入` : event.text.trim();
  if (event.public_before && event.tool) return event.public_before.trim();
  if (event.usage) return `输入 ${number(event.usage.input_tokens)} · 输出 ${number(event.usage.output_tokens)}`;
  if (event.arguments?.command) return Array.isArray(event.arguments.command) ? event.arguments.command.join(" ") : event.arguments.command;
  return event.arguments ? JSON.stringify(event.arguments) : "";
}

function renderList() {
  if (!state.report) return;
  const term = $("search").value.trim().toLowerCase();
  state.filtered = state.report.events.filter((event) => !term || `${event.title} ${event.turn ? "turn " + event.turn : ""} ${event.text || ""} ${json(event.arguments) || ""}`.toLowerCase().includes(term));
  $("event-count").textContent = `${state.filtered.length} / ${state.report.events.length} 条事件`;
  const fragment = document.createDocumentFragment();
  for (const event of state.filtered) {
    const wrapper = node("div"); wrapper.setAttribute("role", "listitem");
    const button = node("button", "event-row" + (event.id === state.selected ? " active" : ""));
    button.type = "button"; button.dataset.event = event.id;
    if (event.id === state.selected) button.setAttribute("aria-current", "true");
    const meta = node("span", "event-meta");
    meta.append(node("span", "", (event.precision === "adjacent" ? "≈ " : "") + clock(event.at_ms)), node("span", "", event.turn ? `Turn ${String(event.turn).padStart(2, "0")}` : ""));
    button.append(meta, node("span", "event-title", event.title));
    const text = preview(event);
    if (text) button.append(node("span", "event-preview", text));
    button.addEventListener("click", () => selectEvent(event.id));
    button.addEventListener("keydown", (key) => {
      if (!["ArrowDown", "ArrowUp"].includes(key.key)) return;
      key.preventDefault(); moveSelection(key.key === "ArrowDown" ? 1 : -1, true);
    });
    wrapper.append(button); fragment.append(wrapper);
  }
  $("event-list").replaceChildren(fragment);
  if (!state.filtered.length) $("event-list").append(node("div", "no-matches", "没有匹配的事件"));
}

function moveSelection(delta, focus = false) {
  const index = state.filtered.findIndex((event) => event.id === state.selected);
  const next = state.filtered[index + delta];
  if (next) {
    selectEvent(next.id);
    if (focus) $("event-list").querySelector(`[data-event="${next.id}"]`)?.focus();
  }
}

function selectEvent(id, scroll = true) {
  const event = state.report.events.find((entry) => entry.id === id);
  if (!event) return;
  state.selected = id;
  $("event-list").querySelectorAll("button").forEach((button) => {
    const active = button.dataset.event === id;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "true"); else button.removeAttribute("aria-current");
    if (active && scroll) {
      const list = $("event-list"), row = button.getBoundingClientRect(), bounds = list.getBoundingClientRect();
      if (row.top < bounds.top) list.scrollTop += row.top - bounds.top;
      else if (row.bottom > bounds.bottom) list.scrollTop += row.bottom - bounds.bottom;
    }
  });
  history.replaceState(null, "", "/?" + new URLSearchParams({ path: state.report.path }) + "#" + id);
  const detail = $("detail"); detail.replaceChildren(); detail.scrollTop = 0;
  const top = node("div", "detail-top");
  top.append(node("span", "detail-turn", event.turn ? `TURN ${String(event.turn).padStart(2, "0")}` : "RUN / 启动与记录"));
  const index = state.filtered.findIndex((entry) => entry.id === id);
  const nav = node("div", "detail-nav"); nav.append(node("span", "", `${index + 1} / ${state.filtered.length}`));
  for (const [label, delta] of [["←", -1], ["→", 1]]) {
    const button = node("button", "", label); button.type = "button"; button.disabled = !state.filtered[index + delta];
    button.setAttribute("aria-label", delta < 0 ? "上一个事件" : "下一个事件"); button.addEventListener("click", () => moveSelection(delta)); nav.append(button);
  }
  top.append(nav); detail.append(top, node("h3", "", event.title));
  const time = node("div", "event-timing");
  time.append(node("span", "", clock(event.at_ms)), node("span", "", event.offset_s === null ? "时间未单独记录" : `启动后 +${seconds(event.offset_s)} s`));
  if (event.duration_ms !== undefined && event.duration_ms !== null) time.append(node("span", "", `用时 ${seconds(event.duration_ms / 1000)} s`));
  if (event.status) time.append(node("span", event.status === "failed" ? "failed" : "", ({ completed: "已完成", failed: "调用失败", inProgress: "进行中" })[event.status] || event.status));
  if (event.precision === "adjacent") time.title = "token 通知没有独立时间戳，此处按邻近事件时间定位。";
  if (event.precision === "second") time.title = "此启动/结束日志的时间精度为秒。";
  detail.append(time);

  if (event.text) {
    if (event.title === "提交初始提示词") detail.append(disclosure(`完整初始输入 · ${number(event.text.length)} 字符`, event.text));
    else if (event.turn) detail.append(publicCopy(event.text));
    else detail.append(disclosure("原始启动日志", event.text));
  }
  if (event.public_before && event.tool) {
    const section = node("section", "detail-section"); section.append(node("div", "detail-label", "最近公开说明"), publicCopy(event.public_before)); detail.append(section);
  }
  if (event.usage) {
    const grid = node("div", "usage-grid");
    for (const [label, key] of [["本 Turn 输入", "input_tokens"], ["本 Turn 输出", "output_tokens"], ["其中缓存输入", "cached_input_tokens"], ["其中 Reasoning", "reasoning_output_tokens"]]) {
      const cell = node("div", "", label); cell.append(node("b", "", number(event.usage[key]))); grid.append(cell);
    }
    detail.append(grid, node("p", "image-note", "缓存已包含在输入中；Reasoning 已包含在输出中。此通知不提供可读的内部推理。"));
  }
  if (event.arguments && Object.values(event.arguments).some((value) => value !== null)) {
    const args = { ...event.arguments };
    if (Array.isArray(args.command)) args.command = args.command.join(" ");
    if (event.tool && JSON.stringify(args).length < 1400) {
      const section = node("section", "detail-section"); section.append(node("div", "detail-label", "调用参数"), node("pre", "code-block", json(args))); detail.append(section);
    } else detail.append(disclosure(event.tool ? "完整调用参数" : "启动命令与配置", args));
  }
  const hasImages = event.before_state !== null && event.before_state !== undefined || event.after_state !== null && event.after_state !== undefined;
  if (hasImages) {
    const imageArea = node("section", "detail-section");
    renderImages(imageArea, event); detail.append(imageArea);
  }
  if (event.result !== undefined) detail.append(disclosure("完整返回结果", event.result));
  if (event.state_result) detail.append(disclosure("环境动作记录", { result: event.state_result, terminated: event.native_success, truncated: event.truncated, execution_s: event.execution_s }));
  if (event.source) detail.append(node("div", "source-note", `${event.source.file}${event.source.line ? ` · 第 ${event.source.line} 行` : ""}`));
}

function renderImages(area, event) {
  area.replaceChildren();
  const pair = Boolean(event.tool);
  const heading = node("div", "detail-label", pair ? "调用前后" : "此时的画面");
  const cameraSwitch = node("div", "camera-switch");
  for (const [camera, label] of [["agentview", "主相机"], ["wrist", "腕部"]]) {
    const button = node("button", camera === state.camera ? "active" : "", label); button.type = "button";
    button.setAttribute("aria-pressed", String(camera === state.camera)); button.addEventListener("click", () => { state.camera = camera; renderImages(area, event); }); cameraSwitch.append(button);
  }
  heading.append(cameraSwitch); area.append(heading);
  const photos = node("div", pair ? "image-pair" : "single-image");
  const entries = pair ? [["调用前", event.before_state], ["调用后", event.after_state]] : [["保存的观测", event.after_state ?? event.before_state]];
  for (const [label, index] of entries) {
    const figure = node("figure");
    const caption = node("figcaption"); caption.append(node("span", "", label), node("span", "", index === null || index === undefined ? "无记录" : `观测 ${String(index).padStart(2, "0")}`)); figure.append(caption);
    const observation = state.report.observations[String(index)];
    const path = observation?.images[state.camera];
    if (path) {
      const button = node("button", "image-button"); button.type = "button";
      const image = node("img"); image.src = asset(path); image.alt = `${label} · ${state.camera === "agentview" ? "主相机" : "腕部"} · 观测 ${index}`;
      image.loading = "lazy"; button.append(image); button.addEventListener("click", () => {
        $("full-image").src = image.src; $("full-image").alt = image.alt; $("full-image-label").textContent = image.alt; $("image-dialog").showModal();
      }); figure.append(button);
    } else figure.append(node("div", "image-missing", "该位置未保存观测图片"));
    photos.append(figure);
  }
  area.append(photos);
  if (pair && event.before_state === event.after_state) area.append(node("div", "image-note", "该工具没有生成新的观测，两侧展示同一份已保存画面。"));
  if (pair && event.after_state === null) area.append(node("div", "image-note", "调用后没有可核对的观测，未推测执行结果。"));
  if (event.reference_state !== undefined && event.reference_state !== event.after_state) area.append(node("div", "image-note", `此工具读取的是历史观测 ${event.reference_state}，没有改变当前场景。`));
  if (state.report.video && event.video_range) {
    const play = node("button", "video-button", `▷ 播放对应动作 · ${seconds(event.video_range.end_s - event.video_range.start_s)} s`); play.type = "button";
    play.addEventListener("click", () => {
      let video = area.querySelector("video");
      if (!video) {
        const shell = node("div", "video-shell"); video = node("video"); video.controls = true; video.preload = "metadata"; video.src = asset(state.report.video.path);
        video.addEventListener("timeupdate", () => { if (video.currentTime >= event.video_range.end_s) video.pause(); }); shell.append(video); area.append(shell);
      }
      const start = () => { video.currentTime = event.video_range.start_s; video.play().catch(() => {}); };
      if (video.readyState >= 1) start(); else video.addEventListener("loadedmetadata", start, { once: true });
    });
    area.append(play);
  }
}

$("path-form").addEventListener("submit", (event) => { event.preventDefault(); $("runs").hidden = true; openPath($("path-input").value); });
$("load-run").addEventListener("click", () => openPath($("run-select").value, false));
$("run-select").addEventListener("change", () => openPath($("run-select").value, false));
$("reload").addEventListener("click", () => state.report && openPath(state.report.path, false));
$("search").addEventListener("input", () => {
  renderList();
  const event = state.filtered.find((entry) => entry.id === state.selected) || state.filtered[0];
  if (event) selectEvent(event.id, false);
  else $("detail").replaceChildren(node("p", "no-matches", "没有匹配的事件"));
});
$("close-image").addEventListener("click", () => $("image-dialog").close());
$("image-dialog").addEventListener("click", (event) => { if (event.target === $("image-dialog")) $("image-dialog").close(); });

fetch("/api/bootstrap").then((response) => response.json()).then((data) => {
  const chosen = new URLSearchParams(location.search).get("path") || localStorage.getItem("run-viewer:path") || data.default_path;
  $("path-input").value = chosen;
  openPath(chosen);
}).catch(() => { $("error").textContent = "本地读取服务未连接，请重新启动 run-viewer.sh。"; $("error").hidden = false; });
