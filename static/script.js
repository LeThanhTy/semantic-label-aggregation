"use strict";

const elements = {
    article: document.querySelector("#article-text"),
    characterCount: document.querySelector("#character-count"),
    clearButton: document.querySelector("#clear-button"),
    error: document.querySelector("#form-error"),
    form: document.querySelector("#predict-form"),
    analyzeButton: document.querySelector("#analyze-button"),
    threshold: document.querySelector("#threshold"),
    thresholdValue: document.querySelector("#threshold-value"),
    results: document.querySelector("#results-section"),
    resultMetrics: document.querySelector("#result-metrics"),
    resultGroups: document.querySelector("#result-groups"),
    resultPredictions: document.querySelector("#result-predictions"),
    topPredictions: document.querySelector("#top-predictions"),
    tokenCount: document.querySelector("#token-count"),
    cleanText: document.querySelector("#clean-text"),
    serviceStatus: document.querySelector("#service-status"),
    modelMetrics: document.querySelector("#model-metrics"),
    labelSearch: document.querySelector("#label-search"),
    groupFilter: document.querySelector("#group-filter"),
    labelCount: document.querySelector("#label-count"),
    labelExplorer: document.querySelector("#label-explorer"),
    semanticMap: document.querySelector("#semantic-map-content"),
    modelComparison: document.querySelector("#model-comparison-content"),
    rawMetrics: document.querySelector("#raw-metrics-content")
};

const state = {
    config: null,
    labels: [],
    semanticLoaded: false,
    experimentsLoaded: false
};

const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
const formatThreshold = (value) => Number(value).toLocaleString("vi-VN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
});

async function requestJson(url, options) {
    const response = await fetch(url, options);
    let payload;
    try {
        payload = await response.json();
    } catch {
        throw new Error(`Máy chủ trả về dữ liệu không hợp lệ (HTTP ${response.status}).`);
    }
    if (!response.ok) {
        throw new Error(payload.error || `Yêu cầu thất bại (HTTP ${response.status}).`);
    }
    return payload;
}

function setText(node, value) {
    node.textContent = value;
}

function updateInputDetails() {
    setText(elements.characterCount, `${elements.article.value.length.toLocaleString("vi-VN")} / 50.000 ký tự`);
    setText(elements.thresholdValue, formatThreshold(elements.threshold.value));
}

function showFormError(message) {
    setText(elements.error, message);
    elements.error.hidden = false;
}

function createMetric(value, label) {
    const card = document.createElement("div");
    card.className = "metric-card";
    const strong = document.createElement("strong");
    strong.textContent = value;
    const caption = document.createElement("span");
    caption.textContent = label;
    card.append(strong, caption);
    return card;
}

function renderPredictions(result) {
    const groups = Array.isArray(result.groups) ? result.groups : [];
    const predictions = Array.isArray(result.predictions) ? result.predictions : [];
    const topPredictions = Array.isArray(result.top_predictions) ? result.top_predictions : [];
    const bestProbability = predictions.reduce((best, item) => Math.max(best, item.probability), 0);

    elements.resultMetrics.replaceChildren(
        createMetric(String(groups.length), "Nhóm chủ đề"),
        createMetric(String(predictions.length), "Nhãn đạt ngưỡng"),
        createMetric(formatPercent(bestProbability), "Độ tin cậy cao nhất")
    );

    elements.resultGroups.replaceChildren();
    if (groups.length === 0) {
        const empty = document.createElement("p");
        empty.className = "empty-inline";
        empty.textContent = "Chưa có nhóm chủ đề nào đạt ngưỡng này.";
        elements.resultGroups.append(empty);
    } else {
        for (const group of groups) {
            const chip = document.createElement("div");
            chip.className = "group-chip";
            chip.textContent = group;
            elements.resultGroups.append(chip);
        }
    }

    elements.resultPredictions.replaceChildren();
    if (predictions.length === 0) {
        const empty = document.createElement("p");
        empty.className = "empty-inline";
        empty.textContent = "Không có nhãn nào đạt ngưỡng hiện tại.";
        elements.resultPredictions.append(empty);
    } else {
        for (const prediction of predictions) {
            const item = document.createElement("div");
            item.className = "prediction-item";
            const label = document.createElement("strong");
            label.textContent = prediction.label;
            const score = document.createElement("span");
            score.textContent = formatPercent(prediction.probability);
            const track = document.createElement("div");
            track.className = "probability-track";
            const fill = document.createElement("div");
            fill.className = "probability-fill";
            fill.style.width = `${Math.max(0, Math.min(100, prediction.probability * 100))}%`;
            track.append(fill);
            item.append(label, score, track);
            elements.resultPredictions.append(item);
        }
    }

    elements.topPredictions.replaceChildren();
    for (const prediction of topPredictions) {
        const row = document.createElement("tr");
        for (const value of [String(prediction.rank), prediction.label, formatPercent(prediction.probability)]) {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.append(cell);
        }
        elements.topPredictions.append(row);
    }

    setText(elements.tokenCount, `${Number(result.token_count).toLocaleString("vi-VN")} token`);
    setText(elements.cleanText, result.clean_text || "");
    elements.results.hidden = false;
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function handlePrediction(event) {
    event.preventDefault();
    elements.error.hidden = true;
    const text = elements.article.value.trim();
    if (!text) {
        showFormError("Vui lòng nhập nội dung bài viết trước khi phân tích.");
        elements.article.focus();
        return;
    }

    elements.analyzeButton.disabled = true;
    setText(elements.analyzeButton.querySelector("span"), "Đang phân tích...");
    try {
        const result = await requestJson("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text,
                threshold: Number(elements.threshold.value)
            })
        });
        renderPredictions(result);
    } catch (error) {
        showFormError(error.message);
    } finally {
        elements.analyzeButton.disabled = false;
        setText(elements.analyzeButton.querySelector("span"), "Phân tích bài viết");
    }
}

function renderModelMetrics(config) {
    const model = config.model;
    const metrics = [
        [String(model.label_count), "Nhãn phân loại"],
        [`${model.embedding_dimensions}-D`, "Word vector"],
        [String(model.lstm_units), "LSTM units"],
        [String(model.sequence_length), "Sequence length"]
    ];
    elements.modelMetrics.replaceChildren(...metrics.map(([value, label]) => createMetric(value, label)));
}

function renderLabels() {
    const query = elements.labelSearch.value.trim().toLocaleLowerCase("vi");
    const selectedGroup = elements.groupFilter.value;
    const filtered = state.labels.filter((item) =>
        item.label.toLocaleLowerCase("vi").includes(query) &&
        (!selectedGroup || item.group === selectedGroup)
    );
    const grouped = new Map();
    for (const item of filtered) {
        const group = item.group || "Chưa ánh xạ nhóm";
        if (!grouped.has(group)) grouped.set(group, []);
        grouped.get(group).push(item.label);
    }

    elements.labelExplorer.replaceChildren();
    if (filtered.length === 0) {
        const empty = document.createElement("div");
        empty.className = "panel empty-state";
        empty.textContent = "Không tìm thấy nhãn phù hợp.";
        elements.labelExplorer.append(empty);
    } else {
        for (const [group, labels] of grouped) {
            const card = document.createElement("section");
            card.className = "label-group-card";
            const heading = document.createElement("div");
            heading.className = "label-group-heading";
            const title = document.createElement("h3");
            title.textContent = group;
            const count = document.createElement("span");
            count.textContent = `${labels.length} nhãn`;
            heading.append(title, count);
            const chips = document.createElement("div");
            chips.className = "label-chip-list";
            for (const label of labels) {
                const chip = document.createElement("span");
                chip.className = "label-chip";
                chip.textContent = label;
                chips.append(chip);
            }
            card.append(heading, chips);
            elements.labelExplorer.append(card);
        }
    }
    setText(elements.labelCount, `${filtered.length} / ${state.labels.length} nhãn`);
}

function renderTable(container, rows, emptyMessage) {
    container.replaceChildren();
    if (!Array.isArray(rows) || rows.length === 0) {
        const empty = document.createElement("div");
        empty.className = "table-empty";
        empty.textContent = emptyMessage;
        container.append(empty);
        return;
    }

    const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    const table = document.createElement("table");
    const header = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const column of columns) {
        const cell = document.createElement("th");
        cell.textContent = column.replaceAll("_", " ").toLocaleUpperCase("vi");
        headerRow.append(cell);
    }
    header.append(headerRow);
    const body = document.createElement("tbody");
    for (const row of rows) {
        const tableRow = document.createElement("tr");
        for (const column of columns) {
            const cell = document.createElement("td");
            cell.textContent = row[column] ?? "";
            tableRow.append(cell);
        }
        body.append(tableRow);
    }
    table.append(header, body);
    wrapper.append(table);
    container.append(wrapper);
}

async function loadSemanticSpace() {
    if (state.semanticLoaded) return;
    try {
        const data = await requestJson("/api/semantic-space");
        if (!data.available) {
            const missing = data.missing_artifacts.join(", ");
            elements.semanticMap.textContent = `Chưa có bản đồ vector ngữ nghĩa. Cần bổ sung ${missing} trong thư mục artifacts. Danh sách nhãn và nhóm bên trên vẫn được tải từ các artifact hiện có.`;
            state.semanticLoaded = true;
            return;
        }
        renderSemanticSpace(data);
        state.semanticLoaded = true;
    } catch (error) {
        elements.semanticMap.textContent = `Không thể tải bản đồ ngữ nghĩa: ${error.message}`;
    }
}

function renderSemanticSpace(data) {
    const width = 760;
    const height = 310;
    const margin = 30;
    const points = data.points;
    if (!Array.isArray(points) || points.length === 0) {
        elements.semanticMap.textContent = "Artifact không chứa điểm nhãn để hiển thị.";
        return;
    }
    const xs = points.map((point) => Number(point.x));
    const ys = points.map((point) => Number(point.y));
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const x = (value) => margin + ((value - minX) / (maxX - minX || 1)) * (width - margin * 2);
    const y = (value) => height - margin - ((value - minY) / (maxY - minY || 1)) * (height - margin * 2);

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Biểu đồ PCA của không gian nhãn ngữ nghĩa");
    svg.classList.add("scatter-plot");
    const colors = ["#5574e8", "#43a98d", "#de9b4c", "#aa75d4", "#e16f7e", "#49a2c5"];
    const clusterColor = new Map();
    for (const point of points) {
        const cluster = String(point.cluster_id ?? "0");
        if (!clusterColor.has(cluster)) clusterColor.set(cluster, colors[clusterColor.size % colors.length]);
        const circle = document.createElementNS(svg.namespaceURI, "circle");
        circle.setAttribute("cx", String(x(Number(point.x))));
        circle.setAttribute("cy", String(y(Number(point.y))));
        circle.setAttribute("r", "5");
        circle.setAttribute("fill", clusterColor.get(cluster));
        circle.setAttribute("fill-opacity", "0.78");
        const title = document.createElementNS(svg.namespaceURI, "title");
        title.textContent = `${point.label} · Nhóm ${cluster}`;
        circle.append(title);
        svg.append(circle);
    }
    elements.semanticMap.replaceChildren(svg);
}

async function loadExperiments() {
    if (state.experimentsLoaded) return;
    try {
        const data = await requestJson("/api/experiments");
        renderTable(elements.modelComparison, data.model_comparison, "Chưa tìm thấy model_comparison.csv trong thư mục artifacts.");
        renderTable(elements.rawMetrics, data.raw_vs_filtered_metrics, "Chưa tìm thấy raw_vs_filtered_metrics.csv trong thư mục artifacts.");
        state.experimentsLoaded = true;
    } catch (error) {
        elements.modelComparison.textContent = `Không thể tải kết quả: ${error.message}`;
        elements.rawMetrics.textContent = `Không thể tải kết quả: ${error.message}`;
    }
}

function navigate(pageName) {
    const page = document.querySelector(`#page-${pageName}`);
    if (!page) return;
    document.querySelectorAll(".page").forEach((section) => {
        const isActive = section === page;
        section.hidden = !isActive;
        section.classList.toggle("active", isActive);
    });
    document.querySelectorAll(".nav-link").forEach((link) => {
        const isActive = link.dataset.page === pageName;
        link.classList.toggle("active", isActive);
        link.setAttribute("aria-current", isActive ? "page" : "false");
    });
    setText(document.querySelector("#breadcrumb-current"), page.dataset.title);
    if (pageName === "semantic") loadSemanticSpace();
    if (pageName === "experiments") loadExperiments();
}

async function initializeDashboard() {
    try {
        const [config, labelData] = await Promise.all([
            requestJson("/api/config"),
            requestJson("/api/labels")
        ]);
        state.config = config;
        state.labels = labelData.labels;
        elements.threshold.value = String(config.default_threshold);
        setText(elements.serviceStatus, "Mô hình đã sẵn sàng");
        renderModelMetrics(config);

        const groups = [...new Set(state.labels.map((item) => item.group).filter(Boolean))]
            .sort((a, b) => a.localeCompare(b, "vi"));
        for (const group of groups) {
            const option = document.createElement("option");
            option.value = group;
            option.textContent = group;
            elements.groupFilter.append(option);
        }
        renderLabels();
        updateInputDetails();
    } catch (error) {
        setText(elements.serviceStatus, "Không thể kết nối với mô hình");
        showFormError(`Không thể khởi tạo giao diện: ${error.message}`);
        elements.analyzeButton.disabled = true;
    }
}

elements.form.addEventListener("submit", handlePrediction);
elements.article.addEventListener("input", updateInputDetails);
elements.threshold.addEventListener("input", updateInputDetails);
elements.clearButton.addEventListener("click", () => {
    elements.article.value = "";
    elements.article.focus();
    updateInputDetails();
    elements.error.hidden = true;
});
elements.labelSearch.addEventListener("input", renderLabels);
elements.groupFilter.addEventListener("change", renderLabels);
document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => navigate(link.dataset.page));
});

initializeDashboard();
