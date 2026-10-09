from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from inference import DEFAULT_THRESHOLD, NewsClassifier
from visualization import load_csv, load_label_semantic_data

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

st.set_page_config(
    page_title="Vietnamese News Intelligence",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --bg: #f4f7fb;
    --panel: rgba(255, 255, 255, 0.9);
    --panel-strong: #ffffff;
    --border: #dfe8f3;
    --primary: #0f172a;
    --primary-soft: #1d4ed8;
    --accent: #22c55e;
    --accent-soft: #dcfce7;
    --muted: #52607a;
    --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #edf5ff 0%, #f8fafc 100%);
    color: var(--primary);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    color: #f8fafc;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
    color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 1.6rem 1.8rem;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #2563eb 100%);
    box-shadow: var(--shadow);
    margin-bottom: 1.3rem;
}

.hero::after {
    content: "";
    position: absolute;
    inset: 0 0 auto auto;
    width: 220px;
    height: 220px;
    background: rgba(255,255,255,0.1);
    border-radius: 50%;
    transform: translate(35%, -25%);
}

.hero-badge {
    display: inline-block;
    padding: .35rem .7rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.18);
    color: #f8fafc;
    border: 1px solid rgba(255,255,255,0.2);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: .8rem;
}

.hero h1 {
    position: relative;
    margin: 0;
    font-size: clamp(2rem, 4vw, 3rem);
    color: #f8fafc;
    line-height: 1.1;
}

.hero p {
    position: relative;
    margin: .55rem 0 0;
    color: rgba(248,250,252,0.8);
    font-size: 1.02rem;
}

.group-card {
    padding: 1rem 1.1rem;
    border-radius: 14px;
    border: 1px solid rgba(37, 99, 235, 0.15);
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.08);
    margin: .4rem 0;
    font-weight: 600;
    color: #1e3a8a;
}

.label-card {
    padding: .8rem 1rem;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--panel-strong);
    margin: .35rem 0;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
}

.info-panel {
    padding: 1rem 1.2rem;
    border-radius: 16px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.72);
    box-shadow: 0 8px 24px rgba(148, 163, 184, 0.08);
}

.metric-shell {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
}

.kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
}

.sidebar-callout {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    padding: .7rem .8rem;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_classifier():
    return NewsClassifier(ARTIFACT_DIR)


try:
    clf = get_classifier()
except Exception as exc:
    st.error("Không thể tải model/artifact.")
    st.code(str(exc))
    st.stop()

with st.sidebar:
    st.title("Vietnamese News AI")
    st.markdown(
        '<div class="sidebar-callout"><strong>AI-based newsroom insight</strong><br>Phân loại tin tức theo nhãn và chủ đề theo ngữ nghĩa.</div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Điều hướng", ["Phân loại", "Mô hình", "Semantic Labels", "Kết quả thực nghiệm"])
    st.divider()
    threshold = st.slider("Threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05)
    st.caption("Threshold mặc định 0.30 được chọn trên validation set trong notebook.")

if page == "Phân loại":
    st.markdown(
        '<div class="hero"><div class="hero-badge">AI-powered</div><h1>Vietnamese News Intelligence</h1><p>Phân tích nội dung tin tức theo nhiều chủ đề cùng lúc với mô hình Word2Vec + LSTM.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-panel">Nhập một bài viết tiếng Việt để xem các chủ đề chính, nhãn dự đoán và mức độ tự tin của mô hình.</div>',
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "Nội dung bài viết",
        height=240,
        placeholder="Dán nội dung bài báo tiếng Việt vào đây...",
    )

    if st.button("Phân tích bài viết", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Vui lòng nhập nội dung bài viết.")
            st.stop()

        with st.spinner("Đang phân tích..."):
            result = clf.predict(text, threshold)

        st.divider()
        st.subheader("Kết quả")

        best_probability = max((item["probability"] for item in result["predictions"]), default=0.0)
        summary_cols = st.columns(3)
        summary_cols[0].markdown('<div class="metric-shell"><div class="kpi-value">%d</div><div>Nhóm chủ đề</div></div>' % len(result["groups"]), unsafe_allow_html=True)
        summary_cols[1].markdown('<div class="metric-shell"><div class="kpi-value">%d</div><div>Nhãn đạt threshold</div></div>' % len(result["predictions"]), unsafe_allow_html=True)
        summary_cols[2].markdown('<div class="metric-shell"><div class="kpi-value">%.1f%%</div><div>Độ tin cao nhất</div></div>' % (best_probability * 100), unsafe_allow_html=True)

        if result["groups"]:
            st.markdown("**Nhóm chủ đề**")
            cols = st.columns(min(3, len(result["groups"])))
            for col, group in zip(cols, result["groups"]):
                col.markdown(f'<div class="group-card"><b>{group}</b></div>', unsafe_allow_html=True)
        else:
            st.info("Chưa có nhóm chủ đề nào đạt threshold.")

        st.markdown("**Nhãn dự đoán**")
        if result["predictions"]:
            for item in result["predictions"]:
                c1, c2 = st.columns([5, 1])
                c1.markdown(f'<div class="label-card"><b>{item["label"]}</b></div>', unsafe_allow_html=True)
                c2.metric("Xác suất", f'{item["probability"]:.1%}')
                st.progress(item["probability"])
        else:
            st.warning("Không có nhãn nào đạt threshold hiện tại.")

        st.subheader("Top nhãn")
        top = pd.DataFrame(clf.top_predictions(result["probabilities"], 10))
        top["probability"] = top["probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(top, use_container_width=True, hide_index=True)

        with st.expander("Chi tiết tiền xử lý"):
            st.write(f"Số token sau tiền xử lý: {result['token_count']:,}")
            st.code(result["clean_text"][:5000])

elif page == "Mô hình":
    st.title("Mô hình và pipeline")
    st.markdown(
        '<div class="info-panel">Pipeline inference: từ văn bản đầu vào đến nhóm chủ đề cuối cùng, tất cả đều được xử lý trong mô hình đã huấn luyện sẵn.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
**Pipeline inference**

`Văn bản → Unicode NFC/lowercase → TextVectorization → Word2Vec-initialized Embedding → LSTM → Sigmoid → 88 nhãn → threshold → semantic groups`

Website chỉ thực hiện inference với model đã huấn luyện; không huấn luyện lại khi người dùng dự đoán.
""")
    cols = st.columns(4)
    cols[0].markdown(f'<div class="metric-shell"><div class="kpi-value">{len(clf.labels)}</div><div>Số nhãn</div></div>', unsafe_allow_html=True)
    cols[1].markdown('<div class="metric-shell"><div class="kpi-value">100-D</div><div>Word vector</div></div>', unsafe_allow_html=True)
    cols[2].markdown('<div class="metric-shell"><div class="kpi-value">128</div><div>LSTM</div></div>', unsafe_allow_html=True)
    cols[3].markdown(f'<div class="metric-shell"><div class="kpi-value">{clf.sequence_length}</div><div>Sequence length</div></div>', unsafe_allow_html=True)

    st.subheader("Multi-label prediction")
    st.write("Mỗi nhãn có một xác suất độc lập. Một bài viết có thể đồng thời nhận nhiều nhãn.")
    st.subheader("Semantic Label Aggregation")
    st.write("Các nhãn vượt threshold được ánh xạ qua label_to_group.json để tạo nhóm chủ đề dễ đọc. Việc này không thay thế 88 nhãn gốc.")

elif page == "Semantic Labels":
    st.title("Semantic Label Space")
    st.markdown(
        '<div class="info-panel">Không gian nhãn ngữ nghĩa cho phép thấy cách các chủ đề liên quan tới nhau trong không gian vector và các cụm phân nhóm tương ứng.</div>',
        unsafe_allow_html=True,
    )
    centroids, assignments = load_label_semantic_data(ARTIFACT_DIR)
    if centroids is None:
        st.info("Chưa có đủ artifact để hiển thị semantic map. Cần label_centroids.npy và label_cluster_assignments.csv.")
    else:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        points = PCA(n_components=2, random_state=42).fit_transform(centroids)
        assignments = assignments.copy()
        assignments["x"] = points[:, 0]
        assignments["y"] = points[:, 1]

        fig, ax = plt.subplots(figsize=(11, 7))
        for cluster_id, g in assignments.groupby("cluster_id"):
            ax.scatter(g["x"], g["y"], label=f"Nhóm {cluster_id}", alpha=.8)
            for _, row in g.iterrows():
                ax.annotate(str(row["label"]), (row["x"], row["y"]), fontsize=7, alpha=.8)
        ax.set_title("2D visualization of label semantic space")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend(fontsize=8, ncol=2)
        st.pyplot(fig)
        st.caption("PCA ở đây chỉ dùng để trực quan hóa; không dùng khoảng cách 2D làm tiêu chí phân nhóm.")

elif page == "Kết quả thực nghiệm":
    st.title("Kết quả thực nghiệm")
    comparison = load_csv(ARTIFACT_DIR / "model_comparison.csv")
    metrics = load_csv(ARTIFACT_DIR / "raw_vs_filtered_metrics.csv")

    if comparison is not None:
        st.subheader("So sánh mô hình")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa tìm thấy model_comparison.csv")

    if metrics is not None:
        st.subheader("Raw vs. filtered metrics")
        st.dataframe(metrics, use_container_width=True, hide_index=True)

st.divider()
st.caption("Research demo — Vietnamese multi-label text classification with Word2Vec + LSTM")
