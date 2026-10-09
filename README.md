# Vietnamese News AI — Word2Vec + LSTM

Web demo cho mô hình phân loại văn bản tiếng Việt đa nhãn.

## 1. Cấu trúc thư mục

```text
vietnamese-news-classifier/
├── app.py
├── inference.py
├── preprocessing.py
├── semantic_groups.py
├── visualization.py
├── requirements.txt
├── README.md
└── artifacts/
    ├── word2vec_lstm.keras
    ├── vocab.json
    ├── labels.json
    ├── label_to_group.json
    ├── cluster_names.json                 # tùy chọn
    ├── label_centroids.npy                # tùy chọn
    ├── label_cluster_assignments.csv      # tùy chọn
    ├── model_comparison.csv               # tùy chọn
    └── raw_vs_filtered_metrics.csv        # tùy chọn
```

## 2. Cài đặt trên Windows

Mở PowerShell tại thư mục project:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Chạy web

```powershell
streamlit run app.py
```

Sau đó mở địa chỉ Streamlit hiện trong terminal, thường là `http://localhost:8501`.

## 4. Artifact bắt buộc

Chỉ cần 4 file để chạy trang dự đoán:

- `word2vec_lstm.keras`
- `vocab.json`
- `labels.json`
- `label_to_group.json`

Các CSV/NPY còn lại phục vụ trang giải thích và thực nghiệm.

## 5. Pipeline inference

```text
Raw Vietnamese text
        ↓
Unicode NFC + lowercase + punctuation removal
        ↓
TextVectorization (vocabulary từ training set)
        ↓
Word2Vec-initialized Embedding
        ↓
LSTM(128)
        ↓
Dense(88) + sigmoid
        ↓
threshold = 0.30
        ↓
fine-grained labels
        ↓
label_to_group.json
        ↓
semantic topic groups
```

Web không train lại model.
