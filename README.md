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

## 3. Chạy dashboard HTML/CSS/JavaScript

Chạy giao diện dashboard mới (phục vụ HTML và API dự đoán tại cùng một địa chỉ):

```powershell
python server.py
```

Mở `http://127.0.0.1:8000`. JavaScript gửi nội dung và ngưỡng tới `POST /api/predict`; server dùng lại `NewsClassifier` và các artifact đã huấn luyện. Có thể đổi địa chỉ bind/cổng bằng `--host` và `--port`.

Các trang Semantic Labels và Kết quả thực nghiệm tự hiển thị trạng thái chưa có dữ liệu nếu artifact tùy chọn tương ứng chưa được tạo.

## 4. Chạy giao diện Streamlit hiện có

Giao diện Streamlit cũ vẫn có thể chạy độc lập:

```powershell
streamlit run app.py
```

Sau đó mở địa chỉ Streamlit hiện trong terminal, thường là `http://localhost:8501`.

## 5. Artifact bắt buộc

Chỉ cần 4 file để chạy trang dự đoán:

- `word2vec_lstm.keras`
- `vocab.json`
- `labels.json`
- `label_to_group.json`

Các CSV/NPY còn lại phục vụ trang giải thích và thực nghiệm.

## 6. Pipeline inference

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
