# 📧 Email Spam Classification

Dự án xây dựng hệ thống **phân loại email spam / không spam** dựa trên nội dung văn bản.

---

## 🎯 Mục tiêu

* Xây dựng pipeline xử lý email từ dữ liệu thô
* Phân tích dữ liệu (EDA) để hiểu đặc trưng spam
* Chuẩn hóa dữ liệu cho mô hình học máy
* Phát triển mô hình phân loại (Machine Learning / Deep Learning)

---

## 📊 Trạng thái hiện tại

### ✅ Đã hoàn thành

* Dataset raw và processed
* Notebook EDA để khám phá dữ liệu
* Phân tích phân bố nhãn (spam / ham)
* Kiểm tra dữ liệu trùng lặp và nhiễu

### 🚧 Đang thực hiện

* Data cleaning và chuẩn hóa dữ liệu
* Xây dựng pipeline tiền xử lý email

### ❌ Chưa thực hiện

* Huấn luyện model
* Xây dựng baseline
* Đánh giá mô hình (accuracy, F1, ROC-AUC)

---

## 📦 Phạm vi hiện tại

* Quản lý dataset (raw & processed)
* Phân tích dữ liệu (EDA)
* Kiểm tra chất lượng dữ liệu:

  * Missing values
  * Duplicate samples
  * Noise (HTML, ký tự đặc biệt, URL)
* Chuẩn bị cho:

  * Train / Validation / Test split
  * Baseline model

---

## 🗂️ Dữ liệu

| Loại              | Đường dẫn                                               |
| ----------------- | ------------------------------------------------------- |
| Raw dataset       | `datasets/raw/email_dataset_github.csv`                 |
| Processed dataset | `datasets/processed/email_dataset_github_processed.csv` |
| EDA notebook      | `notebooks/EDA/`                                        |

---

## 🏗️ Cấu trúc dự án

```id="1m4p6r"
.
├── config/                # Cấu hình xử lý dữ liệu
├── datasets/
│   ├── raw/               # Dữ liệu gốc
│   └── processed/         # Dữ liệu đã xử lý
├── notebooks/
│   └── EDA/               # Notebook phân tích dữ liệu
├── src/
│   └── data/              # Module xử lý & tiền xử lý dữ liệu
└── README.md
```

---

## ⚙️ Hướng dẫn chạy

### 1. Cài đặt môi trường

```id="a7p3t2"
pip install -r requirements.txt
```

---

### 2. Chạy EDA

```id="g0w5k9"
jupyter notebook notebooks/EDA/
```

---

### 3. Chạy script xử lý dữ liệu (nếu có)

```id="l9x2v1"
python -m src.data.preprocessing
```

---

## 🧠 Pipeline (dự kiến)

```id="k4c9vn"
Raw Email
   ↓
Cleaning (remove HTML, URL, noise)
   ↓
Tokenization
   ↓
Vectorization (TF-IDF / Word2Vec)
   ↓
Model (Logistic Regression / LSTM)
   ↓
Evaluation
```

---

## 🔍 Định hướng mô hình

### Baseline (truyền thống)

* Logistic Regression
* Naive Bayes
* SVM

### Deep Learning (dự kiến)

* LSTM
* BiLSTM
* Attention-based models

---

## 📊 Metrics dự kiến

* Accuracy
* Precision / Recall
* F1-score
* ROC-AUC

---

## 🚀 Hướng phát triển tiếp theo

* [ ] Hoàn thiện data cleaning
* [ ] Chuẩn hóa pipeline preprocessing
* [ ] Tách train / validation / test
* [ ] Xây dựng baseline model
* [ ] Đánh giá và so sánh mô hình
* [ ] Triển khai demo UI (LSTM Visualization)

---

## 💡 Ghi chú

* Dataset hiện tại có thể chứa:

  * Nội dung HTML
  * URL spam
  * Ký tự đặc biệt
* Cần chuẩn hóa kỹ trước khi huấn luyện model

---

## 👨‍💻 Author

* Huỳnh Mai

---
