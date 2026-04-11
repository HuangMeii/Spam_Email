# Email Spam Classification

Dự án phân loại email spam và không spam dựa trên nội dung email.

## Mục tiêu

Xây dựng pipeline phân loại email từ dữ liệu thô, tập trung ở giai đoạn hiện tại vào dataset, EDA và data cleaning.

## Trạng thái hiện tại

- Đã có dataset raw và processed
- Đã có notebook EDA để khám phá dữ liệu
- Đang ở giai đoạn data cleaning và chuẩn hóa dữ liệu
- Chưa có code huấn luyện model
- Chưa có baseline, thử nghiệm hay đánh giá mô hình

## Phạm vi hiện tại

- Quản lý dataset raw và processed
- Phân tích phân bố nhãn spam / không spam
- Kiểm tra chất lượng dữ liệu, trùng lặp và dữ liệu nhiễu
- Chuẩn bị nền tảng cho bước tách tập dữ liệu và chọn baseline model

## Dữ liệu hiện có

- Raw dataset: `datasets/raw/email_dataset_github.csv`
- Processed dataset: `datasets/processed/email_dataset_github_processed.csv`
- EDA notebook: `notebooks/EDA/`

## Cấu trúc chính

- `config/`: cấu hình liên quan đến dữ liệu
- `datasets/`: dữ liệu raw và processed
- `notebooks/EDA/`: notebook khám phá dữ liệu
- `src/data/`: các module xử lý dữ liệu, chuẩn bị cho bước tiền xử lý và chia tập

## Hướng phát triển tiếp theo

- Hoàn thiện data cleaning
- Chuẩn hóa pipeline tiền xử lý email
- Tách train / validation / test
- Chọn baseline model cho bài toán phân loại email
