# Chủ đề 5 — Trích xuất thông tin từ ảnh hóa đơn

Python 3.10+, CPU, Tesseract eng. Đầu ra company/date/address/total, văn bản OCR, khung chữ, JSON gốc và ngày/tiền chuẩn hóa.

## Chạy trên Windows
Mở terminal tại thư mục có app.py:
```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m streamlit run app.py
```
Cài chương trình Tesseract riêng. Nếu cài ổ D, đặt `$env:TESSERACT_CMD = "D:\duong_dan_that\tesseract.exe"` trước khi chạy. Giao diện không cần dataset: có thể tải ảnh được phép dùng.

## Tái lập thực nghiệm
```
python scripts/download_source_reference.py
python scripts/01_explore_data.py
python scripts/run_experiment.py
```
Notebook `Bao_cao_thuc_nghiem.ipynb` hướng dẫn cùng quy trình. Kết quả chính: `outputs/experiment/metrics.csv`, `per_field.csv`, `predictions.json`, `splits.json`, `environment.json`. JSON gồm giá trị gốc, chuẩn hóa, OCR và nhãn đối chiếu. Các script 02/03/04 là công cụ cũ; số liệu báo cáo dùng run_experiment.py, không dùng cache cũ.

## Dữ liệu và quyền sử dụng
Nguồn do giảng viên dẫn: https://github.com/metatatt/sorie
Commit cố định: 27be4271b251c256f695acbade9a801bffe85994, truy cập 01/11/2026.
Bản thực tế có 626 ảnh và các nhãn đã được bên phân phối hiệu chỉnh. Không sửa nhãn trong thực nghiệm này. Chọn 300 ID bằng seed42: 200 hold-out nội bộ, 100 phát triển tách 80 train/20 val. Không gọi đây là test chính thức ICDAR. Chưa kiểm soát tách theo doanh nghiệp; khả năng cùng cửa hàng xuất hiện ở nhiều tập là hạn chế.
Giữ LICENSE nguồn trong docs/SOURCE_LICENSE.txt. LICENSE của repo là MIT cho software và tài liệu kèm theo; không suy rộng thành xác nhận quyền của mọi bên thứ ba trên hóa đơn. Trang nguồn gốc: https://rrc.cvc.uab.es/?ch=13 . Không thu thập hóa đơn riêng tư; kiểm tra điều khoản trước khi tái phân phối dữ liệu ngoài lớp học.

## Phương pháp và đánh giá
basic: xám và resize; improved: khử nhiễu, CLAHE, deskew và threshold. Hai bộ luật baseline/improved được cố định trước khi chạy test. Không fine-tune OCR, không dùng nhãn làm đầu vào OCR. Exact match sau chuẩn hóa từng trường; mean là trung bình bốn trường. F1 dùng correct/nonempty-prediction cho precision và correct/labeled cho recall. all4 là tỷ lệ hóa đơn đúng cả bốn trường. Phân loại lỗi theo sự xuất hiện của nhãn trong OCR chỉ là heuristic.
Khóa cache thực nghiệm chứa SHA256 ảnh, variant, PSM, ngôn ngữ và phiên bản pipeline; tăng phiên bản v1 trong runner khi sửa xử lý ảnh. Khi sửa OCR engine/version, xóa cache `outputs/experiment/ocr` hoặc tăng phiên bản. Thời gian gồm OCR và tiền xử lý khi chạy đồng thời 4 worker, không phải benchmark độ trễ một yêu cầu.

## Tài liệu
Yêu cầu giảng viên: trang 23–27, 33–36 của Sau_chu_de...pdf.
Russell & Norvig, AIMA Global Edition 4e, chương 27, mục 27.1 và 27.3.
https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
https://github.com/metatatt/sorie

## Giới hạn
Không phải phần mềm kế toán. Luật còn phụ thuộc tiếng Anh và bố cục Malaysia. improved không mặc nhiên tốt hơn basic. Địa chỉ có thể dừng sớm ở mã bưu chính; tổng tiền có thể nhầm CASH ở dòng sau; chưa hỗ trợ tốt tiền nguyên VND. Xem phân tích lỗi thật trong báo cáo.


## Machine Learning: TF-IDF + Logistic Regression
Pipeline: OpenCV -> Tesseract -> word/char TF-IDF -> LogisticRegression -> JSON 4 fields. Two rule-based extractors remain available.

### Windows PowerShell
Commands (from the repo root, where app.py lives):

    git pull origin main
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m pytest tests -q

Place labeled SROIE data in data/raw/SROIE2019, with:
- train/img, train/box, train/entities
- test/img, test/box, test/entities (if available)

If stored on drive D, set SROIE_DIR in PowerShell, for example:

    $env:SROIE_DIR = "D:\data\SROIE2019"

Train, report metrics, and run the web app:

    python scripts/train_model.py
    python -m streamlit run app.py

Choose the **ml** option in the "Phương pháp trích trường" sidebar selector. Output model: models/receipt_model.joblib. Scores: outputs/ml/metrics.json. Both output directories are ignored by Git.

### Evaluation and limitations
Training only on the train split. Validation and test held out from fitting. Metrics include weak line-label accuracy and macro F1, per-field exact match and all-four-correct. No verified score can be claimed without running training on your local SROIE data.

The line labels are *automatically aligned* from SROIE entities to box text; they are not manual line-level annotations. Training and evaluation use ground-truth box transcription, NOT OCR predictions, so the reported scores are not end-to-end OCR accuracy. The loader may create an internal holdout if a labeled test folder is absent. The model must be reevaluated on real OCR output before making accuracy claims about uploaded images. Only load your own joblib model files.
