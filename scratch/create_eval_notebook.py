import json
import os
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Study\HTK\final_proj\Sino-Nom-to-Vietnamese-Transliteration")
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_Evaluate_HCMUS_API.ipynb"

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# --- Cell 1: Intro ---
add_markdown("""# 🔬 Đánh Giá Mô Hình Phiên Âm Hán-Nôm CLC HCMUS API Trên Tập Test

Notebook này cung cấp quy trình toàn diện để:
1. **Khâu "Load Model":** Tạo Model Wrapper cho API dịch Hán-Nôm của Đại học Khoa học Tự nhiên TP.HCM (`https://tools.clc.hcmus.edu.vn/api/web/clc-sinonom/sinonom-transliteration`).
2. **Khâu Suy Luận (Inference):** Chạy phiên âm trên toàn bộ 1,449 mẫu của tập kiểm thử `data/processed/test.csv` (có cơ chế rate-limiting, error handling và caching thông minh).
3. **Khâu Đánh Giá (Evaluation):** Tính toán đồng bộ các độ đo theo đúng tiêu chuẩn dự án:
   - **Exact Match (EM %)**
   - **Character Accuracy (%)** & **Syllable Accuracy (%)**
   - **BLEU Score** (SacreBLEU)
   - **ROUGE-1, ROUGE-2, ROUGE-L (%)**
   - **Character Error Rate (CER %)** & **Word Error Rate (WER %)**
4. **Xuất Kết Quả:** Xuất ra thư mục `data/result/hcmus/` với cấu trúc tương tự `data/result/mbart/`:
   - `test_predictions.csv` (`nom,vietnamese_clean,prediction,is_exact_match`)
   - `evaluation_metrics.csv`
5. **So Sánh Đối Chuẩn (Benchmark Leaderboard):** So sánh trực tiếp hiệu năng giữa **mBART-5epoch**, **mT5-5epoch** và **HCMUS API**.""")

# --- Cell 2: Imports ---
add_code("""import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import sacrebleu
from rouge_score import rouge_scorer

# Thiết lập đường dẫn gốc dự án
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()

print(f"[*] Project Root: {PROJECT_ROOT}")""")

# --- Cell 3: Paths ---
add_code("""# Cấu hình đường dẫn dữ liệu và thư mục kết quả
TEST_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
RESULT_DIR = PROJECT_ROOT / "data" / "result" / "hcmus"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREDICTIONS_CSV = RESULT_DIR / "test_predictions.csv"
OUT_METRICS_CSV = RESULT_DIR / "evaluation_metrics.csv"

print(f"[+] Test CSV: {TEST_CSV_PATH}")
print(f"[+] Output Directory: {RESULT_DIR}")""")

# --- Cell 4: Markdown Model explanation ---
add_markdown("""## 1. Khâu "Load Model" (HCMUS Transliterator Model Wrapper)

Khác với các mô hình PyTorch/HuggingFace (như mT5, mBART tải qua `from_pretrained`), mô hình của CLC HCMUS được triển khai dưới dạng **Online REST API**. 

Để khâu kiểm thử hoạt động tương tự như một model Transformer thông thường, ta xây dựng lớp `HCMUSTranslatorModel` đóng vai trò là một **Callable Model Interface**. Lớp này đóng gói:
- Thiết lập Endpoint URL và Headers giả lập trình duyệt.
- Payload định dạng JSON: `{"text": text, "lang_type": 1, "font_type": 1}`.
- Cơ chế **Rate Limiting** (`time.sleep(delay)`) chống bị chặn IP (HTTP 429).
- Cơ chế **Retry & Exception Handling** khi gặp sự cố mạng.""")

# --- Cell 5: Code Model wrapper ---
add_code("""import requests

class HCMUSTranslatorModel:
    \"\"\"
    Model Wrapper đóng vai trò interface tương đương với mô hình Transformer,
    thực hiện phiên âm Hán-Nôm sang Tiếng Việt thông qua API của CLC HCMUS.
    \"\"\"
    def __init__(self, delay: float = 0.1, timeout: int = 15, max_retries: int = 3):
        self.url = 'https://tools.clc.hcmus.edu.vn/api/web/clc-sinonom/sinonom-transliteration'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
            'Content-Type': 'application/json',
        }
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        print(f"[*] HCMUS Model initialized at endpoint: {self.url}")

    def predict(self, text: str) -> str:
        \"\"\"Phiên âm một câu Hán-Nôm sang Tiếng Việt.\"\"\"
        text = str(text).strip()
        if not text:
            return ""

        payload = {
            'text': text,
            'lang_type': 1,  # 1: Hán-Nôm -> Quốc ngữ
            'font_type': 1   # 1: Bảng mã Unicode
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                time.sleep(self.delay)

                if response.status_code == 200:
                    data = response.json().get('data', {})
                    transcriptions = data.get('result_text_transcription', [])
                    if transcriptions:
                        return transcriptions[0].strip()
                    return ""
                else:
                    time.sleep(0.5)
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"[ERR] Thử lại {attempt}/{self.max_retries} thất bại cho câu: '{text[:20]}...' | Lỗi: {e}")
                time.sleep(0.5)

        return "Cannot translate this text."

    def __call__(self, text: str) -> str:
        \"\"\"Cho phép gọi dạng callable: output = model(text)\"\"\"
        return self.predict(text)

# Khởi tạo mô hình ("Load Model")
model = HCMUSTranslatorModel(delay=0.1)

# Thử nghiệm nhanh 1 câu
sample = "是日文曉有德并二營属将走出真福安塲"
print(f"Input : {sample}")
print(f"Output: {model(sample)}")""")

# --- Cell 6: Load Test Dataset ---
add_markdown("""## 2. Nạp Tập Dữ Liệu Kiểm Thử (Test Dataset)""")

add_code("""df_test = pd.read_csv(TEST_CSV_PATH)
print(f"[+] Đã nạp thành công {len(df_test):,} mẫu kiểm thử.")
display(df_test.head(5))""")

# --- Cell 7: Inference / Cache logic ---
add_markdown("""## 3. Khâu Suy Luận (Inference) & Tạo Tệp `test_predictions.csv`

Nhằm tối ưu thời gian và tránh gửi trùng 1,449 request qua Internet, notebook hỗ trợ:
- Tự động nạp kết quả dự đoán có sẵn trong thư mục nếu đã từng chạy.
- Nếu muốn chạy mới lại toàn bộ từ API, chỉ cần đặt `FORCE_RE_RUN = True`.""")

add_code("""FORCE_RE_RUN = False  # Đổi thành True nếu bạn muốn chạy request lại toàn bộ từ API

if OUT_PREDICTIONS_CSV.exists() and not FORCE_RE_RUN:
    print(f"[*] Phát hiện tệp dự đoán đã tồn tại tại {OUT_PREDICTIONS_CSV}. Nạp từ cache...")
    df_pred = pd.read_csv(OUT_PREDICTIONS_CSV)
    predictions = df_pred['prediction'].fillna('').astype(str).tolist()
else:
    print(f"[*] Bắt đầu gửi {len(df_test):,} câu lên HCMUS API để dự đoán...")
    predictions = []
    for text in tqdm(df_test['nom'], desc="Inference HCMUS API"):
        pred = model(text)
        predictions.append(pred)

# Thêm kết quả vào DataFrame
df_results = df_test[['nom', 'vietnamese_clean']].copy()
df_results['prediction'] = predictions
df_results['is_exact_match'] = (df_results['vietnamese_clean'].astype(str).str.strip() == df_results['prediction'].astype(str).str.strip())

# Lưu file kết quả chuẩn format tương tự data/result/mbart/test_predictions.csv
df_results.to_csv(OUT_PREDICTIONS_CSV, index=False, encoding='utf-8')
print(f"[+] Đã lưu tệp dự đoán: {OUT_PREDICTIONS_CSV}")
display(df_results.head(10))""")

# --- Cell 8: Metric definition ---
add_markdown("""## 4. Định Nghĩa Độ Đo Đánh Giá (Evaluation Metrics)

Các hàm đo lường được xây dựng đồng nhất với chuẩn đánh giá của các mô hình trong bài toán Transliteration:
- **Exact Match (EM %)**: Tỷ lệ câu dự đoán trùng khớp hoàn toàn 100% với ground truth.
- **Accuracy Char (%)**: Tỷ lệ ký tự trùng khớp tại từng vị trí trên tổng độ dài tối đa của câu.
- **Accuracy Syllable (%)**: Tỷ lệ âm tiết/từ trùng khớp tại từng vị trí trên tổng số âm tiết tối đa.
- **BLEU**: Điểm SacreBLEU cho toàn bộ tập dữ liệu (Corpus-level).
- **ROUGE-1, ROUGE-2, ROUGE-L (%)**: Điểm F1 trùng lặp n-gram và chuỗi con chung dài nhất.
- **CER (%)**: Character Error Rate tính bằng khoảng cách Levenshtein ký tự.
- **WER (%)**: Word Error Rate tính bằng khoảng cách Levenshtein âm tiết.""")

add_code("""def levenshtein_distance(s1, s2):
    \"\"\"Tính khoảng cách Levenshtein giữa 2 chuỗi hoặc 2 danh sách từ.\"\"\"
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def evaluate_predictions(df: pd.DataFrame, model_name: str = "HCMUS-Online-API") -> pd.DataFrame:
    \"\"\"Tính toán toàn bộ bảng metrics theo format của data/result/mbart/evaluation_metrics.csv.\"\"\"
    refs = df['vietnamese_clean'].fillna('').astype(str).tolist()
    preds = df['prediction'].fillna('').astype(str).tolist()
    total_samples = len(df)

    # 1. Exact Match
    em_acc = (df['vietnamese_clean'].str.strip() == df['prediction'].str.strip()).mean() * 100

    # 2. BLEU Score
    bleu_score = sacrebleu.corpus_bleu(preds, [[r for r in refs]]).score

    # 3. Accuracy Char & Accuracy Syllable
    tot_char_match = sum(sum(1 for c1, c2 in zip(r, p) if c1 == c2) for r, p in zip(refs, preds))
    tot_char_max = sum(max(len(r), len(p)) for r, p in zip(refs, preds))
    char_acc = (tot_char_match / tot_char_max * 100) if tot_char_max > 0 else 0.0

    tot_syl_match = sum(sum(1 for w1, w2 in zip(r.split(), p.split()) if w1 == w2) for r, p in zip(refs, preds))
    tot_syl_max = sum(max(len(r.split()), len(p.split())) for r, p in zip(refs, preds))
    syl_acc = (tot_syl_match / tot_syl_max * 100) if tot_syl_max > 0 else 0.0

    # 4. ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    r1_list, r2_list, rl_list = [], [], []
    for ref, pred in zip(refs, preds):
        s = scorer.score(ref, pred)
        r1_list.append(s['rouge1'].fmeasure)
        r2_list.append(s['rouge2'].fmeasure)
        rl_list.append(s['rougeL'].fmeasure)

    rouge1 = (sum(r1_list) / len(r1_list)) * 100
    rouge2 = (sum(r2_list) / len(r2_list)) * 100
    rougel = (sum(rl_list) / len(rl_list)) * 100

    # 5. CER và WER
    tot_cer_dist = sum(levenshtein_distance(ref, pred) for ref, pred in zip(refs, preds))
    tot_cer_len = sum(len(ref) for ref in refs)
    cer = (tot_cer_dist / tot_cer_len * 100) if tot_cer_len > 0 else 0.0

    tot_wer_dist = sum(levenshtein_distance(ref.split(), pred.split()) for ref, pred in zip(refs, preds))
    tot_wer_len = sum(len(ref.split()) for ref in refs)
    wer = (tot_wer_dist / tot_wer_len * 100) if tot_wer_len > 0 else 0.0

    metrics_df = pd.DataFrame([{
        'Model': model_name,
        'Total_Samples': total_samples,
        'BLEU': round(bleu_score, 2),
        'Accuracy_EM_%': round(em_acc, 2),
        'Accuracy_Char_%': round(char_acc, 2),
        'Accuracy_Syllable_%': round(syl_acc, 2),
        'ROUGE1_%': round(rouge1, 2),
        'ROUGE2_%': round(rouge2, 2),
        'ROUGEL_%': round(rougel, 2),
        'CER_%': round(cer, 2),
        'WER_%': round(wer, 2)
    }])
    return metrics_df""")

# --- Cell 9: Compute & Save Metrics ---
add_markdown("""## 5. Xuất Kết Quả `data/result/hcmus/evaluation_metrics.csv`""")

add_code("""metrics_df = evaluate_predictions(df_results, model_name="HCMUS-Online-API")

# Lưu ra file CSV
metrics_df.to_csv(OUT_METRICS_CSV, index=False, encoding='utf-8')
print(f"[+] Đã lưu tệp độ đo đánh giá: {OUT_METRICS_CSV}")

# Hiển thị bảng metrics
display(metrics_df)""")

# --- Cell 10: Comparative Leaderboard ---
add_markdown("""## 6. So Sánh Đối Chuẩn Toàn Diện (Leaderboard: mBART vs mT5 vs HCMUS API)

Tổng hợp kết quả đánh giá từ cả 3 thư mục `data/result/mbart/`, `data/result/mt5/` và `data/result/hcmus/` để lập bảng so sánh hiệu năng trực tiếp.""")

add_code("""all_metrics = []

# Nạp metrics từ mbart
mbart_file = PROJECT_ROOT / "data" / "result" / "mbart" / "evaluation_metrics.csv"
if mbart_file.exists():
    all_metrics.append(pd.read_csv(mbart_file))

# Nạp metrics từ mt5
mt5_file = PROJECT_ROOT / "data" / "result" / "mt5" / "evaluation_metrics.csv"
if mt5_file.exists():
    all_metrics.append(pd.read_csv(mt5_file))

# Thêm metrics HCMUS
all_metrics.append(metrics_df)

leaderboard = pd.concat(all_metrics, ignore_index=True)

# Sắp xếp theo thứ tự BLEU giảm dần
leaderboard = leaderboard.sort_values(by='BLEU', ascending=False).reset_index(drop=True)

print("=== BẢNG SO SÁNH HIỆU NĂNG MÔ HÌNH (TEST SET) ===")
display(leaderboard)""")

notebook_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, ensure_ascii=False, indent=2)

print(f"Created notebook successfully at {NOTEBOOK_PATH}")
