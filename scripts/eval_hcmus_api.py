import os
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time

# Thêm thư mục gốc vào PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hcmus_translate import hcmus_translate

def levenshtein_distance(s1, s2):
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

def compute_metrics(reference, hypothesis):
    """Tính toán Character Error Rate (CER) và Word Error Rate (WER)."""
    # Xử lý chuỗi
    ref = str(reference).strip().lower()
    hyp = str(hypothesis).strip().lower()
    
    # Exact match
    exact_match = int(ref == hyp)
    
    # CER: Bỏ khoảng trắng để tính lỗi ký tự chính xác
    ref_chars = ref.replace(" ", "")
    hyp_chars = hyp.replace(" ", "")
    
    if len(ref_chars) == 0:
        cer = 0.0 if len(hyp_chars) == 0 else 1.0
    else:
        cer = levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)
        
    # WER: Tính toán trên từng từ (âm tiết)
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    if len(ref_words) == 0:
        wer = 0.0 if len(hyp_words) == 0 else 1.0
    else:
        wer = levenshtein_distance(ref_words, hyp_words) / len(ref_words)
        
    return exact_match, cer, wer

def evaluate_hcmus_api(test_csv_path, output_csv_path, max_samples=None):
    print(f"Đang nạp tập test từ: {test_csv_path}")
    df = pd.read_csv(test_csv_path)
    
    if max_samples:
        df = df.head(max_samples)
        print(f"Rút gọn test set còn {max_samples} mẫu để thử nghiệm.")
    
    total = len(df)
    results = []
    
    total_cer = 0.0
    total_wer = 0.0
    total_exact = 0
    valid_count = 0
    
    print(f"Bắt đầu dịch và đánh giá {total} câu bằng HCMUS API...")
    
    # Tự động lưu ngắt quãng để đề phòng lỗi mạng
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in tqdm(df.iterrows(), total=total, desc="Translating"):
        nom_text = row['nom']
        ref_text = row['vietnamese_clean']
        
        # Gọi API
        pred_text = hcmus_translate(nom_text)
        
        # Nếu API trả về thông báo lỗi, ta có thể bỏ qua tính điểm cho câu đó hoặc tính max error
        if pred_text == "Cannot translate this text.":
            cer, wer, exact_match = 1.0, 1.0, 0
            is_valid = False
        else:
            exact_match, cer, wer = compute_metrics(ref_text, pred_text)
            total_cer += cer
            total_wer += wer
            total_exact += exact_match
            valid_count += 1
            is_valid = True
            
        results.append({
            'nom': nom_text,
            'ground_truth': ref_text,
            'prediction': pred_text,
            'exact_match': exact_match,
            'cer': round(cer, 4),
            'wer': round(wer, 4),
            'is_valid': is_valid
        })
        
    # Tạo DataFrame kết quả và lưu lại
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    # Tính trung bình trên các câu hợp lệ (API dịch thành công)
    if valid_count > 0:
        avg_cer = total_cer / valid_count
        avg_wer = total_wer / valid_count
        acc = total_exact / valid_count
        
        # --- Tính BLEU và ROUGE ---
        try:
            import sacrebleu
            from rouge_score import rouge_scorer
            
            preds = [r['prediction'] for r in results if r['is_valid']]
            refs = [[r['ground_truth'] for r in results if r['is_valid']]] # sacrebleu cần dạng list of lists
            
            bleu_score = sacrebleu.corpus_bleu(preds, refs).score
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            r1, r2, rL = 0, 0, 0
            for r in results:
                if r['is_valid']:
                    scores = scorer.score(r['ground_truth'], r['prediction'])
                    r1 += scores['rouge1'].fmeasure
                    r2 += scores['rouge2'].fmeasure
                    rL += scores['rougeL'].fmeasure
            r1 = (r1 / valid_count) * 100
            r2 = (r2 / valid_count) * 100
            rL = (rL / valid_count) * 100
            
            bleu_str = f"{bleu_score:.2f}"
            rouge_str = f"R1: {r1:.2f} | R2: {r2:.2f} | RL: {rL:.2f}"
        except ImportError:
            bleu_str = "Chưa cài sacrebleu"
            rouge_str = "Chưa cài rouge-score"
        
        print("\n" + "="*50)
        print("KẾT QUẢ ĐÁNH GIÁ (HCMUS API)")
        print("="*50)
        print(f"Tổng số câu     : {total}")
        print(f"Dịch thành công : {valid_count}")
        print("-" * 50)
        print(f"Exact Match (ACC) : {acc*100:.2f}% ({total_exact}/{valid_count})")
        print(f"Average CER       : {avg_cer*100:.2f}%")
        print(f"Average WER       : {avg_wer*100:.2f}%")
        print(f"BLEU Score        : {bleu_str}")
        print(f"ROUGE Score       : {rouge_str}")
        print("="*50)
        print(f"Chi tiết (Predictions) đã lưu tại: {output_csv_path}")
    else:
        print("Không có câu nào được dịch thành công!")

if __name__ == "__main__":
    TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"
    OUTPUT_FILE = PROJECT_ROOT / "data" / "result" / "eval_hcmus_test.csv"
    
    # Bạn có thể đổi max_samples = None để chạy toàn bộ tập dữ liệu (1400+ câu)
    # Tuy nhiên để test thử tool, tôi khuyến nghị để số nhỏ trước (vd: 10)
    evaluate_hcmus_api(TEST_FILE, OUTPUT_FILE, max_samples=None)
