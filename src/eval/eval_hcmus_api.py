import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

def evaluate_hcmus_api(test_csv_path, prediction_txt_path, output_csv_path, max_samples=None):
    print(f"[*] Nạp tập test: {Path(test_csv_path).name}")
    df_test = pd.read_csv(test_csv_path)
    
    print(f"[*] Nạp cache dự đoán: {Path(prediction_txt_path).name}")
    # Parse txt thủ công để tránh lỗi khoảng trắng header
    preds_dict = {}
    with open(prediction_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]: # Bỏ qua header
            parts = line.strip().split(',')
            if len(parts) >= 2:
                nom_text = parts[0]
                pred_text = ','.join(parts[1:]) # Đề phòng có dấu phẩy trong câu dịch
                preds_dict[nom_text] = pred_text
    
    if max_samples:
        df_test = df_test.head(max_samples)
        print(f"[*] Chạy test nhanh trên {max_samples} mẫu.")
    
    total = len(df_test)
    results = []
    
    total_cer = 0.0
    total_wer = 0.0
    total_exact = 0
    valid_count = 0
    
    print(f"[*] Bắt đầu đánh giá cho {total} câu")
    
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in tqdm(df_test.iterrows(), total=total, desc="Evaluating"):
        nom_text = row['nom']
        ref_text = row['vietnamese_clean']
        
        # Kéo kết quả từ cache (không có thì văng lỗi dịch)
        pred_text = preds_dict.get(nom_text, "Cannot translate this text.")
        
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
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    # Tính điểm trung bình 
    if valid_count > 0:
        avg_cer = total_cer / valid_count
        avg_wer = total_wer / valid_count
        acc = total_exact / valid_count
        
        # Tính BLEU và ROUGE
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
        summary_text = (
            f"KẾT QUẢ ĐÁNH GIÁ HCMUS API\n"
            f"==================================================\n"
            f"Tổng số câu     : {total}\n"
            f"Tính điểm cho   : {valid_count} câu\n"
            f"--------------------------------------------------\n"
            f"Exact Match (ACC) : {acc*100:.2f}% ({total_exact}/{valid_count})\n"
            f"Average CER       : {avg_cer*100:.2f}%\n"
            f"Average WER       : {avg_wer*100:.2f}%\n"
            f"BLEU Score        : {bleu_str}\n"
            f"ROUGE Score       : {rouge_str}\n"
        )
        
        print("\n" + summary_text)
        print(f"[+] Đã lưu file chi tiết: {Path(output_csv_path).name}")
        
        summary_file_path = str(output_csv_path).replace('.csv', '_summary.txt')
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"[+] Đã lưu file summary: {Path(summary_file_path).name}")
    else:
        print("[!] Không có kết quả hợp lệ nào để chấm điểm.")

if __name__ == "__main__":
    TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"
    PRED_CACHE_FILE = PROJECT_ROOT / "data" / "result" / "hcmus_predictions.txt"
    OUTPUT_FILE = PROJECT_ROOT / "data" / "result" / "eval_hcmus_test.csv"
    
    evaluate_hcmus_api(TEST_FILE, PRED_CACHE_FILE, OUTPUT_FILE, max_samples=None)
