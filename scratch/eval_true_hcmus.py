import subprocess
import io
import sys
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

sys.stdout.reconfigure(encoding='utf-8')

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

# Load true HCMUS predictions from git hcmus_predictions.txt
raw_txt = subprocess.check_output(
    ['git', 'show', '2473960d9587c93270077afe4477f12362645f72:data/result/hcmus_predictions.txt'],
    encoding='utf-8'
)

preds_dict = {}
for line in raw_txt.strip().split('\n')[1:]:
    parts = line.strip().split(',')
    if len(parts) >= 2:
        nom = parts[0]
        pred = ','.join(parts[1:]).strip()
        preds_dict[nom] = pred

df_test = pd.read_csv('data/processed/test.csv')

predictions = []
for nom in df_test['nom']:
    p = preds_dict.get(nom, "")
    predictions.append(p)

df_eval = df_test[['nom', 'vietnamese_clean']].copy()
df_eval['prediction'] = predictions
df_eval['is_exact_match'] = (df_eval['vietnamese_clean'].str.strip() == df_eval['prediction'].str.strip())

print(f"Total samples: {len(df_eval)}")
print(f"Exact match: {df_eval['is_exact_match'].sum()} / {len(df_eval)} ({df_eval['is_exact_match'].mean()*100:.2f}%)")

refs = df_eval['vietnamese_clean'].fillna('').astype(str).tolist()
preds = df_eval['prediction'].fillna('').astype(str).tolist()

bleu_score = sacrebleu.corpus_bleu(preds, [[r for r in refs]]).score

tot_char_match = sum(sum(1 for c1, c2 in zip(r, p) if c1 == c2) for r, p in zip(refs, preds))
tot_char_max = sum(max(len(r), len(p)) for r, p in zip(refs, preds))
char_acc = (tot_char_match / tot_char_max * 100) if tot_char_max > 0 else 0.0

tot_syl_match = sum(sum(1 for w1, w2 in zip(r.split(), p.split()) if w1 == w2) for r, p in zip(refs, preds))
tot_syl_max = sum(max(len(r.split()), len(p.split())) for r, p in zip(refs, preds))
syl_acc = (tot_syl_match / tot_syl_max * 100) if tot_syl_max > 0 else 0.0

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

tot_cer_dist = sum(levenshtein_distance(ref, pred) for ref, pred in zip(refs, preds))
tot_cer_len = sum(len(ref) for ref in refs)
cer = (tot_cer_dist / tot_cer_len * 100) if tot_cer_len > 0 else 0.0

tot_wer_dist = sum(levenshtein_distance(ref.split(), pred.split()) for ref, pred in zip(refs, preds))
tot_wer_len = sum(len(ref.split()) for ref in refs)
wer = (tot_wer_dist / tot_wer_len * 100) if tot_wer_len > 0 else 0.0

metrics_df = pd.DataFrame([{
    'Model': 'HCMUS-Online-API',
    'Total_Samples': len(df_eval),
    'BLEU': round(bleu_score, 2),
    'Accuracy_EM_%': round(df_eval['is_exact_match'].mean()*100, 2),
    'Accuracy_Char_%': round(char_acc, 2),
    'Accuracy_Syllable_%': round(syl_acc, 2),
    'ROUGE1_%': round(rouge1, 2),
    'ROUGE2_%': round(rouge2, 2),
    'ROUGEL_%': round(rougel, 2),
    'CER_%': round(cer, 2),
    'WER_%': round(wer, 2)
}])

print("\n--- TRUE HCMUS METRICS ---")
print(metrics_df.to_string())
