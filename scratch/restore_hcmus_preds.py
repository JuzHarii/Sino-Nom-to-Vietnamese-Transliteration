import subprocess
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Study\HTK\final_proj\Sino-Nom-to-Vietnamese-Transliteration")
RESULT_DIR = PROJECT_ROOT / "data" / "result" / "hcmus"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Restore authentic hcmus_predictions.txt
raw_txt = subprocess.check_output(
    ['git', 'show', '2473960d9587c93270077afe4477f12362645f72:data/result/hcmus_predictions.txt'],
    encoding='utf-8'
)

with open(RESULT_DIR / "hcmus_predictions.txt", "w", encoding="utf-8") as f:
    f.write(raw_txt)
print(f"[+] Restored {RESULT_DIR / 'hcmus_predictions.txt'}")

# 2. Build dictionary
preds_dict = {}
for line in raw_txt.strip().split('\n')[1:]:
    parts = line.strip().split(',')
    if len(parts) >= 2:
        nom = parts[0]
        pred = ','.join(parts[1:]).strip()
        preds_dict[nom] = pred

# 3. Create authentic test_predictions.csv
df_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
predictions = [preds_dict.get(nom, "") for nom in df_test['nom']]

df_clean_preds = df_test[['nom', 'vietnamese_clean']].copy()
df_clean_preds['prediction'] = predictions
df_clean_preds['is_exact_match'] = (df_clean_preds['vietnamese_clean'].str.strip() == df_clean_preds['prediction'].str.strip())

out_csv = RESULT_DIR / "test_predictions.csv"
df_clean_preds.to_csv(out_csv, index=False, encoding='utf-8')
print(f"[+] Saved authentic {out_csv} with {len(df_clean_preds)} rows.")
print(f"[+] Exact Match Count: {df_clean_preds['is_exact_match'].sum()} / {len(df_clean_preds)} ({df_clean_preds['is_exact_match'].mean()*100:.2f}%)")
