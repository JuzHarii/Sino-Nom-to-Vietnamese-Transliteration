import sys
import os
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"c:\Study\HTK\final_proj\Sino-Nom-to-Vietnamese-Transliteration")
test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
test_df = pd.read_csv(test_path)

hcmus_df = pd.read_csv(PROJECT_ROOT / "data" / "result" / "hcmus" / "test_predictions.csv")
mbart_df = pd.read_csv(PROJECT_ROOT / "data" / "result" / "mbart" / "test_predictions.csv")
mt5_df = pd.read_csv(PROJECT_ROOT / "data" / "result" / "mt5" / "test_predictions.csv")

hcmus_metrics = pd.read_csv(PROJECT_ROOT / "data" / "result" / "hcmus" / "evaluation_metrics.csv")
mbart_metrics = pd.read_csv(PROJECT_ROOT / "data" / "result" / "mbart" / "evaluation_metrics.csv")
mt5_metrics = pd.read_csv(PROJECT_ROOT / "data" / "result" / "mt5" / "evaluation_metrics.csv")

combined_metrics = pd.concat([hcmus_metrics, mbart_metrics, mt5_metrics], ignore_index=True)
combined_metrics['Model_Short'] = ['HCMUS API', 'mBART-5ep', 'mT5-5ep']
print("=== Overall Metrics ===")
print(combined_metrics[['Model_Short', 'BLEU', 'Accuracy_EM_%', 'Accuracy_Char_%', 'Accuracy_Syllable_%', 'ROUGEL_%', 'CER_%', 'WER_%']])

# Analyze per source
models = [('HCMUS API', hcmus_df), ('mBART-5ep', mbart_df), ('mT5-5ep', mt5_df)]
source_names = test_df['source'].unique()

print("\n=== Exact Match by Source ===")
em_data = []
for short_name, df in models:
    df['source'] = test_df['source']
    for src in source_names:
        sub = df[df['source'] == src]
        em_pct = sub['is_exact_match'].mean() * 100
        em_data.append({'Model': short_name, 'Source': src, 'Count': len(sub), 'EM_%': em_pct})

em_df = pd.DataFrame(em_data)
print(em_df.pivot(index='Source', columns='Model', values='EM_%'))

# Check BLEU or Syllable Acc by source if needed
from collections import Counter
def calc_syllable_acc(preds, refs):
    correct, total = 0, 0
    for p, r in zip(preds, refs):
        p_toks = str(p).split()
        r_toks = str(r).split()
        total += max(len(p_toks), len(r_toks))
        for pt, rt in zip(p_toks, r_toks):
            if pt == rt:
                correct += 1
    return (correct / total * 100) if total > 0 else 0

print("\n=== Syllable Accuracy by Source ===")
syl_data = []
for short_name, df in models:
    for src in source_names:
        sub = df[df['source'] == src]
        syl_acc = calc_syllable_acc(sub['prediction'], sub['vietnamese_clean'])
        syl_data.append({'Model': short_name, 'Source': src, 'Syllable_Acc_%': syl_acc})

syl_df = pd.DataFrame(syl_data)
print(syl_df.pivot(index='Source', columns='Model', values='Syllable_Acc_%'))
