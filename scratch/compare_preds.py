import subprocess
import io
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Get committed eval_hcmus_test.csv from git
raw_csv = subprocess.check_output(
    ['git', 'show', '2473960d9587c93270077afe4477f12362645f72:data/result/eval_hcmus_test.csv'],
    encoding='utf-8'
)
df_old = pd.read_csv(io.StringIO(raw_csv))

print(f"Old eval_hcmus_test: {len(df_old)} rows")
print(f"Old exact_match sum: {df_old['exact_match'].sum()} / {len(df_old)} ({df_old['exact_match'].mean()*100:.2f}%)")

# Compare with current data/result/hcmus/test_predictions.csv
df_curr = pd.read_csv('data/result/hcmus/test_predictions.csv')
print(f"Current test_predictions: {len(df_curr)} rows")

matches = 0
diffs = 0
for i in range(min(len(df_old), len(df_curr))):
    p_old = str(df_old.iloc[i]['prediction']).strip().lower()
    p_curr = str(df_curr.iloc[i]['prediction']).strip().lower()
    if p_old == p_curr:
        matches += 1
    else:
        diffs += 1
        if diffs <= 5:
            print(f"\nDiff at line {i}:")
            print(f"  Nom     : {df_curr.iloc[i]['nom']}")
            print(f"  Ref     : {df_curr.iloc[i]['vietnamese_clean']}")
            print(f"  Old Pred: {p_old}")
            print(f"  Cur Pred: {p_curr}")

print(f"\nTotal identical predictions between old and current: {matches}")
print(f"Total different predictions between old and current: {diffs}")
