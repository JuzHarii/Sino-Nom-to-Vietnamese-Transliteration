import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"c:\Study\HTK\final_proj\Sino-Nom-to-Vietnamese-Transliteration")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.loader import load_raw_corpus
from data.preprocessor import DataPreprocessor, compute_length_features

out_lines = []
def log(s=""):
    out_lines.append(str(s))

df_raw = load_raw_corpus(PROJECT_ROOT / "data" / "raw")
df_raw = compute_length_features(df_raw, vn_col='vietnamese')

preprocessor = DataPreprocessor(df_raw)
df_clean = preprocessor.clean_corpus(augment=False)
df_clean_filtered = df_clean[df_clean['align_diff'] <= 2].copy()

train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

log("=== 1. TỔNG QUAN TỆP VÀ NGUỒN DỮ LIỆU ===")
log(f"Tổng số dòng RAW: {len(df_raw):,}")
log("\nSố dòng theo tệp:")
for f, c in df_raw['file_name'].value_counts().sort_index().items():
    log(f"  - {f}: {c:,} dòng ({c/len(df_raw):.2%})")

log("\nSố dòng theo nguồn tác phẩm:")
for s, c in df_raw['source'].value_counts().items():
    log(f"  - {s}: {c:,} dòng ({c/len(df_raw):.2%})")

log("\n=== 2. THỐNG KÊ ĐỘ DÀI (RAW) ===")
cols = ['nom_char_len', 'vn_word_len', 'vn_char_len', 'ratio_vn_word_per_nom_char']
for col in cols:
    s = df_raw[col].dropna()
    log(f"\n--- {col} ---")
    log(f"  Mean: {s.mean():.2f}, Std: {s.std():.2f}, Min: {s.min()}, Max: {s.max()}")
    log(f"  Quantiles: 25%={s.quantile(0.25):.2f}, 50%={s.median():.2f}, 75%={s.quantile(0.75):.2f}, 90%={s.quantile(0.90):.2f}, 95%={s.quantile(0.95):.2f}, 99%={s.quantile(0.99):.2f}")

log("\n=== 3. ĐỘ DÀI THEO NGUỒN TÁC PHẨM (MEAN & MEDIAN & MAX) ===")
for src, grp in df_raw.groupby('source'):
    log(f"\nNguồn: {src} ({len(grp):,} dòng)")
    log(f"  Nom char len - Mean: {grp['nom_char_len'].mean():.2f}, Median: {grp['nom_char_len'].median():.1f}, Max: {grp['nom_char_len'].max()}")
    log(f"  VN word len  - Mean: {grp['vn_word_len'].mean():.2f}, Median: {grp['vn_word_len'].median():.1f}, Max: {grp['vn_word_len'].max()}")
    log(f"  Ratio (VN/Nom)- Mean: {grp['ratio_vn_word_per_nom_char'].mean():.2f}")

log("\n=== 4. BẤT THƯỜNG VÀ NHIỄU (ANOMALIES) ===")
anom = preprocessor.detect_anomalies()
for k, v in anom.items():
    log(f"  {k}: {v}")

log("\nPhân bố Align Diff:")
for grp, cnt in df_raw['align_group'].value_counts().items():
    log(f"  - {grp}: {cnt:,} dòng ({cnt/len(df_raw):.2%})")

log("\nTop 5 câu có Align Diff lớn nhất trong RAW:")
for idx, r in df_raw.nlargest(5, 'align_diff').iterrows():
    log(f"  [{r['file_name']}] Nom len={r['nom_char_len']}, VN words={r['vn_word_len']}, diff={r['align_diff']}")
    log(f"     Nom: {str(r['nom'])[:50]}")
    log(f"     VN : {str(r['vietnamese'])[:60]}")

log("\n=== 5. VỐN TỪ (VOCABULARY) VÀ KÝ TỰ HIẾM ===")
vstats = preprocessor.analyze_vocab_distribution()
for k, v in vstats.items():
    log(f"  {k}: {v}")

log("\n=== 6. CHỮ NÔM ĐA ÂM (POLYPHONES) ===")
poly_df = preprocessor.analyze_polyphones(min_occurrences=5)
log(f"Tổng số chữ Nôm đa âm (xuất hiện >= 5 lần): {len(poly_df)}")
log("Top 15 chữ Nôm đa âm hàng đầu:")
for idx, r in poly_df.head(15).iterrows():
    log(f"  - {r['nom_char']}: {r['num_readings']} âm đọc, tổng xuất hiện={r['total_occurrences']}, âm chính: '{r['top_reading']}' ({r['top_reading_pct']}%), Chi tiết: {r['readings_detail']}")

log("\n=== 7. TẬP DỮ LIỆU CHẾ BIẾN (TRAIN/VAL/TEST) ===")
total_split = len(train_df) + len(val_df) + len(test_df)
log(f"Train: {len(train_df):,} dòng ({len(train_df)/total_split:.2%})")
log(f"Val  : {len(val_df):,} dòng ({len(val_df)/total_split:.2%})")
log(f"Test : {len(test_df):,} dòng ({len(test_df)/total_split:.2%})")

with open(PROJECT_ROOT / "scratch" / "eda_metrics_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Wrote detailed stats to scratch/eda_metrics_summary.txt successfully!")
