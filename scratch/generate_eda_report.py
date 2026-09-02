import os
import sys
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set encoding for Windows stdout
if sys.platform == 'win32':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

PROJECT_ROOT = Path(r"c:\Study\HTK\final_proj\Sino-Nom-to-Vietnamese-Transliteration")
ARTIFACT_DIR = Path(r"C:\Users\HP Victus\.gemini\antigravity-ide\brain\60a0e0ae-e385-419a-8231-39365ba9453a")
IMG_DIR = PROJECT_ROOT / "docs" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACT_DIR / "images").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.loader import load_raw_corpus
from data.preprocessor import DataPreprocessor, get_source_category, clean_text, compute_length_features

print("--- Step 1: Loading Raw and Processed Datasets ---")
df_raw = load_raw_corpus(PROJECT_ROOT / "data" / "raw")
print(f"Loaded raw dataset: {len(df_raw):,} rows from {df_raw['file_name'].nunique()} files.")

train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
val_path = PROJECT_ROOT / "data" / "processed" / "val.csv"
test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"

df_train = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
df_val = pd.read_csv(val_path) if val_path.exists() else pd.DataFrame()
df_test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()

print(f"Loaded processed splits: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")

# Extract features for raw
df_raw = compute_length_features(df_raw, vn_col='vietnamese')

# Preprocessing to clean
preprocessor = DataPreprocessor(df_raw)
df_clean = preprocessor.clean_corpus(augment=False)
df_clean_filtered = df_clean[df_clean['align_diff'] <= 2].copy()

print(f"Cleaned dataset: {len(df_clean):,} rows (Filtered align_diff<=2: {len(df_clean_filtered):,} rows)")

# Configure plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans', 'Calibri']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Helper to save figures to both docs/images and artifact/images
def save_figure(fig, name):
    path1 = IMG_DIR / name
    path2 = ARTIFACT_DIR / "images" / name
    fig.savefig(path1, dpi=200, bbox_inches='tight')
    fig.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {name}")

# --- Plot 1: Source & File Distribution ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
source_counts = df_raw['source'].value_counts()
colors1 = ['#1f77b4', '#e377c2', '#2ca02c', '#ff7f0e']
ax1.bar(source_counts.index, source_counts.values, color=colors1[:len(source_counts)], width=0.55, edgecolor='black', alpha=0.85)
ax1.set_title('Số Dòng Theo Nguồn Tác Phẩm (Raw Data)', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('Số lượng dòng', fontsize=11)
for i, v in enumerate(source_counts.values):
    ax1.text(i, v + max(source_counts.values)*0.015, f"{v:,}\n({v/len(df_raw):.1%})", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax1.set_ylim(0, max(source_counts.values)*1.15)

file_counts = df_raw['file_name'].value_counts().sort_index()
ax2.barh(file_counts.index, file_counts.values, color='#3498db', height=0.6, edgecolor='black', alpha=0.85)
ax2.set_title('Số Dòng Theo Từng Tệp (.txt)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Số lượng dòng', fontsize=11)
for i, v in enumerate(file_counts.values):
    ax2.text(v + max(file_counts.values)*0.01, i, f" {v:,}", ha='left', va='center', fontsize=9, fontweight='bold')
ax2.set_xlim(0, max(file_counts.values)*1.15)

plt.tight_layout()
save_figure(fig, 'source_file_distribution.png')

# --- Plot 2: Length Distributions ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Phân Phối Độ Dài Dữ Liệu Raw (Ký Tự & Âm Tiết)', fontsize=15, fontweight='bold', y=0.98)

configs = [
    ('nom_char_len', 'Số Ký Tự Hán-Nôm (Câu)', 'Số ký tự', '#2980b9', axes[0, 0]),
    ('vn_word_len', 'Số Âm Tiết Tiếng Việt (Câu)', 'Số từ/âm tiết', '#27ae60', axes[0, 1]),
    ('vn_char_len', 'Số Ký Tự Tiếng Việt (Chuỗi)', 'Số ký tự Quốc ngữ', '#e67e22', axes[1, 0]),
    ('ratio_vn_word_per_nom_char', 'Tỷ Lệ (Âm Tiết VN / Ký Tự Nôm)', 'Tỷ lệ', '#8e44ad', axes[1, 1])
]

for col, title, xlabel, color, ax in configs:
    data = df_raw[col].dropna()
    sns.histplot(data, bins=45, kde=True, ax=ax, color=color, edgecolor='black', alpha=0.6)
    ax.set_title(title, fontsize=11.5, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Tần suất', fontsize=10)
    med = data.median()
    mean = data.mean()
    fmt = ".2f" if col == 'ratio_vn_word_per_nom_char' else ".1f"
    ax.axvline(med, color='red', linestyle='--', linewidth=1.5, label=f"Trung vị: {med:{fmt}}")
    ax.axvline(mean, color='darkgreen', linestyle=':', linewidth=1.5, label=f"Trung bình: {mean:{fmt}}")
    ax.legend(fontsize=9, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_figure(fig, 'seq_length_distributions.png')

# --- Plot 3: Boxplot by Source ---
df_raw['short_source'] = df_raw['source'].apply(lambda x: 'DVSKTT' if 'DVSKTT' in x else ('KIEU' if 'KIEU' in x else ('LVT' if 'LVT' in x else 'Other')))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle('So Sánh Phân Phối Độ Dài Theo Nguồn Tác Phẩm', fontsize=14, fontweight='bold')
palette = {'DVSKTT': '#2980b9', 'KIEU': '#e84393', 'LVT': '#27ae60'}

sns.boxplot(data=df_raw, x='short_source', y='nom_char_len', ax=axes[0], palette=palette, hue='short_source', legend=False)
axes[0].set_title('Số Ký Tự Hán-Nôm', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Số ký tự', fontsize=10)
axes[0].set_xlabel('')

sns.boxplot(data=df_raw, x='short_source', y='vn_word_len', ax=axes[1], palette=palette, hue='short_source', legend=False)
axes[1].set_title('Số Âm Tiết Tiếng Việt', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Số từ/âm tiết', fontsize=10)
axes[1].set_xlabel('')

sns.boxplot(data=df_raw, x='short_source', y='ratio_vn_word_per_nom_char', ax=axes[2], palette=palette, hue='short_source', legend=False)
axes[2].set_title('Tỷ Lệ (Âm Tiết VN / Ký Tự Nôm)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Tỷ lệ', fontsize=10)
axes[2].set_xlabel('')

plt.tight_layout()
save_figure(fig, 'seq_length_by_source.png')

# --- Plot 4: Alignment Difference Distribution ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
align_counts = df_raw['align_group'].value_counts()
order = ['0 (Không lệch)', '1 (Lệch 1)', '2 (Lệch 2)', '3-5 (Lệch vừa)', '>5 (Lệch nặng/Lỗi)']
align_counts = align_counts.reindex([o for o in order if o in align_counts.index])

colors_align = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
ax1.bar(align_counts.index, align_counts.values, color=colors_align, edgecolor='black', alpha=0.85)
ax1.set_title('Mức Độ Chênh Lệch Căn Chỉnh (Align Diff = |Nôm - VN|)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Số câu', fontsize=10)
plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')
for i, v in enumerate(align_counts.values):
    ax1.text(i, v + max(align_counts.values)*0.01, f"{v:,}\n({v/len(df_raw):.1%})", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax1.set_ylim(0, max(align_counts.values)*1.18)

# Pie chart for perfect match vs small vs large diff
exact_0 = (df_raw['align_diff'] == 0).sum()
diff_1_2 = (df_raw['align_diff'].isin([1, 2])).sum()
diff_gt2 = (df_raw['align_diff'] > 2).sum()
labels_pie = ['Khớp hoàn toàn (diff=0)', 'Lệch nhẹ (diff 1-2)', 'Lệch nhiều/Nhiễu (diff >2)']
sizes_pie = [exact_0, diff_1_2, diff_gt2]
ax2.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', startangle=140, colors=['#2ecc71', '#3498db', '#e74c3c'], explode=(0.03, 0.03, 0.08), textprops={'fontsize': 9.5, 'fontweight': 'bold'})
ax2.set_title('Tỷ Lệ Chất Lượng Căn Chỉnh Dữ Liệu Raw', fontsize=12, fontweight='bold')

plt.tight_layout()
save_figure(fig, 'align_diff_distribution.png')

# --- Plot 5: Cumulative Distribution Function (CDF) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Phân Phối Tích Lũy (CDF) Để Xác Định MAX_SEQ_LENGTH', fontsize=14, fontweight='bold')

for idx, (col, label, color) in enumerate([('nom_char_len', 'Ký Tự Hán-Nôm', '#2980b9'), ('vn_word_len', 'Âm Tiết Tiếng Việt', '#27ae60')]):
    ax = axes[idx]
    sorted_vals = np.sort(df_raw[col].values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, color=color, linewidth=2.5, label='CDF')
    for pct in [0.90, 0.95, 0.99]:
        val = np.percentile(sorted_vals, pct * 100)
        ax.axhline(y=pct, color='gray', linestyle=':', alpha=0.6)
        ax.axvline(x=val, color='#d63031', linestyle='--', alpha=0.7)
        ax.annotate(f'{pct:.0%}: {int(val)}', xy=(val, pct), xytext=(val + 2, pct - 0.04),
                    fontsize=9, color='#d63031', fontweight='bold')
    ax.set_title(f'Phân Phối Tích Lũy - {label}', fontsize=12, fontweight='bold')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Tỷ lệ tích lũy (Cumulative Ratio)', fontsize=10)
    ax.set_ylim(0, 1.02)

plt.tight_layout()
save_figure(fig, 'seq_length_cdf.png')

# --- Plot 6: Polyphone Characters Analysis ---
poly_df = preprocessor.analyze_polyphones(min_occurrences=5)
print(f"Found {len(poly_df):,} polyphonic Nom characters with >= 5 occurrences.")

if not poly_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    top_poly = poly_df.head(12).iloc[::-1]
    y_pos = np.arange(len(top_poly))
    
    bars = ax.barh(y_pos, top_poly['total_occurrences'], color='#8e44ad', height=0.65, edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    labels_ytick = [f"{row['nom_char']} ({row['num_readings']} âm, chính: '{row['top_reading']}' {row['top_reading_pct']}%)" for _, row in top_poly.iterrows()]
    ax.set_yticklabels(labels_ytick, fontsize=10, fontweight='bold')
    ax.set_xlabel('Tổng số lần xuất hiện', fontsize=11)
    ax.set_title('Top 12 Chữ Nôm Đa Âm Phổ Biến Nhất', fontsize=13, fontweight='bold')
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(top_poly['total_occurrences'])*0.01, bar.get_y() + bar.get_height()/2, f"{int(w):,}", ha='left', va='center', fontsize=9, fontweight='bold')
        
    ax.set_xlim(0, max(top_poly['total_occurrences'])*1.12)
    plt.tight_layout()
    save_figure(fig, 'polyphone_top_chars.png')

# --- Plot 7: Vocab Long-Tail Analysis ---
vocab_stats = preprocessor.analyze_vocab_distribution()
nom_chars = [c for text in df_raw['nom'].astype(str) for c in text]
counts = Counter(nom_chars)
freqs = sorted(counts.values(), reverse=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(range(1, len(freqs)+1), freqs, color='#e67e22', linewidth=2)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_title('Phân Phối Tần Suất Ký Tự Hán-Nôm (Log-Log Scale)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Xếp hạng tần suất (Rank)', fontsize=10)
ax1.set_ylabel('Tần suất (Frequency - Log scale)', fontsize=10)

# Frequency categories
freq_1 = sum(1 for c, f in counts.items() if f == 1)
freq_2 = sum(1 for c, f in counts.items() if f == 2)
freq_3_10 = sum(1 for c, f in counts.items() if 3 <= f <= 10)
freq_gt10 = sum(1 for c, f in counts.items() if f > 10)

cats = ['1 lần (Đơn độc)', '2 lần', '3-10 lần', '>10 lần (Thường gặp)']
cat_counts = [freq_1, freq_2, freq_3_10, freq_gt10]
colors_cat = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']

ax2.bar(cats, cat_counts, color=colors_cat, edgecolor='black', alpha=0.85)
ax2.set_title('Phân Phối Tần Suất Chữ Hán-Nôm Theo Nhóm', fontsize=12, fontweight='bold')
ax2.set_ylabel('Số chữ unique', fontsize=10)
for i, v in enumerate(cat_counts):
    ax2.text(i, v + max(cat_counts)*0.015, f"{v:,}\n({v/len(counts):.1%})", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_ylim(0, max(cat_counts)*1.18)

plt.tight_layout()
save_figure(fig, 'vocab_long_tail.png')

# --- Plot 8: Dataset Split Breakdown ---
if not df_train.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    splits = ['Train', 'Validation', 'Test']
    split_counts = [len(df_train), len(df_val), len(df_test)]
    total_split = sum(split_counts)
    
    bars = ax.bar(splits, split_counts, color=['#2980b9', '#f39c12', '#27ae60'], width=0.5, edgecolor='black', alpha=0.85)
    ax.set_title('Phân Phối Số Câu Theo Tập Dữ Liệu (Split)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Số lượng mẫu', fontsize=11)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + total_split*0.01, f"{h:,}\n({h/total_split:.1%})", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    ax.set_ylim(0, max(split_counts)*1.18)
    plt.tight_layout()
    save_figure(fig, 'split_distribution.png')

print("--- EDA Calculations Finished Successfully! ---")
