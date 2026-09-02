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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(r"c:\Study\HTK\final_proj\Sino-Nom-to-Vietnamese-Transliteration")
ARTIFACT_DIR = Path(r"C:\Users\HP Victus\.gemini\antigravity-ide\brain\319c95fb-328e-448f-92b3-432cfe19811c")
IMG_DIR = PROJECT_ROOT / "docs" / "images"
NOTEBOOK_RAW_DIR = PROJECT_ROOT / "notebooks" / "eda_plots_raw"
NOTEBOOK_CLEAN_DIR = PROJECT_ROOT / "notebooks" / "eda_plots_clean"

for p in [IMG_DIR, ARTIFACT_DIR / "images", NOTEBOOK_RAW_DIR, NOTEBOOK_CLEAN_DIR]:
    p.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.loader import load_raw_corpus
from data.preprocessor import DataPreprocessor, compute_length_features

print("--- Step 1: Loading Raw and Processed Datasets ---")
df_raw = load_raw_corpus(PROJECT_ROOT / "data" / "raw")
print(f"Loaded raw dataset: {len(df_raw):,} rows from {df_raw['file_name'].nunique()} files.")

train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
val_path = PROJECT_ROOT / "data" / "processed" / "val.csv"
test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"

df_train = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
df_val = pd.read_csv(val_path) if val_path.exists() else pd.DataFrame()
df_test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()

df_raw = compute_length_features(df_raw, vn_col='vietnamese')

preprocessor = DataPreprocessor(df_raw)
df_clean = preprocessor.clean_corpus(augment=False)
df_clean_filtered = df_clean[df_clean['align_diff'] <= 2].copy()

# Configure plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Microsoft YaHei', 'SimSun', 'Arial', 'DejaVu Sans', 'Calibri']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def save_figure(fig, name):
    path1 = IMG_DIR / name
    path2 = ARTIFACT_DIR / "images" / name
    fig.savefig(path1, dpi=250, bbox_inches='tight')
    fig.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {name}")

# --- Plot 1: Source & File Distribution ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
source_counts = df_raw['source'].value_counts()
source_display_names = {
    'DVSKTT': 'DVSKTT (Chronicle)',
    'KIEU (Truyện Kiều)': 'The Tale of Kieu',
    'LVT (Lục Vân Tiên)': 'Luc Van Tien'
}
labels_s = [source_display_names.get(s, s) for s in source_counts.index]
colors1 = ['#1f77b4', '#e377c2', '#2ca02c', '#ff7f0e']
ax1.bar(labels_s, source_counts.values, color=colors1[:len(source_counts)], width=0.55, edgecolor='black', alpha=0.85)
ax1.set_title('Sample Count by Literary Source (Raw Data)', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('Number of Samples', fontsize=11)
for i, v in enumerate(source_counts.values):
    ax1.text(i, v + max(source_counts.values)*0.015, f"{v:,}\n({v/len(df_raw):.1%})", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax1.set_ylim(0, max(source_counts.values)*1.15)

file_counts = df_raw['file_name'].value_counts().sort_index()
ax2.barh(file_counts.index, file_counts.values, color='#3498db', height=0.6, edgecolor='black', alpha=0.85)
ax2.set_title('Sample Count by Text File (.txt)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Number of Samples', fontsize=11)
for i, v in enumerate(file_counts.values):
    ax2.text(v + max(file_counts.values)*0.01, i, f" {v:,}", ha='left', va='center', fontsize=9, fontweight='bold')
ax2.set_xlim(0, max(file_counts.values)*1.15)

plt.tight_layout()
save_figure(fig, 'source_file_distribution.png')

# --- Plot 2: Length Distributions ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Raw Data Sequence Length Distributions (Characters & Syllables)', fontsize=15, fontweight='bold', y=0.98)

configs = [
    ('nom_char_len', 'Sino-Nom Character Count (Sentence)', 'Characters', '#2980b9', axes[0, 0]),
    ('vn_word_len', 'Vietnamese Syllable Count (Sentence)', 'Words / Syllables', '#27ae60', axes[0, 1]),
    ('vn_char_len', 'Vietnamese Character Count (String)', 'Characters', '#e67e22', axes[1, 0]),
    ('ratio_vn_word_per_nom_char', 'Ratio (VN Syllables / Nom Characters)', 'Ratio', '#8e44ad', axes[1, 1])
]

for col, title, xlabel, color, ax in configs:
    data = df_raw[col].dropna()
    sns.histplot(data, bins=45, kde=True, ax=ax, color=color, edgecolor='black', alpha=0.6)
    ax.set_title(title, fontsize=11.5, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    med = data.median()
    mean = data.mean()
    fmt = ".2f" if col == 'ratio_vn_word_per_nom_char' else ".1f"
    ax.axvline(med, color='red', linestyle='--', linewidth=1.5, label=f"Median: {med:{fmt}}")
    ax.axvline(mean, color='darkgreen', linestyle=':', linewidth=1.5, label=f"Mean: {mean:{fmt}}")
    ax.legend(fontsize=9, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_figure(fig, 'seq_length_distributions.png')

# --- Plot 3: Boxplot by Source ---
df_raw['short_source'] = df_raw['source'].apply(lambda x: 'DVSKTT' if 'DVSKTT' in x else ('KIEU' if 'KIEU' in x else ('LVT' if 'LVT' in x else 'Other')))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle('Sequence Length Distribution Comparison by Literary Source', fontsize=14, fontweight='bold')
palette = {'DVSKTT': '#2980b9', 'KIEU': '#e84393', 'LVT': '#27ae60'}

sns.boxplot(data=df_raw, x='short_source', y='nom_char_len', ax=axes[0], palette=palette, hue='short_source', legend=False)
axes[0].set_title('Sino-Nom Character Count', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Characters', fontsize=10)
axes[0].set_xlabel('Source')

sns.boxplot(data=df_raw, x='short_source', y='vn_word_len', ax=axes[1], palette=palette, hue='short_source', legend=False)
axes[1].set_title('Vietnamese Syllable Count', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Words / Syllables', fontsize=10)
axes[1].set_xlabel('Source')

sns.boxplot(data=df_raw, x='short_source', y='ratio_vn_word_per_nom_char', ax=axes[2], palette=palette, hue='short_source', legend=False)
axes[2].set_title('Ratio (VN Syllables / Nom Characters)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Ratio', fontsize=10)
axes[2].set_xlabel('Source')

plt.tight_layout()
save_figure(fig, 'seq_length_by_source.png')

# --- Plot 4: Alignment Difference Distribution ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
align_counts = df_raw['align_group'].value_counts()
mapping_eng = {
    '0 (Không lệch)': '0 (Exact 1:1)',
    '1 (Lệch 1)': '1 (Diff 1)',
    '2 (Lệch 2)': '2 (Diff 2)',
    '3-5 (Lệch vừa)': '3-5 (Moderate Diff)',
    '>5 (Lệch nặng/Lỗi)': '>5 (Severe Error)'
}
order_eng = ['0 (Exact 1:1)', '1 (Diff 1)', '2 (Diff 2)', '3-5 (Moderate Diff)', '>5 (Severe Error)']
align_counts_eng = pd.Series({mapping_eng.get(k, k): v for k, v in align_counts.items()}).reindex(order_eng).dropna()

colors_align = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
ax1.bar(align_counts_eng.index, align_counts_eng.values, color=colors_align, edgecolor='black', alpha=0.85)
ax1.set_title('Alignment Difference (|Nom Chars - VN Words|)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Sentences', fontsize=10)
plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')
for i, v in enumerate(align_counts_eng.values):
    ax1.text(i, v + max(align_counts_eng.values)*0.01, f"{v:,}\n({v/len(df_raw):.1%})", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax1.set_ylim(0, max(align_counts_eng.values)*1.18)

# Pie chart for perfect match vs small vs large diff
exact_0 = (df_raw['align_diff'] == 0).sum()
diff_1_2 = (df_raw['align_diff'].isin([1, 2])).sum()
diff_gt2 = (df_raw['align_diff'] > 2).sum()
labels_pie = ['Exact 1:1 (diff=0)', 'Minor Diff (diff 1-2)', 'Severe Error / Noise (diff >2)']
sizes_pie = [exact_0, diff_1_2, diff_gt2]
ax2.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', startangle=140, colors=['#2ecc71', '#3498db', '#e74c3c'], explode=(0.03, 0.03, 0.08), textprops={'fontsize': 9.5, 'fontweight': 'bold'})
ax2.set_title('Raw Data Alignment Quality Breakdown', fontsize=12, fontweight='bold')

plt.tight_layout()
save_figure(fig, 'align_diff_distribution.png')

# --- Plot 5: Cumulative Distribution Function (CDF) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Cumulative Distribution Function (CDF) for Determining MAX_SEQ_LENGTH', fontsize=14, fontweight='bold')

for idx, (col, label, color) in enumerate([('nom_char_len', 'Sino-Nom Characters', '#2980b9'), ('vn_word_len', 'Vietnamese Syllables', '#27ae60')]):
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
    ax.set_title(f'Cumulative Distribution — {label}', fontsize=12, fontweight='bold')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Cumulative Ratio', fontsize=10)
    ax.set_ylim(0, 1.02)

plt.tight_layout()
save_figure(fig, 'seq_length_cdf.png')

# --- Plot 6: Polyphone Characters Analysis ---
poly_df = preprocessor.analyze_polyphones(min_occurrences=5)
if not poly_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    top_poly = poly_df.head(12).iloc[::-1]
    y_pos = np.arange(len(top_poly))
    
    bars = ax.barh(y_pos, top_poly['total_occurrences'], color='#8e44ad', height=0.65, edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    labels_ytick = [f"{row['nom_char']} ({row['num_readings']} readings, top: '{row['top_reading']}' {row['top_reading_pct']}%)" for _, row in top_poly.iterrows()]
    ax.set_yticklabels(labels_ytick, fontsize=10, fontweight='bold')
    ax.set_xlabel('Total Occurrences', fontsize=11)
    ax.set_title('Top 12 Most Frequent Polyphonic Sino-Nom Characters', fontsize=13, fontweight='bold')
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(top_poly['total_occurrences'])*0.01, bar.get_y() + bar.get_height()/2, f"{int(w):,}", ha='left', va='center', fontsize=9, fontweight='bold')
        
    ax.set_xlim(0, max(top_poly['total_occurrences'])*1.12)
    plt.tight_layout()
    save_figure(fig, 'polyphone_top_chars.png')

# --- Plot 7: Vocab Long-Tail Analysis ---
nom_chars = [c for text in df_raw['nom'].astype(str) for c in text]
counts = Counter(nom_chars)
freqs = sorted(counts.values(), reverse=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(range(1, len(freqs)+1), freqs, color='#e67e22', linewidth=2)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_title('Sino-Nom Character Frequency Distribution (Log-Log Scale)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Frequency Rank', fontsize=10)
ax1.set_ylabel('Frequency (Log scale)', fontsize=10)

freq_1 = sum(1 for c, f in counts.items() if f == 1)
freq_2 = sum(1 for c, f in counts.items() if f == 2)
freq_3_10 = sum(1 for c, f in counts.items() if 3 <= f <= 10)
freq_gt10 = sum(1 for c, f in counts.items() if f > 10)

cats = ['1 time (Singleton)', '2 times', '3-10 times', '>10 times (Frequent)']
cat_counts = [freq_1, freq_2, freq_3_10, freq_gt10]
colors_cat = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']

ax2.bar(cats, cat_counts, color=colors_cat, edgecolor='black', alpha=0.85)
ax2.set_title('Sino-Nom Character Frequency by Group', fontsize=12, fontweight='bold')
ax2.set_ylabel('Unique Characters', fontsize=10)
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
    ax.set_title('Dataset Split Distribution (Train / Validation / Test)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=11)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + total_split*0.01, f"{h:,}\n({h/total_split:.1%})", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    ax.set_ylim(0, max(split_counts)*1.18)
    plt.tight_layout()
    save_figure(fig, 'split_distribution.png')

# --- Plot 9: Scatter plot (Nom vs VN) ---
fig, ax = plt.subplots(figsize=(8, 7))
palette_sc = {'DVSKTT': '#2196F3', 'KIEU': '#E91E63', 'LVT': '#4CAF50'}
for src_name, group in df_clean_filtered.groupby(df_clean_filtered['source'].apply(lambda x: 'DVSKTT' if 'DVSKTT' in x else ('KIEU' if 'KIEU' in x else 'LVT'))):
    ax.scatter(group['nom_char_len'], group['vn_word_len'], alpha=0.3, s=12, label=src_name, color=palette_sc.get(src_name, 'gray'))
max_val = max(df_clean_filtered['nom_char_len'].max(), df_clean_filtered['vn_word_len'].max())
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, label='Identity Line 1:1')
ax.set_xlabel('Sino-Nom Character Count', fontsize=11, fontweight='bold')
ax.set_ylabel('Vietnamese Syllable Count', fontsize=11, fontweight='bold')
ax.set_title('Sequence Length Correlation: Sino-Nom vs. Vietnamese', fontsize=13, fontweight='bold', pad=10)
ax.legend(frameon=True, fontsize=10)
ax.set_aspect('equal', adjustable='datalim')

plt.tight_layout()
save_figure(fig, 'nom_vs_vn_scatter.png')
# Save to notebook dirs too
fig_sc_path1 = NOTEBOOK_CLEAN_DIR / 'nom_vs_vn_scatter.png'
fig_sc_path2 = NOTEBOOK_RAW_DIR / 'nom_vs_vn_scatter.png'

print("--- All English EDA Plots Generated Successfully! ---")
