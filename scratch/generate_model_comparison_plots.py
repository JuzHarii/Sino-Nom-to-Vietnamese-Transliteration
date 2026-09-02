import os
import sys
import math
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(r"c:\Study\HTK\final_proj\Sino-Nom-to-Vietnamese-Transliteration")
ARTIFACT_DIR = Path(r"C:\Users\HP Victus\.gemini\antigravity-ide\brain\319c95fb-328e-448f-92b3-432cfe19811c")
DOCS_IMG_DIR = PROJECT_ROOT / "docs" / "images"
RESULT_IMG_DIR = PROJECT_ROOT / "data" / "result"
ARTIFACT_IMG_DIR = ARTIFACT_DIR / "images"

for d in [DOCS_IMG_DIR, RESULT_IMG_DIR, ARTIFACT_IMG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans', 'Calibri']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.autolayout'] = False

# Load data
hcmus_m = pd.read_csv(PROJECT_ROOT / "data" / "result" / "hcmus" / "evaluation_metrics.csv")
mbart_m = pd.read_csv(PROJECT_ROOT / "data" / "result" / "mbart" / "evaluation_metrics.csv")
mt5_m = pd.read_csv(PROJECT_ROOT / "data" / "result" / "mt5" / "evaluation_metrics.csv")

hcmus_pred = pd.read_csv(PROJECT_ROOT / "data" / "result" / "hcmus" / "test_predictions.csv")
mbart_pred = pd.read_csv(PROJECT_ROOT / "data" / "result" / "mbart" / "test_predictions.csv")
mt5_pred = pd.read_csv(PROJECT_ROOT / "data" / "result" / "mt5" / "test_predictions.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

hcmus_pred['source'] = test_df['source']
mbart_pred['source'] = test_df['source']
mt5_pred['source'] = test_df['source']

metrics_all = pd.concat([hcmus_m, mbart_m, mt5_m], ignore_index=True)
metrics_all['Model_Name'] = ['HCMUS Online API', 'mBART-50 (5 Epochs)', 'mT5-Small (5 Epochs)']
metrics_all['Model_Short'] = ['HCMUS API', 'mBART-5ep', 'mT5-5ep']

def save_fig(fig, filename):
    fig.savefig(DOCS_IMG_DIR / filename, dpi=250, bbox_inches='tight')
    fig.savefig(ARTIFACT_IMG_DIR / filename, dpi=200, bbox_inches='tight')
    fig.savefig(RESULT_IMG_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")

# Define consistent model palette
MODEL_COLORS = {
    'HCMUS API': '#2b5c8f',        # Deep Blue
    'mBART-5ep': '#d9534f',        # Crimson / Coral Red
    'mT5-5ep': '#f0ad4e',          # Amber / Orange
}
MODEL_NAMES = ['HCMUS API', 'mBART-5ep', 'mT5-5ep']

# ==============================================================================
# PLOT 1: Overall Performance Comparison (Grouped Bar Chart)
# ==============================================================================
fig, ax = plt.subplots(figsize=(13, 6))

metrics_to_compare = [
    ('BLEU', 'BLEU Score'),
    ('Accuracy_EM_%', 'Exact Match (EM %)'),
    ('Accuracy_Syllable_%', 'Syllable Acc (%)'),
    ('Accuracy_Char_%', 'Char Acc (%)'),
    ('ROUGEL_%', 'ROUGE-L (%)'),
    ('ROUGE1_%', 'ROUGE-1 (%)'),
]

x = np.arange(len(metrics_to_compare))
width = 0.26
offsets = [-width, 0, width]

for i, (m_short, offset) in enumerate(zip(MODEL_NAMES, offsets)):
    row = metrics_all[metrics_all['Model_Short'] == m_short].iloc[0]
    vals = [row[col] for col, _ in metrics_to_compare]
    bars = ax.bar(x + offset, vals, width, label=metrics_all[metrics_all['Model_Short'] == m_short]['Model_Name'].values[0],
                  color=MODEL_COLORS[m_short], edgecolor='black', linewidth=0.8, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}", ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', rotation=0)

ax.set_ylabel('Score / Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison of Models on Test Set (1,449 Samples)', fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([name for _, name in metrics_to_compare], fontsize=11, fontweight='bold')
ax.set_ylim(0, 108)
ax.legend(loc='upper right', frameon=True, fontsize=10.5, shadow=True)
ax.grid(axis='y', linestyle='--', alpha=0.7)

save_fig(fig, 'model_comparison_overview.png')

# ==============================================================================
# PLOT 2: Error Rates (CER vs WER) - Lower is Better
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

err_metrics = [('CER_%', 'Character Error Rate (CER %)'), ('WER_%', 'Word Error Rate (WER %)')]
x_err = np.arange(len(err_metrics))
width_err = 0.24

for i, (m_short, offset) in enumerate(zip(MODEL_NAMES, [-width_err, 0, width_err])):
    row = metrics_all[metrics_all['Model_Short'] == m_short].iloc[0]
    vals = [row[col] for col, _ in err_metrics]
    bars = ax.bar(x_err + offset, vals, width_err, label=metrics_all[metrics_all['Model_Short'] == m_short]['Model_Name'].values[0],
                  color=MODEL_COLORS[m_short], edgecolor='black', linewidth=0.8, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.6, f"{h:.2f}%", ha='center', va='bottom',
                fontsize=10, fontweight='bold')

ax.set_ylabel('Error Rate (%) [Lower is Better]', fontsize=11, fontweight='bold')
ax.set_title('Error Rate Comparison: CER vs. WER (Lower is Better)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_err)
ax.set_xticklabels([name for _, name in err_metrics], fontsize=11.5, fontweight='bold')
ax.set_ylim(0, 45)
ax.legend(loc='upper left', frameon=True, fontsize=10, shadow=True)
ax.annotate('Lower is better', xy=(0.02, 0.92), xycoords='axes fraction', fontsize=11, color='#c0392b', fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.7)

save_fig(fig, 'model_error_rates.png')

# ==============================================================================
# PLOT 3: Radar Chart (Spider Chart)
# ==============================================================================
radar_categories = ['BLEU', 'Exact Match', 'Char Acc', 'Syllable Acc', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Word Accuracy\n(100 - WER)']
N = len(radar_categories)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for m_short in MODEL_NAMES:
    row = metrics_all[metrics_all['Model_Short'] == m_short].iloc[0]
    values = [
        row['BLEU'],
        row['Accuracy_EM_%'],
        row['Accuracy_Char_%'],
        row['Accuracy_Syllable_%'],
        row['ROUGE1_%'],
        row['ROUGE2_%'],
        row['ROUGEL_%'],
        100.0 - row['WER_%']
    ]
    values += values[:1]
    
    full_name = metrics_all[metrics_all['Model_Short'] == m_short]['Model_Name'].values[0]
    ax.plot(angles, values, linewidth=2.2, linestyle='solid', label=full_name, color=MODEL_COLORS[m_short])
    ax.fill(angles, values, color=MODEL_COLORS[m_short], alpha=0.18)

ax.set_theta_offset(math.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_categories, fontsize=10, fontweight='bold')
ax.set_rlabel_position(30)
plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=9)
plt.ylim(0, 105)
plt.title('Multidimensional Performance Radar Chart', size=14, fontweight='bold', y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.32, 1.12), fontsize=10, frameon=True)

save_fig(fig, 'model_radar_chart.png')

# ==============================================================================
# PLOT 4: Breakdown by Literary Source (DVSKTT vs Kiều vs Lục Vân Tiên)
# ==============================================================================
models_pred = [
    ('HCMUS API', hcmus_pred),
    ('mBART-5ep', mbart_pred),
    ('mT5-5ep', mt5_pred)
]

source_labels = ['DVSKTT (Chronicle)', 'The Tale of Kieu (Poetry)', 'Luc Van Tien (Southern Verse)']
source_keys = ['DVSKTT', 'KIEU (Truyện Kiều)', 'LVT (Lục Vân Tiên)']

# Calculate EM and Syllable Acc per source
em_per_source = {m: [] for m in MODEL_NAMES}
syl_per_source = {m: [] for m in MODEL_NAMES}

def calc_syl_acc(preds, refs):
    correct, total = 0, 0
    for p, r in zip(preds, refs):
        p_t = str(p).split()
        r_t = str(r).split()
        total += max(len(p_t), len(r_t))
        for pt, rt in zip(p_t, r_t):
            if pt == rt:
                correct += 1
    return (correct / total * 100) if total > 0 else 0

for m_short, df in models_pred:
    for src in source_keys:
        sub = df[df['source'] == src]
        em = sub['is_exact_match'].mean() * 100
        syl = calc_syl_acc(sub['prediction'], sub['vietnamese_clean'])
        em_per_source[m_short].append(em)
        syl_per_source[m_short].append(syl)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle('Model Performance Breakdown by Literary Genre & Source', fontsize=15, fontweight='bold', y=0.98)

x_src = np.arange(len(source_labels))
width_src = 0.25

# Subplot 1: Exact Match by Source
for i, m_short in enumerate(MODEL_NAMES):
    offset = (i - 1) * width_src
    bars = ax1.bar(x_src + offset, em_per_source[m_short], width_src, label=m_short,
                   color=MODEL_COLORS[m_short], edgecolor='black', linewidth=0.8, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%", ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold')

ax1.set_title('Exact Match Accuracy (EM %)', fontsize=12, fontweight='bold', pad=10)
ax1.set_ylabel('Exact Match (%)', fontsize=11, fontweight='bold')
ax1.set_xticks(x_src)
ax1.set_xticklabels(source_labels, fontsize=10.5, fontweight='bold')
ax1.set_ylim(0, 65)
ax1.legend(loc='upper right', frameon=True, fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 2: Syllable Accuracy by Source
for i, m_short in enumerate(MODEL_NAMES):
    offset = (i - 1) * width_src
    bars = ax2.bar(x_src + offset, syl_per_source[m_short], width_src, label=m_short,
                   color=MODEL_COLORS[m_short], edgecolor='black', linewidth=0.8, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold')

ax2.set_title('Syllable Accuracy (%)', fontsize=12, fontweight='bold', pad=10)
ax2.set_ylabel('Syllable Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_xticks(x_src)
ax2.set_xticklabels(source_labels, fontsize=10.5, fontweight='bold')
ax2.set_ylim(0, 112)
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.94])
save_fig(fig, 'model_performance_by_genre.png')

# ==============================================================================
# PLOT 5: Length Ratio Distribution (Prediction vs Ground Truth Alignment)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
fig.suptitle('Inference Length Ratio Distribution: Predicted Syllables / Reference Syllables (L_pred / L_ref)', fontsize=14, fontweight='bold', y=0.98)

for idx, (m_short, df) in enumerate(models_pred):
    ax = axes[idx]
    len_pred = df['prediction'].astype(str).apply(lambda s: len(s.split()))
    len_ref = df['vietnamese_clean'].astype(str).apply(lambda s: len(s.split()))
    ratio = len_pred / len_ref
    
    sns.histplot(ratio, bins=40, kde=True, ax=ax, color=MODEL_COLORS[m_short], edgecolor='black', alpha=0.6)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Exact 1:1 (L_pred = L_ref)')
    
    exact_ratio = (ratio == 1.0).mean() * 100
    mean_ratio = ratio.mean()
    std_ratio = ratio.std()
    
    ax.set_title(f"{m_short}\n(Exact 1:1: {exact_ratio:.1f}% | Mean: {mean_ratio:.2f})", fontsize=11, fontweight='bold')
    ax.set_xlabel('Length Ratio (L_pred / L_ref)', fontsize=10)
    if idx == 0:
        ax.set_ylabel('Sample Count', fontsize=10)
    ax.set_xlim(0.4, 1.8)
    ax.legend(fontsize=9, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.94])
save_fig(fig, 'model_prediction_length_ratio.png')

print("All 5 comparison plots generated with English text successfully!")
