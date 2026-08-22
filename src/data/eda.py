import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import RawCorpusLoader
from data.preprocessor import DataPreprocessor


def print_statistics(df: pd.DataFrame) -> None:
    """In thống kê tổng quan và theo từng nguồn dữ liệu."""
    print("---- Thống kê độ dài chuỗi ----\n")
    cols = {
        'nom_char_len': 'Số ký tự Hán-Nôm',
        'vn_word_len': 'Số âm tiết Quốc ngữ',
        'vn_char_len': 'Số ký tự Quốc ngữ',
        'ratio_vn_word_per_nom_char': 'Tỷ lệ (Âm tiết VN / Ký tự Nôm)',
    }
    percentiles = [0.25, 0.50, 0.75, 0.95, 0.99]
    
    for col, label in cols.items():
        s = df[col].dropna()
        print(f"\n--- {label} ---")
        print(f"  Số lượng : {len(s):,}\n  Trung bình : {s.mean():.2f}\n  Độ lệch chuẩn: {s.std():.2f}\n  Nhỏ nhất  : {s.min():.2f}")
        for p in percentiles:
            print(f"  Phân vị {int(p*100)}% : {s.quantile(p):.2f}")
        print(f"  Lớn nhất  : {s.max():.2f}")

    print("\n--- Thống kê theo nguồn tác phẩm ---")
    
    stats = df.groupby('source').agg(
        count=('nom_char_len', 'count'),
        nom_mean=('nom_char_len', 'mean'),
        nom_median=('nom_char_len', 'median'),
        nom_max=('nom_char_len', 'max'),
        vn_word_mean=('vn_word_len', 'mean'),
        vn_word_median=('vn_word_len', 'median'),
        vn_word_max=('vn_word_len', 'max'),
        ratio_mean=('ratio_vn_word_per_nom_char', 'mean'),
        ratio_std=('ratio_vn_word_per_nom_char', 'std'),
    ).round(2)
    print(f"\n{stats.to_string()}\n\n--- Số dòng theo tệp ---")
    for fname, count in df['file_name'].value_counts().sort_index().items():
        print(f"  {fname}: {count:,} dòng")
    print(f"  TỔNG CỘNG: {len(df):,} dòng")


def detect_outliers(df: pd.DataFrame, col: str, label: str, n_show: int = 10) -> None:
    """Phát hiện và in các câu ngoại lệ (outliers) dựa trên phương pháp IQR."""
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"\n--- Ngoại lệ đối với {label} (Phương pháp IQR) ---")
    print(f"  Khoảng IQR hợp lệ: [{lower:.1f}, {upper:.1f}]")
    print(f"  Tổng số ngoại lệ: {len(outliers):,} / {len(df):,} ({100*len(outliers)/len(df):.1f}%)")

    if not outliers.empty:
        print(f"  Top {min(n_show, len(outliers))} câu dài nhất:")
        for _, row in outliers.nlargest(n_show, col).iterrows():
            nom_p = row['nom'][:40] + ('...' if len(row['nom']) > 40 else '')
            vn_p = str(row['vietnamese'])[:50] + ('...' if len(str(row['vietnamese'])) > 50 else '')
            print(f"    [{row['file_name']}] {col}={row[col]:.0f} | {nom_p} → {vn_p}")


def plot_length_distributions(df: pd.DataFrame, save_dir: Path) -> None:
    """Vẽ và lưu 4 biểu đồ phân phối độ dài chuỗi."""
    save_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Biểu đồ tần suất (Histograms)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phân Phối Độ Dài Chuỗi Dữ Liệu', fontsize=16, fontweight='bold')
    hist_configs = [
        ('nom_char_len', 'Số ký tự Hán-Nôm', 'Ký tự', '#2196F3', axes[0, 0]),
        ('vn_word_len', 'Số âm tiết Tiếng Việt', 'Từ/Âm tiết', '#4CAF50', axes[0, 1]),
        ('vn_char_len', 'Số ký tự Tiếng Việt', 'Ký tự', '#FF9800', axes[1, 0]),
        ('ratio_vn_word_per_nom_char', 'Tỷ lệ: Âm tiết VN / Ký tự Nôm', 'Tỷ lệ', '#9C27B0', axes[1, 1])
    ]
    for col, title, xlabel, color, ax in hist_configs:
        data = df[col].dropna()
        sns.histplot(data, bins=50, kde=True, ax=ax, color=color)
        ax.set(title=title, xlabel=xlabel)
        fmt = ".2f" if col == 'ratio_vn_word_per_nom_char' else ".0f"
        ax.axvline(data.median(), color='red', linestyle='--', label=f"Trung vị={data.median():{fmt}}")
        ax.legend()

    plt.tight_layout()
    fig.savefig(save_dir / 'seq_length_distributions.png', dpi=150, bbox_inches='tight')

    # 2. Biểu đồ hộp (Boxplots) theo nguồn tác phẩm
    df_clean_source = df.copy()
    df_clean_source['short_source'] = df_clean_source['source'].str.split(' ').str[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Độ Dài Chuỗi Theo Nguồn Tác Phẩm', fontsize=16, fontweight='bold')
    palette = {'DVSKTT': '#2196F3', 'KIEU': '#E91E63', 'LVT': '#4CAF50'}
    
    box_configs = [
        ('nom_char_len', 'Số ký tự Hán-Nôm', 'Ký tự', axes[0]),
        ('vn_word_len', 'Số âm tiết Tiếng Việt', 'Từ/Âm tiết', axes[1]),
        ('ratio_vn_word_per_nom_char', 'Tỷ lệ (Âm tiết VN / Ký tự Nôm)', 'Tỷ lệ', axes[2])
    ]
    for col, title, ylabel, ax in box_configs:
        sns.boxplot(data=df_clean_source, x='short_source', y=col, hue='short_source', ax=ax, palette=palette, legend=False)
        ax.set(title=title, xlabel='', ylabel=ylabel)

    plt.tight_layout()
    fig.savefig(save_dir / 'seq_length_by_source.png', dpi=150, bbox_inches='tight')

    # 3. Biểu đồ phân tán (Scatter plot)
    fig, ax = plt.subplots(figsize=(10, 7))
    for src_name, group in df_clean_source.groupby('short_source'):
        ax.scatter(group['nom_char_len'], group['vn_word_len'], alpha=0.3, s=10, label=src_name, color=palette.get(src_name, 'gray'))
    max_val = max(df['nom_char_len'].max(), df['vn_word_len'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Đường 1:1')
    ax.set(xlabel='Số ký tự Hán-Nôm', ylabel='Số âm tiết Tiếng Việt', title='Tương Quan Độ Dài: Nôm vs Tiếng Việt')
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    fig.savefig(save_dir / 'nom_vs_vn_scatter.png', dpi=150, bbox_inches='tight')

    # 4. Biểu đồ tích lũy (CDF)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phân Phối Tích Lũy (Dùng Để Chọn MAX_SEQ_LENGTH)', fontsize=14, fontweight='bold')
    cdf_configs = [
        ('nom_char_len', 'Số ký tự Hán-Nôm', '#2196F3', axes[0]),
        ('vn_word_len', 'Số âm tiết Tiếng Việt', '#4CAF50', axes[1])
    ]
    for col, label, color, ax in cdf_configs:
        sorted_vals = np.sort(df[col].values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color=color, linewidth=2)
        for pct in [0.90, 0.95, 0.99]:
            val = np.percentile(sorted_vals, pct * 100)
            ax.axhline(y=pct, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=val, color='red', linestyle='--', alpha=0.5)
            ax.annotate(f'{pct:.0%}: {val:.0f}', xy=(val, pct), fontsize=9, color='red', ha='left', va='bottom')
        ax.set(xlabel=label, ylabel='Tỷ lệ tích lũy', ylim=(0, 1.02))

    plt.tight_layout()
    fig.savefig(save_dir / 'seq_length_cdf.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"\n[Đã lưu] Các biểu đồ được ghi thành công vào {save_dir}")


def run_sequence_length_analysis(df: pd.DataFrame, save_dir: Path) -> pd.DataFrame:
    """Chạy toàn bộ pipeline phân tích EDA độ dài chuỗi dữ liệu."""
    print("---- Phân Tích Độ Dài Chuỗi Dữ Liệu ----\n")
    preprocessor = DataPreprocessor(df)
    df = preprocessor.extract_features()
    print_statistics(df)

    print("\n---- Phát Hiện Ngoại Lệ (Outliers) ----\n")
    for col, label in [
        ('nom_char_len', 'Số ký tự Hán-Nôm'),
        ('vn_word_len', 'Số âm tiết Tiếng Việt'),
        ('ratio_vn_word_per_nom_char', 'Tỷ lệ (Âm tiết VN / Ký tự Nôm)')
    ]:
        detect_outliers(df, col, label)

    print("\n---- Vẽ Biểu Đồ Phân Phối ----\n")
    plot_length_distributions(df, save_dir)
    return df


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    loader = RawCorpusLoader(data_dir=str(PROJECT_ROOT / "data" / "raw"))
    df = loader.load()
    print(f"Đã nạp {len(df):,} dòng từ {df['file_name'].nunique()} tệp dữ liệu.\n")
    run_sequence_length_analysis(df, save_dir=PROJECT_ROOT / "notebooks" / "eda_plots")