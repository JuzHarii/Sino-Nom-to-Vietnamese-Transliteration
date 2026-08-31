import re
import unicodedata
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Các mẫu biểu thức chính quy (Regex) đã biên dịch sẵn để tối ưu tốc độ xử lý
RE_QUESTION_MARK = re.compile(r'\[\?\]')
RE_BRACKETS = re.compile(r'\[|\]')
RE_PUNCTUATION = re.compile(r'[\.,;\!\?:\-\"“\'”]')
RE_WHITESPACE = re.compile(r'\s+')
RE_SLASH_OPTION = re.compile(r'\[([^\]]+)/([^\]]+)\]')


def get_source_category(filename: str) -> str:
    """Phân loại nguồn văn bản (thơ/lịch sử) dựa trên tên tệp tin."""
    if filename.startswith('DVSKTT'):
        return 'DVSKTT'
    if filename.startswith('KIEU'):
        return 'KIEU (Truyện Kiều)'
    if filename.startswith('LVT'):
        return 'LVT (Lục Vân Tiên)'
    return 'Other'


def categorize_align_diff(diff: int) -> str:
    """Phân loại độ chênh lệch giữa số ký tự Nôm và số âm tiết Quốc ngữ thành các nhóm."""
    if diff == 0:
        return '0 (Không lệch)'
    if diff == 1:
        return '1 (Lệch 1)'
    if diff == 2:
        return '2 (Lệch 2)'
    if 3 <= diff <= 5:
        return '3-5 (Lệch vừa)'
    return '>5 (Lệch nặng/Lỗi)'


def compute_length_features(df: pd.DataFrame, vn_col: str = 'vietnamese') -> pd.DataFrame:
    """Tính toán các chỉ số độ dài và mức độ chênh lệch căn chỉnh (alignment) cho dữ liệu."""
    res = df.copy()
    if 'source' not in res.columns and 'file_name' in res.columns:
        res['source'] = res['file_name'].apply(get_source_category)

    nom_str = res['nom'].astype(str)
    vn_str = res[vn_col].astype(str)

    res['nom_char_len'] = nom_str.str.len()
    res['vn_word_len'] = vn_str.str.split().str.len()
    res['vn_char_len'] = vn_str.str.len()

    res['align_diff'] = (res['nom_char_len'] - res['vn_word_len']).abs()
    res['align_group'] = res['align_diff'].apply(categorize_align_diff)
    res['ratio_vn_word_per_nom_char'] = res['vn_word_len'] / res['nom_char_len'].replace(0, np.nan)
    return res


def clean_text(text: str) -> str:
    """Làm sạch văn bản Quốc ngữ: loại bỏ dấu câu, normalize khoảng trắng và lowercase."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = RE_QUESTION_MARK.sub('', text)
    text = RE_BRACKETS.sub('', text)
    text = RE_PUNCTUATION.sub('', text)
    text = text.lower()
    return RE_WHITESPACE.sub(' ', text).strip()


class DataPreprocessor:
    """Class chuyên tính toán feature, phân tích chất lượng data và extract chỉ số cho bài toán Transliteration."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _get_source_category(self, filename: str) -> str:
        return get_source_category(filename)

    def _categorize_align_diff(self, diff: int) -> str:
        return categorize_align_diff(diff)

    def extract_features(self, vn_col: str = 'vietnamese') -> pd.DataFrame:
        """Trích xuất các đặc trưng độ dài và alignment giữa Hán-Nôm và Quốc ngữ."""
        self.df = compute_length_features(self.df, vn_col=vn_col)
        return self.df

    def clean_text(self, text: str) -> str:
        return clean_text(text)

    def clean_corpus(self, augment: bool = True) -> pd.DataFrame:
        """Thực hiện pipeline làm sạch dữ liệu theo 5 tiêu chuẩn chuẩn hóa:
        1. Dấu ... (thiếu phiên âm): Loại bỏ dòng.
        2. Dấu câu cuối không nhất quán: Loại bỏ toàn bộ dấu câu.
        3. Ngoặc [/] (ví dụ: [búa/vó]):
           - augment=True  (Train): tách thành 2 mẫu ('búa' và 'vó').
           - augment=False (Val/Test): chỉ giữ phương án đầu tiên ('búa').
        4. Unicode normalization: NFC normalization (giữ nguyên PUA).
        5. Khoảng trắng thừa: Normalize whitespace và strip.
        """
        rows = []
        df_raw = self.df.copy()

        for _, row in df_raw.iterrows():
            nom_text = str(row['nom'])
            vn_text = str(row['vietnamese'])
            file_name = row['file_name']
            source = row.get('source', '')

            nom_text = unicodedata.normalize('NFC', nom_text)
            vn_text = unicodedata.normalize('NFC', vn_text)

            # 1. Dấu ... (thiếu phiên âm) -> Bỏ qua dòng
            if '...' in vn_text:
                continue

            # 3. Ký tự ngoặc [/] (ví dụ: [búa/vó])
            match_slash = RE_SLASH_OPTION.search(vn_text)
            if match_slash:
                opt1 = match_slash.group(1).strip()
                vn1 = vn_text[:match_slash.start()] + opt1 + vn_text[match_slash.end():]
                if augment:
                    opt2 = match_slash.group(2).strip()
                    vn2 = vn_text[:match_slash.start()] + opt2 + vn_text[match_slash.end():]
                    vn_list = [vn1, vn2]
                else:
                    vn_list = [vn1]
            else:
                vn_list = [vn_text]

            for vn in vn_list:
                vn_clean = clean_text(vn)
                nom_clean = RE_WHITESPACE.sub('', nom_text).strip()

                if nom_clean and vn_clean:
                    rows.append({
                        'file_name': file_name,
                        'source': source if source else get_source_category(file_name),
                        'nom': nom_clean,
                        'vietnamese': row['vietnamese'],
                        'vietnamese_clean': vn_clean
                    })

        result_df = pd.DataFrame(rows)
        result_df = result_df.drop_duplicates(subset=['nom', 'vietnamese_clean']).reset_index(drop=True)
        return compute_length_features(result_df, vn_col='vietnamese_clean')

    def detect_anomalies(self) -> dict:
        """Phát hiện các bất thường trong dữ liệu raw (dấu câu, ngoặc, rỗng, trùng lặp)."""
        df = self.df
        total_rows = len(df)

        has_ellipsis = df['vietnamese'].astype(str).str.contains(r'\.\.\.', regex=True).sum()
        has_brackets = df['vietnamese'].astype(str).str.contains(r'\[|\]', regex=True).sum()
        has_punctuation = df['vietnamese'].astype(str).str.contains(r'[\.,;\!\?]', regex=True).sum()
        null_count = df.isnull().sum().to_dict()
        duplicate_count = df.duplicated(subset=['nom', 'vietnamese']).sum()

        return {
            'total_rows': total_rows,
            'null_count': null_count,
            'duplicate_count': duplicate_count,
            'has_ellipsis': int(has_ellipsis),
            'has_brackets': int(has_brackets),
            'has_punctuation': int(has_punctuation),
        }

    def analyze_vocab_distribution(self) -> dict:
        """Phân tích vốn từ (Vocabulary) và ký tự hiếm (Long-tail analysis)."""
        if 'nom_char_len' not in self.df.columns:
            self.extract_features()

        vn_col = 'vietnamese_clean' if 'vietnamese_clean' in self.df.columns else 'vietnamese'

        nom_chars = [char for text in self.df['nom'].astype(str) for char in text]
        vn_syllables = [syllable for text in self.df[vn_col].astype(str) for syllable in text.split() if syllable]

        nom_char_counts = Counter(nom_chars)
        vn_syllable_counts = Counter(vn_syllables)
        rare_nom_chars = {char: count for char, count in nom_char_counts.items() if count < 3}

        return {
            'total_nom_chars_tokens': len(nom_chars),
            'unique_nom_chars': len(nom_char_counts),
            'total_vn_syllables_tokens': len(vn_syllables),
            'unique_vn_syllables': len(vn_syllable_counts),
            'rare_nom_chars_count': len(rare_nom_chars),
            'rare_nom_chars_pct': (len(rare_nom_chars) / len(nom_char_counts) * 100) if nom_char_counts else 0,
            'top_10_nom_chars': nom_char_counts.most_common(10),
            'top_10_vn_syllables': vn_syllable_counts.most_common(10),
        }

    def analyze_polyphones(self, min_occurrences: int = 5) -> pd.DataFrame:
        """Phân tích các ký tự Hán-Nôm đa âm (Polyphonic characters)."""
        if 'align_diff' not in self.df.columns:
            self.extract_features()

        aligned_df = self.df[self.df['align_diff'] == 0]
        char_to_readings = defaultdict(Counter)
        vn_col = 'vietnamese_clean' if 'vietnamese_clean' in self.df.columns else 'vietnamese'

        for _, row in aligned_df.iterrows():
            nom_text = str(row['nom'])
            vn_words = [w for w in str(row[vn_col]).split() if w]

            if len(nom_text) == len(vn_words):
                for char, word in zip(nom_text, vn_words):
                    if word:
                        char_to_readings[char][word] += 1

        polyphone_data = []
        for char, readings in char_to_readings.items():
            total_occ = sum(readings.values())
            if len(readings) > 1 and total_occ >= min_occurrences:
                sorted_readings = readings.most_common()
                readings_str = ", ".join([f"{w} ({c})" for w, c in sorted_readings])
                polyphone_data.append({
                    'nom_char': char,
                    'num_readings': len(readings),
                    'total_occurrences': total_occ,
                    'readings_detail': readings_str,
                    'top_reading': sorted_readings[0][0],
                    'top_reading_pct': round(sorted_readings[0][1] / total_occ * 100, 1)
                })

        poly_df = pd.DataFrame(polyphone_data)
        if not poly_df.empty:
            poly_df = poly_df.sort_values(by='total_occurrences', ascending=False).reset_index(drop=True)
        return poly_df
