import pandas as pd
from pathlib import Path
from typing import Optional, Union

def load_raw_corpus(data_dir: Union[str, Path]) -> pd.DataFrame:
    """Tải toàn bộ các file .txt trong thư mục dữ liệu thô vào một DataFrame theo thứ tự cố định."""
    dir_path = Path(data_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Thư mục {dir_path} không tồn tại.")

    all_dfs = []
    for file_path in sorted(dir_path.glob("*.txt")):
        df_file = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=['nom', 'vietnamese'],
            encoding='utf-8'
        )
        df_file['file_name'] = file_path.name
        all_dfs.append(df_file)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


class RawCorpusLoader:
    """Lớp tương thích cũ (Wrapper) dùng để đọc dữ liệu thô."""
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.dataframe: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        self.dataframe = load_raw_corpus(self.data_dir)
        return self.dataframe


if __name__ == "__main__":
    df = load_raw_corpus("data/raw")
    print(f"Đã đọc {len(df):,} dòng dữ liệu.")
    print(df.head(5))
