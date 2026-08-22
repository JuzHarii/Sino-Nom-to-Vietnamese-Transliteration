from pathlib import Path
import kagglehub
from src.data.loader import load_raw_corpus

output_dir = Path("data/raw/")
dataset_path = kagglehub.dataset_download('quandang/nomnaocr', path=str(output_dir))
print(f"Đã tải bộ dữ liệu về thư mục: {dataset_path}")

df = load_raw_corpus(dataset_path)
print(f"Đã nạp thành công {len(df):,} dòng dữ liệu. 5 dòng đầu tiên:")
print(df.head(5))