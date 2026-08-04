import kagglehub

file_path = "data/raw/"

# df = kagglehub.load_dataset(
#   KaggleDatasetAdapter.PANDAS,
#   "quandang/nomnaocr",
#   file_path,
# )

df = kagglehub.dataset_download('quandang/nomnaocr', file_path)
print("First 5 records:", df.head())