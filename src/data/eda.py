import pandas as pd 
from pathlib import Path 

class DatasetSinoNom:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.df = None

    def load_data(self, data_dir=None):
        data_dir = Path(data_dir) if data_dir else Path(self.data_dir)
        all_dfs = []

        for file_path in data_dir.glob("*.txt"):
            df_file = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                names=['nom', 'vietnamese'],
                encoding='utf-8'
            )
            df_file['file_name'] = file_path.name
            all_dfs.append(df_file)
        full_df = pd.concat(all_dfs, ignore_index=True)
        self.df = full_df
        return self.df
    
    def compute_length(self):
        if self.df is None:
            self.load_data()

        self.df['nom_char_len'] = self.df['nom'].apply(len)
        self.df['vn_word_len'] = self.df['vietnamese'].apply(lambda x: len(str(x).split()))
        self.df['vn_char_len'] = self.df['vietnamese'].apply(len)
        return self.df


if __name__ == "__main__":
    dataset = DatasetSinoNom(data_dir="data/raw")
    df = dataset.load_data()
    dataset.compute_length()

    print(df.head(5))
    print(df[['nom_char_len', 'vn_word_len', 'vn_char_len']].describe())
