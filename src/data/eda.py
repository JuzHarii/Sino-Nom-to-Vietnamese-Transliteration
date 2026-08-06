import pandas as pd 
from pathlib import Path 

class Dataloader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.df = None

    def load_dataframe(self, data_dir=None):
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
        
    def get_nom_char_len(self):
        return self.df['nom'].apply(len)

    def get_vn_word_len(self):
        return self.df['vietnamese'].apply(lambda x: len(str(x).split()))

    def get_vn_char_len(self):
        return self.df['vietnamese'].apply(len)

    def get_length(self):
        if self.df is None:
            self.load_dataframe()
        nom_char_len = self.get_nom_char_len()     
        vn_word_len = self.get_vn_word_len()
        vn_char_len = self.get_vn_char_len()
        return nom_char_len, vn_word_len, vn_char_len

if __name__ == "__main__":
    dataset = Dataloader(data_dir="data/raw")
    df = dataset.load_dataframe()
    print(df.head(5))
    dataset.compute_length()
    print(df.head(5))
    # print(df[['nom_char_len', 'vn_word_len', 'vn_char_len']].describe())
