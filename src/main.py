import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc dự án vào python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.hcmus_translate import hcmus_translate
from src.models.mt5_translate import mt5_translate

if __name__ == "__main__":
    # text = "是日文曉有德"
    text = "是日文曉有德并二營属将走出真福安塲"
    
    res_hcmus = hcmus_translate(text)
    res_hf = mt5_translate(text)
    
    print(f"Chữ Nôm: {text}")
    print(f"HCMUS API : {res_hcmus}")
    print(f"HF Local  : {res_hf}")
