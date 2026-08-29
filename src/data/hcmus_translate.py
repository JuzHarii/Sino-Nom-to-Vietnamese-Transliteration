import requests
import json
import time
import sys
import io
from tqdm import tqdm

def hcmus_translate(text):
    url = 'https://tools.clc.hcmus.edu.vn/api/web/clc-sinonom/sinonom-transliteration'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0',
        'content-type': 'application/json',
    }
    response = requests.request('POST', url, headers=headers, data=json.dumps({'text': text, 'lang_type': 1, 'font_type': 1}))
    time.sleep(0.1)     
    
    try:
        result = json.loads(response.text)['data']
        return result['result_text_transcription'][0].strip()
    except:
        print(f'[ERR] "{text}": {response.text}')
        return 'Cannot translate this text.'

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    # # Test thử với một từ Hán Nôm
    # sample_text = "是日文曉有德并二營属将走出真福安塲"
    # print(f"Đang dịch: {sample_text}")
    # translated = hcmus_translate(sample_text)
    # print(f"Kết quả: {translated}")

    input_path = 'data/result/nom_only.txt'
    output_path = 'data/result/comparison.txt'
    
    #Dem so dong
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'a', encoding='utf-8') as fout:
        fout.write("Nom, Vietnamese translated by hcmus\n")
        
        with tqdm(total=total_lines, desc="Translating Nom to Vietnamese", unit="line") as pbar:
            for line in fin:
                text = line.strip()
                if not text:
                    continue
                translated_text = hcmus_translate(text)
                fout.write(f"{text},{translated_text}\n")
                pbar.update(1)

