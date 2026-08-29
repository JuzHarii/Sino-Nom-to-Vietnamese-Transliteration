import requests
import json
import time
import sys
import io

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

# Đoạn code dưới đây sẽ chỉ chạy khi bạn chạy trực tiếp file này
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Test thử với một từ Hán Nôm
    sample_text = "是日文曉有德并二營属将走出真福安塲"
    print(f"Đang dịch: {sample_text}")
    translated = hcmus_translate(sample_text)
    print(f"Kết quả: {translated}")
