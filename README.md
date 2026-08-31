# Sino-Nom to Vietnamese Transliteration

This project focused on transliterating Hán-Nôm (Sino-Nom) characters into modern Vietnamese using Transformer models.

You can download our raw dataset here: [Download](https://www.mediafire.com/file/yfp1po9jrxegdyg/raw.zip/file). 
> **Important:** Extract the contents of `raw.zip` into the `data/raw/` directory before running any data scripts.

## Project Structure

```text
Sino-Nom-to-Vietnamese-Transliteration/
├── app/                    # Web application
│   ├── server/            
│   └── ui/                 
├── data/                   # Data management
│   ├── external/           
│   ├── interim/            
│   ├── processed/          # Datasets for model training
│   └── raw/                # Original raw data dumps
├── docs/                   
├── models/                 # Model checkpoints & weights 
├── notebooks/              # Notebooks for experimentation, EDA and Fine-tuning
├── src/                    # Primary source code for the project
│   └── data/               # Scripts for data preprocessing 
├── .env.example            
├── .gitignore              
├── README.md               
└── requirements.txt        
```

## Getting Started

### 1. Environment Setup

It is highly recommended to create and activate a virtual environment first:
```bash
# Create a virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate
```

Install required Python packages:
```bash
pip install -r requirements.txt
```
> **Windows/GPU Users:** If you want to use GPU acceleration, make sure to install the appropriate PyTorch version with CUDA support before installing the requirements. For example:
> `pip install torch --index-url https://download.pytorch.org/whl/cu118`


### 2. Running the Web Application

The web app is split into a backend API and a frontend UI. You need two terminal windows.

**Backend (FastAPI):**
```bash
cd app/server
uvicorn main:app --reload
```
*(The API will run at http://localhost:8000)*

> **Note:** The first time you run the backend, it will automatically download the mT5 and mBART model weights (several GBs) from HuggingFace. This may take a few minutes and the terminal might appear paused during the download.

**Frontend (Vite + Tailwind):**
```bash
cd app/ui
npm install
npm run dev
```
*(The UI will run at http://localhost:5173)*

### 3. Data Preparation & EDA

If you want to run the data pipelines or Exploratory Data Analysis (EDA):
1. Make sure you extracted the raw dataset to `data/raw/`.
2. Run the EDA and cleaning script:
```bash
python src/data/eda.py
```

### 4. Evaluation

To evaluate model predictions (Exact Match, CER, WER, BLEU, ROUGE):
1. Ensure your predictions are formatted correctly and cached.
2. Run the evaluation script:
```bash
python src/eval/eval_hcmus_api.py
```
This will output a summary to the console and save detailed metrics to `data/result/eval_hcmus_test.csv` and a summary file.
