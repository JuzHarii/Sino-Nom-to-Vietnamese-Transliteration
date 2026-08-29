# Sino-Nom to Vietnamese Transliteration

This project focused on transliterating Hán-Nôm (Sino-Nom) characters into modern Vietnamese using Transformer models.

You can download our dataset here: [Download](https://www.mediafire.com/file/yfp1po9jrxegdyg/raw.zip/file)

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

Install required Python packages:
```bash
pip install -r requirements.txt
```

Create environment configuration file from the template:
```bash
cp .env.example .env
```

### 2. Running the Web Application

The web app is split into a backend API and a frontend UI. You need two terminal windows.

**Backend (FastAPI):**
```bash
cd app/server
uvicorn main:app --reload
```
*(The API will run at http://localhost:8000)*

**Frontend (Vite + Tailwind):**
```bash
cd app/ui
npm install
npm run dev
```
*(The UI will run at http://localhost:5173)*

### 3. Data Collection
