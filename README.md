# Sino-Nom to Vietnamese Transliteration

This project focused on transliterating Hán-Nôm (Sino-Nom) characters into modern Vietnamese using Transformer models.

You can download our dataset here: 

## Project Structure

```text
Sino-Nom-to-Vietnamese-Transliteration/
├── app/                    # Web application
│   ├── server/            
│   └── ui/                 
├── data/                   # Data management
│   ├── external/           
│   ├── interim/            
│   ├── processed/          # Clean datasets ready for model training
│   └── raw/                # Original raw data dumps
├── docs/                   # Project documentation and research papers
├── models/                 # Trained model checkpoints & weights 
├── notebooks/              # Jupyter notebooks for experimentation, EDA and Fine-tuning
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

### 2. Data Collection

