# Transformer-Based Fake News & Stance Detection System

SEHS4713 AI Application — Group Project 2

A complete NLP pipeline that fine-tunes **DistilBERT** for two tasks:

1. **Fake News Detection** — classify a news article as **Fake** or **Real**
2. **Stance Detection** — classify the relationship between a headline and article body as **Support**, **Against**, or **Neutral**

Includes a Flask web app for interactive inference with demo examples.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Clone & Setup](#clone--setup)
- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Training Pipeline (Notebooks)](#training-pipeline-notebooks)
- [Running the Web App](#running-the-web-app)
- [Standalone Inference](#standalone-inference)
- [Key Dependencies](#key-dependencies)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
gp2/
├── app.py                          # Flask web app (main entry point)
├── src/
│   ├── predict.py                  # Fake news inference module
│   └── predict_stance.py           # Stance detection inference module
├── models/
│   ├── fake_news_model/            # Fine-tuned DistilBERT (fake news)
│   │   ├── model.safetensors       # Final model weights (~256 MB)
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   ├── checkpoint-856/         # Training checkpoint (step 856)
│   │   ├── checkpoint-1712/        # Training checkpoint (step 1712)
│   │   └── checkpoint-2568/        # Training checkpoint (step 2568)
│   └── stance_model/               # Fine-tuned DistilBERT (stance)
│       ├── model.safetensors       # Final model weights (~256 MB)
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       ├── checkpoint-4373/        # Training checkpoint (step 4373)
│       └── checkpoint-8746/        # Training checkpoint (step 8746)
├── templates/
│   └── index.html                  # Web UI template
├── static/
│   └── style.css                   # Web UI styles
├── data/
│   ├── fake_news_full_clean.csv    # Cleaned full dataset (39k rows)
│   └── fake_news_lite_clean.csv    # Lite subset for quick demo (2k rows)
├── Fake.csv                        # Raw fake news dataset
├── True.csv                        # Raw real news dataset
├── train_bodies.csv                # FNC-1 stance dataset (bodies)
├── train_stances.csv               # FNC-1 stance dataset (stances)
├── 01_fake_news_pipeline.ipynb     # Baseline: TF-IDF + Logistic Regression
├── 02_fake_news_transformer.ipynb  # Fine-tune DistilBERT for fake news
├── 03_create_lite_dataset.ipynb    # Create lite dataset for demo
├── 04_bias_analysis.ipynb          # Model bias analysis
├── 05_lite_demo.ipynb              # Lite demo notebook
├── 05_stance_baseline.ipynb        # Baseline: TF-IDF stance detection
├── 06_stance_transformer.ipynb     # Fine-tune DistilBERT for stance
├── requirements.txt                # Python dependencies (pip)
├── environment.yml                 # Conda environment config
└── README.md                       # This file
```

---

## Prerequisites

- **Git LFS** — required to clone this repo (large model & data files are tracked via Git LFS)
- **Python 3.10** (recommended; developed and tested on 3.10)
- **Conda** (Anaconda or Miniconda) — recommended for environment management
- **~5 GB disk space** for model weights (including training checkpoints) + datasets
- **No GPU required** — all inference runs on CPU

---

## Clone & Setup

```bash
# 1. Install Git LFS (only needed once)
git lfs install

# 2. Clone the repository (will automatically download all large files)
git clone https://github.com/Frosty-00/gp2.git
cd gp2

# 3. Install dependencies (choose one of the options below)
```

> **Note:** If you cloned without Git LFS installed, large files will be pointer files (~1 KB each). Run `git lfs pull` inside the repo to download the actual files.

---

## Environment Setup

### Option 1: Conda (Recommended)

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate gp2_env
```

### Option 2: Conda + pip

```bash
# Create a new conda environment with Python 3.10
conda create -n gp2_env python=3.10 -y
conda activate gp2_env

# Install all dependencies
pip install -r requirements.txt
```

### Option 3: pip only (venv)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; import transformers; import flask; print('All dependencies OK')"
```

---

## Datasets

### Fake News Detection

| File | Description | Rows |
|------|-------------|------|
| `Fake.csv` | Raw fake news articles | 23,489 |
| `True.csv` | Raw real news articles | 21,417 |
| `data/fake_news_full_clean.csv` | Cleaned & merged dataset | 39,103 |
| `data/fake_news_lite_clean.csv` | Lite subset for quick demo | 2,000 |

**Columns:** `title`, `text`, `date`, `source`, `author`, `category`, `label` (fake / real)

### Stance Detection (FNC-1)

| File | Description | Rows |
|------|-------------|------|
| `train_bodies.csv` | Article bodies | 37,726 |
| `train_stances.csv` | Headline-body stance labels | 49,972 |

**Labels:** Support, Against, Neutral (3-class, excludes "unrelated" from original FNC-1)

---

## Training Pipeline (Notebooks)

Run the notebooks **in order** to reproduce the full pipeline from raw data to trained models.

| # | Notebook | Description | Output |
|---|----------|-------------|--------|
| 01 | `01_fake_news_pipeline.ipynb` | Baseline fake news detection (TF-IDF + Logistic Regression) | Baseline metrics |
| 02 | `02_fake_news_transformer.ipynb` | Fine-tune DistilBERT for fake news detection | `models/fake_news_model/` |
| 03 | `03_create_lite_dataset.ipynb` | Create lite dataset (2k rows) for quick demo | `data/fake_news_lite_clean.csv` |
| 04 | `04_bias_analysis.ipynb` | Analyze model bias (topic, source, length) | Analysis report |
| 05 | `05_stance_baseline.ipynb` | Baseline stance detection (TF-IDF) | Baseline metrics |
| 05 | `05_lite_demo.ipynb` | Lite demo notebook for presentation | — |
| 06 | `06_stance_transformer.ipynb` | Fine-tune DistilBERT for stance detection | `models/stance_model/` |

**Important:** After running notebook 02 and 06, make sure the model files are saved to `models/fake_news_model/` and `models/stance_model/` respectively. The web app depends on these directories.

To run a notebook:

```bash
jupyter notebook 02_fake_news_transformer.ipynb
```

Or open in VS Code / Cursor and select the `gp2_env` kernel.

---

## Running the Web App

### Step 1: Ensure Models Exist

Verify the trained models are in place:

```
models/fake_news_model/
  ├── model.safetensors
  ├── config.json
  ├── tokenizer.json
  └── tokenizer_config.json

models/stance_model/
  ├── model.safetensors
  ├── config.json
  ├── tokenizer.json
  └── tokenizer_config.json
```

If the model directories are missing, run notebooks `02` and `06` first to train and save the models.

### Step 2: Start the Server

```bash
python app.py
```

You should see:

```
[predict] Loading model from .../models/fake_news_model ...
[predict] Model loaded on cpu
[predict_stance] Loading model from .../models/stance_model ...
[predict_stance] Model loaded on cpu
==================================================
  Fake News & Stance Detector
  Open http://localhost:5000 in your browser
==================================================
```

### Step 3: Open the Web App

Open **http://localhost:5000** in your browser.

The web app provides two modules:

- **Fake News Detection** — Paste any news article and click "Analyze" to get a Fake/Real prediction with confidence score.
- **Stance Detection** — Enter a headline and article body, click "Analyze Stance" to get Support/Against/Neutral prediction.

Both modules support **Custom Input** and **Lite Demo Examples** mode with preset examples.

---

## Standalone Inference

You can also run inference directly from the command line without the web app.

### Fake News Detection

```bash
python -m src.predict
```

### Stance Detection

```bash
python -m src.predict_stance
```

Both scripts include built-in test examples and will print predictions to the terminal.

### Using in Your Own Code

```python
from src.predict import predict_text
from src.predict_stance import predict_stance

# Fake news detection
result = predict_text("Your news article text here...")
print(result)  # {"label": "Fake" or "Real", "confidence": 0.95}

# Stance detection
result = predict_stance("Headline here", "Article body text here...")
print(result)  # {"label": "Support" or "Against" or "Neutral", "confidence": 0.88}
```

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Runtime |
| torch | 2.11.0 | Deep learning framework |
| transformers | 5.3.0 | Hugging Face (DistilBERT) |
| flask | 3.1.3 | Web application |
| scikit-learn | 1.7.2 | Baseline models & metrics |
| pandas | 2.3.3 | Data processing |
| numpy | 2.2.6 | Numerical computing |
| matplotlib | 3.10.8 | Visualization |
| seaborn | 0.13.2 | Statistical plots |
| accelerate | 1.13.0 | HuggingFace training |
| safetensors | 0.7.0 | Model weight format |

Full list in [requirements.txt](requirements.txt).

---

## Troubleshooting

### `OSError: Repo id must use alphanumeric chars`

This means the model directory is missing. Make sure:
1. You have run notebook `02` (fake news) and/or `06` (stance) to train and save the models.
2. The `models/fake_news_model/` and `models/stance_model/` directories exist and contain `model.safetensors`, `config.json`, `tokenizer.json`, and `tokenizer_config.json`.

### `OMP: Error #15: Initializing libiomp5md.dll`

This OpenMP conflict is already handled in the code via `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`. If it still occurs, set the environment variable before running:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE   # Windows
export KMP_DUPLICATE_LIB_OK=TRUE  # macOS/Linux
```

### Port 5000 Already in Use

Change the port in `app.py` (last line) or run:

```bash
python -c "from app import app; app.run(port=5001)"
```

### Model Loading Is Slow

First launch loads ~256 MB model weights into memory. This is a one-time cost — subsequent predictions are fast. All inference runs on CPU; no GPU required.
