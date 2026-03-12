# Bengali–English Multilingual Sentiment Annotation Pipeline

An end-to-end sentiment annotation pipeline for Bengali and English text using XLM-RoBERTa, with a Streamlit-based review UI.

## Overview

This project annotates Bengali and English sentences with three sentiment labels — POSITIVE, NEGATIVE, and NEUTRAL — using a multilingual transformer model (XLM-R). It includes a full annotation pipeline, dataset export in multiple formats, and an interactive review UI.

## Labels

| Label | Description | Example |
|-------|-------------|---------|
| `POSITIVE` 😊 | Happy, excited, satisfied | *"I loved every moment of it!"* |
| `NEGATIVE` 😞 | Angry, sad, disappointed | *"খুব হতাশ হয়ে গেলাম।"* |
| `NEUTRAL` 😐 | Factual, no strong emotion | *"Today's weather is okay."* |

## Project Structure

```
sentiment_pipeline/
├── data/
│   ├── sentences.json              # 60 raw Bengali + English sentences
│   ├── annotated_sentiment.json    # Annotated output
│   ├── sentiment_train.csv         # CSV training format
│   └── sentiment.conll             # CoNLL IOB format
├── sentiment_annotation.py         # Annotation pipeline
├── app.py                          # Streamlit review UI
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 1 — Annotate the dataset
```bash
python sentiment_annotation.py
```

### Step 2 — Launch the review UI
```bash
streamlit run app.py
```

## Features

- **Multilingual** — handles Bengali (বাংলা) and English in the same pipeline
- **XLM-R model** — uses `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- **Rule-based fallback** — works even without internet using keyword matching
- **3 export formats** — JSON, CSV, CoNLL IOB
- **Streamlit UI** — filter by language/sentiment, correct labels, download dataset

## Sample Output

```json
{
  "id": 2,
  "text": "এই সিনেমাটা সত্যিই অসাধারণ ছিল, আমি খুব উপভোগ করেছি।",
  "language": "Bengali",
  "sentiment": "POSITIVE",
  "confidence": 0.94,
  "method": "transformer"
}
```
