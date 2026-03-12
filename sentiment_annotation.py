"""
sentiment_annotation.py
------------------------
Annotates Bengali and English sentences with sentiment labels:
  - POSITIVE
  - NEGATIVE
  - NEUTRAL

Uses XLM-RoBERTa (multilingual) for sentiment prediction,
with a rule-based fallback for robustness.

Exports:
  - data/annotated_sentiment.json
  - data/sentiment.conll
  - data/sentiment_train.csv
"""

import json
import re
from pathlib import Path

# ── Keyword dictionaries (Bengali + English) ─────────────────────────────────

POSITIVE_EN = [
    "fantastic", "amazing", "excellent", "love", "loved", "great", "happy",
    "wonderful", "best", "awesome", "thrilling", "excited", "joy", "joyful",
    "grand", "caring", "professional", "resolved", "approved", "won", "winning",
    "highest", "beautiful", "dearun", "paid off", "top of the world"
]

NEGATIVE_EN = [
    "terrible", "rude", "bad", "worst", "horrible", "disappointed", "broke",
    "miserable", "unacceptable", "hopeless", "heartbreaking", "slow", "unbearable",
    "noise", "potholes", "regret", "overheats", "failed", "lost", "useless",
    "bored", "misery", "poor", "awful"
]

POSITIVE_BN = [
    "অসাধারণ", "দারুণ", "ভালো", "চমৎকার", "সুন্দর", "আনন্দ", "আনন্দময়",
    "উত্তেজিত", "রোমাঞ্চ", "সেরা", "জমকালো", "খুশি", "ভালো লাগছে",
    "ভালো লাগে", "উপভোগ", "জিতে"
]

NEGATIVE_BN = [
    "বাজে", "খারাপ", "অসহ্য", "হতাশ", "নষ্ট", "ধীর", "অসহায়",
    "বেহাল", "গরম", "ফেল", "হেরে", "যানজট", "লোডশেডিং", "ভুল"
]


def detect_language(text):
    """Detect if text is Bengali or English based on Unicode range."""
    bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    return "Bengali" if bengali_chars > 2 else "English"


def rule_based_sentiment(text, language):
    """Rule-based sentiment using keyword matching."""
    text_lower = text.lower()

    if language == "Bengali":
        pos_score = sum(1 for w in POSITIVE_BN if w in text)
        neg_score = sum(1 for w in NEGATIVE_BN if w in text)
    else:
        pos_score = sum(1 for w in POSITIVE_EN if w in text_lower)
        neg_score = sum(1 for w in NEGATIVE_EN if w in text_lower)

    if pos_score > neg_score:
        return "POSITIVE", round(0.65 + min(pos_score * 0.08, 0.30), 2)
    elif neg_score > pos_score:
        return "NEGATIVE", round(0.65 + min(neg_score * 0.08, 0.30), 2)
    else:
        return "NEUTRAL", 0.55


def try_transformer_sentiment(texts):
    """
    Try to use XLM-R transformer model for sentiment.
    Falls back to rule-based if transformers not available.
    """
    try:
        from transformers import pipeline
        print("  Loading XLM-R multilingual sentiment model...")
        classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            truncation=True,
            max_length=128
        )
        label_map = {"positive": "POSITIVE", "negative": "NEGATIVE", "neutral": "NEUTRAL"}
        results = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            preds = classifier(batch)
            for p in preds:
                label = label_map.get(p["label"].lower(), "NEUTRAL")
                results.append((label, round(p["score"], 2)))
        print("  XLM-R model loaded and predictions complete.")
        return results, True
    except Exception as e:
        print(f"  Transformer unavailable ({e}), using rule-based fallback.")
        return None, False


def annotate(sentences):
    """Annotate all sentences with sentiment labels."""
    texts = [s["text"] for s in sentences]

    # Try transformer first
    transformer_results, used_transformer = try_transformer_sentiment(texts)

    annotated = []
    for i, item in enumerate(sentences):
        lang = item.get("language") or detect_language(item["text"])

        if used_transformer and transformer_results:
            label, confidence = transformer_results[i]
        else:
            label, confidence = rule_based_sentiment(item["text"], lang)

        annotated.append({
            "id":         item["id"],
            "text":       item["text"],
            "language":   lang,
            "sentiment":  label,
            "confidence": confidence,
            "method":     "transformer" if used_transformer else "rule-based"
        })

    return annotated


def to_conll(annotated):
    """Convert to CoNLL format: word \\t sentiment_label."""
    lines = []
    for item in annotated:
        words = item["text"].split()
        for j, word in enumerate(words):
            tag = f"B-{item['sentiment']}" if j == 0 else f"I-{item['sentiment']}"
            lines.append(f"{word}\t{tag}")
        lines.append("")
    return lines


def to_csv(annotated):
    """Convert to CSV format for easy ML training."""
    lines = ["id,language,sentiment,confidence,text"]
    for item in annotated:
        text_escaped = item["text"].replace('"', '""')
        lines.append(
            f"{item['id']},{item['language']},{item['sentiment']},"
            f"{item['confidence']},\"{text_escaped}\""
        )
    return lines


def print_summary(annotated):
    """Print annotation summary."""
    from collections import Counter
    sentiments = Counter(a["sentiment"] for a in annotated)
    languages  = Counter(a["language"]  for a in annotated)
    method     = annotated[0]["method"] if annotated else "unknown"

    print("\n── Annotation Summary ──────────────────────")
    print(f"  Total sentences : {len(annotated)}")
    print(f"  Method used     : {method}")
    print(f"\n  Sentiment Distribution:")
    for label, count in sentiments.items():
        bar = "█" * count
        print(f"    {label:<10} : {count:>3}  {bar}")
    print(f"\n  Language Distribution:")
    for lang, count in languages.items():
        print(f"    {lang:<10} : {count:>3}")
    print("────────────────────────────────────────────\n")


def main():
    print("Loading sentences...")
    with open("data/sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)
    print(f"Loaded {len(sentences)} sentences.\n")

    print("Annotating sentiments...")
    annotated = annotate(sentences)
    print_summary(annotated)

    # Save JSON
    out_json = Path("data/annotated_sentiment.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out_json}")

    # Save CoNLL
    out_conll = Path("data/sentiment.conll")
    with open(out_conll, "w", encoding="utf-8") as f:
        f.write("\n".join(to_conll(annotated)))
    print(f"Saved → {out_conll}")

    # Save CSV
    out_csv = Path("data/sentiment_train.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(to_csv(annotated)))
    print(f"Saved → {out_csv}")

    # Preview
    print("\n── Sample Annotations (first 6) ────────────")
    for item in annotated[:6]:
        flag = "🇬🇧" if item["language"] == "English" else "🇧🇩"
        icon = "😊" if item["sentiment"] == "POSITIVE" else "😞" if item["sentiment"] == "NEGATIVE" else "😐"
        print(f"\n{flag} [{item['id']}] {item['text'][:60]}...")
        print(f"     {icon} {item['sentiment']} (confidence: {item['confidence']})")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()
