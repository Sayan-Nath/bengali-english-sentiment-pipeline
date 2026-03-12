"""
app.py
------
Streamlit UI for reviewing Bengali + English sentiment annotations.

Features:
  - Browse all annotated sentences with sentiment labels
  - Filter by language and sentiment
  - Correct labels manually
  - View statistics dashboard
  - Export corrected dataset

Usage:
    streamlit run app.py
"""

import json
from pathlib import Path
from collections import Counter

import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

SENTIMENT_COLORS = {
    "POSITIVE": "#A8E6CF",
    "NEGATIVE": "#FF6B6B",
    "NEUTRAL":  "#FFD93D",
}

SENTIMENT_ICONS = {
    "POSITIVE": "😊",
    "NEGATIVE": "😞",
    "NEUTRAL":  "😐",
}

DATA_PATH      = Path("data/annotated_sentiment.json")
CORRECTED_PATH = Path("data/corrected_sentiment.json")


@st.cache_data
def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_corrected(data):
    with open(CORRECTED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sentiment_badge(sentiment):
    color = SENTIMENT_COLORS.get(sentiment, "#DDD")
    icon  = SENTIMENT_ICONS.get(sentiment, "")
    return (
        f'<span style="background:{color};padding:4px 12px;border-radius:20px;'
        f'font-weight:bold;font-size:14px">{icon} {sentiment}</span>'
    )


def language_badge(language):
    color = "#D6E4F0" if language == "English" else "#F0D6E4"
    flag  = "🇬🇧" if language == "English" else "🇧🇩"
    return (
        f'<span style="background:{color};padding:3px 10px;border-radius:20px;'
        f'font-size:13px">{flag} {language}</span>'
    )


def main():
    st.set_page_config(
        page_title="Bengali-English Sentiment Reviewer",
        page_icon="🌏",
        layout="wide"
    )

    st.title("🌏 Bengali–English Sentiment Annotation Reviewer")
    st.markdown("Review, correct, and export multilingual sentiment annotations.")

    data = load_data()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["📋 Review", "📊 Statistics", "📤 Export"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Sentiment Labels")
    for sentiment, color in SENTIMENT_COLORS.items():
        icon = SENTIMENT_ICONS[sentiment]
        st.sidebar.markdown(
            f'<span style="background:{color};padding:4px 12px;border-radius:20px;'
            f'font-weight:bold">{icon} {sentiment}</span>',
            unsafe_allow_html=True
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    lang_filter      = st.sidebar.selectbox("Language", ["All", "English", "Bengali"])
    sentiment_filter = st.sidebar.selectbox("Sentiment", ["All", "POSITIVE", "NEGATIVE", "NEUTRAL"])

    # Apply filters
    filtered = data
    if lang_filter != "All":
        filtered = [d for d in filtered if d["language"] == lang_filter]
    if sentiment_filter != "All":
        filtered = [d for d in filtered if d["sentiment"] == sentiment_filter]

    # ── Review Page ───────────────────────────────────────────────────────────
    if page == "📋 Review":
        st.subheader(f"Annotation Review — {len(filtered)} sentences")

        if not filtered:
            st.warning("No sentences match your filters.")
            return

        idx  = st.slider("Select sentence", 1, len(filtered), 1) - 1
        item = filtered[idx]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f'<div style="background:#1E1E2E;padding:20px;border-radius:10px;'
                f'font-size:18px;line-height:1.8;margin-bottom:10px">'
                f'{item["text"]}</div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown("**Language:**")
            st.markdown(language_badge(item["language"]), unsafe_allow_html=True)
            st.markdown("**Sentiment:**")
            st.markdown(sentiment_badge(item["sentiment"]), unsafe_allow_html=True)
            st.markdown(f"**Confidence:** `{item['confidence']}`")
            st.markdown(f"**Method:** `{item.get('method', 'N/A')}`")

        st.markdown("---")

        # Correction
        with st.expander("✏️ Correct this label"):
            new_sentiment = st.selectbox(
                "Change sentiment to:",
                ["POSITIVE", "NEGATIVE", "NEUTRAL"],
                index=["POSITIVE", "NEGATIVE", "NEUTRAL"].index(item["sentiment"])
            )
            if st.button("Save correction"):
                # Find and update in original data
                for d in data:
                    if d["id"] == item["id"]:
                        d["sentiment"] = new_sentiment
                        d["method"]    = "manual"
                        break
                save_corrected(data)
                st.success(f"Updated to {new_sentiment} and saved!")

        # Navigation buttons
        col_prev, col_next = st.columns(2)
        st.markdown("---")
        st.markdown("**All sentences in current filter:**")
        for i, d in enumerate(filtered):
            color = SENTIMENT_COLORS.get(d["sentiment"], "#DDD")
            icon  = SENTIMENT_ICONS.get(d["sentiment"], "")
            flag  = "🇬🇧" if d["language"] == "English" else "🇧🇩"
            st.markdown(
                f'<div style="padding:8px;margin:4px 0;border-left:4px solid {color};'
                f'background:#1E1E2E;border-radius:4px">'
                f'{flag} <b>[{d["id"]}]</b> {d["text"][:70]}... '
                f'<span style="float:right">{icon} {d["sentiment"]}</span></div>',
                unsafe_allow_html=True
            )

    # ── Statistics Page ───────────────────────────────────────────────────────
    elif page == "📊 Statistics":
        st.subheader("Annotation Statistics")

        sentiments = Counter(d["sentiment"] for d in data)
        languages  = Counter(d["language"]  for d in data)
        methods    = Counter(d.get("method", "unknown") for d in data)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sentences", len(data))
        col2.metric("English",  languages.get("English", 0))
        col3.metric("Bengali",  languages.get("Bengali", 0))
        col4.metric("Annotation Method", list(methods.keys())[0])

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Sentiment Distribution**")
            total = len(data)
            for sentiment, count in sentiments.items():
                color = SENTIMENT_COLORS.get(sentiment, "#DDD")
                icon  = SENTIMENT_ICONS.get(sentiment, "")
                pct   = count / total * 100
                st.markdown(
                    f'<div style="display:flex;align-items:center;margin:8px 0">'
                    f'<span style="background:{color};padding:4px 12px;border-radius:20px;'
                    f'font-weight:bold;width:130px;display:inline-block;text-align:center">'
                    f'{icon} {sentiment}</span>'
                    f'<div style="background:{color};height:22px;width:{pct*4:.0f}px;'
                    f'margin-left:10px;border-radius:4px"></div>'
                    f'<span style="margin-left:10px">{count} ({pct:.1f}%)</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with col_b:
            st.markdown("**Language Distribution**")
            for lang, count in languages.items():
                flag = "🇬🇧" if lang == "English" else "🇧🇩"
                pct  = count / total * 100
                color = "#D6E4F0" if lang == "English" else "#F0D6E4"
                st.markdown(
                    f'<div style="display:flex;align-items:center;margin:8px 0">'
                    f'<span style="background:{color};padding:4px 12px;border-radius:20px;'
                    f'width:130px;display:inline-block;text-align:center">'
                    f'{flag} {lang}</span>'
                    f'<div style="background:{color};height:22px;width:{pct*4:.0f}px;'
                    f'margin-left:10px;border-radius:4px"></div>'
                    f'<span style="margin-left:10px">{count} ({pct:.1f}%)</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("**Sentiment by Language**")
        for lang in ["English", "Bengali"]:
            lang_data  = [d for d in data if d["language"] == lang]
            lang_sents = Counter(d["sentiment"] for d in lang_data)
            flag = "🇬🇧" if lang == "English" else "🇧🇩"
            st.markdown(f"**{flag} {lang}**")
            cols = st.columns(3)
            for i, (sentiment, color) in enumerate(SENTIMENT_COLORS.items()):
                count = lang_sents.get(sentiment, 0)
                cols[i].markdown(
                    f'<div style="background:{color};padding:10px;border-radius:8px;'
                    f'text-align:center"><b>{SENTIMENT_ICONS[sentiment]} {sentiment}</b>'
                    f'<br><span style="font-size:24px;font-weight:bold">{count}</span></div>',
                    unsafe_allow_html=True
                )
            st.markdown("")

    # ── Export Page ───────────────────────────────────────────────────────────
    elif page == "📤 Export":
        st.subheader("Export Annotated Dataset")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**JSON Format**")
            st.download_button(
                label="⬇️ Download JSON",
                data=json.dumps(data, indent=2, ensure_ascii=False),
                file_name="bengali_english_sentiment.json",
                mime="application/json"
            )

        with col2:
            st.markdown("**CSV Format**")
            lines = ["id,language,sentiment,confidence,text"]
            for item in data:
                text_escaped = item["text"].replace('"', '""')
                lines.append(
                    f"{item['id']},{item['language']},{item['sentiment']},"
                    f"{item['confidence']},\"{text_escaped}\""
                )
            st.download_button(
                label="⬇️ Download CSV",
                data="\n".join(lines),
                file_name="bengali_english_sentiment.csv",
                mime="text/csv"
            )

        with col3:
            st.markdown("**CoNLL Format**")
            conll_lines = []
            for item in data:
                words = item["text"].split()
                for j, word in enumerate(words):
                    tag = f"B-{item['sentiment']}" if j == 0 else f"I-{item['sentiment']}"
                    conll_lines.append(f"{word}\t{tag}")
                conll_lines.append("")
            st.download_button(
                label="⬇️ Download CoNLL",
                data="\n".join(conll_lines),
                file_name="bengali_english_sentiment.conll",
                mime="text/plain"
            )

        st.markdown("---")
        st.markdown("**Dataset Preview**")
        for item in data[:10]:
            flag  = "🇬🇧" if item["language"] == "English" else "🇧🇩"
            color = SENTIMENT_COLORS.get(item["sentiment"], "#DDD")
            icon  = SENTIMENT_ICONS.get(item["sentiment"], "")
            st.markdown(
                f'<div style="padding:10px;margin:4px 0;border-left:4px solid {color};'
                f'background:#1E1E2E;border-radius:4px">'
                f'{flag} {item["text"]}<br>'
                f'<small style="color:#AAA">{icon} {item["sentiment"]} '
                f'| confidence: {item["confidence"]}</small></div>',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
