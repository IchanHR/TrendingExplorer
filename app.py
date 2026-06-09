import streamlit as st
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
from utils import (
    load_models, load_dataset, filter_by_niche,
    predict_trending, get_best_model, get_tfidf_keywords,
    get_hook_patterns, get_trending_factors,
    get_top_hashtags, get_content_list
)

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Trending Content Explorer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; letter-spacing: -0.5px;
    }
    .subtitle { color: #888; font-size: 1rem; margin-bottom: 2rem; }
    .trending-badge {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white; padding: 4px 14px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px;
    }
    .not-trending-badge {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white; padding: 4px 14px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px;
    }
    .content-card {
        background: #1a1a2e; border: 1px solid #2d2d44;
        border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem;
    }
    .metric-label { font-size: 0.75rem; color: #888; }
    .metric-value { font-size: 1.1rem; font-weight: 600; color: #fff; }
    .keyword-chip {
        display: inline-block; background: #2d2d44;
        color: #a78bfa; padding: 5px 14px; border-radius: 20px;
        font-size: 0.82rem; margin: 3px; font-weight: 500;
    }
    .keyword-chip-top {
        display: inline-block;
        background: linear-gradient(135deg, #a78bfa, #7c3aed);
        color: #fff; padding: 5px 14px; border-radius: 20px;
        font-size: 0.85rem; margin: 3px; font-weight: 600;
    }
    .hook-chip {
        display: inline-block; background: #1e3a5f;
        color: #60a5fa; padding: 5px 14px; border-radius: 20px;
        font-size: 0.82rem; margin: 3px; font-weight: 500;
    }
    .hashtag-chip {
        display: inline-block; background: #1a2a1a;
        color: #4ade80; padding: 5px 14px; border-radius: 20px;
        font-size: 0.82rem; margin: 3px; font-weight: 500;
    }
    .factor-item {
        background: #1a1a2e; border-left: 3px solid #a78bfa;
        padding: 9px 14px; margin: 5px 0; border-radius: 4px;
        font-size: 0.9rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 600;
        color: #fff; margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2d2d44;
    }
    .section-desc {
        font-size: 0.82rem; color: #888;
        margin-bottom: 0.8rem; font-style: italic;
    }
    .insight-box {
        background: #16213e; border: 1px solid #2d2d44;
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .accuracy-box {
        background: #1a1a2e; border: 1px solid #2d2d44;
        border-radius: 8px; padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stProgress"] > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS (cached) ─────────────────────────────────
@st.cache_resource
def get_models():
    return load_models()

@st.cache_data
def get_data():
    return load_dataset()

# ── CHECK IF MODELS EXIST ────────────────────────────────
if not os.path.exists("models/random_forest.pkl"):
    st.error("Models not found. Please run `python train_model.py` first.")
    st.stop()

models, scaler, tfidf, feature_cols, accuracy = get_models()
df = get_data()
best_model_name = get_best_model(accuracy)

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Trending Explorer")
    st.markdown("---")

    st.markdown("### Model Accuracy")
    for name, acc in accuracy.items():
        color = "#4ade80" if name == best_model_name else "#888"
        star  = "  ★" if name == best_model_name else ""
        st.markdown(f"""
        <div class="accuracy-box">
            <span style="color:{color}; font-weight:600">{name}{star}</span><br>
            <span style="color:#a78bfa">{acc:.2%}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset Info")
    st.markdown(f"**Total videos:** {len(df):,}")
    st.markdown(f"**Platforms:** TikTok & YouTube")
    st.markdown(f"**Year:** 2025")

    st.markdown("---")
    st.markdown("### Popular Niches")
    niche_suggestions = [
        "Tech", "Comedy",
        "Travel", "Education", "Entertainment"
    ]
    for n in niche_suggestions:
        st.markdown(f"• {n}")

# ── MAIN HEADER ──────────────────────────────────────────
st.markdown('<div class="main-title">Trending Content Explorer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML-powered prediction for viral content — TikTok & YouTube Shorts 2025</div>',
            unsafe_allow_html=True)

# ── SEARCH BAR ───────────────────────────────────────────
if "niche_input" not in st.session_state:
    st.session_state.niche_input = ""

def set_niche(selected):
    st.session_state.niche_input = selected

def reset_niche():
    st.session_state.niche_input = ""

col1, col2 = st.columns([4, 1])
with col1:
    niche = st.text_input(
        "",
        placeholder="Enter a niche... e.g. Gaming, Tech, Comedy, Travel",
        label_visibility="collapsed",
        key="niche_input"
    )
with col2:
    search_btn = st.button("Explore", use_container_width=True, type="primary")

# ── TABS ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Content List", "Insights", "Feedback"])

# ── MAIN LOGIC ───────────────────────────────────────────
if niche.strip():
    st.button("← Back to Home", on_click=reset_niche, type="secondary")
    with st.spinner(f"Analyzing '{niche}' content..."):

        # Filter & predict
        filtered_df  = filter_by_niche(df, niche)
        no_match     = filtered_df["_no_match"].iloc[0] if "_no_match" in filtered_df.columns else False
        predictions  = predict_trending(filtered_df, models, scaler, tfidf, feature_cols)
        content_list = get_content_list(filtered_df, predictions, best_model_name)

        # Keywords & insights
        keywords       = get_tfidf_keywords(filtered_df, tfidf, top_n=12)
        hook_patterns  = get_hook_patterns(filtered_df, top_n=8)
        factors        = get_trending_factors(filtered_df)
        top_hashtags   = get_top_hashtags(niche, top_n=8)

        # Summary stats
        trending_count  = sum(1 for c in content_list if c["is_trending"])
        avg_probability = np.mean([c["probability"] for c in content_list])

    # ── WARNING IF NO MATCH ──────────────────────────────
    if no_match:
        st.warning(f"No exact match for '{niche}'. Showing a random sample instead.")
    else:
        st.success(f"Found **{len(content_list)} videos** matching **'{niche}'**")

    # ── SUMMARY METRICS ──────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Videos Found",    len(content_list))
    m2.metric("Trending",        f"{trending_count}/{len(content_list)}")
    m3.metric("Avg Probability", f"{avg_probability:.1f}%")
    m4.metric("Best Model",      best_model_name.split()[0])

    # ════════════════════════════════════════════════════
    # TAB 1 — CONTENT LIST
    # ════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">Content Analysis Results</div>',
                    unsafe_allow_html=True)

        for i, content in enumerate(content_list):
            prob        = content["probability"]
            is_trending = content["is_trending"]

            with st.expander(
                f"{content['title'][:70]}...  |  {content['platform']}  |  {prob:.0f}%",
                expanded=(i < 3)
            ):
                col_a, col_b = st.columns([3, 2])

                with col_a:
                    if is_trending:
                        st.markdown('<span class="trending-badge">TRENDING</span>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="not-trending-badge">NOT TRENDING</span>',
                                    unsafe_allow_html=True)

                    st.markdown(f"**{content['title']}**")
                    st.markdown(f"{content['category']} • {content['genre']} • {content['platform']}")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Views",    content["views"])
                    c2.metric("Likes",    content["likes"])
                    c3.metric("Comments", content["comments"])
                    c4.metric("Shares",   content["shares"])

                    if content["hashtag"]:
                        st.markdown(
                            f'<span class="hashtag-chip">{content["hashtag"]}</span>',
                            unsafe_allow_html=True
                        )

                with col_b:
                    st.markdown("**ML Prediction**")
                    st.progress(int(prob))
                    st.markdown(
                        f"<h2 style='color:{'#4ade80' if is_trending else '#f87171'};"
                        f"text-align:center'>{prob:.1f}%</h2>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<p style='text-align:center;color:#888'>"
                        f"Trending probability<br><small>by {best_model_name}</small></p>",
                        unsafe_allow_html=True
                    )

                    st.markdown("**All models:**")
                    for model_name, preds in predictions.items():
                        p = float(preds["proba"][i]) * 100 if i < len(preds["proba"]) else 50
                        st.markdown(
                            f"<small>{model_name}: **{p:.1f}%**</small>",
                            unsafe_allow_html=True
                        )

    # ════════════════════════════════════════════════════
    # TAB 2 — INSIGHTS  (ENHANCED)
    # ════════════════════════════════════════════════════
    with tab2:
        st.markdown(f'<div class="section-header">Content Insights for "{niche}"</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Data-driven signals extracted from the matching content in this niche.</div>',
                    unsafe_allow_html=True)

        col_ins1, col_ins2 = st.columns(2)

        # ── LEFT COLUMN ──────────────────────────────────
        with col_ins1:
            # ── Important Keywords ──
            st.markdown('<div class="section-header">Important Keywords (TF-IDF)</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-desc">Terms that most strongly characterize titles in this niche. The highlighted one is the most significant.</div>',
                        unsafe_allow_html=True)
            if keywords:
                top_kw = keywords[0][0]
                chips = []
                for idx, (kw, sc) in enumerate(keywords):
                    cls = "keyword-chip-top" if idx == 0 else "keyword-chip"
                    chips.append(f'<span class="{cls}">{kw} · {sc:.3f}</span>')
                st.markdown(" ".join(chips), unsafe_allow_html=True)
                st.markdown(
                    f'<div class="insight-box">💡 <b>Takeaway:</b> Titles featuring '
                    f'<b style="color:#a78bfa">"{top_kw}"</b> tend to perform best in this niche. '
                    f'Consider including high-ranking keywords early in your title.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("No keywords found for this niche.")

            # ── Hook Patterns ──
            st.markdown('<div class="section-header">Hook Patterns</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-desc">Opening phrases that frequently appear in titles — these are proven attention grabbers.</div>',
                        unsafe_allow_html=True)
            if hook_patterns:
                total_hooks = sum(c for _, c in hook_patterns)
                top_hook    = hook_patterns[0][0]
                hook_html = " ".join(
                    [f'<span class="hook-chip">"{hook}" · {count}×</span>'
                     for hook, count in hook_patterns]
                )
                st.markdown(hook_html, unsafe_allow_html=True)
                st.markdown(
                    f'<div class="insight-box">💡 <b>Most used hook:</b> '
                    f'<b style="color:#60a5fa">"{top_hook}"</b> appeared '
                    f'{hook_patterns[0][1]} times across analyzed titles. '
                    f'Hooks set the tone and drive the click — use them deliberately.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("No hook patterns detected.")

        # ── RIGHT COLUMN ─────────────────────────────────
        with col_ins2:
            # ── Recommended Hashtags ──
            st.markdown('<div class="section-header">Recommended Hashtags</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-desc">Popular hashtags associated with this niche to maximize discoverability.</div>',
                        unsafe_allow_html=True)
            if top_hashtags:
                ht_html = " ".join(
                    [f'<span class="hashtag-chip">{ht}</span>'
                     for ht in top_hashtags]
                )
                st.markdown(ht_html, unsafe_allow_html=True)
                st.markdown(
                    f'<div class="insight-box">💡 <b>Tip:</b> Mix broad reach tags '
                    f'(like #fyp, #viral) with niche-specific ones to balance '
                    f'visibility and relevance.</div>',
                    unsafe_allow_html=True
                )

            # ── Trending Factors ──
            st.markdown('<div class="section-header">Trending Factors</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-desc">Patterns observed in this niche that correlate with higher engagement.</div>',
                        unsafe_allow_html=True)
            for factor in factors:
                st.markdown(
                    f'<div class="factor-item">{factor}</div>',
                    unsafe_allow_html=True
                )

            # ── Platform breakdown ──
            if "platform" in filtered_df.columns:
                st.markdown('<div class="section-header">Platform Breakdown</div>',
                            unsafe_allow_html=True)
                platform_counts = filtered_df["platform"].value_counts()
                st.bar_chart(platform_counts)

            # ── Trend label breakdown ──
            if "trend_label" in filtered_df.columns:
                st.markdown('<div class="section-header">Trend Label Distribution</div>',
                            unsafe_allow_html=True)
                trend_counts = filtered_df["trend_label"].value_counts()
                st.bar_chart(trend_counts)

    # ════════════════════════════════════════════════════
    # TAB 3 — FEEDBACK
    # ════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">Rate This Analysis</div>',
                    unsafe_allow_html=True)
        st.markdown("Help us improve by sharing your feedback.")

        with st.form("feedback_form"):
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                user_name = st.text_input("Your Name (optional)", placeholder="Anonymous")
                rating    = st.slider("Overall Rating", 1, 5, 3)
                stars     = "★" * rating

            with col_f2:
                accuracy_rating = st.select_slider(
                    "Prediction Accuracy",
                    options=["Very Poor", "Poor", "Average", "Good", "Excellent"]
                )
                usefulness = st.select_slider(
                    "Usefulness for Content Creation",
                    options=["Not Useful", "Slightly", "Moderately", "Very", "Extremely"]
                )

            comment = st.text_area(
                "Comments / Suggestions",
                placeholder="What did you like? What can be improved?",
                height=100
            )

            submitted = st.form_submit_button("Submit Feedback", type="primary")

            if submitted:
                feedback_data = {
                    "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "niche"          : niche,
                    "user_name"      : user_name or "Anonymous",
                    "rating"         : rating,
                    "accuracy_rating": accuracy_rating,
                    "usefulness"     : usefulness,
                    "comment"        : comment,
                }

                feedback_file = "feedback.csv"
                file_exists   = os.path.exists(feedback_file)
                with open(feedback_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=feedback_data.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(feedback_data)

                st.success(f"Thank you! {stars} Your feedback has been saved.")
                st.balloons()

        if os.path.exists("feedback.csv"):
            st.markdown('<div class="section-header">Recent Feedback</div>',
                        unsafe_allow_html=True)
            fb_df = pd.read_csv("feedback.csv")
            if len(fb_df) > 0:
                avg_rating = fb_df["rating"].mean()
                st.markdown(f"**Average Rating: {'★' * round(avg_rating)} ({avg_rating:.1f}/5)**")
                st.dataframe(
                    fb_df[["timestamp", "niche", "user_name", "rating", "comment"]].tail(5),
                    use_container_width=True
                )

# ── PLACEHOLDER WHEN NO SEARCH ───────────────────────────
elif not niche.strip():
    st.markdown("---")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.info("**Enter a niche** above to start exploring trending content")
    with col_p2:
        st.info("**ML Models** will predict trending probability for each video")
    with col_p3:
        st.info("**Get insights** on keywords, hooks, and trending factors")

    st.markdown("### Try these popular niches:")
    cols = st.columns(4)
    niches = [("Tech", "Tech"),
              ("Comedy", "Comedy"),
                ("Travel", "Travel"),
                  ("Education", "Education")]
    for i, (label, value) in enumerate(niches):
        cols[i % 4].button(label, key=f"niche_{i}", use_container_width=True,
                           on_click=set_niche, args=(value,))