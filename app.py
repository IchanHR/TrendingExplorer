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

# ── THEME-AWARE CSS ─────────────────────────────────────
st.markdown("""
<style>
    /* ── Force dark background ── */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    [data-testid="stHeader"] {
        background-color: #0e1117;
    }

    /* ── Typography ── */
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B, #FF8E53, #fbbf24);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0; letter-spacing: -1px;
        line-height: 1.2;
    }
    .subtitle {
        color: #9ca3af; font-size: 1.05rem;
        margin-bottom: 1.5rem; font-weight: 400;
    }

    /* ── Badges ── */
    .trending-badge {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white; padding: 6px 16px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 700; letter-spacing: 0.5px;
        display: inline-block;
    }
    .not-trending-badge {
        background: linear-gradient(135deg, #6b7280, #9ca3af);
        color: white; padding: 6px 16px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 700; letter-spacing: 0.5px;
        display: inline-block;
    }

    /* ── Chips ── */
    .keyword-chip {
        display: inline-block; background: rgba(167,139,250,0.15);
        color: #c4b5fd; padding: 6px 14px; border-radius: 20px;
        font-size: 0.82rem; margin: 3px; font-weight: 500;
        border: 1px solid rgba(167,139,250,0.3);
    }
    .keyword-chip-top {
        display: inline-block;
        background: linear-gradient(135deg, #a78bfa, #7c3aed);
        color: #fff; padding: 6px 16px; border-radius: 20px;
        font-size: 0.85rem; margin: 3px; font-weight: 700;
        box-shadow: 0 2px 8px rgba(124,58,237,0.3);
    }
    .hook-chip {
        display: inline-block; background: rgba(96,165,250,0.15);
        color: #93c5fd; padding: 6px 14px; border-radius: 20px;
        font-size: 0.82rem; margin: 3px; font-weight: 500;
        border: 1px solid rgba(96,165,250,0.3);
    }
    .hashtag-chip {
        display: inline-block; background: rgba(74,222,128,0.15);
        color: #86efac; padding: 6px 14px; border-radius: 20px;
        font-size: 0.82rem; margin: 3px; font-weight: 500;
        border: 1px solid rgba(74,222,128,0.3);
    }

    /* ── Cards & Sections ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(22,33,62,0.8), rgba(26,26,46,0.8));
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 12px; padding: 1rem 1.2rem; margin: 0.8rem 0;
        color: #e5e7eb;
    }
    .factor-item {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #a78bfa;
        padding: 10px 16px; margin: 6px 0; border-radius: 0 8px 8px 0;
        font-size: 0.9rem; color: #e5e7eb;
    }
    .section-header {
        font-size: 1.15rem; font-weight: 700;
        color: #f3f4f6; margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(167,139,250,0.3);
    }
    .section-desc {
        font-size: 0.82rem; color: #9ca3af;
        margin-bottom: 0.8rem; font-style: italic;
    }

    /* ── Sidebar ── */
    .sidebar-model-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-model-card.best {
        border-color: #4ade80;
        background: rgba(74,222,128,0.08);
    }

    /* ── Metric cards ── */
    .stat-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem; font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.78rem; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* ── Prediction gauge ── */
    .prob-display {
        text-align: center; padding: 0.5rem 0;
    }
    .prob-number {
        font-size: 2.5rem; font-weight: 800;
        line-height: 1;
    }
    .prob-label {
        font-size: 0.75rem; color: #9ca3af;
        margin-top: 0.3rem;
    }
    .model-score {
        display: inline-block; background: rgba(255,255,255,0.05);
        border-radius: 8px; padding: 4px 10px; margin: 2px;
        font-size: 0.78rem; color: #d1d5db;
    }

    /* ── Home page cards ── */
    .feature-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: rgba(167,139,250,0.4); }
    .feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .feature-title {
        font-size: 1rem; font-weight: 700; color: #f3f4f6;
        margin-bottom: 0.3rem;
    }
    .feature-desc { font-size: 0.82rem; color: #9ca3af; }

    /* ── Progress bar ── */
    div[data-testid="stProgress"] > div { border-radius: 10px; }

    /* ── Fix all text to white ── */
    .stMarkdown, .stText, p, span, label, .stSelectbox label,
    .stTextInput label, .stSlider label { color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS (cached) ─────────────────────────────────
@st.cache_resource
def get_models():
    return load_models()

@st.cache_data
def get_data():
    return load_dataset()

if not os.path.exists("models/random_forest.pkl"):
    st.error("Models not found. Please run `python train_model.py` first.")
    st.stop()

models, scaler, tfidf, feature_cols, accuracy = get_models()
df = get_data()
best_model_name = get_best_model(accuracy)

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔥 Explorer")
    st.caption("ML-Powered Trend Analysis")
    st.markdown("---")

    st.markdown("##### Model Performance")
    for name, acc in accuracy.items():
        is_best = name == best_model_name
        cls = "sidebar-model-card best" if is_best else "sidebar-model-card"
        star = " ⭐" if is_best else ""
        color = "#4ade80" if is_best else "#d1d5db"
        st.markdown(f"""
        <div class="{cls}">
            <div style="font-weight:600;color:{color};font-size:0.9rem">{name}{star}</div>
            <div style="font-size:1.3rem;font-weight:800;color:#a78bfa">{acc:.1%}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📊 Dataset")
    st.markdown(f"""
    <div class="sidebar-model-card">
        <div style="color:#9ca3af;font-size:0.8rem">Total Videos</div>
        <div style="font-size:1.2rem;font-weight:700;color:#f3f4f6">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Platforms: TikTok & YouTube")
    st.caption("Data Year: 2025")

# ── MAIN HEADER ──────────────────────────────────────────
st.markdown('<div class="main-title">Trending Content Explorer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover what makes content go viral — powered by machine learning analysis of 236K+ videos</div>',
            unsafe_allow_html=True)

# ── SEARCH BAR ───────────────────────────────────────────
if "niche_input" not in st.session_state:
    st.session_state.niche_input = ""

def set_niche(selected):
    st.session_state.niche_input = selected

def reset_niche():
    st.session_state.niche_input = ""

col1, col2 = st.columns([5, 1])
with col1:
    niche = st.text_input(
        "Search",
        placeholder="🔍 Search a niche... e.g. Gaming, Tech, Comedy, Travel",
        label_visibility="collapsed",
        key="niche_input"
    )
with col2:
    search_btn = st.button("🚀 Explore", use_container_width=True, type="primary")

# ── MAIN LOGIC ───────────────────────────────────────────
if niche.strip():
    with st.spinner(f"Analyzing **{niche}** content..."):
        filtered_df  = filter_by_niche(df, niche)
        no_match     = filtered_df["_no_match"].iloc[0] if "_no_match" in filtered_df.columns else False
        predictions  = predict_trending(filtered_df, models, scaler, tfidf, feature_cols)
        content_list = get_content_list(filtered_df, predictions, best_model_name)
        keywords       = get_tfidf_keywords(filtered_df, tfidf, top_n=12)
        hook_patterns  = get_hook_patterns(filtered_df, top_n=8)
        factors        = get_trending_factors(filtered_df)
        top_hashtags   = get_top_hashtags(niche, top_n=8)
        trending_count  = sum(1 for c in content_list if c["is_trending"])
        avg_probability = np.mean([c["probability"] for c in content_list])

    if no_match:
        st.warning(f"No exact match for '{niche}'. Showing a random sample instead.")
    else:
        st.success(f"Found **{len(content_list)} videos** matching **'{niche}'**")

    # ── STAT CARDS ──────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, f"{len(content_list)}", "VIDEOS FOUND"),
        (m2, f"{trending_count}/{len(content_list)}", "TRENDING"),
        (m3, f"{avg_probability:.1f}%", "AVG PROBABILITY"),
        (m4, best_model_name.split()[0], "BEST MODEL"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── TABS ─────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 Content Analysis", "💡 Insights & Strategy", "📝 Feedback"])

    # ════════════════════════════════════════════════════
    # TAB 1 — CONTENT LIST
    # ════════════════════════════════════════════════════
    with tab1:
        st.markdown(f'<div class="section-header">Analysis Results for "{niche}"</div>',
                    unsafe_allow_html=True)
        st.caption(f"Showing {len(content_list)} videos ranked by trending probability")

        for i, content in enumerate(content_list):
            prob        = content["probability"]
            is_trending = content["is_trending"]
            badge_cls   = "trending-badge" if is_trending else "not-trending-badge"
            badge_text  = "🔥 TRENDING" if is_trending else "— NOT TRENDING"
            prob_color  = "#4ade80" if is_trending else "#9ca3af"

            with st.expander(
                f"{'🔥' if is_trending else '○'} {content['title'][:65]}  —  {content['platform']}  —  {prob:.0f}%",
                expanded=(i < 2)
            ):
                col_a, col_b = st.columns([3, 2])

                with col_a:
                    st.markdown(f'<span class="{badge_cls}">{badge_text}</span>',
                                unsafe_allow_html=True)
                    st.markdown(f"**{content['title']}**")
                    st.caption(f"{content['category']} • {content['genre']} • {content['platform']}")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("👁 Views",    content["views"])
                    c2.metric("❤ Likes",    content["likes"])
                    c3.metric("💬 Comments", content["comments"])
                    c4.metric("↗ Shares",   content["shares"])

                    if content["hashtag"]:
                        st.markdown(
                            f'<span class="hashtag-chip">{content["hashtag"]}</span>',
                            unsafe_allow_html=True
                        )

                with col_b:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="prob-display">
                            <div class="prob-number" style="color:{prob_color}">{prob:.1f}%</div>
                            <div class="prob-label">Trending Probability</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    st.progress(min(int(prob), 100))

                    model_html = ""
                    for model_name, preds in predictions.items():
                        p = float(preds["proba"][i]) * 100 if i < len(preds["proba"]) else 50
                        model_html += f'<span class="model-score">{model_name.split()[0]}: <b>{p:.1f}%</b></span> '
                    st.markdown(model_html, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # TAB 2 — INSIGHTS
    # ════════════════════════════════════════════════════
    with tab2:
        st.markdown(f'<div class="section-header">Content Strategy for "{niche}"</div>',
                    unsafe_allow_html=True)
        st.caption("Data-driven recommendations based on ML analysis of matching content")

        col_ins1, col_ins2 = st.columns(2)

        with col_ins1:
            # ── Keywords ──
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 🔑 Top Keywords (TF-IDF)")
            st.caption("Terms that most strongly characterize titles in this niche")
            if keywords:
                top_kw = keywords[0][0]
                chips = []
                for idx, (kw, sc) in enumerate(keywords):
                    cls = "keyword-chip-top" if idx == 0 else "keyword-chip"
                    chips.append(f'<span class="{cls}">{kw} · {sc:.3f}</span>')
                st.markdown(" ".join(chips), unsafe_allow_html=True)
                st.markdown(
                    f'<div class="insight-box">💡 Titles featuring '
                    f'<b style="color:#c4b5fd">"{top_kw}"</b> tend to perform best. '
                    f'Place high-ranking keywords early in your title.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("No keywords found for this niche.")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Hook Patterns ──
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 🎣 Hook Patterns")
            st.caption("Opening phrases that grab attention")
            if hook_patterns:
                top_hook = hook_patterns[0][0]
                hook_html = " ".join(
                    [f'<span class="hook-chip">"{hook}" · {count}×</span>'
                     for hook, count in hook_patterns]
                )
                st.markdown(hook_html, unsafe_allow_html=True)
                st.markdown(
                    f'<div class="insight-box">💡 <b>"{top_hook}"</b> appeared '
                    f'{hook_patterns[0][1]}× across titles. '
                    f'Strong hooks drive higher click-through rates.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("No hook patterns detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_ins2:
            # ── Hashtags ──
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### #️⃣ Recommended Hashtags")
            st.caption("Popular tags to maximize discoverability")
            if top_hashtags:
                ht_html = " ".join(
                    [f'<span class="hashtag-chip">{ht}</span>' for ht in top_hashtags]
                )
                st.markdown(ht_html, unsafe_allow_html=True)
                st.markdown(
                    f'<div class="insight-box">💡 Mix broad tags (#fyp, #viral) with '
                    f'niche-specific ones for best reach-to-relevance balance.</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Trending Factors ──
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 📈 Trending Factors")
            st.caption("Patterns that correlate with higher engagement")
            for factor in factors:
                st.markdown(f'<div class="factor-item">{factor}</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Charts row ──
        st.markdown("---")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            if "platform" in filtered_df.columns:
                st.markdown("##### 📊 Platform Breakdown")
                platform_counts = filtered_df["platform"].value_counts()
                st.bar_chart(platform_counts)

        with col_c2:
            if "trend_label" in filtered_df.columns:
                st.markdown("##### 📊 Trend Label Distribution")
                trend_counts = filtered_df["trend_label"].value_counts()
                st.bar_chart(trend_counts)

    # ════════════════════════════════════════════════════
    # TAB 3 — FEEDBACK
    # ════════════════════════════════════════════════════
    with tab3:
        st.markdown("##### ⭐ Rate This Analysis")
        st.caption("Help us improve by sharing your experience")

        with st.form("feedback_form"):
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                user_name = st.text_input("Your Name (optional)", placeholder="Anonymous")
                rating    = st.slider("Overall Rating", 1, 5, 3)

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

            submitted = st.form_submit_button("Submit Feedback", type="primary",
                                              use_container_width=True)

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

                st.success(f"Thank you! {'⭐' * rating} Your feedback has been saved.")
                st.balloons()

        if os.path.exists("feedback.csv"):
            st.markdown("##### Recent Feedback")
            fb_df = pd.read_csv("feedback.csv")
            if len(fb_df) > 0:
                avg_rating = fb_df["rating"].mean()
                st.markdown(f"**Average Rating: {'⭐' * round(avg_rating)} ({avg_rating:.1f}/5)**")
                st.dataframe(
                    fb_df[["timestamp", "niche", "user_name", "rating", "comment"]].tail(5),
                    use_container_width=True
                )

    # ── Back button at bottom ──
    st.markdown("---")
    st.button("← Back to Home", on_click=reset_niche, type="secondary")

# ── HOME PAGE (no search) ────────────────────────────────
elif not niche.strip():
    st.markdown("")

    # Feature cards
    col_p1, col_p2, col_p3 = st.columns(3)
    features = [
        ("🔍", "Explore Niches", "Search any content niche to discover trending patterns and insights"),
        ("🤖", "ML Predictions", "3 trained models predict trending probability for every video"),
        ("💡", "Get Strategy", "Keywords, hooks, hashtags, and factors that drive virality"),
    ]
    for col, (icon, title, desc) in zip([col_p1, col_p2, col_p3], features):
        col.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("##### Try a popular niche:")
    cols = st.columns(5)
    niches = ["Tech", "Comedy", "Travel", "Education", "Entertainment"]
    for i, n in enumerate(niches):
        cols[i].button(f"🔥 {n}", key=f"niche_{i}", use_container_width=True,
                       on_click=set_niche, args=(n,))

    st.markdown("---")
    st.caption("Built with Streamlit • Powered by Scikit-learn • Data: 236K+ TikTok & YouTube videos (2025)")
