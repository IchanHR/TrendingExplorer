import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# ── LOAD ALL SAVED MODELS ────────────────────────────────
def load_models():
    """Load all trained ML models and preprocessors."""
    models = {}
    model_files = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "Random Forest"      : "models/random_forest.pkl",
        "SVM"                : "models/svm.pkl",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)

    scaler       = pickle.load(open("models/scaler.pkl",       "rb"))
    tfidf        = pickle.load(open("models/tfidf.pkl",        "rb"))
    feature_cols = pickle.load(open("models/feature_cols.pkl", "rb"))
    accuracy     = pickle.load(open("models/accuracy.pkl",     "rb"))

    return models, scaler, tfidf, feature_cols, accuracy


# ── LOAD DATASET ─────────────────────────────────────────
def load_dataset():
    """Load and return the main dataset."""
    df = pd.read_csv("youtube_shorts_tiktok_trends_2025.csv")
    return df


# ── FILTER BY NICHE ──────────────────────────────────────
def filter_by_niche(df, niche):
    """Filter dataset by niche (category or genre)."""
    niche_lower = niche.strip().lower()

    mask = (
        df["category"].str.lower().str.contains(niche_lower, na=False) |
        df["genre"].str.lower().str.contains(niche_lower, na=False)    |
        df["hashtag"].str.lower().str.contains(niche_lower, na=False)  |
        df["title"].str.lower().str.contains(niche_lower, na=False)
    )
    filtered = df[mask].copy()

    # If no match, return random sample with a flag
    if len(filtered) == 0:
        filtered = df.sample(min(20, len(df)), random_state=42).copy()
        filtered["_no_match"] = True
    else:
        filtered["_no_match"] = False

    return filtered.head(20)


# ── BUILD FEATURES FOR PREDICTION ────────────────────────
def build_features(row_df, tfidf, feature_cols):
    """Build feature matrix from a filtered dataframe."""
    df = row_df.copy()

    # Fill missing
    df["title"]    = df["title"].fillna("")
    df["category"] = df["category"].fillna("Unknown")
    df["platform"] = df["platform"].fillna("Unknown")
    df["genre"]    = df["genre"].fillna("Unknown")
    df["region"]   = df["region"].fillna("Unknown")

    numeric_cols = ["views", "likes", "comments", "shares",
                    "saves", "engagement_rate", "duration_sec",
                    "like_dislike_ratio", "engagement_velocity"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derived
    df["title_length"] = df["title"].apply(len)
    df["has_emoji"]    = df["title"].apply(
        lambda x: 1 if any(ord(c) > 127 for c in str(x)) else 0
    )
    df["like_view_ratio"]    = np.where(df["views"] > 0, df["likes"]    / df["views"], 0)
    df["comment_view_ratio"] = np.where(df["views"] > 0, df["comments"] / df["views"], 0)

    # Encode categoricals
    le = LabelEncoder()
    for col in ["platform", "category", "genre", "region"]:
        if col in df.columns:
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    # TF-IDF
    title_tfidf = tfidf.transform(df["title"]).toarray()
    tfidf_df    = pd.DataFrame(
        title_tfidf,
        columns=[f"tfidf_{i}" for i in range(title_tfidf.shape[1])],
        index=df.index
    )

    # Assemble feature matrix
    existing_cols = [c for c in feature_cols if c in df.columns]
    X_base  = df[existing_cols].values
    X_tfidf = tfidf_df.values
    X       = np.hstack([X_base, X_tfidf])

    return X


# ── PREDICT TRENDING ─────────────────────────────────────
def predict_trending(filtered_df, models, scaler, tfidf, feature_cols):
    """Run all 3 models and return predictions per row."""
    X       = build_features(filtered_df, tfidf, feature_cols)
    X_scaled = scaler.transform(X)

    predictions = {}
    for name, model in models.items():
        proba = model.predict_proba(X_scaled)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        predictions[name] = {"proba": proba, "pred": pred}

    return predictions


# ── GET BEST MODEL ───────────────────────────────────────
def get_best_model(accuracy_dict):
    """Return name of the best model by accuracy."""
    return max(accuracy_dict, key=accuracy_dict.get)


# ── GET TF-IDF KEYWORDS ──────────────────────────────────
def get_tfidf_keywords(filtered_df, tfidf, top_n=10):
    """Extract top TF-IDF keywords from filtered titles."""
    titles = filtered_df["title"].fillna("").tolist()
    if not titles:
        return []

    tfidf_matrix = tfidf.transform(titles).toarray()
    feature_names = tfidf.get_feature_names_out()
    mean_scores   = tfidf_matrix.mean(axis=0)

    top_indices = mean_scores.argsort()[::-1][:top_n]
    keywords    = [(feature_names[i], round(float(mean_scores[i]), 4))
                   for i in top_indices if mean_scores[i] > 0]
    return keywords


# ── GET HOOK PATTERNS ────────────────────────────────────
def get_hook_patterns(filtered_df, top_n=8):
    """Extract common hook patterns from titles."""
    hook_starters = [
        "how to", "easy", "quick", "simple", "best", "top",
        "my", "i tried", "you need", "why", "what", "the truth",
        "secret", "hack", "tips", "guide", "tutorial", "review",
        "vs", "challenge", "asmr", "pov", "day in", "this",
        "watch", "never", "always", "stop", "start", "learn",
    ]

    titles  = filtered_df["title"].fillna("").str.lower().tolist()
    pattern_counts = {}

    for title in titles:
        for hook in hook_starters:
            if hook in title:
                pattern_counts[hook] = pattern_counts.get(hook, 0) + 1

    sorted_patterns = sorted(pattern_counts.items(),
                             key=lambda x: x[1], reverse=True)
    return sorted_patterns[:top_n]


# ── GET TRENDING FACTORS ─────────────────────────────────
def get_trending_factors(filtered_df):
    """Analyze what factors contribute to trending in filtered data."""
    trending = filtered_df[
        filtered_df["trend_label"].str.lower() == "rising"
    ] if "trend_label" in filtered_df.columns else filtered_df

    all_df = filtered_df

    factors = []

    # Engagement rate
    if "engagement_rate" in all_df.columns:
        avg_eng_trending = trending["engagement_rate"].mean() if len(trending) > 0 else 0
        avg_eng_all      = all_df["engagement_rate"].mean()
        if avg_eng_trending > avg_eng_all:
            factors.append(f"High engagement rate (avg {avg_eng_trending:.2%})")

    # Views
    if "views" in all_df.columns:
        avg_views = all_df["views"].mean()
        factors.append(f"Average views: {avg_views:,.0f}")

    # Duration
    if "duration_sec" in all_df.columns:
        avg_dur = all_df["duration_sec"].mean()
        if avg_dur < 60:
            factors.append(f"Short content wins ({avg_dur:.0f}s avg duration)")
        else:
            factors.append(f"Longer content ({avg_dur:.0f}s avg duration)")

    # Upload hour
    if "upload_hour" in all_df.columns:
        best_hour = all_df.groupby("upload_hour")["views"].mean().idxmax()
        factors.append(f"Best upload hour: {best_hour}:00")

    # Weekend
    if "is_weekend" in all_df.columns:
        weekend_views = all_df[all_df["is_weekend"] == 1]["views"].mean()
        weekday_views = all_df[all_df["is_weekend"] == 0]["views"].mean()
        if weekend_views > weekday_views:
            factors.append("Weekend uploads perform better")
        else:
            factors.append("Weekday uploads perform better")

    # Emoji in title
    if "has_emoji" in all_df.columns:
        emoji_rate = all_df["has_emoji"].mean() * 100
        if emoji_rate > 50:
            factors.append(f"Emoji in title helps ({emoji_rate:.0f}% of trending use emoji)")

    # Top platform
    if "platform" in all_df.columns:
        top_platform = all_df["platform"].value_counts().idxmax()
        factors.append(f"Dominant platform: {top_platform}")

    # Creator tier
    if "creator_tier" in all_df.columns:
        top_tier = all_df["creator_tier"].value_counts().idxmax()
        factors.append(f"Most common creator tier: {top_tier}")

    return factors if factors else ["Not enough data for analysis"]


# ── GET TOP HASHTAGS FOR NICHE ────────────────────────────
def get_top_hashtags(niche, top_n=8):
    """Get top hashtags related to a niche from hashtags dataset."""
    try:
        ht_df = pd.read_csv("top_hashtags_2025.csv")
        niche_lower = niche.lower()

        # Filter relevant hashtags
        mask = ht_df.apply(
            lambda row: niche_lower in str(row).lower(), axis=1
        )
        filtered = ht_df[mask]

        if len(filtered) == 0:
            filtered = ht_df.head(top_n)

        # Try to get hashtag column
        hashtag_col = None
        for col in ["hashtag", "tag", "name", "Hashtag"]:
            if col in filtered.columns:
                hashtag_col = col
                break

        if hashtag_col:
            return filtered[hashtag_col].head(top_n).tolist()
        else:
            return filtered.iloc[:, 0].head(top_n).tolist()

    except Exception:
        return [f"#{niche}", f"#{niche}tips", f"#{niche}content",
                "#viral", "#trending", "#fyp", "#foryou", "#shorts"]


# ── FORMAT NUMBER ────────────────────────────────────────
def format_number(n):
    """Format large numbers to readable string."""
    try:
        n = float(n)
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n/1_000:.1f}K"
        else:
            return str(int(n))
    except Exception:
        return "0"


# ── GET CONTENT SUMMARY ───────────────────────────────────
def get_content_list(filtered_df, predictions, best_model_name):
    """Build a clean list of content items with predictions."""
    content_list = []
    best_preds   = predictions.get(best_model_name, {})
    probas       = best_preds.get("proba", [0] * len(filtered_df))

    for i, (_, row) in enumerate(filtered_df.iterrows()):
        prob = float(probas[i]) if i < len(probas) else 0.5
        content_list.append({
            "title"      : str(row.get("title", "Unknown Title")),
            "platform"   : str(row.get("platform", "Unknown")),
            "category"   : str(row.get("category", "Unknown")),
            "views"      : format_number(row.get("views", 0)),
            "likes"      : format_number(row.get("likes", 0)),
            "comments"   : format_number(row.get("comments", 0)),
            "shares"     : format_number(row.get("shares", 0)),
            "hashtag"    : str(row.get("hashtag", "")),
            "trend_label": str(row.get("trend_label", "unknown")),
            "probability": round(prob * 100, 1),
            "is_trending": (prob >= 0.30) or 
                            (float(row.get("views", 0)) > 100000) or
                            (str(row.get("trend_label","")).lower() == "rising"),
            "engagement" : str(row.get("engagement_rate", 0)),
            "duration"   : str(row.get("duration_sec", 0)),
            "genre"      : str(row.get("genre", "")),
        })

    return content_list