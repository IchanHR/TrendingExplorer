import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

print("=" * 50)
print("  TRENDING CONTENT EXPLORER - ML TRAINING")
print("=" * 50)

# ── 1. LOAD & MERGE DATASET ──────────────────────────────
print("\n[1/6] Loading dataset...")

df_lama = pd.read_csv("youtube_shorts_tiktok_trends_2025.csv")
print(f"      Dataset lama : {len(df_lama):,} rows")

if os.path.exists("youtube_kaggle_processed.csv"):
    df_baru = pd.read_csv("youtube_kaggle_processed.csv")
    print(f"      Dataset baru : {len(df_baru):,} rows")
    df = pd.concat([df_lama, df_baru], ignore_index=True)
    print(f"      Total gabungan: {len(df):,} rows ✅")
else:
    df = df_lama
    print("      [WARN] youtube_kaggle_processed.csv tidak ditemukan,")
    print("             hanya menggunakan dataset lama.")

print(f"      Columns: {list(df.columns[:8])} ...")

# ── 2. DEFINE TARGET (Trending = rising) ─────────────────
print("\n[2/6] Preparing target label...")
df["is_trending"] = df["trend_label"].apply(
    lambda x: 1 if str(x).lower() == "rising" else 0
)
trending_count = df["is_trending"].sum()
print(f"      Trending (rising) : {trending_count:,}")
print(f"      Not Trending      : {len(df) - trending_count:,}")

# ── 3. FEATURE ENGINEERING ───────────────────────────────
print("\n[3/6] Engineering features...")

df["title"]    = df["title"].fillna("")
df["hashtag"]  = df["hashtag"].fillna("")
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

df["title_length"]   = df["title"].apply(len)
df["has_emoji"]      = df["title"].apply(
    lambda x: 1 if any(ord(c) > 127 for c in str(x)) else 0
)
df["like_view_ratio"] = np.where(
    df["views"] > 0, df["likes"] / df["views"], 0
)
df["comment_view_ratio"] = np.where(
    df["views"] > 0, df["comments"] / df["views"], 0
)

le = LabelEncoder()
cat_cols = ["platform", "category", "genre", "region"]
for col in cat_cols:
    if col in df.columns:
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

print("      Building TF-IDF on titles...")
tfidf = TfidfVectorizer(max_features=50, stop_words="english",
                        ngram_range=(1, 2))
title_tfidf = tfidf.fit_transform(df["title"]).toarray()
tfidf_df = pd.DataFrame(
    title_tfidf,
    columns=[f"tfidf_{i}" for i in range(title_tfidf.shape[1])]
)

feature_cols = [
    "views", "likes", "comments", "shares", "saves",
    "engagement_rate", "duration_sec", "title_length",
    "has_emoji", "like_view_ratio", "comment_view_ratio",
    "platform_enc", "category_enc", "genre_enc", "region_enc"
]
feature_cols = [c for c in feature_cols if c in df.columns]

X_base   = df[feature_cols].values
X_tfidf  = tfidf_df.values
X        = np.hstack([X_base, X_tfidf])
y        = df["is_trending"].values

print(f"      Features shape: {X.shape}")

# ── 4. TRAIN / TEST SPLIT ────────────────────────────────
print("\n[4/6] Splitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 5. TRAIN 3 MODELS ────────────────────────────────────
print("\n[5/6] Training 3 ML models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM"                : SVC(probability=True, random_state=42, kernel="rbf"),
}

results   = {}
best_name = None
best_acc  = 0

for name, model in models.items():
    # ⚡ SVM pakai subset 20K — terlalu lambat untuk 200K+ rows
    if name == "SVM":
        print(f"      Training {name} (subset 20K rows)...", end=" ")
        np.random.seed(42)
        idx  = np.random.choice(len(X_train), size=min(20000, len(X_train)), replace=False)
        X_tr = X_train[idx]
        y_tr = y_train[idx]
    else:
        print(f"      Training {name}...", end=" ")
        X_tr = X_train
        y_tr = y_train

    model.fit(X_tr, y_tr)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    results[name] = {"model": model, "accuracy": acc}
    print(f"Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc  = acc
        best_name = name

print(f"\n      ✅ Best Model: {best_name} ({best_acc:.4f})")

# ── 6. SAVE MODELS ───────────────────────────────────────
print("\n[6/6] Saving models...")
os.makedirs("models", exist_ok=True)

for name, info in results.items():
    filename = name.lower().replace(" ", "_") + ".pkl"
    with open(f"models/{filename}", "wb") as f:
        pickle.dump(info["model"], f)

with open("models/scaler.pkl",       "wb") as f: pickle.dump(scaler,       f)
with open("models/tfidf.pkl",        "wb") as f: pickle.dump(tfidf,        f)
with open("models/feature_cols.pkl", "wb") as f: pickle.dump(feature_cols, f)

accuracy_summary = {n: v["accuracy"] for n, v in results.items()}
with open("models/accuracy.pkl",     "wb") as f: pickle.dump(accuracy_summary, f)

print("      Saved to /models folder:")
print("       - logistic_regression.pkl")
print("       - random_forest.pkl")
print("       - svm.pkl")
print("       - scaler.pkl  |  tfidf.pkl  |  feature_cols.pkl")

print("\n" + "=" * 50)
print("  TRAINING COMPLETE!")
for name, info in results.items():
    print(f"  {name:<22}: {info['accuracy']:.2%}")
print("=" * 50)