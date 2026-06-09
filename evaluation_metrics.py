import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

print("=" * 55)
print("  EVALUATION METRICS — Trending Content Explorer")
print("=" * 55)

# ── 1. LOAD & PREPARE DATA (same pipeline as train_model.py) ──

print("\n[1/5] Loading and preparing data...")

df_lama = pd.read_csv("youtube_shorts_tiktok_trends_2025.csv")
if os.path.exists("youtube_kaggle_processed.csv"):
    df_baru = pd.read_csv("youtube_kaggle_processed.csv")
    df = pd.concat([df_lama, df_baru], ignore_index=True)
else:
    df = df_lama

df["is_trending"] = df["trend_label"].apply(
    lambda x: 1 if str(x).lower() == "rising" else 0
)

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

df["title_length"]       = df["title"].apply(len)
df["has_emoji"]          = df["title"].apply(lambda x: 1 if any(ord(c) > 127 for c in str(x)) else 0)
df["like_view_ratio"]    = np.where(df["views"] > 0, df["likes"] / df["views"], 0)
df["comment_view_ratio"] = np.where(df["views"] > 0, df["comments"] / df["views"], 0)

le = LabelEncoder()
for col in ["platform", "category", "genre", "region"]:
    if col in df.columns:
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

tfidf = TfidfVectorizer(max_features=50, stop_words="english", ngram_range=(1, 2))
title_tfidf = tfidf.fit_transform(df["title"]).toarray()
tfidf_df = pd.DataFrame(title_tfidf, columns=[f"tfidf_{i}" for i in range(title_tfidf.shape[1])])

feature_cols = [
    "views", "likes", "comments", "shares", "saves",
    "engagement_rate", "duration_sec", "title_length",
    "has_emoji", "like_view_ratio", "comment_view_ratio",
    "platform_enc", "category_enc", "genre_enc", "region_enc"
]
feature_cols = [c for c in feature_cols if c in df.columns]

X = np.hstack([df[feature_cols].values, tfidf_df.values])
y = df["is_trending"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"      Class distribution — Trending: {y.sum():,} | Not: {len(y) - y.sum():,}")

# ── 2. LOAD TRAINED MODELS ──────────────────────────────────

print("\n[2/5] Loading trained models...")

models = {}
model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest":       "models/random_forest.pkl",
    "SVM":                 "models/svm.pkl",
}

for name, path in model_files.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"      Loaded {name}")

# ── 3. COMPUTE ALL METRICS ──────────────────────────────────

print("\n[3/5] Computing evaluation metrics...")

os.makedirs("models/evaluation", exist_ok=True)

all_metrics = {}

for name, model in models.items():
    print(f"\n  ── {name} ──")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_proba)

    report = classification_report(y_test, y_pred,
                                   target_names=["Not Trending", "Trending"],
                                   output_dict=True)

    cm = confusion_matrix(y_test, y_pred)

    print(f"      Accuracy:  {acc:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1 Score:  {f1:.4f}")
    print(f"      ROC-AUC:   {roc_auc:.4f}")

    all_metrics[name] = {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1_score":  f1,
        "roc_auc":   roc_auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "y_pred":  y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }

    # ── Confusion Matrix Plot ──
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Trending", "Trending"],
                yticklabels=["Not Trending", "Trending"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    fig.savefig(f"models/evaluation/cm_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)
    print(f"      Saved confusion matrix plot")

    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#a78bfa", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#555", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(f"models/evaluation/roc_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)
    print(f"      Saved ROC curve plot")

# ── 4. CROSS-VALIDATION ─────────────────────────────────────

print("\n[4/5] Running 5-fold cross-validation...")

for name, model in models.items():
    if name == "SVM":
        np.random.seed(42)
        idx = np.random.choice(len(X_train), size=min(20000, len(X_train)), replace=False)
        X_cv, y_cv = X_train[idx], y_train[idx]
    else:
        X_cv, y_cv = X_train, y_train

    cv_scores = cross_val_score(model, X_cv, y_cv, cv=5, scoring="accuracy")
    all_metrics[name]["cv_scores"]    = cv_scores.tolist()
    all_metrics[name]["cv_mean"]      = float(cv_scores.mean())
    all_metrics[name]["cv_std"]       = float(cv_scores.std())
    print(f"      {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ── 5. SAVE & COMPARISON CHART ──────────────────────────────

print("\n[5/5] Saving results...")

with open("models/evaluation/metrics.pkl", "wb") as f:
    pickle.dump(all_metrics, f)

# Comparison bar chart
model_names = list(all_metrics.keys())
metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
metric_labels   = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

x = np.arange(len(metric_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#a78bfa", "#4ade80", "#f97316"]

for i, model_name in enumerate(model_names):
    values = [all_metrics[model_name][m] for m in metrics_to_plot]
    ax.bar(x + i * width, values, width, label=model_name, color=colors[i])

ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Metrics")
ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig("models/evaluation/model_comparison.png", dpi=150)
plt.close(fig)

print("\n" + "=" * 55)
print("  EVALUATION COMPLETE!")
print("=" * 55)
print(f"\n  Files saved to models/evaluation/:")
print(f"    - metrics.pkl (all metrics data)")
print(f"    - cm_*.png (confusion matrices)")
print(f"    - roc_*.png (ROC curves)")
print(f"    - model_comparison.png (comparison chart)")

print(f"\n  Summary:")
print(f"  {'Model':<24} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print(f"  {'-'*64}")
for name, m in all_metrics.items():
    print(f"  {name:<24} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
          f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} {m['roc_auc']:>8.4f}")
print("=" * 55)
