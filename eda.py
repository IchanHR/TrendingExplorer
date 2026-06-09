"""
eda.py — Exploratory Data Analysis
====================================
Generates all EDA visualizations as PNG files for the PPT.
Output folder: eda_output/

EDA Pipeline:
  1. Data Loading & Overview
  2. Missing Values Analysis
  3. Target Distribution (Class Imbalance)
  4. Numerical Feature Distributions
  5. Categorical Feature Distributions
  6. Correlation Analysis
  7. Feature vs Target Analysis
  8. Engagement Analysis
  9. Platform & Content Analysis
 10. Temporal Patterns
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── SETUP ────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#1a1a2e",
    "axes.labelcolor":  "white",
    "axes.titlesize":   14,
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "white",
    "grid.color":       "#2d2d44",
})

OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, name):
    fig.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {name}.png")

# ══════════════════════════════════════════════════════════
# 1. DATA LOADING & OVERVIEW
# ══════════════════════════════════════════════════════════

print("=" * 55)
print("  EDA — Trending Content Explorer")
print("=" * 55)

print("\n[1/10] Loading data...")

df_main = pd.read_csv("youtube_shorts_tiktok_trends_2025.csv")
print(f"  Main dataset: {df_main.shape[0]:,} rows x {df_main.shape[1]} columns")

if os.path.exists("youtube_kaggle_processed.csv"):
    df_kaggle = pd.read_csv("youtube_kaggle_processed.csv")
    print(f"  Kaggle dataset: {df_kaggle.shape[0]:,} rows x {df_kaggle.shape[1]} columns")
    df = pd.concat([df_main, df_kaggle], ignore_index=True)
    print(f"  Combined: {df.shape[0]:,} rows")
else:
    df = df_main

df["is_trending"] = df["trend_label"].apply(
    lambda x: 1 if str(x).lower() == "rising" else 0
)

for col in ["views", "likes", "comments", "shares", "saves",
            "engagement_rate", "duration_sec"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ── Dataset overview summary ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")
summary_text = (
    f"Dataset Overview\n"
    f"{'─' * 40}\n"
    f"Total Samples:    {len(df):,}\n"
    f"Features:         {df.shape[1]}\n"
    f"Platforms:         {df['platform'].nunique() if 'platform' in df.columns else 'N/A'}\n"
    f"Categories:       {df['category'].nunique() if 'category' in df.columns else 'N/A'}\n"
    f"Trending (rising): {df['is_trending'].sum():,} ({df['is_trending'].mean():.1%})\n"
    f"Not Trending:     {(~df['is_trending'].astype(bool)).sum():,} ({1 - df['is_trending'].mean():.1%})"
)
ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=14,
        verticalalignment="center", horizontalalignment="center",
        fontfamily="monospace", color="white",
        bbox=dict(boxstyle="round,pad=1", facecolor="#1a1a2e", edgecolor="#a78bfa"))
save_fig(fig, "01_dataset_overview")

# ══════════════════════════════════════════════════════════
# 2. MISSING VALUES ANALYSIS
# ══════════════════════════════════════════════════════════

print("\n[2/10] Missing values analysis...")

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0].head(20)

if len(missing_pct) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_pct.plot(kind="barh", ax=ax, color="#a78bfa")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values by Column (Top 20)")
    ax.invert_yaxis()
    save_fig(fig, "02_missing_values")
else:
    print("  No missing values found!")

# ══════════════════════════════════════════════════════════
# 3. TARGET DISTRIBUTION (CLASS IMBALANCE)
# ══════════════════════════════════════════════════════════

print("\n[3/10] Target distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
counts = df["is_trending"].value_counts()
labels = ["Not Trending", "Trending"]
colors = ["#e74c3c", "#4ade80"]
axes[0].bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
    axes[0].text(i, v + len(df)*0.01, f"{v:,}", ha="center", fontweight="bold", color="white")

# Pie chart
axes[1].pie([counts.get(0, 0), counts.get(1, 0)], labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90, textprops={"color": "white"})
axes[1].set_title("Class Balance Ratio")

fig.suptitle("Target Variable: Trending vs Not Trending", fontsize=15, fontweight="bold")
plt.tight_layout()
save_fig(fig, "03_target_distribution")

# Trend label breakdown
if "trend_label" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    trend_counts = df["trend_label"].value_counts()
    colors_trend = ["#4ade80", "#60a5fa", "#f97316", "#e74c3c"]
    trend_counts.plot(kind="bar", ax=ax, color=colors_trend[:len(trend_counts)])
    ax.set_title("Trend Label Distribution")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for i, v in enumerate(trend_counts):
        ax.text(i, v + len(df)*0.005, f"{v:,}", ha="center", fontsize=9, color="white")
    plt.tight_layout()
    save_fig(fig, "03b_trend_label_breakdown")

# ══════════════════════════════════════════════════════════
# 4. NUMERICAL FEATURE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════

print("\n[4/10] Numerical feature distributions...")

num_features = ["views", "likes", "comments", "shares", "saves",
                "engagement_rate", "duration_sec"]
num_features = [c for c in num_features if c in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(num_features):
    data = df[col].clip(upper=df[col].quantile(0.99))
    axes[i].hist(data, bins=50, color="#a78bfa", alpha=0.8, edgecolor="none")
    axes[i].set_title(col, fontsize=11)
    axes[i].set_ylabel("Count")

for j in range(len(num_features), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distribution of Numerical Features (clipped at 99th percentile)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "04_numerical_distributions")

# Log-scale views distribution
fig, ax = plt.subplots(figsize=(10, 5))
views_nonzero = df["views"][df["views"] > 0]
ax.hist(np.log10(views_nonzero), bins=60, color="#60a5fa", alpha=0.8, edgecolor="none")
ax.set_xlabel("log10(Views)")
ax.set_ylabel("Count")
ax.set_title("Views Distribution (Log Scale)")
ax.axvline(np.log10(views_nonzero.median()), color="#f97316", linestyle="--",
           label=f"Median: {views_nonzero.median():,.0f}")
ax.legend()
plt.tight_layout()
save_fig(fig, "04b_views_log_distribution")

# ══════════════════════════════════════════════════════════
# 5. CATEGORICAL FEATURE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════

print("\n[5/10] Categorical feature distributions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Platform
if "platform" in df.columns:
    platform_counts = df["platform"].value_counts()
    axes[0, 0].bar(platform_counts.index, platform_counts.values, color=["#e74c3c", "#60a5fa"])
    axes[0, 0].set_title("Platform Distribution")
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate(platform_counts.values):
        axes[0, 0].text(i, v + len(df)*0.005, f"{v:,}", ha="center", fontsize=9, color="white")

# Top 10 Categories
if "category" in df.columns:
    cat_counts = df["category"].value_counts().head(10)
    axes[0, 1].barh(cat_counts.index, cat_counts.values, color="#a78bfa")
    axes[0, 1].set_title("Top 10 Categories")
    axes[0, 1].invert_yaxis()

# Top 10 Genres
if "genre" in df.columns:
    genre_counts = df["genre"].value_counts().head(10)
    axes[1, 0].barh(genre_counts.index, genre_counts.values, color="#4ade80")
    axes[1, 0].set_title("Top 10 Genres")
    axes[1, 0].invert_yaxis()

# Creator Tier
if "creator_tier" in df.columns:
    tier_order = ["Nano", "Micro", "Mid", "Macro", "Mega"]
    tier_counts = df["creator_tier"].value_counts()
    tier_counts = tier_counts.reindex([t for t in tier_order if t in tier_counts.index])
    colors_tier = ["#888", "#60a5fa", "#a78bfa", "#f97316", "#e74c3c"]
    axes[1, 1].bar(tier_counts.index, tier_counts.values,
                   color=colors_tier[:len(tier_counts)])
    axes[1, 1].set_title("Creator Tier Distribution")
    axes[1, 1].set_ylabel("Count")

fig.suptitle("Categorical Feature Distributions", fontsize=15, fontweight="bold")
plt.tight_layout()
save_fig(fig, "05_categorical_distributions")

# ══════════════════════════════════════════════════════════
# 6. CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════

print("\n[6/10] Correlation analysis...")

corr_cols = ["views", "likes", "comments", "shares", "saves",
             "engagement_rate", "duration_sec", "title_length",
             "is_trending"]
corr_cols = [c for c in corr_cols if c in df.columns]

corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5,
            annot_kws={"size": 9, "color": "white"})
ax.set_title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
save_fig(fig, "06_correlation_heatmap")

# Correlation with target
fig, ax = plt.subplots(figsize=(8, 5))
target_corr = corr_matrix["is_trending"].drop("is_trending").sort_values()
colors_corr = ["#4ade80" if v > 0 else "#e74c3c" for v in target_corr]
target_corr.plot(kind="barh", ax=ax, color=colors_corr)
ax.set_title("Feature Correlation with Target (is_trending)")
ax.set_xlabel("Correlation Coefficient")
ax.axvline(0, color="white", linewidth=0.5)
plt.tight_layout()
save_fig(fig, "06b_correlation_with_target")

# ══════════════════════════════════════════════════════════
# 7. FEATURE vs TARGET ANALYSIS
# ══════════════════════════════════════════════════════════

print("\n[7/10] Feature vs target analysis...")

# ── Individual large charts per key feature (easier to read on slides) ──
key_features = [
    ("views",           "Views Distribution"),
    ("likes",           "Likes Distribution"),
    ("engagement_rate", "Engagement Rate Distribution"),
    ("duration_sec",    "Video Duration Distribution"),
]

for col, title in key_features:
    if col not in df.columns:
        continue
    fig, ax = plt.subplots(figsize=(10, 6))
    trending     = df[df["is_trending"] == 1][col].clip(upper=df[col].quantile(0.95))
    not_trending = df[df["is_trending"] == 0][col].clip(upper=df[col].quantile(0.95))

    ax.hist(not_trending, bins=50, alpha=0.65, color="#e74c3c", label="Not Trending", density=True)
    ax.hist(trending, bins=50, alpha=0.65, color="#4ade80", label="Trending", density=True)

    ax.axvline(trending.median(), color="#4ade80", linestyle="--", linewidth=2,
               label=f"Trending Median: {trending.median():,.1f}")
    ax.axvline(not_trending.median(), color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Not Trending Median: {not_trending.median():,.1f}")

    ax.set_title(f"{title}: Trending vs Not Trending", fontsize=14, fontweight="bold")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, f"07_{col}_comparison")

# ── Violin plots (more visual than box plots) ──
violin_features = ["views", "likes", "engagement_rate"]
violin_features = [c for c in violin_features if c in df.columns]

fig, axes = plt.subplots(1, len(violin_features), figsize=(6 * len(violin_features), 7))
if len(violin_features) == 1:
    axes = [axes]

for i, col in enumerate(violin_features):
    data_plot = df[[col, "is_trending"]].copy()
    data_plot[col] = data_plot[col].clip(upper=data_plot[col].quantile(0.95))
    data_plot["Status"] = data_plot["is_trending"].map({0: "Not Trending", 1: "Trending"})

    parts = axes[i].violinplot(
        [data_plot[data_plot["Status"] == "Not Trending"][col].values,
         data_plot[data_plot["Status"] == "Trending"][col].values],
        positions=[0, 1], showmeans=True, showmedians=True
    )
    for j, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#e74c3c", "#4ade80"][j])
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("white")
    parts["cmedians"].set_color("#f97316")
    parts["cmins"].set_color("white")
    parts["cmaxes"].set_color("white")
    parts["cbars"].set_color("white")

    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(["Not Trending", "Trending"], fontsize=12)
    axes[i].set_title(f"{col.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    axes[i].set_ylabel(col.replace("_", " ").title())

    mean_nt = data_plot[data_plot["Status"] == "Not Trending"][col].mean()
    mean_t  = data_plot[data_plot["Status"] == "Trending"][col].mean()
    axes[i].text(0, mean_nt, f"  avg: {mean_nt:,.1f}", color="white", fontsize=9, va="bottom")
    axes[i].text(1, mean_t, f"  avg: {mean_t:,.1f}", color="white", fontsize=9, va="bottom")

fig.suptitle("Feature Comparison: Violin Plots (Trending vs Not Trending)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
save_fig(fig, "07b_violin_plots")

# ══════════════════════════════════════════════════════════
# 8. ENGAGEMENT ANALYSIS
# ══════════════════════════════════════════════════════════

print("\n[8/10] Engagement analysis...")

# ── Scatter: Views vs Likes (separate panels for clarity) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sample_nt = df[df["is_trending"] == 0].sample(min(3000, (df["is_trending"] == 0).sum()), random_state=42)
sample_t  = df[df["is_trending"] == 1].sample(min(3000, (df["is_trending"] == 1).sum()), random_state=42)

for ax, data, color, label in [
    (axes[0], sample_nt, "#e74c3c", "Not Trending"),
    (axes[1], sample_t,  "#4ade80", "Trending"),
]:
    ax.scatter(
        data["views"].clip(upper=data["views"].quantile(0.95)),
        data["likes"].clip(upper=data["likes"].quantile(0.95)),
        c=color, alpha=0.4, s=15, edgecolors="none"
    )
    ax.set_xlabel("Views", fontsize=12)
    ax.set_ylabel("Likes", fontsize=12)
    ax.set_title(f"{label}", fontsize=13, fontweight="bold", color=color)

fig.suptitle("Views vs Likes: Trending vs Not Trending", fontsize=15, fontweight="bold")
plt.tight_layout()
save_fig(fig, "08_scatter_views_likes")

# ── Average metrics comparison (bar chart) ──
fig, ax = plt.subplots(figsize=(10, 6))

compare_cols = ["views", "likes", "comments", "shares", "engagement_rate"]
compare_cols = [c for c in compare_cols if c in df.columns]

avg_trending     = df[df["is_trending"] == 1][compare_cols].mean()
avg_not_trending = df[df["is_trending"] == 0][compare_cols].mean()

x = np.arange(len(compare_cols))
width = 0.35

bars1 = ax.bar(x - width/2, avg_not_trending, width, label="Not Trending",
               color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x + width/2, avg_trending, width, label="Trending",
               color="#4ade80", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", " ").title() for c in compare_cols], fontsize=11)
ax.set_ylabel("Average Value", fontsize=12)
ax.set_title("Average Metrics: Trending vs Not Trending", fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_fig(fig, "08b_avg_metrics_comparison")

# ══════════════════════════════════════════════════════════
# 9. PLATFORM & CONTENT ANALYSIS
# ══════════════════════════════════════════════════════════

print("\n[9/10] Platform & content analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Trending rate by platform
if "platform" in df.columns:
    platform_trending = df.groupby("platform")["is_trending"].mean() * 100
    platform_trending.plot(kind="bar", ax=axes[0], color=["#e74c3c", "#60a5fa"])
    axes[0].set_title("Trending Rate by Platform (%)")
    axes[0].set_ylabel("Trending %")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    for i, v in enumerate(platform_trending):
        axes[0].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=10, color="white")

# Trending rate by top categories
if "category" in df.columns:
    top_cats = df["category"].value_counts().head(8).index
    cat_trending = df[df["category"].isin(top_cats)].groupby("category")["is_trending"].mean() * 100
    cat_trending = cat_trending.sort_values(ascending=True)
    cat_trending.plot(kind="barh", ax=axes[1], color="#a78bfa")
    axes[1].set_title("Trending Rate by Category (%)")
    axes[1].set_xlabel("Trending %")

fig.suptitle("Platform & Content Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "09_platform_content_analysis")

# ══════════════════════════════════════════════════════════
# 10. TEMPORAL PATTERNS
# ══════════════════════════════════════════════════════════

print("\n[10/10] Temporal patterns...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Upload hour distribution
if "upload_hour" in df.columns:
    hour_counts = df.groupby("upload_hour")["is_trending"].agg(["count", "mean"])
    axes[0].bar(hour_counts.index, hour_counts["count"], color="#60a5fa", alpha=0.6, label="Total")
    ax2 = axes[0].twinx()
    ax2.plot(hour_counts.index, hour_counts["mean"] * 100, color="#4ade80",
             linewidth=2, marker="o", markersize=4, label="Trending %")
    ax2.set_ylabel("Trending %", color="#4ade80")
    axes[0].set_xlabel("Upload Hour")
    axes[0].set_ylabel("Total Videos")
    axes[0].set_title("Upload Hour: Volume & Trending Rate")
    axes[0].legend(loc="upper left")
    ax2.legend(loc="upper right")

# Weekend vs Weekday
if "is_weekend" in df.columns:
    weekend_data = df.groupby("is_weekend").agg(
        count=("is_trending", "count"),
        trending_rate=("is_trending", "mean")
    )
    labels_w = ["Weekday", "Weekend"]
    x_w = range(len(labels_w))
    bars = axes[1].bar(x_w, weekend_data["count"], color=["#60a5fa", "#f97316"])
    axes[1].set_xticks(x_w)
    axes[1].set_xticklabels(labels_w)
    axes[1].set_ylabel("Total Videos")
    axes[1].set_title("Weekday vs Weekend")

    ax3 = axes[1].twinx()
    ax3.plot(x_w, weekend_data["trending_rate"] * 100, color="#4ade80",
             linewidth=2, marker="o", markersize=8)
    ax3.set_ylabel("Trending %", color="#4ade80")

fig.suptitle("Temporal Patterns", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "10_temporal_patterns")

# ══════════════════════════════════════════════════════════
# 11. WORD CLOUD — TRENDING TITLES
# ══════════════════════════════════════════════════════════

print("\n[BONUS] Word cloud of trending titles...")

try:
    from wordcloud import WordCloud

    trending_titles = " ".join(df[df["is_trending"] == 1]["title"].fillna("").tolist())
    not_trending_titles = " ".join(df[df["is_trending"] == 0]["title"].fillna("").tolist())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    wc_trending = WordCloud(
        width=800, height=400, background_color="#1a1a2e",
        colormap="Greens", max_words=80, contour_width=1,
        contour_color="#4ade80"
    ).generate(trending_titles if trending_titles.strip() else "no data")

    wc_not = WordCloud(
        width=800, height=400, background_color="#1a1a2e",
        colormap="Reds", max_words=80, contour_width=1,
        contour_color="#e74c3c"
    ).generate(not_trending_titles if not_trending_titles.strip() else "no data")

    axes[0].imshow(wc_trending, interpolation="bilinear")
    axes[0].set_title("Trending Titles", fontsize=14, fontweight="bold", color="#4ade80")
    axes[0].axis("off")

    axes[1].imshow(wc_not, interpolation="bilinear")
    axes[1].set_title("Not Trending Titles", fontsize=14, fontweight="bold", color="#e74c3c")
    axes[1].axis("off")

    fig.suptitle("Word Cloud: What Words Appear in Trending vs Not Trending Titles?",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "11_wordcloud_comparison")

except ImportError:
    print("  [SKIP] wordcloud package not installed. Run: pip install wordcloud")

# ══════════════════════════════════════════════════════════
# 12. TOP CATEGORIES TRENDING RATE (HEATMAP STYLE)
# ══════════════════════════════════════════════════════════

print("\n[BONUS] Category x Platform trending heatmap...")

if "category" in df.columns and "platform" in df.columns:
    top_cats = df["category"].value_counts().head(10).index
    heatmap_data = df[df["category"].isin(top_cats)].groupby(
        ["category", "platform"]
    )["is_trending"].mean().unstack(fill_value=0) * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Trending Rate (%)"},
                annot_kws={"size": 11, "color": "white"})
    ax.set_title("Trending Rate (%) by Category & Platform", fontsize=14, fontweight="bold")
    ax.set_ylabel("Category")
    ax.set_xlabel("Platform")
    plt.tight_layout()
    save_fig(fig, "12_category_platform_heatmap")

# ══════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  EDA COMPLETE!")
print("=" * 55)
print(f"\n  All plots saved to: {OUTPUT_DIR}/")
print(f"  Files generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".png"):
        print(f"    - {f}")
print(f"\n  Use these in your PPT slide 4 (EDA)")
print("=" * 55)
