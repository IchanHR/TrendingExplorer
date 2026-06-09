"""
prepare_new_dataset.py
======================
Menggabungkan DUA sumber dataset YouTube Trending dari Kaggle:

  FOLDER 1 — 10 negara (US, CA, GB, DE, FR, IN, JP, KR, MX, RU)
             Kolom: video_id, trending_date, title, channel_title,
                    category_id, publish_time, tags, views, likes,
                    dislikes, comment_count, ...

  FOLDER 2 — US & GB + file comments terpisah
             Kolom: video_id, title, channel_title, category_id, tags,
                    views, likes, dislikes, comment_total, thumbnail_link, date
             Comments: video_id, comment_text, likes, replies

Output: youtube_kaggle_processed.csv  ← siap digabung ke dataset lama

Cara pakai:
    1. Taruh semua CSV & JSON di folder yang sama dengan script ini
    2. Jalankan: python prepare_new_dataset.py
"""

import pandas as pd
import numpy as np
import json
import os

# ── KONFIGURASI ───────────────────────────────────────────────────────────────

DATA_DIR    = "."   # ganti jika file ada di subfolder
OUTPUT_FILE = "youtube_kaggle_processed.csv"

# Folder 1 — 10 negara
FOLDER1_COUNTRIES = ["US", "CA", "GB", "DE", "FR", "IN", "JP", "KR", "MX", "RU"]

# Folder 2 — hanya US & GB (dengan comments)
FOLDER2_COUNTRIES = ["US", "GB"]

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def load_category_map(country_code):
    path = os.path.join(DATA_DIR, f"{country_code}_category_id.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(item["id"]): item["snippet"]["title"] for item in data["items"]}
    except Exception as e:
        print(f"  [WARN] Gagal baca {path}: {e}")
        return {}

def extract_first_tag(tags_str):
    if pd.isna(tags_str) or str(tags_str).strip() in ["[none]", ""]:
        return ""
    tags = str(tags_str).replace('"', '').split("|")
    tag  = tags[0].strip()
    return f"#{tag.replace(' ', '')}" if tag else ""

def assign_trend_label(views, engagement_rate):
    if views >= 1_000_000 and engagement_rate >= 0.05:
        return "rising"
    elif views >= 500_000:
        return "stable"
    elif views >= 100_000:
        return "emerging"
    else:
        return "declining"

def creator_tier(views):
    if views >= 10_000_000: return "Mega"
    elif views >= 1_000_000: return "Macro"
    elif views >= 100_000:   return "Mid"
    elif views >= 10_000:    return "Micro"
    else:                    return "Nano"

def standardize_df(df):
    """Standarisasi kolom ke format yang dibutuhkan app."""
    np.random.seed(42)
    n = len(df)

    # Numerik
    for col in ["views", "likes", "dislikes", "comments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Engagement
    df["engagement_rate"] = np.where(
        df["views"] > 0,
        (df["likes"] + df["comments"]) / df["views"], 0
    )
    dislikes_col = df["dislikes"] if "dislikes" in df.columns else pd.Series([0]*n, index=df.index)
    dislikes_col = pd.to_numeric(dislikes_col, errors="coerce").fillna(0)
    df["like_dislike_ratio"] = pd.Series(
        np.where(
            dislikes_col > 0,
            df["likes"] / dislikes_col.replace(0, np.nan),
            df["likes"].clip(upper=9999)
        ), index=df.index
    ).fillna(df["likes"].clip(upper=9999))
    df["engagement_velocity"] = np.where(
        df["views"] > 0,
        (df["likes"] / df["views"]) * 1000, 0
    )

    # Kolom yang tidak ada di dataset → isi default
    df["shares"] = 0
    df["saves"]  = 0
    df["duration_sec"] = np.random.randint(60, 601, size=n)

    # Upload time features
    if "publish_time" in df.columns:
        dt = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        dt = pd.Series([pd.NaT] * n)

    df["upload_hour"] = dt.dt.hour.fillna(12).astype(int)
    df["is_weekend"]  = dt.dt.dayofweek.isin([5, 6]).astype(int)

    # Title features
    df["title"]        = df["title"].fillna("").astype(str)
    df["title_length"] = df["title"].str.len()
    df["has_emoji"]    = df["title"].apply(lambda x: 1 if any(ord(c) > 127 for c in x) else 0)

    # Derived columns
    df["hashtag"]      = df.get("tags", pd.Series([""] * n)).apply(extract_first_tag)
    df["genre"]        = df["category"]
    df["trend_label"]  = df.apply(lambda r: assign_trend_label(r["views"], r["engagement_rate"]), axis=1)
    df["creator_tier"] = df["views"].apply(creator_tier)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 1 — FOLDER 1 (10 negara, format lama: trending_date + publish_time)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("📂 FOLDER 1 — Dataset 10 Negara")
print("=" * 60)

folder1_dfs = []

for country in FOLDER1_COUNTRIES:
    csv_path = os.path.join(DATA_DIR, f"{country}videos.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} tidak ditemukan")
        continue

    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="latin-1")

    # Rename comment_count → comments
    if "comment_count" in df.columns:
        df = df.rename(columns={"comment_count": "comments"})

    cat_map          = load_category_map(country)
    df["category"]   = df["category_id"].map(cat_map).fillna("Unknown")
    df["region"]     = country
    df["platform"]   = "YouTube"

    df = standardize_df(df)
    folder1_dfs.append(df)
    print(f"  [OK] {country}videos.csv → {len(df):,} baris")

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 2 — FOLDER 2 (US & GB, format baru: date + comment_total + comments file)
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("📂 FOLDER 2 — Dataset US & GB + Comments")
print("=" * 60)

folder2_dfs = []

for country in FOLDER2_COUNTRIES:
    vid_path = os.path.join(DATA_DIR, f"{country}videos.csv")
    com_path = os.path.join(DATA_DIR, f"{country}comments.csv")

    # Cek apakah ini memang folder-2 format (ada kolom 'comment_total')
    try:
        test = pd.read_csv(vid_path, on_bad_lines="skip", nrows=1)
        if "comment_total" not in test.columns:
            print(f"  [SKIP] {country}videos.csv → bukan format folder 2, sudah diproses di folder 1")
            continue
    except Exception:
        print(f"  [SKIP] {vid_path} tidak ditemukan")
        continue

    print(f"  [LOAD] {country}videos.csv ...")
    try:
        vid = pd.read_csv(vid_path, on_bad_lines="skip", encoding="utf-8")
    except UnicodeDecodeError:
        vid = pd.read_csv(vid_path, on_bad_lines="skip", encoding="latin-1")

    # Rename kolom ke standard
    vid = vid.rename(columns={"comment_total": "comments"})

    # Join dengan comments untuk dapat total comment_likes per video
    if os.path.exists(com_path):
        print(f"  [LOAD] {country}comments.csv ...")
        try:
            com = pd.read_csv(com_path, on_bad_lines="skip",
                              dtype={"likes": str, "replies": str}, low_memory=False)
        except Exception:
            com = pd.DataFrame()

        if len(com) > 0:
            com["likes"]   = pd.to_numeric(com["likes"],   errors="coerce").fillna(0)
            com["replies"] = pd.to_numeric(com["replies"], errors="coerce").fillna(0)

            # Agregasi per video: total comment likes & avg replies
            com_agg = com.groupby("video_id").agg(
                comment_likes_total = ("likes",   "sum"),
                avg_replies         = ("replies", "mean"),
                comment_count_check = ("likes",   "count")
            ).reset_index()

            vid = vid.merge(com_agg, on="video_id", how="left")
            print(f"    → Berhasil join dengan {len(com_agg):,} unique video dari comments")
    else:
        print(f"  [WARN] {com_path} tidak ditemukan, skip join")

    cat_map        = load_category_map(country)
    vid["category"] = vid["category_id"].map(cat_map).fillna("Unknown")
    vid["region"]   = f"{country}_v2"   # tandai sebagai folder 2 agar tidak duplikat
    vid["platform"] = "YouTube"

    vid = standardize_df(vid)
    folder2_dfs.append(vid)
    print(f"  [OK] {country} folder2 → {len(vid):,} baris")

# ─────────────────────────────────────────────────────────────────────────────
# GABUNGKAN SEMUA
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("🔗 MENGGABUNGKAN SEMUA DATASET")
print("=" * 60)

all_dfs = folder1_dfs + folder2_dfs

if not all_dfs:
    raise RuntimeError("Tidak ada data yang berhasil dimuat!")

# Kolom final yang diinginkan
FINAL_COLS = [
    "title", "platform", "category", "genre", "region",
    "views", "likes", "dislikes", "comments", "shares", "saves",
    "engagement_rate", "like_dislike_ratio", "engagement_velocity",
    "duration_sec", "upload_hour", "is_weekend",
    "title_length", "has_emoji",
    "hashtag", "trend_label", "creator_tier",
    "channel_title"
]

combined_list = []
for df in all_dfs:
    cols = [c for c in FINAL_COLS if c in df.columns]
    combined_list.append(df[cols])

combined = pd.concat(combined_list, ignore_index=True)

# Hapus baris views = 0
combined = combined[combined["views"] > 0].reset_index(drop=True)

# Hapus duplikat berdasarkan title + platform
before = len(combined)
combined = combined.drop_duplicates(subset=["title", "platform"]).reset_index(drop=True)
after = len(combined)
print(f"  Duplikat dihapus: {before - after:,} baris")

# Simpan
combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"\n✅ SELESAI!")
print(f"   File: {OUTPUT_FILE}")
print(f"   Total baris: {len(combined):,}")
print(f"\n📊 Trend Label:")
print(combined["trend_label"].value_counts().to_string())
print(f"\n🌍 Region:")
print(combined["region"].value_counts().to_string())
print(f"\n📂 Top 10 Category:")
print(combined["category"].value_counts().head(10).to_string())

