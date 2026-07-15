# ==========================================================
# CodeAlpha Internship
# Task 3 - Data Visualization Dashboard
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs("outputs/graphs", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "netflix_titles.csv")

df = pd.read_csv(DATA_PATH)

# ==========================================================
# Data Cleaning
# ==========================================================

clean_df = df.copy()

clean_df["director"] = clean_df["director"].fillna("Unknown")
clean_df["cast"] = clean_df["cast"].fillna("Not Available")
clean_df["country"] = clean_df["country"].fillna("Unknown")
clean_df["rating"] = clean_df["rating"].fillna(clean_df["rating"].mode()[0])
clean_df["duration"] = clean_df["duration"].fillna(clean_df["duration"].mode()[0])

clean_df.dropna(subset=["date_added"], inplace=True)

# ==========================================================
# Dashboard Summary
# ==========================================================

print("=" * 60)
print("           NETFLIX DATA VISUALIZATION DASHBOARD")
print("=" * 60)

print(f"\nTotal Titles      : {len(clean_df)}")
print(f"Movies            : {(clean_df['type'] == 'Movie').sum()}")
print(f"TV Shows          : {(clean_df['type'] == 'TV Show').sum()}")
print(f"Countries         : {clean_df['country'].nunique()}")
print(f"Directors         : {clean_df['director'].nunique()}")
print(f"Genres            : {clean_df['listed_in'].nunique()}")
print(f"Ratings           : {clean_df['rating'].nunique()}")

print("\nDashboard Loaded Successfully!")

print("=" * 60)

# ==========================================================
# PROFESSIONAL NETFLIX DASHBOARD
# ==========================================================

fig, axes = plt.subplots(3, 2, figsize=(20, 15))

fig.suptitle(
    "Netflix Data Visualization Dashboard",
    fontsize=24,
    fontweight="bold",
    y=0.99
)

# ----------------------------------------------------------
# 1. Movies vs TV Shows
# ----------------------------------------------------------
sns.countplot(
    data=clean_df,
    x="type",
    hue="type",
    palette="Set2",
    legend=False,
    ax=axes[0,0]
)

axes[0,0].set_title("Movies vs TV Shows", fontsize=16)
axes[0,0].set_xlabel("")
axes[0,0].set_ylabel("Count", fontsize=12)

# ----------------------------------------------------------
# 2. Rating Distribution
# ----------------------------------------------------------
clean_df["rating"].value_counts().head(10).plot(
    kind="bar",
    color="mediumpurple",
    ax=axes[0,1]
)

axes[0,1].set_title("Top Ratings", fontsize=16)
axes[0,1].set_xlabel("Rating", fontsize=12)
axes[0,1].set_ylabel("Count", fontsize=12)
axes[0,1].tick_params(axis="x", rotation=45)

# ----------------------------------------------------------
# 3. Top 10 Countries
# ----------------------------------------------------------
country = (
    clean_df["country"]
    .dropna()
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

country.plot(
    kind="barh",
    color="teal",
    ax=axes[1,0]
)

axes[1,0].set_title("Top 10 Countries", fontsize=16)
axes[1,0].set_xlabel("Number of Titles", fontsize=12)
axes[1,0].set_ylabel("")

# ----------------------------------------------------------
# 4. Top 10 Genres
# ----------------------------------------------------------
genres = (
    clean_df["listed_in"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

genres.plot(
    kind="barh",
    color="coral",
    ax=axes[1,1]
)

axes[1,1].set_title("Top 10 Genres", fontsize=16)
axes[1,1].set_xlabel("Number of Titles", fontsize=12)
axes[1,1].set_ylabel("")

# ----------------------------------------------------------
# 5. Content Released Over Years
# ----------------------------------------------------------
clean_df["release_year"].value_counts().sort_index().plot(
    ax=axes[2,0],
    color="royalblue",
    linewidth=2
)

axes[2,0].set_title("Content Released Over Years", fontsize=16)
axes[2,0].set_xlabel("Year", fontsize=12)
axes[2,0].set_ylabel("Titles", fontsize=12)

# ----------------------------------------------------------
# 6. Movie Duration Distribution
# ----------------------------------------------------------
movie_duration = clean_df[
    clean_df["type"] == "Movie"
]["duration"].str.replace(" min", "", regex=False)

movie_duration = pd.to_numeric(movie_duration, errors="coerce")

axes[2,1].hist(
    movie_duration.dropna(),
    bins=25,
    color="steelblue",
    edgecolor="black"
)

axes[2,1].set_title("Movie Duration Distribution", fontsize=16)
axes[2,1].set_xlabel("Minutes", fontsize=12)
axes[2,1].set_ylabel("Movies", fontsize=12)

# ==========================================================
# Improve Spacing
# ==========================================================

plt.subplots_adjust(
    top=0.92,
    bottom=0.07,
    left=0.07,
    right=0.98,
    hspace=0.45,
    wspace=0.30
)

# ==========================================================
# Save Dashboard
# ==========================================================

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "netflix_dashboard.png"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()