import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs("Task 2 EDA/outputs/graphs", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("datasets/netflix_titles.csv")

print("=" * 60)
print("NETFLIX DATASET INFORMATION")
print("=" * 60)

# ==========================================================
# 1. Dataset Shape
# ==========================================================

print("\n1. Dataset Shape")
print(df.shape)

# ==========================================================
# 2. Column Names
# ==========================================================

print("\n2. Column Names")
print(df.columns)

# ==========================================================
# 3. Data Types
# ==========================================================

print("\n3. Data Types")
print(df.dtypes)

# ==========================================================
# 4. Dataset Information
# ==========================================================

print("\n4. Dataset Information")
df.info()

# ==========================================================
# 5. Missing Values
# ==========================================================

print("\n" + "=" * 60)
print("5. Missing Values")
print("=" * 60)

print(df.isnull().sum())

# ==========================================================
# 6. Duplicate Records
# ==========================================================

print("\n" + "=" * 60)
print("6. Duplicate Records")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Total Duplicate Records:", duplicates)

# ==========================================================
# 7. Statistical Summary
# ==========================================================

print("\n" + "=" * 60)
print("7. Statistical Summary")
print("=" * 60)

print(df.describe())

# ==========================================================
# 8. Unique Values
# ==========================================================

print("\n" + "=" * 60)
print("8. Unique Values")
print("=" * 60)

print("\nContent Types:")
print(df["type"].unique())

print("\nRatings:")
print(df["rating"].unique())

print("\nTotal Unique Countries:")
print(df["country"].nunique())

print("\nTotal Unique Directors:")
print(df["director"].nunique())

print("\nTotal Unique Genres:")
print(df["listed_in"].nunique())

# ==========================================================
# 9. Data Cleaning
# ==========================================================

print("\n" + "=" * 60)
print("9. Data Cleaning")
print("=" * 60)

clean_df = df.copy()

clean_df["director"] = clean_df["director"].fillna("Unknown")
clean_df["cast"] = clean_df["cast"].fillna("Not Available")
clean_df["country"] = clean_df["country"].fillna("Unknown")

clean_df["rating"] = clean_df["rating"].fillna(clean_df["rating"].mode()[0])
clean_df["duration"] = clean_df["duration"].fillna(clean_df["duration"].mode()[0])

clean_df = clean_df.dropna(subset=["date_added"])

print("\nData Cleaning Completed Successfully!")

print("\nRemaining Missing Values:")
print(clean_df.isnull().sum())

# ==========================================================
# 10. Visualization 1 - Movies vs TV Shows
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=clean_df,
    x="type",
    hue="type",
    palette="Set2",
    legend=False
)

plt.title("Distribution of Movies and TV Shows on Netflix")
plt.xlabel("Content Type")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "Task 2 EDA/outputs/graphs/movies_vs_tvshows.png",
    dpi=300
)

plt.close()

# ==========================================================
# 11. Visualization 2 - Top 10 Countries
# ==========================================================

country_df = clean_df[clean_df["country"] != "Unknown"]

top_countries = (
    country_df["country"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index,
    hue=top_countries.index,
    palette="viridis",
    legend=False
)

plt.title("Top 10 Countries Producing Netflix Content")
plt.xlabel("Number of Movies / TV Shows")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig(
    "Task 2 EDA/outputs/graphs/top10_countries.png",
    dpi=300
)

plt.close()

# ==========================================================
# 12. Visualization 3 - Content Released by Year
# ==========================================================

year_counts = (
    clean_df["release_year"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12,6))

plt.plot(
    year_counts.index,
    year_counts.values,
    linewidth=2
)

plt.title("Netflix Content Released Over the Years")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Task 2 EDA/outputs/graphs/content_by_year.png",
    dpi=300
)

plt.close()

print("\nAll three visualizations have been generated successfully!")

# ==========================================================
# 13. Visualization 4 - Rating Distribution
# ==========================================================

rating_counts = clean_df["rating"].value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=rating_counts.index,
    y=rating_counts.values,
    hue=rating_counts.index,
    palette="magma",
    legend=False
)

plt.title("Top Netflix Content Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "Task 2 EDA/outputs/graphs/rating_distribution.png",
    dpi=300
)

plt.close()

print("Visualization 4 completed.")

# ==========================================================
# 14. Visualization 5 - Top 10 Genres
# ==========================================================

genre_counts = (
    clean_df["listed_in"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=genre_counts.values,
    y=genre_counts.index,
    hue=genre_counts.index,
    palette="coolwarm",
    legend=False
)

plt.title("Top 10 Netflix Genres")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")

plt.tight_layout()

plt.savefig(
    "Task 2 EDA/outputs/graphs/top10_genres.png",
    dpi=300
)

plt.close()

print("Visualization 5 completed.")

# ==========================================================
# 15. Visualization 6 - Top 10 Directors
# ==========================================================

director_counts = (
    clean_df[clean_df["director"] != "Unknown"]["director"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=director_counts.values,
    y=director_counts.index,
    hue=director_counts.index,
    palette="crest",
    legend=False
)

plt.title("Top 10 Directors on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Director")

plt.tight_layout()

plt.savefig(
    "Task 2 EDA/outputs/graphs/top10_directors.png",
    dpi=300
)

plt.close()

print("Visualization 6 completed.")

# ==========================================================
# 16. Visualization 7 - Movie Duration Distribution
# ==========================================================

movie_df = clean_df[clean_df["type"] == "Movie"].copy()

movie_df["duration"] = (
    movie_df["duration"]
    .str.replace(" min", "", regex=False)
)

movie_df["duration"] = pd.to_numeric(movie_df["duration"], errors="coerce")

plt.figure(figsize=(10,6))

sns.histplot(
    movie_df["duration"].dropna(),
    bins=30,
    kde=True,
    color="skyblue"
)

plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Movies")

plt.tight_layout()

plt.savefig(
    "Task 2 EDA/outputs/graphs/movie_duration_distribution.png",
    dpi=300
)

plt.close()

print("Visualization 7 completed.")

# ==========================================================
# 17. Visualization 8 - Movies vs TV Shows by Release Year
# ==========================================================

plt.figure(figsize=(12,6))

sns.countplot(
    data=clean_df,
    x="release_year",
    hue="type"
)

plt.xticks(rotation=90)

plt.title("Movies vs TV Shows Released Each Year")
plt.xlabel("Release Year")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "Task 2 EDA/outputs/graphs/movies_vs_tvshows_year.png",
    dpi=300
)

plt.close()

print("Visualization 8 completed.")

# ==========================================================
# 18. Visualization 9 - Top 10 Release Years
# ==========================================================

top_years = (
    clean_df["release_year"]
    .value_counts()
    .head(10)
    .sort_index()
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_years.index.astype(str),
    y=top_years.values,
    hue=top_years.index.astype(str),
    palette="Blues",
    legend=False
)

plt.title("Top 10 Years with Highest Netflix Content")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    "Task 2 EDA/outputs/graphs/top_release_years.png",
    dpi=300
)

plt.close()

print("Visualization 9 completed.")

