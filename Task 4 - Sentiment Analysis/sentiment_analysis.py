# ==========================================================
# CodeAlpha Internship
# Task 4 - Sentiment Analysis
# ==========================================================

import os
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# ==========================================================
# Create Output Folder
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "graphs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

DATA_PATH = os.path.join(BASE_DIR, "datasets", "Reviews.csv")

df = pd.read_csv(DATA_PATH)

# ==========================================================
# Display Dataset Information
# ==========================================================

print("=" * 60)
print(" AMAZON REVIEWS SENTIMENT ANALYSIS ")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Loaded Successfully!")

print("=" * 60)
print("Performing Sentiment Analysis...")
print("=" * 60)

# Use first 5000 reviews for faster execution
reviews = df[["Text"]].dropna().head(5000).copy()


def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


reviews["Sentiment"] = reviews["Text"].apply(get_sentiment)

print("\nSentiment Counts:\n")
print(reviews["Sentiment"].value_counts())

plt.figure(figsize=(8,5))

sns.countplot(
    data=reviews,
    x="Sentiment",
    order=["Positive","Neutral","Negative"]
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "sentiment_distribution.png"
    ),
    dpi=300
)

plt.show()

plt.figure(figsize=(7,7))

reviews["Sentiment"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.ylabel("")
plt.title("Sentiment Percentage")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "sentiment_pie.png"
    ),
    dpi=300
)

plt.show()

# ==========================================================
# WORD CLOUD
# ==========================================================

print("\nGenerating Word Cloud...")

text = " ".join(reviews["Text"].astype(str))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    stopwords=stopwords
).generate(text)

plt.figure(figsize=(14,7))

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Reviews", fontsize=18)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "wordcloud.png"
    ),
    dpi=300
)

plt.show()

# ==========================================================
# SAMPLE REVIEWS
# ==========================================================

print("\n")
print("="*60)
print("Sample Positive Reviews")
print("="*60)

positive_reviews = reviews[
    reviews["Sentiment"]=="Positive"
]["Text"].head(3)

for i, review in enumerate(positive_reviews,1):
    print(f"\n{i}. {review[:180]}...")

print("\n")
print("="*60)
print("Sample Negative Reviews")
print("="*60)

negative_reviews = reviews[
    reviews["Sentiment"]=="Negative"
]["Text"].head(3)

for i, review in enumerate(negative_reviews,1):
    print(f"\n{i}. {review[:180]}...")

print("\n" + "="*70)
print("SAMPLE POSITIVE REVIEWS")
print("="*70)

for i, review in enumerate(reviews[reviews["Sentiment"]=="Positive"]["Text"].head(3),1):
    print(f"\n{i}. {review[:200]}...")

print("\n" + "="*70)
print("SAMPLE NEGATIVE REVIEWS")
print("="*70)

for i, review in enumerate(reviews[reviews["Sentiment"]=="Negative"]["Text"].head(3),1):
    print(f"\n{i}. {review[:200]}...")

print("\n" + "="*70)
print("TASK 4 COMPLETED SUCCESSFULLY")
print("="*70)      