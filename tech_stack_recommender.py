"""
DecodeLabs — Project 3: AI Recommendation Logic
Tech Stack Recommender

Content-based filtering engine that maps a user's raw skills to the
closest-matching job roles, using TF-IDF weighted vectors and Cosine
Similarity — built from scratch (no ML libraries required) so every
step of the IPO (Input -> Process -> Output) pipeline is visible.

Pipeline:
    1. Ingestion  — load the job-role dataset, capture user skills
    2. Vectorize  — build a shared vocabulary, compute TF-IDF weights
    3. Scoring    — cosine similarity between user vector & every item
    4. Sorting    — rank items by descending similarity score
    5. Filtering  — truncate to the Top-N list (default N = 3)
"""

import csv
import math
from collections import Counter


# --------------------------------------------------------------------------- #
# 1. INGESTION
# --------------------------------------------------------------------------- #

def load_dataset(path):
    """Read job_role,skills rows into a list of (role, [skill tags]) tuples."""
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags = [tag.strip().lower() for tag in row["skills"].split(",") if tag.strip()]
            items.append((row["job_role"].strip(), tags))
    return items


def get_user_skills(min_inputs=3):
    """
    Prompt the user for skills interactively.
    Enforces the Project 3 requirement of >= 3 inputs for sufficient
    data density.
    """
    print(f"Enter at least {min_inputs} skills or interests (comma-separated):")
    raw = input("> ")
    skills = [s.strip().lower() for s in raw.split(",") if s.strip()]

    while len(skills) < min_inputs:
        print(f"Please enter at least {min_inputs} skills.")
        raw = input("> ")
        skills = [s.strip().lower() for s in raw.split(",") if s.strip()]

    return skills


# --------------------------------------------------------------------------- #
# 2. VECTOR MAPPING (TF-IDF)
# --------------------------------------------------------------------------- #

def build_vocabulary(items):
    """Every unique skill tag across all items defines a dimension."""
    vocab = set()
    for _, tags in items:
        vocab.update(tags)
    return sorted(vocab)


def compute_tf(tags, vocab):
    """Term Frequency: count of term / total terms in this document."""
    counts = Counter(tags)
    total = len(tags) if tags else 1
    return {term: counts.get(term, 0) / total for term in vocab}


def compute_idf(items, vocab):
    """
    Inverse Document Frequency: log(total docs / docs containing term).
    Penalizes generic skills that appear across many job roles, rewards
    specific/rare ones.
    """
    n_docs = len(items)
    idf = {}
    for term in vocab:
        docs_with_term = sum(1 for _, tags in items if term in tags)
        # +1 guards against division by zero / log(0) for unseen terms
        idf[term] = math.log(n_docs / (docs_with_term + 1)) + 1
    return idf


def tfidf_vector(tags, vocab, idf):
    """Combine TF and IDF into a single weighted vector for one document."""
    tf = compute_tf(tags, vocab)
    return [tf[term] * idf[term] for term in vocab]


# --------------------------------------------------------------------------- #
# 3. SIMILARITY ENGINE (Cosine Similarity)
# --------------------------------------------------------------------------- #

def cosine_similarity(vec_a, vec_b):
    """
    cos(theta) = (A . B) / (||A|| * ||B||)
    Measures the angle between two vectors, ignoring magnitude — so a
    short user profile can still align strongly with a rich job-role
    profile.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0  # Cold Start: a zero vector can't be compared
    return dot / (norm_a * norm_b)


# --------------------------------------------------------------------------- #
# 4. RANKING PIPELINE (Scoring -> Sorting -> Filtering)
# --------------------------------------------------------------------------- #

def recommend(user_skills, items, top_n=3):
    vocab = build_vocabulary(items)
    idf = compute_idf(items, vocab)

    # Cold-start check: warn if none of the user's skills exist in the vocabulary
    known_skills = [s for s in user_skills if s in vocab]
    if not known_skills:
        return [], "cold_start"

    user_vector = tfidf_vector(user_skills, vocab, idf)

    # Step 2: Scoring — compare user vector against every item
    scored = []
    for role, tags in items:
        item_vector = tfidf_vector(tags, vocab, idf)
        score = cosine_similarity(user_vector, item_vector)
        scored.append((role, score))

    # Step 3: Sorting — descending by similarity score
    scored.sort(key=lambda pair: pair[1], reverse=True)

    # Step 4: Filtering — truncate to Top-N to prevent choice overload
    return scored[:top_n], "ok"


# --------------------------------------------------------------------------- #
# ENTRY POINT
# --------------------------------------------------------------------------- #

def main():
    items = load_dataset("raw_skills.csv")
    user_skills = get_user_skills(min_inputs=3)

    results, status = recommend(user_skills, items, top_n=3)

    print("\n--- Top 3 Recommended Career Paths ---")
    if status == "cold_start":
        print("None of your listed skills matched our dataset (Cold Start).")
        print("Falling back to trending / most common roles instead:\n")
        # Trending fallback: rank roles by how many skills they define (proxy for popularity)
        fallback = sorted(items, key=lambda item: len(item[1]), reverse=True)[:3]
        for i, (role, tags) in enumerate(fallback, start=1):
            print(f"{i}. {role}")
    else:
        for i, (role, score) in enumerate(results, start=1):
            print(f"{i}. {role:<28} match: {score * 100:.1f}%")


if __name__ == "__main__":
    main()
