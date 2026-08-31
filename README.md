# decodelab-task-3-Aleena-Murtaza
Tech Stack Recommender — a content-based recommendation engine that maps a user's input skills to relevant job roles using TF-IDF weighted vectors and Cosine Similarity, without relying on any historical user data.

It follows a full Input-Process-Output pipeline (ingest skills → vectorize & score → sort → filter) and returns the Top 3 best-matching career paths, with a fallback for cold-start cases where no skills overlap the dataset.
