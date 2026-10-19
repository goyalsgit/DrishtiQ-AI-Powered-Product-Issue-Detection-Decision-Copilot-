"""Lesson 1: analyse five fictional reviews using basic Python.

Run from the project folder: python3 lessons/01_review_basics.py
These examples are learning data, not real customer reviews.
"""

# A dictionary stores one review as named fields (key-value pairs).
# A list holds several reviews. Each rating is an integer.
reviews = [
    {"review_id": "DEMO001", "product": "Moisturizer", "rating": 5, "text": "Feels gentle on my skin."},
    {"review_id": "DEMO002", "product": "Moisturizer", "rating": 2, "text": "The jar arrived cracked."},
    {"review_id": "DEMO003", "product": "Moisturizer", "rating": 1, "text": "The lid leaked during delivery."},
    {"review_id": "DEMO004", "product": "Moisturizer", "rating": 4, "text": "Easy to apply."},
    {"review_id": "DEMO005", "product": "Moisturizer", "rating": 3, "text": "It was okay."},
]

# Our explicit project rule: ratings of 1 or 2 count as negative.
# This is a rating-based label, not AI analysis of the review text.
negative_count = 0
rating_total = 0

# A loop processes each review in the list, one at a time.
for review in reviews:
    rating_total = rating_total + review["rating"]
    if review["rating"] <= 2:
        negative_count = negative_count + 1

total_reviews = len(reviews)

# Avoid division by zero if the list is empty.
if total_reviews == 0:
    print("No reviews available. Add reviews before calculating metrics.")
else:
    average_rating = rating_total / total_reviews
    negative_share = (negative_count / total_reviews) * 100

    print("OmniSight — Lesson 1 (fictional learning data)")
    print(f"Total reviews: {total_reviews}")
    print(f"Average rating: {average_rating:.2f} / 5")
    print(f"Negative reviews: {negative_count}")
    print(f"Negative-review share: {negative_share:.1f}%")
