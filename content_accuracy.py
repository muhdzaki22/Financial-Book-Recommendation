import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
financialBookData = pd.read_excel('removedata.xlsx', 'Sheet1')

# Preprocess the descriptions
def preprocess(text):
    # Implement your text preprocessing steps here (e.g., lowercasing, removing punctuation, etc.)
    # For demonstration, we'll just convert text to lowercase
    return text.lower()

# Apply preprocessing to the 'description' column to create 'preprocessed_description'
financialBookData['preprocessed_description'] = financialBookData['description'].apply(preprocess)

def get_tfidf_matrix():
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(financialBookData['preprocessed_description'])
    return tfidf_matrix, vectorizer

def get_item_similarities(tfidf_matrix):
    item_similarities = cosine_similarity(tfidf_matrix)
    return item_similarities

def get_content_based_scores(user_vector, tfidf_matrix):
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)
    content_based_scores = cosine_similarities.flatten()
    return content_based_scores

def get_item_sim_recommendations(user_title, item_similarities, top_k=5):
    # Find the index of the user_title in the dataset
    idx = financialBookData[financialBookData['title'].str.lower() == user_title.lower()].index
    if len(idx) == 0:
        return None
    idx = idx[0]
    
    # Get similarity scores for the item
    similarity_scores = item_similarities[idx]
    similar_indices = similarity_scores.argsort()[-top_k-1:-1][::-1]
    
    return financialBookData.iloc[similar_indices][['title']].values.ravel()

def get_hybrid_recommendations(user_title, alpha=0.6, beta=0.4, k=5, tfidf_matrix=None, item_similarities=None, vectorizer=None):
    if tfidf_matrix is None or vectorizer is None:
        tfidf_matrix, vectorizer = get_tfidf_matrix()
    if item_similarities is None:
        item_similarities = get_item_similarities(tfidf_matrix)

    user_vector = vectorizer.transform([user_title.lower()])
    content_recommendations = financialBookData.iloc[get_content_based_scores(user_vector, tfidf_matrix).argsort()[-k:]][['title']].values.ravel()  # Top k content-based recommendations
    collaborative_recommendations = get_item_sim_recommendations(user_title, item_similarities)  # Top k collaborative recommendations (if book found)

    merged_recommendations = get_unique_recommendations(user_title, content_recommendations, collaborative_recommendations)
    
    return merged_recommendations

def get_unique_recommendations(user_title, content_recommendations, collaborative_recommendations):
    merged_recommendations = []

    if collaborative_recommendations is None:
        merged_recommendations = content_recommendations  # Use only content-based if collaborative recommendations are not available
    else:
        merged_recommendations = list(content_recommendations[:4]) + list(collaborative_recommendations[:4])

    # Remove duplicates
    unique_recommendations = []
    encountered_titles = set()
    for title in merged_recommendations:
        if title.lower() not in encountered_titles:
            unique_recommendations.append(title)
            encountered_titles.add(title.lower())

    # Filter out duplicate results from the user's search 
    merged_recommendations = [title for title in unique_recommendations if title.lower() != user_title.lower()]

    return merged_recommendations

# Example ground truth data for evaluation
ground_truth_recommendations = ["Global Financial Accounting and Reporting", "Contemporary Issues in Accounting"]

# Example usage:
user_title = "Intermediate Accounting IFRS"
tfidf_matrix, vectorizer = get_tfidf_matrix()
item_similarities = get_item_similarities(tfidf_matrix)

predicted_recommendations = get_hybrid_recommendations(user_title, k=5, tfidf_matrix=tfidf_matrix, item_similarities=item_similarities, vectorizer=vectorizer)
print(f"Predicted Recommendations: {predicted_recommendations}")

# Calculate RMSE (assuming binary relevance: 1 if recommended, 0 if not)
def calculate_rmse(predicted, ground_truth):
    y_true = [1 if title in ground_truth else 0 for title in predicted]
    y_pred = [1] * len(predicted)  # Since we consider all recommendations as positive
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

rmse = calculate_rmse(predicted_recommendations, ground_truth_recommendations)
print(f"RMSE: {rmse}")
