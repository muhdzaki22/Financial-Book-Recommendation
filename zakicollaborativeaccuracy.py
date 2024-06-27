from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define the descriptions
input_description = "A good business model should describe how an organization creates and delivers value, meaning that financial modelling is a vital tool for business strategy, allowing hypotheses and scenarios to be translated into numbers. It enables a company to experiment with different ideas and scenarios in a safe, low-risk environment, to consider what it is aiming to achieve, and to prioritize accordingly."

recommended_descriptions = [
    "The classic guide to warehouse operations-now fully revised and updated with the latest strategies, best practices, and case studies",
    "The updated edition of the bestselling, essential guide to real estate financial calculations",
    "A completely revised and updated edition of an investing classic to help readers make sense of investing today, full of solid information and advice for individual investors (The Washington Post). Today, anyone can be an informed investor, and once you learn to tune out the hype and focus on meaningful factors, you can beat the Street. The Motley Fool Investment Guide,  completely revised and updated with clear and witty explanations, deciphers all the current information--from evaluating individual stocks to creating a diverse investment portfolio. David and Tom Gardner have investing ideas for you, no matter how much time or money you have. This new edition of The Motley Fool Investment Guide is designed for today's investor, sophisticate and novice alike, with the latest information on: ",
    "FULLY REVISED AND UPDATED SECOND EDITIONSeasoned investor Andy Bell shows you how to plan your financial future in this updated edition of his bestselling guide to do-it-yourself investing.This book will show you how to build an investment portfolio using a range of low-cost, tax-efficient strategies. With expert guidance and industry insights suitable for both first-time investors and those who are more experienced, The DIY Investor will teach you the skills and strategies required to take control of your investments."
]

# Vectorize the descriptions using TF-IDF
vectorizer = TfidfVectorizer()
descriptions = [input_description] + recommended_descriptions
tfidf_matrix = vectorizer.fit_transform(descriptions)

# Calculate cosine similarity between input description and each recommended description
cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

similarity_percentages = cosine_similarities * 100

# Print the cosine similarity scores
print("Cosine Similarity Scores:")
for i, score in enumerate(similarity_percentages):
    print(f"Recommendation {i+1}: {score:.3f}")
