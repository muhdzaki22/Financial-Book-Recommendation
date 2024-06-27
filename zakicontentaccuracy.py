from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define the descriptions
input_description = "A good business model should describe how an organization creates and delivers value, meaning that financial modelling is a vital tool for business strategy, allowing hypotheses and scenarios to be translated into numbers. It enables a company to experiment with different ideas and scenarios in a safe, low-risk environment, to consider what it is aiming to achieve, and to prioritize accordingly."

recommended_descriptions = [
    "A Best Business Book of 2017 -- The Financial Times",
    "From the field's leading authority, the most authoritative and comprehensive advanced-level textbook on asset pricing  Financial Decisions and Markets is a graduate-level textbook that provides a broad overview of the field of asset pricing. John Campbell, one of the field's most respected authorities, introduces students to leading theories of portfolio choice, their implications for asset prices, and empirical patterns of risk and return in financial markets. Campbell emphasizes the interplay of theory and evidence, as theorists respond to empirical puzzles by developing models with new testable implications. Increasingly these models make predictions not only about asset prices but also about investors' financial positions, and they often draw on insights from behavioral economics. After a careful introduction to single-period models, Campbell develops multiperiod models with time-varying discount rates, reviews the leading approaches to consumption-based asset pricing, and integrates the study of equities and fixed-income securities.",
    "A fully up-to-date, cutting-edge guide to the measurement and management of liquidity risk Written for front and middle office risk management and quantitative practitioners, this book provides the ground-level knowledge, tools, and techniques for effective liquidity risk management. Highly practical, though thoroughly grounded in theory, the book begins with the basics of liquidity risks and, using examples pulled from the recent financial crisis, how they manifest themselves in financial institutions. The book then goes on to look at tools which can be used to measure liquidity risk, discussing risk monitoring and the different models used, notably financial variables models, credit variables models, and behavioural variables models, and then at managing these risks. As well as looking at the tools necessary for effective measurement and management, the book also looks at and discusses current regulation and the implication of new Basel regulations on management procedures and tools.",
    "Portfolio risk forecasting has been and continues to be an active research field for both academics and practitioners. Almost all institutional investment management firms use quantitative models for their portfolio forecasting, and researchers have explored models' econometric foundations, relative performance, and implications for capital market behavior and asset pricing equilibrium. Portfolio Risk Analysis provides an insightful and thorough overview of financial risk modeling, with an emphasis on practical applications, empirical reality, and historical perspective. Beginning with mean-variance analysis and the capital asset pricing model, the authors give a comprehensive and detailed account of factor models, which are the key to successful risk analysis in every economic climate. Topics range from the relative merits of fundamental, statistical, and macroeconomic models, to GARCH and other time series models, to the properties of the VIX volatility index. The book covers both mainstream and alternative asset classes, and includes in-depth treatments of model integration and evaluation."
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
