from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re

# Assuming bookData.xlsx is in the same directory as your Python file
financialBookData = pd.read_excel('removedata.xlsx', 'Sheet1')

# Function for data preprocessing (already defined)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove non-alphanumeric characters
    words = [word for word in text.split() if word not in english_stopwords]  # Remove stopwords
    return " ".join(words)

# Preprocess descriptions (already defined)
english_stopwords = stopwords.words('english')
financialBookData['preprocessed_description'] = financialBookData['description'].apply(preprocess_text)

# Functions for content-based and item-to-item recommendations (already defined)
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

def get_hybrid_recommendations(user_title, alpha=0.6, beta=0.4, k=5, tfidf_matrix=None, item_similarities=None, vectorizer=None):
    if tfidf_matrix is None:
        tfidf_matrix, _ = get_tfidf_matrix()
    if item_similarities is None:
        item_similarities = get_item_similarities(tfidf_matrix)

    user_vector = vectorizer.transform([user_title.lower()])
    content_recommendations = financialBookData.iloc[get_content_based_scores(user_vector, tfidf_matrix).argsort()[-k:]][['title']].values.ravel()  # Top k content-based recommendations
    collaborative_recommendations = get_item_sim_recommendations(user_title, item_similarities)  # Top k collaborative recommendations (if book found)

    if collaborative_recommendations is None:
        return content_recommendations  # Use only content-based if book not found
    else:
        merged_recommendations = list(content_recommendations[:4]) + \
                                 list(collaborative_recommendations[:4])
                                 
    # Filter out duplicate results from the user's search 
    merged_recommendations = [title for title in merged_recommendations if title.lower() != user_title.lower()]

    return merged_recommendations

def get_item_sim_recommendations(user_title, item_similarities, k=5):
    try:
        # Using get_loc() for potentially duplicate titles
        user_index = financialBookData['title'].eq(user_title).idxmax()
        similar_items = item_similarities[user_index]
        similar_items_sorted = similar_items.argsort()[-k:]  # Sort for top k similar items
        return financialBookData.loc[similar_items_sorted[1:]]['title'].tolist()

    except KeyError:
        print(f"Book '{user_title}' not found in data. Returning empty recommendations.")
        return []

# Flask app setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Default route
@app.route("/")
def default():
    return redirect(url_for('login'))  # Redirect to login page by default

# Register route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")  # Render registration form on GET request
    else:
        username = request.form["username"]
        password = request.form["password"]
        # Check if the username already exists
        if username in session:
            return "User already exists!"
        else:
            # Hash the password before storing it
            hashed_password = generate_password_hash(password)
            session[username] = hashed_password
            return redirect(url_for('login'))

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")  # Render login form on GET request
    else:
        username = request.form["username"]
        password = request.form["password"]
        # Check if the user exists and the password is correct
        if username in session and check_password_hash(session[username], password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('search'))  # Redirect to search page on successful login
        else:
            return "Invalid username or password!"

# Logout route
@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# Search route
@app.route("/search", methods=["GET", "POST"])
def search():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))

    if request.method == "GET":
        return render_template("search.html")  # Render search form on GET request
    elif request.method == "POST":
        search_query = request.form["search_term"]
        tfidf_matrix, vectorizer = get_tfidf_matrix()
        recommendations = get_hybrid_recommendations(search_query, tfidf_matrix=tfidf_matrix, vectorizer=vectorizer)
        return render_template("results.html", search_query=search_query, recommendations=recommendations)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
