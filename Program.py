from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re

# Load the book data
financialBookData = pd.read_excel(r'C:\Users\ADMIN\Desktop\zaki tm\initiative\book recommendation\Financial-Book-Recommendation\removedata.xlsx', 'Sheet1')

# Data preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove non-alphanumeric characters
    words = [word for word in text.split() if word not in english_stopwords]  # Remove stopwords
    return " ".join(words)

english_stopwords = stopwords.words('english')
financialBookData['preprocessed_description'] = financialBookData['description'].apply(preprocess_text)

# Generate TF-IDF matrix
def get_tfidf_matrix():
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(financialBookData['preprocessed_description'])
    return tfidf_matrix, vectorizer

# Compute recommendations using TF-IDF
def get_tfidf_recommendations(user_query, k=8):  # Changed k to 8
    tfidf_matrix, vectorizer = get_tfidf_matrix()
    user_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-k:][::-1]  # Top k similar items
    return financialBookData.iloc[top_indices][['title']].values.ravel()

# Flask app setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Default route
@app.route("/")
def default():
    return redirect(url_for('login'))

# Register route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    else:
        username = request.form["username"]
        password = request.form["password"]
        if username in session:
            return "User already exists!"
        else:
            session[username] = generate_password_hash(password)
            return redirect(url_for('login'))

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    else:
        username = request.form["username"]
        password = request.form["password"]
        if username in session and check_password_hash(session[username], password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('search'))
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
        return render_template("search.html")
    elif request.method == "POST":
        search_query = request.form["search_term"]
        recommendations = get_tfidf_recommendations(search_query, k=8)  # Pass k=8 explicitly
        return render_template("results.html", search_query=search_query, recommendations=recommendations)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
