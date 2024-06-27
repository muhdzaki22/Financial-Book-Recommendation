from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define the descriptions
input_description = "The easy way to get your head around company finance"

recommended_descriptions = [
    "UK bookkeeping and accounting basics for the rest of us Unless you're one of those rare numbers people, the thought of accounting and bookkeeping probably make your head spin. While these pragmatic and confusing practices may not be fun for the rest of us, mastering them is absolutely essential in order to run and maintain a successful business. Thankfully, Bookkeeping & Accounting All-in-One For Dummies, UK Edition, is here to take the intimidation out of crunching numbers and offers easy-to-follow, step-by-step instruction on keeping your business' finances in order with information specific to a business in the United Kingdom. Written in plain English and packed with loads of helpful instruction, this approachable and all-encompassing guide arms you with everything you need to get up and running on all the latest accounting practices and bookkeeping software. Inside, you'll find out how to prepare financial statements, balance your books, keep the tax inspector off your back, and so much more.",
    "Accounting is truly the language of business. Success or failure is measured in dollars, but in order to make good decisions, you need to understand how finances drive business realities and become fluent in the essential elements of the accounting process. ACCOUNTING DEMYSTIFIED tells you all you need to know about the numbers that drive business. The book uses examples of typical business situations to demonstrate basic financial concepts, including: * The accounting process * Financial statements * Making entries * Accounts payable and accounts receivable * Cashflow statements * Fixed and intangible assets * Inventory * Liabilities * Adjusting and closing entries * Prepaid expenses *Preparing a bank reconciliation * Accounting information systems * Stockholders equity * Ratio analysis ACCOUNTING DEMYSTIFIED transforms a complex and potentially intimidating subject into something anyone can easily comprehend. This useful resource helps you understand the basics of accounting and gives you access to an essential part of any business equation. For new students of accounting, entry-level accounting professionals, and business professionals whose own work relates directly to the numbers on the ledger, a basic understanding of core accounting functions and documents is critical. Accounting Demystified provides a simple and straightforward description of universal elements of the accounting process, plus accessible tutorials in creating, interpreting, and using financial statements. Haber's clear language will let readers: * understand accounting basics * find errors quickly * prepare accurate financial statements * analyze financial documents * determine the financial health of a business * prepare a financial prospectus for potential investors and lenders From the classroom to the back room to the board room, Accounting Demystified serves as a valuable primer on the basics of accounting and the purposes they serve.",
    "This best-selling dictionary includes more than 3,800 entries covering all aspects of accounting, including financial accounting, financial reporting, management accounting, taxation, auditing, corporate finance, and accounting bodies and institutions. Its international coverage includes important terms from UK, US, Australia, India, and Asia-Pacific.",
    "The easy way to master a managerial accounting course Are you enrolled in a managerial accounting class and finding yourself struggling? Fear not! Managerial Accounting For Dummies is the go-to study guide to help you easily master the concepts of this challenging course. You'll discover the basic concepts, terminology, and methods to identify, measure, analyze, interpret, and communicate information in the pursuit of an organization's goals. Tracking to a typical managerial accounting course and packed with easy-to-understand explanations and real-life examples, Managerial Accounting For Dummies explores cost behavior, cost analysis, profit planning and control measures, accounting for decentralized operations, capital budgeting decisions, ethical challenges in managerial accounting, and much more."

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
