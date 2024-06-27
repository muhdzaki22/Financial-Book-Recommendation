import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the CSV file
file_path = r"C:\Users\ROGSTRIX\Documents\FYP\Combine\semuadatacombined_reviews.csv"
data = pd.read_csv(file_path)

# Shuffle the data
data = shuffle(data, random_state=42)

# Pivot the data to create a user-item matrix
user_item_matrix = data.pivot_table(index='username', columns='trailID', values='star_rating').fillna(0)

# Split the data into training and testing sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# Apply Truncated SVD
n_components = min(train_data.shape) - 1  # Number of components for SVD
svd = TruncatedSVD(n_components=n_components, random_state=42)
svd.fit(train_data)

# Transform both train and test sets
train_data_transformed = svd.transform(train_data)
test_data_transformed = svd.transform(test_data)

# Inverse transform to get back to the user-item matrix form
train_data_reconstructed = svd.inverse_transform(train_data_transformed)
test_data_reconstructed = svd.inverse_transform(test_data_transformed)

# Calculate the accuracy (RMSE and MAE) for both training and testing sets
train_rmse = np.sqrt(mean_squared_error(train_data.values, train_data_reconstructed))
test_rmse = np.sqrt(mean_squared_error(test_data.values, test_data_reconstructed))
train_mae = mean_absolute_error(train_data.values, train_data_reconstructed)
test_mae = mean_absolute_error(test_data.values, test_data_reconstructed)

print(f"Training RMSE: {train_rmse}")
print(f"Testing RMSE: {test_rmse}")
print(f"Training MAE: {train_mae}")
print(f"Testing MAE: {test_mae}")

# Convert reconstructed matrices to DataFrames for better comparison
train_data_reconstructed_df = pd.DataFrame(train_data_reconstructed, index=train_data.index, columns=train_data.columns)
test_data_reconstructed_df = pd.DataFrame(test_data_reconstructed, index=test_data.index, columns=test_data.columns)

# Display a portion of the original and reconstructed matrices
print("\nOriginal Training Matrix (portion):")
print(train_data.head())

print("\nReconstructed Training Matrix (portion):")
print(train_data_reconstructed_df.head())

print("\nOriginal Testing Matrix (portion):")
print(test_data.head())

print("\nReconstructed Testing Matrix (portion):")
print(test_data_reconstructed_df.head())
