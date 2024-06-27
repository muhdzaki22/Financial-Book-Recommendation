import pandas as pd
import random

# Load the Excel file into a DataFrame
df = pd.read_excel('C:/Users/User/OneDrive/Desktop/file/study notes/sem 6/CSP650 FYP/website V2/rmvdplct.xlsx')

# Get the total number of rows in the DataFrame
total_rows = len(df)

# Define the number of rows to remove
rows_to_remove = 9000

# Generate a list of random row indices to remove
indices_to_remove = random.sample(range(total_rows), rows_to_remove)

# Remove the selected rows
df = df.drop(indices_to_remove)

# Save the modified DataFrame back to the Excel file
df.to_excel('removedata.xlsx', index=False)
