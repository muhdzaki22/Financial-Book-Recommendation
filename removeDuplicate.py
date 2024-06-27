import pandas as pd

# Read Excel data into a pandas DataFrame
df = pd.read_excel('rmvmsg.xlsx')

# Count missing values in each column
missing_values_count = df.isna().sum()

# Print the number of missing values in each column
print("Number of missing values in each column:")
print(missing_values_count)

# Remove duplicated rows
df = df.drop_duplicates()

# Print the updated shape of the DataFrame after removing duplicated rows
print("Shape of DataFrame after removing duplicated rows:", df.shape)

# Save the DataFrame as an Excel file
df.to_excel('rmvdplct.xlsx', index=False)
