import pandas as pd

# Read Excel data into a pandas DataFrame
df = pd.read_excel('newData.xlsx')

# Count missing values in each column
missing_values_count = df.isna().sum()

# Print the number of missing values in each column
print("Number of missing values in each column:")
print(missing_values_count)

# Remove rows with missing data
df = df.dropna()

# Print the updated shape of the DataFrame after removing rows with missing data
print("Shape of DataFrame after removing rows with missing data:", df.shape)

# Save the DataFrame as an Excel file
df.to_excel('rmvmsg.xlsx', index=False)
