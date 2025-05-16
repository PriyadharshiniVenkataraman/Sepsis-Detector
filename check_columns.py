import pandas as pd

# Load the dataset (make sure Dataset.csv is in the same folder as this script)
data = pd.read_csv('Dataset.csv')

# Display all column names
print("Available columns in Dataset.csv:")
print(data.columns.tolist())
