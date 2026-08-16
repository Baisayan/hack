#=====================
# download_dataset.py
"""
This script is used to download the Xtest dataset from Hugging Face.
"""
#=====================

# Imports
from datasets import load_dataset

# Load Xtest dataset from hugging face
dataset = load_dataset("walledai/XSTest")
print(dataset.keys()) # contains only the test split

# Inspecting the first sample
split_name = list(dataset.keys())[0] #test
print(f"INFO: Using split: {split_name}")

data = dataset[split_name]
print(f"INFO: Column Names: {data.column_names}")
print(f"INFO: First Sample: {data[0]}")

# Converting the dataset to a dataframe to save locally
df = data.to_pandas()

# adding an ID column to link the activation vectors to the prompts to retrieve the labels
df.insert(0, "id", [f"xstest_{i:05d}" for i in range(len(df))])

# saving the file locally to be used in the server
df.to_csv("XSTest.csv", index=False)
print("INFO: Dataset saved as XSTest.csv")




