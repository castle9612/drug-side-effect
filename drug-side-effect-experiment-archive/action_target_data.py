import pandas as pd
import numpy as np
from tqdm import tqdm

# Load the CSV file into a DataFrame
file_path = 'target_action_data_delete_0.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Assuming the columns follow the specified format
data2 = data.iloc[:, 1:]  # Select relevant columns for data2

# Calculate data2_1, data2_2, and data2_3
data2_1 = ((data2 * data2) + data2) / 2
data2_2 = ((data2 * data2) - data2) / 2
data2_3 = 1 - (data2 * data2)
print(data2)
print(data2_1)
print(data2_2)
print(data2_3)

# Concatenate data1, data2_1, data2_2, and data2_3 to form one_hot_vectors
one_hot_vectors = pd.concat([data["Unnamed: 0"], data2_1, data2_2, data2_3], axis=1)

# Define a function to calculate Tanimoto similarity
def tanimoto_similarity(vector1, vector2):
    intersection = np.sum(np.logical_and(vector1, vector2))
    union = np.sum(np.logical_or(vector1, vector2))
    
    if union == 0:
        return 0.0  # Handle the case where both vectors are all zeros
    
    return intersection / union

# Define a function to create the similarity matrix
def create_similarity_matrix(data):
    num_samples, num_features = data.shape
    similarity_matrix = np.zeros((num_samples, num_samples))
    
    for i in tqdm(range(num_samples),desc="Calculating Similarity"):
        for j in range(num_samples):
            similarity_matrix[i, j] = tanimoto_similarity(data.iloc[i, 1:], data.iloc[j, 1:])
    print(similarity_matrix)
    return similarity_matrix

# Calculate Tanimoto similarity using the provided one_hot_vectors
print("start")
similarity_matrix = create_similarity_matrix(one_hot_vectors)
print(similarity_matrix)
similarity_matrix=pd.DataFrame(similarity_matrix,index=one_hot_vectors["Unnamed: 0"], columns=one_hot_vectors["Unnamed: 0"])
similarity_matrix.to_csv('simliarity_action.csv')