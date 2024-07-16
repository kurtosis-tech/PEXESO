import faiss
import numpy as np

def check_column_correlation(left_series, left_vector_array, right_series, right_vector_array, distance_threshold=0.1, relation_ratio_threshold=0.5, feature_dim=300):
    """
    Checks the correlation between two columns using FAISS for vector comparison.
    
    Args:
        left_series (pd.Series): The series from the left table.
        left_vector_array (np.ndarray): The vector array corresponding to the left series.
        right_series (pd.Series): The series from the right table.
        right_vector_array (np.ndarray): The vector array corresponding to the right series.
        distance_threshold (float): The maximum distance to consider vectors as related. Default is 0.1.
        relation_ratio_threshold (float): The minimum ratio of related vectors to consider columns as correlated. Default is 0.5.
        feature_dim (int): The dimensionality of the feature vectors. Default is 300.
        
    Returns:
        bool: True if the columns are considered correlated, False otherwise.
        float: The calculated relation ratio.
    """
    left_column_name_vector = left_vector_array[0]
    left_column_value_vector_array = left_vector_array[1:]
    right_column_name_vector = right_vector_array[0]
    right_column_value_vector_array = right_vector_array[1:]

    # Create a FAISS index
    faiss_index = faiss.IndexFlatL2(feature_dim)

    # Add vectors from the left column to the FAISS index
    for i in range(left_column_value_vector_array.shape[0]):
        vector = left_column_value_vector_array[i].reshape(1, feature_dim).astype(np.float32)
        if np.isnan(vector).any():
            continue  # Skip vectors containing NaN
        faiss_index.add(vector)

    # Reshape and convert right column vectors for FAISS search
    right_column_value_vector_array = right_column_value_vector_array.reshape(-1, feature_dim).astype(np.float32)
    distances, indices = faiss_index.search(right_column_value_vector_array, 1)

    # Count the number of related vectors based on the distance threshold
    relation_count = 0
    for distance in distances:
        if distance < distance_threshold:
            relation_count += 1

    relation_ratio = relation_count / distances.shape[0]

    # Determine if the relation ratio meets the threshold for correlation
    if relation_ratio >= relation_ratio_threshold:
        return True, relation_ratio
    else:
        return False, relation_ratio
