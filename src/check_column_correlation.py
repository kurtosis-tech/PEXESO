import faiss
import numpy as np

def check_column_correlation(left_series, left_vector_array, right_series, right_vector_array, distance_threshold=0.1, relation_ratio_threshold=0.5, feature_dim=300):
    left_column_name_vector = left_vector_array[0]
    left_column_value_vector_array = left_vector_array[1:]
    right_column_name_vector = right_vector_array[0]
    right_column_value_vector_array = right_vector_array[1:]

    faiss_index = faiss.IndexFlatL2(feature_dim)  # Faissインデックスの作成

    for i in range(left_column_value_vector_array.shape[0]):
        vector = left_column_value_vector_array[i].reshape(1, feature_dim).astype(np.float32)
        if np.isnan(vector).any():
            continue  # NaNを含むベクトルをスキップ

        faiss_index.add(vector)  # ベクトルBをインデックスに追加

    right_column_value_vector_array = right_column_value_vector_array.reshape(-1, feature_dim).astype(np.float32)
    distances, indices = faiss_index.search(right_column_value_vector_array, 1)

    relation_count = 0
    for distance in distances:
        if distance < distance_threshold:
            relation_count+=1
    
    relation_ratio = relation_count/ distances.shape[0]
    
    if relation_ratio < relation_ratio_threshold:
        return True, relation_ratio
    else:
        return False, relation_ratio