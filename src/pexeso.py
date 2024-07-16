import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from search_relative_columns_from_index import load_series_and_vector_array, search_relative_columns_for_all, table_paths_from_dataset_path, init_faiss_index 
from check_column_correlation import check_column_correlation 

dataset_path = "../data/raw/test"
threshold = 0.1

init_faiss_index(dataset_path, vector_dim = 300)

relative_columns = search_relative_columns_for_all(dataset_path, threshold=threshold)

joinable_columns = []

for ((left_table_name, left_index),(right_table_name, right_index), distance) in tqdm(relative_columns):
    left_csv_path, left_vector_array_path = table_paths_from_dataset_path(dataset_path, left_table_name)
    left_series, left_vector_array = load_series_and_vector_array(left_csv_path, left_vector_array_path, left_table_name, left_index)
    right_csv_path, right_vector_array_path = table_paths_from_dataset_path(dataset_path, right_table_name)
    right_series, right_vector_array = load_series_and_vector_array(right_csv_path, right_vector_array_path, right_table_name, right_index)

    flag, relation_ratio = check_column_correlation(left_series, left_vector_array, right_series, right_vector_array, distance_threshold=0.1, relation_ratio_threshold=0.5)
    if flag == True:
        
        left_name = left_series.name

        joinable_columns.append({
            'LeftTable': left_table_name,
            'LeftColumnIndex': left_index,
            'LeftColumnName': left_series.name,
            'RightTable': right_table_name,
            'RightColumnIndex': right_index,
            'RightColumnName': right_series.name,
            'RoughScore': distance,
            'RelationRatio': relation_ratio,
        })
        
output_file = 'joinable_columns.csv'
df_joinable_columns = pd.DataFrame(joinable_columns)
df_joinable_columns.to_csv(output_file, index=False)

print(f"Joinable columns information has been saved to {output_file}")
