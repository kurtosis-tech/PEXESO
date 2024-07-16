import os
import pandas as pd
import numpy as np
import pickle
import glob
import faiss
from tqdm import tqdm 

def load_series_and_vector_array(csv_path, vector_array_path, table_name, column_index):
    df = pd.read_csv(csv_path)
    if not os.path.exists(vector_array_path):
        print("Error: npy file does not exist")
    np_data = np.load(vector_array_path)
    series = df.iloc[:, column_index]
    vector_array = np_data[column_index, :]
    return series, vector_array

def paths_from_dataset_path(dataset_path):
    faiss_index_path = os.path.join(dataset_path, "faiss_index.bin")
    index_list_path = os.path.join(dataset_path, "index_dict.pkl")
    csv_path = os.path.join(dataset_path, "csv")
    vector_array_path = os.path.join(dataset_path, "npy")
    
    return faiss_index_path, index_list_path, csv_path, vector_array_path

def table_paths_from_dataset_path(dataset_path, table_name):
    faiss_index_path = os.path.join(dataset_path, "faiss_index.bin")
    index_list_path = os.path.join(dataset_path, "index_dict.pkl")
    csv_path = os.path.join(dataset_path, "csv", table_name + ".csv")
    vector_array_path = os.path.join(dataset_path, "npy", table_name + ".npy")
    
    return csv_path, vector_array_path

def culc_feature_vector(vector_array):
    return (vector_array[0] + np.mean(vector_array[1:], axis=0)).astype(np.float32).reshape(1,-1)

def update_faiss_index(dataset_path, faiss_index, index_list, table_name, column_index):
    csv_path, vector_array_path = table_paths_from_dataset_path(dataset_path, table_name)

    table_index_tuple= (table_name, column_index)
    series, vector_array = load_series_and_vector_array(csv_path, vector_array_path, table_name, column_index)

    feature_vector = culc_feature_vector(vector_array)
    
    faiss_index.add(feature_vector)
    index_list.append(table_index_tuple)

    return faiss_index, index_list

def search_relative_column_with_table_index(dataset_path, faiss_index, index_list, table_name, column_index, threshold=0.1, candidates = 20):
    csv_path, vector_array_path = table_paths_from_dataset_path(dataset_path, table_name)
    series, vector_array = load_series_and_vector_array(csv_path, vector_array_path, table_name, column_index)
    feature_vector = culc_feature_vector(vector_array)
    output = []
    distances, indice = faiss_index.search(feature_vector, candidates)
    for distance, index in zip(distances[0], indice[0]):
        if distance < threshold:
            table_index_tuple = index_list[index]
            if table_index_tuple == (table_name, column_index):
                continue
            output.append((table_index_tuple, distance))
    return output

# 数字を含むかどうかをチェックする関数
def contains_digit(value):
    return any(char.isdigit() for char in str(value))

# 文字列が短すぎないかをチェックする関数
def is_too_short(value, min_length=3):
    return len(str(value)) < min_length

def check_invalid_series(series):
    series = series.dropna()
    if (not any(series.apply(contains_digit)) and not any(series.apply(is_too_short))):
        return True
    unique_values = set(series)
    if len(unique_values) < 3:
        return True
    return False

def init_faiss_index(dataset_path, vector_dim = 300):
    faiss_index_path, index_list_path, csv_path, vector_array_path = paths_from_dataset_path(dataset_path)
    
    faiss_index = faiss.IndexFlatL2(vector_dim) 
    index_list = []

    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    for csv_file in tqdm(csv_files):
        csv_basename= os.path.basename(csv_file)
        table_name = os.path.splitext(csv_basename)[0]
        
        df = pd.read_csv(csv_file)
        for column_index, (column_name, series) in enumerate(df.items()):
            if check_invalid_series(series):
                continue
            
            update_faiss_index(dataset_path, faiss_index, index_list, table_name, column_index)
    
    faiss.write_index(faiss_index, faiss_index_path)
    with open(index_list_path, 'wb') as f:
        pickle.dump(index_list, f)
    return 

def search_relative_columns_for_all(dataset_path, threshold=0.1):

    faiss_index_path, index_list_path, csv_path, vector_array_path = paths_from_dataset_path(dataset_path)

    # FAISSインデックスの読み込み
    faiss_index = faiss.read_index(faiss_index_path)
    
    # index_listの読み込み
    with open(index_list_path, 'rb') as f:
        index_list = pickle.load(f)

    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))

    output = []

    for csv_file in tqdm(csv_files):
        csv_basename= os.path.basename(csv_file)
        table_name = os.path.splitext(csv_basename)[0]
        
        df = pd.read_csv(csv_file)
        for column_index, (column_name, series) in enumerate(df.items()):
            if check_invalid_series(series):
                continue
            relative_columns = search_relative_column_with_table_index(dataset_path, faiss_index, index_list, table_name, column_index, threshold=threshold, candidates = 20)
            for table_index_tuple, distance in relative_columns:
                output.append(((table_name, column_index), table_index_tuple, distance))
    
    return output

