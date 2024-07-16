import os
import pandas as pd
import numpy as np
import pickle
import glob
import faiss
from tqdm import tqdm
from utils import (
    check_invalid_series,
    load_series_and_vector_array,
    paths_from_dataset_path,
    table_paths_from_dataset_path,
    culc_feature_vector
)

def update_faiss_index(dataset_path: str, faiss_index, index_list: list, table_name: str, column_index: int):
    """
    Updates the FAISS index with a new feature vector from the specified column in the table.
    
    Args:
        dataset_path (str): The base path to the dataset directory.
        faiss_index: The FAISS index object.
        index_list (list): A list to store the index information.
        table_name (str): The name of the table.
        column_index (int): The index of the column in the table.
        
    Returns:
        faiss_index: The updated FAISS index.
        index_list (list): The updated index list.
    """
    csv_path, vector_array_path = table_paths_from_dataset_path(dataset_path, table_name)
    series, vector_array = load_series_and_vector_array(csv_path, vector_array_path, column_index)
    feature_vector = culc_feature_vector(vector_array)
    faiss_index.add(feature_vector)
    index_list.append((table_name, column_index))
    return faiss_index, index_list

def search_relative_column_with_table_index(dataset_path: str, faiss_index, index_list: list, table_name: str, column_index: int, threshold=0.1, candidates=20):
    """
    Searches for columns related to the specified column using the FAISS index.
    
    Args:
        dataset_path (str): The base path to the dataset directory.
        faiss_index: The FAISS index object.
        index_list (list): A list of index information.
        table_name (str): The name of the table.
        column_index (int): The index of the column in the table.
        threshold (float): The distance threshold for determining related columns. Default is 0.1.
        candidates (int): The number of candidate columns to consider. Default is 20.
        
    Returns:
        list: A list of tuples containing related columns and their distances.
    """
    csv_path, vector_array_path = table_paths_from_dataset_path(dataset_path, table_name)
    series, vector_array = load_series_and_vector_array(csv_path, vector_array_path, column_index)
    feature_vector = culc_feature_vector(vector_array)
    distances, indices = faiss_index.search(feature_vector, candidates)
    output = []
    for distance, index in zip(distances[0], indices[0]):
        if distance < threshold:
            table_index_tuple = index_list[index]
            if table_index_tuple == (table_name, column_index):
                continue
            output.append((table_index_tuple, distance))
    return output

def init_faiss_index(dataset_path: str, vector_dim=300):
    """
    Initializes the FAISS index with feature vectors from all columns in all tables.
    
    Args:
        dataset_path (str): The base path to the dataset directory.
        vector_dim (int): The dimensionality of the feature vectors. Default is 300.
        
    Returns:
        None
    """
    faiss_index_path, index_list_path, csv_path, vector_array_path = paths_from_dataset_path(dataset_path)
    faiss_index = faiss.IndexFlatL2(vector_dim)
    index_list = []
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    for csv_file in tqdm(csv_files):
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
        df = pd.read_csv(csv_file)
        for column_index, (column_name, series) in enumerate(df.items()):
            if check_invalid_series(series):
                continue
            update_faiss_index(dataset_path, faiss_index, index_list, table_name, column_index)
    faiss.write_index(faiss_index, faiss_index_path)
    with open(index_list_path, 'wb') as f:
        pickle.dump(index_list, f)
    return

def search_candidate_columns(dataset_path: str, threshold=0.1):
    """
    Searches for candidate columns that are related to each column in the dataset.
    
    Args:
        dataset_path (str): The base path to the dataset directory.
        threshold (float): The distance threshold for determining related columns. Default is 0.1.
        
    Returns:
        list: A list of tuples containing pairs of related columns and their distances.
    """
    faiss_index_path, index_list_path, csv_path, vector_array_path = paths_from_dataset_path(dataset_path)
    faiss_index = faiss.read_index(faiss_index_path)
    with open(index_list_path, 'rb') as f:
        index_list = pickle.load(f)
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    output = []
    for csv_file in tqdm(csv_files):
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
        df = pd.read_csv(csv_file)
        for column_index, (column_name, series) in enumerate(df.items()):
            if check_invalid_series(series):
                continue
            relative_columns = search_relative_column_with_table_index(dataset_path, faiss_index, index_list, table_name, column_index, threshold=threshold, candidates=20)
            for table_index_tuple, distance in relative_columns:
                output.append(((table_name, column_index), table_index_tuple, distance))
    return output
