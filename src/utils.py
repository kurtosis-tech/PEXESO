import os
import pandas as pd
import numpy as np

def contains_digit(series: pd.Series) -> pd.Series:
    """
    Checks if the elements of a pandas Series contain any digits.
    
    Args:
        series (pd.Series): The input series to check.
        
    Returns:
        pd.Series: A series of booleans indicating whether each element contains a digit.
    """
    return series.apply(lambda x: any(char.isdigit() for char in str(x)))

def is_too_short(series: pd.Series, min_length=3) -> pd.Series:
    """
    Checks if the elements of a pandas Series are too short.
    
    Args:
        series (pd.Series): The input series to check.
        min_length (int): The minimum length threshold. Default is 3.
        
    Returns:
        pd.Series: A series of booleans indicating whether each element is too short.
    """
    return series.apply(lambda x: len(str(x)) < min_length)

def check_invalid_series(series: pd.Series) -> bool:
    """
    Checks if a pandas Series is invalid based on certain criteria: contains digits, is too short, or has less than 3 unique values.
    
    Args:
        series (pd.Series): The input series to check.
        
    Returns:
        bool: True if the series is invalid, False otherwise.
    """
    series = series.dropna()
    if any(contains_digit(series)) or any(is_too_short(series)):
        return True
    unique_values = set(series)
    return len(unique_values) < 3

def load_series_and_vector_array(csv_path: str, vector_array_path: str, column_index: int):
    """
    Loads a series from a CSV file and its corresponding vector array from an npy file.
    
    Args:
        csv_path (str): The path to the CSV file.
        vector_array_path (str): The path to the npy file containing the vector array.
        column_index (int): The index of the column to load from the CSV.
        
    Returns:
        pd.Series: The loaded series from the CSV.
        np.ndarray: The corresponding vector array from the npy file.
    """
    df = pd.read_csv(csv_path)
    if not os.path.exists(vector_array_path):
        raise FileNotFoundError("Error: npy file does not exist")
    vector_data = np.load(vector_array_path)
    series = df.iloc[:, column_index]
    vector_array = vector_data[column_index, :]
    return series, vector_array

def paths_from_dataset_path(dataset_path: str):
    """
    Generates paths for FAISS index, index list, CSV directory, and vector array directory based on the dataset path.
    
    Args:
        dataset_path (str): The base path to the dataset directory.
        
    Returns:
        str: Path to the FAISS index file.
        str: Path to the index list file.
        str: Path to the CSV directory.
        str: Path to the vector array directory.
    """
    faiss_index_path = os.path.join(dataset_path, "faiss_index.bin")
    index_list_path = os.path.join(dataset_path, "index_dict.pkl")
    csv_path = os.path.join(dataset_path, "csv")
    vector_array_path = os.path.join(dataset_path, "npy")
    return faiss_index_path, index_list_path, csv_path, vector_array_path

def table_paths_from_dataset_path(dataset_path: str, table_name: str):
    """
    Generates paths for the CSV file and vector array file for a specific table name based on the dataset path.
    
    Args:
        dataset_path (str): The base path to the dataset directory.
        table_name (str): The name of the table.
        
    Returns:
        str: Path to the CSV file for the specified table.
        str: Path to the vector array file for the specified table.
    """
    csv_path = os.path.join(dataset_path, "csv", table_name + ".csv")
    vector_array_path = os.path.join(dataset_path, "npy", table_name + ".npy")
    return csv_path, vector_array_path

def culc_feature_vector(vector_array: np.ndarray) -> np.ndarray:
    """
    Calculates a feature vector from a vector array.
    
    Args:
        vector_array (np.ndarray): The input vector array.
        
    Returns:
        np.ndarray: The calculated feature vector.
    """
    return (vector_array[0] + np.mean(vector_array[1:], axis=0)).astype(np.float32).reshape(1, -1)
