import pandas as pd
import fasttext
import numpy as np
import os
import glob
import time

model = fasttext.load_model('cc.en.300.bin')

def text_to_vector(text):
    words = text.split()
    word_vectors = [model.get_word_vector(word) for word in words]
    # print(f"Vector {word_vectors} for words {words}")
    if len(word_vectors) == 0:
        return np.zeros(300)
    return np.mean(word_vectors, axis=0)

def series_to_vec(series, column_name):
    output = [text_to_vector(column_name)] 
    for index, value in series.items():
        if is_number(value):
            value = str(value)
        output.append(text_to_vector(value))
    return np.array(output)

def is_number(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def preprocess_file(df, base_name, output_dir):
    npy_base_name = base_name.replace(".csv",".npy")
    output_path = os.path.join(output_dir, npy_base_name)
    
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping...")
        return
    
    vectors = []
    for column in df.columns:
        series = df[column]
        vector = series_to_vec(series, column)
        vectors.append(vector)
    
    vectors_array = np.array(vectors)

    np.save(output_path, vectors_array)
    
    return 

def main():
    base_path = "../data/raw/test"

    csv_path = os.path.join(base_path, "csv")
    npy_path = os.path.join(base_path, "npy")

    file_paths = glob.glob(os.path.join(csv_path,"*.csv"))

    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    table_count = 0
    column_count = 0
    for file_path in file_paths:
        base_name = os.path.basename(file_path)

        print(file_path)
        df = pd.read_csv(file_path)
        column_count += df.shape[1]
        preprocess_file(df, base_name, npy_path)
        table_count += 1
    
    print(f"{column_count} columns across {table_count} tables")

if __name__ == '__main__':
    main()
