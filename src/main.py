import numpy as np
import pandas as pd
from itertools import combinations
from search_relative_columns import search_relative_columns
import glob
import os
import random
from tqdm import tqdm

# CSVファイルパスの設定
csv_path = "../data/raw/baseball-databank/csv"
csv_files = glob.glob(os.path.join(csv_path, "*.csv"))

# ファイルをソートして前半の500ファイルだけを対象とする
csv_files = sorted(csv_files)[:500]

def csv_file_name_to_npy_file_name(csv_file):
    npy_file = csv_file.replace(".csv", ".npy").replace("/csv/", "/npy/")
    return npy_file

# 2つのペアの全ての組み合わせを計算
combinations_list = list(combinations(csv_files, 2))
random.shuffle(combinations_list)

joinable_columns = []

# 結果を表示
for combination in tqdm(combinations_list, desc="Processing combinations"):
    
    csv_file_name_A = combination[0]
    npy_file_name_A = csv_file_name_to_npy_file_name(csv_file_name_A)

    csv_A = pd.read_csv(csv_file_name_A)
    npy_A = np.load(npy_file_name_A, allow_pickle=True)

    csv_file_name_B = combination[1]
    npy_file_name_B = csv_file_name_to_npy_file_name(csv_file_name_B)
    csv_B = pd.read_csv(csv_file_name_B)
    npy_B = np.load(npy_file_name_B, allow_pickle=True)

    # 関連する列を検索
    relative_columns, distance_for_relative_columns = search_relative_columns(csv_A, npy_A, csv_B, npy_B)
    if len(relative_columns) != 0:
        for (index_A, index_B), distance in zip(relative_columns, distance_for_relative_columns):
            column_name_A = csv_A.columns[index_A]
            column_name_B = csv_B.columns[index_B]
            joinable_columns.append({
                'Table1': csv_file_name_A,
                'Column1': column_name_A,
                'Table2': csv_file_name_B,
                'Column2': column_name_B,
                'Score': distance,
            })

# 結果をCSVファイルに保存
output_file = 'joinable_columns_pexeso_mean_baseball.csv'
df_joinable_columns = pd.DataFrame(joinable_columns)
df_joinable_columns.to_csv(output_file, index=False)

print(f"Joinable columns information has been saved to {output_file}")
