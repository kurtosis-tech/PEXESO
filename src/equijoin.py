import pandas as pd
import os
from itertools import combinations
from tqdm import tqdm

# CSVファイルが保存されているディレクトリを指定
dataset_path = "../data/raw/test/"
csv_path = os.path.join(dataset_path, "csv")

# 一致率の閾値を設定（例：70%）
threshold = 0.5

# ディレクトリ内のCSVファイルを読み込んでデータフレームに変換
tables = {}
csv_files = sorted([f for f in os.listdir(csv_path) if f.endswith('.csv')])

print(csv_files)

for filename in csv_files:
    table_name = os.path.splitext(filename)[0]
    file_path = os.path.join(csv_path, filename)
    tables[table_name] = pd.read_csv(file_path)

# 数字を含むかどうかをチェックする関数
def contains_digit(value):
    return any(char.isdigit() for char in str(value))

# 文字列が短すぎないかをチェックする関数
def is_too_short(value, min_length=3):
    return len(str(value)) < min_length

# 全てのテーブルのペアの組み合わせを計算
combinations_list = list(combinations(tables.keys(), 2))

# Equijoinが可能な列を探索
joinable_columns = []

for table1, table2 in tqdm(combinations_list, desc="Processing combinations"):
    df1 = tables[table1]
    df2 = tables[table2]
    for column1 in df1.columns:
        for column2 in df2.columns:
            # 数字を含む値が存在する場合や文字列が短すぎる場合はスキップ
            if (not any(df1[column1].dropna().apply(contains_digit)) and
                not any(df2[column2].dropna().apply(contains_digit)) and
                not any(df1[column1].dropna().apply(is_too_short)) and
                not any(df2[column2].dropna().apply(is_too_short))):
                
                unique_values1 = set(df1[column1].dropna())
                unique_values2 = set(df2[column2].dropna())
                if len(unique_values1) < 3 or len(unique_values2) < 3:
                    continue
                intersection = unique_values1.intersection(unique_values2)
                intersection_rate1 = len(intersection) / len(unique_values1) if len(unique_values1) > 0 else 0
                intersection_rate2 = len(intersection) / len(unique_values2) if len(unique_values2) > 0 else 0
                
                # 両方のテーブルで閾値を超える場合に結合可能と判断
                if intersection_rate1 >= threshold and intersection_rate2 >= threshold:
                    joinable_columns.append({
                        'Table1': table1,
                        'Column1': column1,
                        'Table2': table2,
                        'Column2': column2,
                        'Intersection_Rate1': intersection_rate1,
                        'Intersection_Rate2': intersection_rate2
                    })

# 結果をCSVファイルに保存
output_file = 'equijoin_joinable_columns.csv'
df_joinable_columns = pd.DataFrame(joinable_columns)
df_joinable_columns.to_csv(output_file, index=False)

print(f"Joinable columns information has been saved to {output_file}")
