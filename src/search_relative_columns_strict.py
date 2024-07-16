import faiss
import numpy as np

def contains_digit(value):
    """
    値が数字を含むかどうかを判定する関数。
    
    Args:
        value: 判定する値。
        
    Returns:
        bool: 数字を含むならTrue、それ以外ならFalse。
    """
    return any(char.isdigit() for char in str(value))

def is_too_short(value, min_length=3):
    """
    文字列が短すぎるかどうかを判定する関数。
    
    Args:
        value: 判定する値。
        min_length (int): 最小文字数。
        
    Returns:
        bool: 短すぎるならTrue、それ以外ならFalse。
    """
    return len(str(value)) < min_length

def search_relative_columns(df_A, vectors_matrix_A, df_B, vectors_matrix_B, threshold_1=0.1):
    """
    2つのベクトル行列間の相対的な列を検索する関数。
    各行列内のユニークなベクトルが4つ以上ない場合、その列をスキップする。
    また、数字を含むか短すぎる文字列を含む場合もスキップする。

    Args:
        vectors_matrix_A (np.ndarray): 行列Aのベクトル。形状は (column数, row数, 次元数)。
        df_A (pd.DataFrame): 行列Aのデータフレーム。
        vectors_matrix_B (np.ndarray): 行列Bのベクトル。形状は (column数, row数, 次元数)。
        df_B (pd.DataFrame): 行列Bのデータフレーム。
        threshold_1 (float): 距離のしきい値。

    Returns:
        list: 相対的な列のインデックスペア。
        list: 相対的な列の距離。
    """
    faiss_indices = []  # Faissインデックスのリスト
    relative_columns = []  # 相対的な列のインデックスペアのリスト
    distance_for_relative_columns = []  # 相対的な列の距離のリスト

    distances_dict = {}  # 各ペアの距離を保存する辞書

    # 行列Bの各列を処理
    for index_B in range(vectors_matrix_B.shape[0]):
        column_B = df_B.iloc[:, index_B]
        
        # ユニークな値の数が3以下、または数字を含む値や短すぎる値が含まれる場合はスキップ
        if column_B.nunique() <= 3 or any(contains_digit(value) or is_too_short(value) for value in column_B):
            continue

        # ベクトルBの計算
        vector_B = vectors_matrix_B[index_B, 0]
        vector_B = np.hstack((vectors_matrix_B[index_B, 0], np.mean(vectors_matrix_B[index_B, 1:], axis=0)))
        vector_B = vector_B.astype('float32').reshape(1, -1)

        faiss_index = faiss.IndexFlatL2(vector_B.shape[1])  # Faissインデックスの作成

        if np.isnan(vector_B).any():
            continue  # NaNを含むベクトルをスキップ

        faiss_index.add(vector_B)  # ベクトルBをインデックスに追加
        faiss_indices.append((index_B, faiss_index))  # インデックスと列インデックスのペアを保存

    # 行列Aの各列を処理
    for index_A in range(vectors_matrix_A.shape[0]):
        column_A = df_A.iloc[:, index_A]

        # 数字を含む値や短すぎる値が含まれる場合はスキップ
        if column_A.nunique() <= 3 or any(contains_digit(value) or is_too_short(value) for value in column_A):
            continue

        # ベクトルAの計算（結合）
        vector_A = np.hstack((vectors_matrix_A[index_A, 0], np.mean(vectors_matrix_A[index_A, 1:], axis=0)))
        vector_A = vector_A.astype('float32').reshape(1, -1)

        if np.isnan(vector_A).any():
            continue  # NaNを含むベクトルをスキップ

        # 行列Bの各列との距離を計算
        for index_B, faiss_index in faiss_indices:
            distances, indices = faiss_index.search(vector_A, 1)
            distance_value = distances[0][0]
            distances_dict[(index_A, index_B)] = distance_value  # 各ペアの距離を保存

            if distance_value < threshold_1:
                relative_columns.append((index_A, index_B))  # 相対的な列のインデックスペアを保存
                distance_for_relative_columns.append(distance_value)  # 相対的な列の距離を保存
    
    return relative_columns, distance_for_relative_columns
