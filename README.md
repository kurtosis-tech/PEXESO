# PEXESO: Efficient Joinable Table Discovery in Data Lakes

## 概要

PEXESOは、高次元の類似性に基づいてデータレイク内のジョイン可能なテーブルを効率的に発見するためのフレームワークです。
このリポジトリは、論文「Efficient Joinable Table Discovery in Data Lakes: A High-Dimensional Similarity-Based Approach」に基づいています。
しかし、このコードは単にジョイン可能なテーブルを探索するためのものであり、PEXESOを完全に再現したコードではないことに注意してください。

## How to use

1. "/data" にcsvデータを配置する
   csvデータは "<dataset_path>/csv"という形で配置する。

2. dataset_pathを入力してpreprocess.pyを実行する。
   npyフォルダがcsvと同列に作成されたら成功。

3. dataset_pathを入力してpexeso.py，equijoin.pyを実行する。
   resultフォルダ中に結果が保存される。

### 必要条件

- Python 3.7以上
- 必要なPythonライブラリは `requirements.txt` に記載されています。
