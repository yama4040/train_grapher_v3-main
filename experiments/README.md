# experiments/

シミュレーション実験の設定・スクリプト・結果を管理するディレクトリです。

## ディレクトリ構成

```
experiments/
├── README.md              # このファイル
├── configs/               # 実験設定（Gitで管理）
│   └── exp001_example/
│       ├── model.json     # シミュレーションモデル（datas/ と同形式）
│       └── meta.json      # 実験の目的・タグ・備考
├── scripts/               # 共通スクリプト（Gitで管理）
│   ├── run.py             # 実験実行
│   └── list_results.py    # 結果一覧表示
└── results/               # 実験結果（.gitignore 対象）
    └── exp001_example/
        └── 20260219_143022/   # 実行ごとにタイムスタンプで記録
            ├── run_info.json          # 実行メタデータ（再現情報）
            ├── initial_conditions.json
            ├── results_summary.json
            └── train_data/
                └── 列車名.csv
```

## 実験の追加方法

1. `configs/` に `expNNN_説明/` ディレクトリを作成する
2. `model.json`（シミュレーションモデル）を作成する
3. `meta.json`（実験の説明）を作成する

### 命名規則

```
expNNN_説明（英数字・ハイフン・アンダースコアのみ）
例: exp002_moving_block
    exp003_multi_train_headway
    exp004_timetable_control
```

### meta.json の形式

```json
{
  "name": "実験の表示名",
  "description": "実験の目的・概要",
  "author": "担当者名",
  "tags": ["固定閉塞", "複数列車"]
}
```

## 実験の実行

```bash
# 実験を実行して results/ に保存
uv run python experiments/scripts/run.py exp001_example

# 保存せず動作確認のみ
uv run python experiments/scripts/run.py exp001_example --no-save
```

## 結果の確認

```bash
# 全実験の一覧
uv run python experiments/scripts/list_results.py

# 特定実験の詳細（全実行を表示）
uv run python experiments/scripts/list_results.py exp001_example
```

## 注意事項

- `results/` は `.gitignore` で除外されています。大容量の結果ファイルはGit管理しません。
- `configs/` と `scripts/` はGitで管理します。実験設定は必ずここに保存してください。
- 同じ実験を複数回実行しても、タイムスタンプで別ディレクトリに保存されます（上書きなし）。
