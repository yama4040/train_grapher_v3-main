# base_model: 15秒安全確認付き基準モデル

移動閉塞環境において、駅間停車後の再発車時に15秒間の安全確認（`CHECK_SAFETY`）を挟む基準モデル。

## 概要

移動閉塞（`MovingBlockSystem`）では、先行列車との距離が安全距離を下回ると後続列車に `instruction_speed=0` が返される。このとき後続列車は駅間で停止する（駅間停車）。

閉塞が解除された（`instruction_speed > 0` に変化した）際、実際の運転では安全確認のため15秒の待機を挟む。`BaseModelDrivingDecision` はこの挙動を再現する。

## ファイル構成

```
experiments/base_model/
├── driving_decision.py   # BaseModelDrivingDecision クラス
├── runner.py             # バッチ実行CLI（並列対応）
└── README.md             # このファイル
```

## `BaseModelDrivingDecision` の動作

1. 駅間停車中（`velocity ≈ 0`、`!is_station_stopping`）かつ閉塞解除を検出
2. `CHECK_SAFETY` を返す（`velocity=0` が強制される）
3. 15秒経過後、`DefaultDrivingDecision` へ引き継ぐ

### ステータス値

| ステータス | 値 | 意味 |
|---|---|---|
| `CHECK_SAFETY` | 9 | 安全確認中（`velocity=0`、`acceleration=0`） |

## 使い方

```bash
# テスト実行（experiments/datas/ からランダム 1 ファイル）
uv run python experiments/base_model/runner.py --test

# 全ファイル実行
uv run python experiments/base_model/runner.py --select all

# 全ファイル 4並列実行
uv run python experiments/base_model/runner.py --select all --workers 4

# 最初の 5 ファイルのみ
uv run python experiments/base_model/runner.py --select first:5

# ランダムに 3 ファイル
uv run python experiments/base_model/runner.py --select random:3

# インデックス指定（0-based）
uv run python experiments/base_model/runner.py --select index:0,1,5

# データディレクトリを明示指定
uv run python experiments/base_model/runner.py --datas experiments/datas --select all
```

## 結果の場所

```
experiments/results/base_model/
└── <バッチタイムスタンプ>/       # バッチ実行ごとにグループ化
    └── <モデル名>/
        ├── run_info.json         # 実行メタデータ（CHECK_SAFETY ステップ数含む）
        ├── initial_conditions.json
        ├── results_summary.json
        └── train_data/
            └── <列車名>.csv      # ステップごとのデータ（status_name=CHECK_SAFETY で確認可能）
```

## 注意事項

- `runner.py` は JSON の `block_system_type` に関わらず `MovingBlockSystem` を強制使用する
- 1 列車のみの JSON では `CHECK_SAFETY` が発生しない（駅間停車の原因がないため）
- `BaseModelDrivingDecision` のインスタンスは列車ごとに個別に作成する（状態管理のため）
