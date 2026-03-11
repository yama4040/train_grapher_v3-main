# ダイヤ制御機能の使用ガイド

## 概要

train_grapher_v3では、列車にダイヤ（運行時刻表）情報を持たせることで、以下の制御が可能になります：

- **停車時間の保証**: 最低停車時間を満たしながら停車
- **ダイヤ厳守**: 指定された出発時刻より早出ししない
- **最適な出発タイミング**: 両条件を満たす最適な出発時刻を自動決定

## 使用方法

### 1. Python コードでの使用

```python
from train_grapher_v3.core.train import StationStopTime, Train

# ダイヤ情報を含む駅停車時間を作成
station_stop_times = [
    StationStopTime(
        station_id="station_A",
        default_value=60.0,        # 最小停車時間: 60秒
        departure_time=110.0,      # ダイヤ出発時刻: 110秒
    ),
]

# 列車を作成（use_timetable=Trueでダイヤ制御を有効化）
train = Train(
    name="特急列車",
    line_shape=line_shape,
    route=route,
    train_parameter=params,
    station_stop_times=station_stop_times,
    use_timetable=True,  # ★ ダイヤ制御を有効化
)
```

### 2. JSON形式での指定

```json
{
  "trains": [
    {
      "id": "train_1",
      "name": "特急列車",
      "use_timetable": true,
      "station_stop_times": [
        {
          "station_id": "station_A",
          "default_value": 60.0,
          "departure_time": 110.0
        }
      ]
    }
  ]
}
```

## 動作原理

### 停車時間の判定

`is_stop(step, step_size)` メソッドが以下の条件を評価します：

```
current_time = step * step_size

if use_timetable == False:
    停車中 = (停車時間カウント > 0)
else:
    停車中 = (停車時間カウント > 0) OR (現在時刻 < ダイヤ出発時刻)
```

**例:**
- 到着時刻: 50秒
- 最小停車時間: 60秒 → 110秒まで停車必須
- ダイヤ出発時刻: 120秒 → 120秒より早出し禁止
- 結果: **120秒で出発** （両条件の最大値）

## 設定値

### `use_timetable` フラグ

| 値 | 動作 |
|---|---|
| `False` (デフォルト) | 従来通り、`default_value` の停車時間のみ適用 |
| `True` | `departure_time` を有効にし、ダイヤ制御を適用 |

### `StationStopTime` パラメータ

| パラメータ | 型 | 説明 |
|---|---|---|
| `station_id` | str | 駅ID |
| `default_value` | float | 最小停車時間（秒） |
| `departure_time` | float\|None | ダイヤ出発時刻（秒、Noneはダイヤなし） |
| `use_timetable` | bool | ダイヤ制御を使用するか（Trainから自動設定） |

## サンプル

### サンプルファイル

[sample/timetable_control_example.py](sample/timetable_control_example.py) では、以下を実演しています：

- **ダイヤ制御列車**: A駅で110秒、B駅で260秒に厳密に出発
- **通常列車**: 60秒停車後、可能な限り早く出発

実行方法：
```bash
uv run python sample/timetable_control_example.py
```

### JSON例

[datas/simple_simulation_model_with_timetable.json](datas/simple_simulation_model_with_timetable.json) に、ダイヤ情報を含むシミュレーションモデルの例があります。

## JSON形式の拡張

### バージョン: 1.0

#### メタデータ
```json
{
  "metadata": {
    "version": "1.0"
  }
}
```

#### trains フィールド

新しく追加されたフィールド：

| フィールド | 型 | 説明 |
|---|---|---|
| `use_timetable` | bool | ダイヤ制御を使用するか（デフォルト: false） |

#### station_stop_times 内のフィールド

新しく追加されたフィールド：

| フィールド | 型 | 説明 |
|---|---|---|
| `departure_time` | float\|null | ダイヤ出発時刻（秒、nullはダイヤなし） |

## 実装の詳細

### クラス変更

#### `StationStopTime`
- `use_timetable: bool` フラグを追加
- `is_stop()` メソッドでダイヤ制御ロジックを実装
- `set_arrival_time()` メソッドで到着時刻を記録可能

#### `Train`
- コンストラクタに `use_timetable: bool` パラメータを追加
- 初期化時に全駅停車時間に `use_timetable` フラグを設定

### ファイル変更

- **core/train.py**: StationStopTime, Train クラスを拡張
- **util/simulation_model_io.py**: JSON入出力にダイヤ情報を対応
- **document/SIMULATION_JSON_FORMAT.md**: JSON形式仕様を更新

## 注意点

### 到着遅延への対応

現在の実装では、到着遅延時のダイヤ逸脱は自動的に処理されます：

```
到着時刻 = 120秒（予定より20秒遅延）
最小停車時間 = 60秒 → 180秒まで停車必須
ダイヤ出発時刻 = 150秒
結果: 180秒で出発（両条件の最大値、ダイヤより30秒遅延）
```

ダイヤ通りの運行が保証されるわけではなく、**実績に基づいた最適な運用判断** となります。

### 互換性

- `use_timetable=False` の場合、既存の動作と完全に互換
- 既存のシミュレーションモデルは影響を受けない

## トラブルシューティング

### ダイヤ制御が反映されない場合

1. **`use_timetable` が `True` に設定されているか確認**
   ```python
   print(train._use_timetable)  # True であることを確認
   ```

2. **`departure_time` が設定されているか確認**
   ```python
   for sst in train._station_stop_times:
       print(f"{sst.station_id}: {sst.departure_time}")
   ```

3. **JSON読み込み時のバリデーション**
   - JSONスキーマが正しく、必須フィールドが揃っているか確認
   - `use_timetable` フィールドが存在するか確認

## 今後の拡張可能性

- 複数の出発候補時刻（優先度付き）
- ダイヤ遅延時の自動再計画
- 接続列車との乗換制約
- 車両数制限下での運用計画

