# シミュレーションモデル JSON フォーマット仕様

このドキュメントは、train_grapher_v3でシミュレーションモデルをJSON形式で保存・読込する際のフォーマット仕様です。

## 概要

シミュレーションモデルは以下の要素で構成されます：

- **メタデータ**: モデルの名前、説明、バージョン等
- **路線形状（LineShape）**: ノード、エッジ、駅、勾配、曲線
- **列車（Trains）**: 複数の列車定義
- **閉塞システム（BlockSystem）**: 固定閉塞または移動閉塞
- **シミュレーション設定**: ステップサイズ、総ステップ数等

## JSON スキーマ

```json
{
  "metadata": {
    "version": "1.0",
    "name": "シミュレーションモデル名",
    "description": "モデルの説明",
    "author": "作成者名",
    "created_at": "2026-01-28T00:00:00",
    "updated_at": "2026-01-28T00:00:00"
  },
  "simulation_config": {
    "step_size": 0.1,
    "total_steps": 3000,
    "block_system_type": "fixed"
  },
  "line_shape": {
    "nodes": [
      {
        "id": "node_start",
        "offset": 0.0
      }
    ],
    "edges": [
      {
        "id": "edge1",
        "length": 10.0,
        "start_node_id": "node_start",
        "end_node_id": "node_end",
        "grades": [
          {
            "start": 0.0,
            "end": 10.0,
            "grade": 0.0
          }
        ],
        "curves": [
          {
            "start": 0.0,
            "end": 5.0,
            "curve": 0.0
          }
        ],
        "stations": [
          {
            "id": "station1",
            "value": 2.0,
            "name": "第一駅"
          }
        ],
        "blocks": [
          {
            "start": 0.0,
            "speed_limits": [25, 45, 80, 120]
          }
        ]
      }
    ]
  },
  "trains": [
    {
      "id": "train_1",
      "name": "列車1号",
      "route_edge_ids": ["edge1"],
      "initial_position": 0.0,
      "initial_velocity": 0.0,
      "use_timetable": false,
      "end_edge_id": "edge1",
      "end_position_value": 10.0,
      "parameters": {
        "wight": 389.85,
        "factor_of_inertia": 24.54,
        "decelerating_acceleration": -3.0,
        "decelerating_acceleration_station": -4.0,
        "fast_margine": 6.0,
        "slow_margine": 15.0
      },
      "station_stop_times": [
        {
          "station_id": "station1",
          "default_value": 30.0,
          "departure_time": null
        }
      ],
      "start_condition": {
        "step": 0,
        "position": 0.0,
        "edge_id": "edge1"
      },
      "end_condition": {
        "position": 10.0,
        "edge_id": "edge1"
      }
    }
  ]
}
```

## 詳細な要素説明

### メタデータ (metadata)

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| version | string | ✓ | JSON形式のバージョン (現在は "1.0") |
| name | string | ✓ | シミュレーションモデルの名前 |
| description | string | ✗ | モデルの詳細な説明 |
| author | string | ✗ | 作成者名 |
| created_at | string | ✗ | 作成日時 (ISO 8601形式) |
| updated_at | string | ✗ | 更新日時 (ISO 8601形式) |

### シミュレーション設定 (simulation_config)

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| step_size | float | ✓ | 1ステップのサイズ（秒）。通常は0.1 |
| total_steps | integer | ✓ | シミュレーション総ステップ数 |
| block_system_type | string | ✓ | 閉塞システムの種類。"fixed" または "moving" |

### 路線形状 (line_shape)

#### ノード (nodes)

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| id | string | ✓ | ノードの一意なID |
| offset | float | ✗ | 累積オフセット（デフォルト: 0.0） |

#### エッジ (edges)

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| id | string | ✓ | エッジの一意なID |
| length | float | ✓ | エッジの長さ（km） |
| start_node_id | string | ✓ | 開始ノードのID |
| end_node_id | string | ✓ | 終了ノードのID |
| grades | array | ✓ | 勾配情報の配列 |
| curves | array | ✓ | 曲線情報の配列 |
| stations | array | ✗ | 駅情報の配列 |
| blocks | array | ✓ | 閉塞情報の配列 |

**grades内の要素：**

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| start | float | ✓ | 勾配開始位置（km） |
| end | float | ✓ | 勾配終了位置（km） |
| grade | float | ✓ | 勾配（%）、0 = フラット |

**curves内の要素：**

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| start | float | ✓ | 曲線開始位置（km） |
| end | float | ✓ | 曲線終了位置（km） |
| curve | float | ✓ | 曲線（度数）、0 = 直線 |

**stations内の要素：**

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| id | string | ✓ | 駅の一意なID |
| value | float | ✓ | 駅の位置（エッジ上の距離、km） |
| name | string | ✓ | 駅名 |

**blocks内の要素：**

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| start | float | ✓ | 閉塞開始位置（km） |
| speed_limits | array | ✓ | 速度制限リスト [0個前, 1個前, 2個前, ...] (km/h) |

### 列車 (trains)

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| id | string | ✓ | 列車の一意なID |
| name | string | ✗ | 列車の名前 |
| route_edge_ids | array | ✓ | ルートを構成するエッジIDの配列（順序重要） |
| initial_position | float | ✗ | 初期位置（km、デフォルト: 0.0） |
| initial_velocity | float | ✗ | 初期速度（km/h、デフォルト: 0.0） |
| use_timetable | boolean | ✗ | ダイヤ制御を使用するか（デフォルト: false） |
| end_edge_id | string | ✗ | 終了位置のエッジID（Noneの場合はルート終了位置） |
| end_position_value | float | ✗ | 終了位置のエッジ上の位置(km) |
| parameters | object | ✓ | 列車性能パラメータ（詳細は下記） |
| station_stop_times | array | ✗ | 駅停車時間情報 |
| use_timetable | boolean | ✗ | ダイヤ制御を使用するか（デフォルト: false） |
| end_edge_id | string | ✗ | 終了位置のエッジID（Noneの場合はルート終了位置） |
| end_position_value | float | ✗ | 終了位置のエッジ上の位置(km) |
| start_condition | object | ✗ | 列車の開始条件（詳細は下記） |
| end_condition | object | ✗ | 列車の終了条件（詳細は下記） |

#### 列車性能パラメータ (parameters)

| フィールド | 型 | 必須 | デフォルト | 説明 |
| --- | --- | --- | --- | --- |
| wight | float | ✗ | 389.85 | 列車重量（ton） |
| factor_of_inertia | float | ✗ | 24.54 | 慣性係数 |
| decelerating_acceleration | float | ✗ | -3.0 | 制限時の加速度（m/s²） |
| decelerating_acceleration_station | float | ✗ | -4.0 | 駅停車時の減速加速度（m/s²） |
| fast_margine | float | ✗ | 6.0 | 速度マージン（高速側） |
| slow_margine | float | ✗ | 15.0 | 速度マージン（低速側） |

#### 駅停車時間 (station_stop_times)

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| station_id | string | ✓ | 駅ID（line_shapeで定義されたもの） |
| default_value | float | ✓ | 停車時間（秒） |
| departure_time | float/null | ✗ | ダイヤ出発時刻（秒、null=ダイヤ制御なし、use_timetableが有効の場合に使用） |

#### 開始条件 (start_condition)

列車がシミュレーションを開始する条件を指定します。すべてのフィールドが省略可能です。

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| step | integer | ✗ | 列車が開始するステップ番号。省略時はnull |
| position | float | ✗ | 開始位置（エッジ上の距離、km）。edge_idと同時に指定必須 |
| edge_id | string | ✗ | 開始位置のエッジID。positionと同時に指定必須 |

**例:**
```json
"start_condition": {
  "step": 100,
  "position": 2.5,
  "edge_id": "edge1"
}
```

**注意:**
- `step`のみ指定した場合、そのステップからルートの開始位置で開始します。
- `position`と`edge_id`は一緒に指定する必要があります。
- すべて省略するとステップ0、位置nullとして扱われます。

#### 終了条件 (end_condition)

列車がシミュレーションを終了する位置を指定します。省略するとルートの終端が使用されます。

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| position | float | ✓ | 終了位置（エッジ上の距離、km） |
| edge_id | string | ✓ | 終了位置のエッジID |

**例:**
```json
"end_condition": {
  "position": 9.5,
  "edge_id": "edge1"
}
```

**注意:**
- 省略するとルートの最後のエッジの終端が終了位置になります。
- 指定する場合は`position`と`edge_id`の両方が必須です。
- `end_edge_id`と`end_position_value`を使用する方法もありますが、`end_condition`がより柔軟です。

## ダイヤ制御について

`use_timetable` が `true` に設定されている場合、各駅の `departure_time` を指定することで以下の制御が実現します：

1. **停車時間の保証**: デフォルト停車時間を満たしながら停車
2. **ダイヤ出発時刻の厳守**: `departure_time` より早く出発しない
3. **最小ターンアラウンドタイム**: 停車時間とダイヤを両立させた最適な出発

例：
- 到着時刻：120秒、デフォルト停車時間：60秒、ダイヤ出発時刻：200秒
- → 停車時間条件：120 + 60 = 180秒以降に出発可能
- → ダイヤ条件：200秒以降に出発
- → 実結果：200秒で出発（両条件を満たす最終時刻）

`use_timetable` が `false` の場合は、従来通りデフォルト停車時間のみ適用されます。

## バリデーション規則

JSON読込時に以下のバリデーションが行われます：

1. **必須フィールド**: すべての必須フィールドが存在すること
2. **ノード参照**: エッジのstart_node_id・end_node_idはnodes配列に存在すること
3. **エッジ参照**: trainsのroute_edge_idsはedges配列に存在すること
4. **駅参照**: station_stop_timesのstation_idはedgesのstations配列に存在すること
5. **位置の妥当性**: 駅とブロックの位置がエッジの長さ内であること
6. **IDの一意性**: ノード、エッジ、駅、列車のIDが重複していないこと

## 拡張可能性

このフォーマットは将来の拡張を想定しています：

- 新しい駆動戦略（DrivingDecision）の追加
- 移動閉塞システムの詳細設定
- シミュレーション結果のJSON出力形式
- 複雑な路線ネットワーク（複数エッジの連結）
