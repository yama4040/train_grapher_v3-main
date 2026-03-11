# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを操作する際のガイドラインを提供します。

## プロジェクト概要

`train-grapher-v3` は、Pythonで記述されたバックエンド向けの鉄道運行シミュレーション・ライブラリです。複数列車の移動、閉塞信号システム、およびダイヤグラム（時刻表）ベースのスケジューリングをモデル化します。Webアプリケーションへの組み込み、または単体での利用を想定して設計されています。

* **Python バージョン:** 3.12.3
* **パッケージマネージャー:** `uv` (`pip`や`poetry`は不使用)
* **リンター/フォーマッタ:** `ruff`

## コマンド

```bash
# 依存関係のインストール
uv sync

# サンプルスクリプトの実行
uv run python sample/basic/simple_simulation_example.py
uv run python sample/json_io/json_model_runner.py
uv run python sample/timetable_control_example.py

# 静的解析 (Lint) / フォーマット
uv run ruff check src/
uv run ruff format src/

```

現在、自動テストは実装されていません。

## アーキテクチャ

### コアコンポーネント (`src/train_grapher_v3/core/`)

シミュレーションは以下のレイヤー構造で設計されています：

1. **`line_shape.py`** — 静的なトポロジー：ノード、エッジ、駅、閉塞区間、勾配、曲線。物理的な路線を定義します。
2. **`block_system.py`** — 信号・安全層。`FixedBlockSystem`（固定閉塞）および `MovingBlockSystem`（移動閉塞）が列車に対して `SignalInstruction` オブジェクトを発行し、速度制限の適用と衝突回避を強制します。
3. **`train.py`** — `Train` インスタンス。位置、速度、駅の停車時間、およびダイヤグラムのスケジュールを追跡します。シミュレーションへの進入・退出条件をサポートしています。`use_timetable=True` に設定することで、ダイヤ制御（予定時刻に基づく出発制御）が可能です。
4. **`driving_decision.py`** — 運転動作のためのストラテジーパターン。`DefaultDrivingDecision` が標準的なロジックを実装しています。独自の運転ロジックを作成する場合は、`DrivingDecision` を継承して `decide()` メソッドをオーバーライドします。
5. **`line.py`** — 列車と閉塞システムを統合して `Line`（路線）を構成します。中央のコーディネーターとして機能します。
6. **`simulation.py`** — タイムステップ実行型のシミュレーションエンジン。設定されたステップサイズ（例：0.1秒）で `Line` を進行させます。進捗コールバックを受け付けます。
7. **`runningdata.py`** — ステップごとの列車状態（ステータス、速度、加速度、位置）の記録。ステップごとにインデックス化されます。
8. **`train_parameter.py`** — 物理パラメータ：重量、慣性係数、減速度、速度余裕など。
9. **`status.py`** — `TrainStatus` 定数：`POWER_RUN`（力行）、`COASTING`（惰行）、`BRAKE_*`（ブレーキ）、`STOPPING_STATION`（駅停車中）、`NONE_SIMURATION`（シミュレーション外）など。

### ユーティリティ (`src/train_grapher_v3/util/`)

* **`simulation_model_io.py`** — シミュレーションモデル全体（トポロジー + 列車 + 閉塞システム + 設定）をJSON形式でシリアライズ・デシリアライズするための `SimulationModelEncoder` / `SimulationModelDecoder` です。
* **`simple_viewer.py`** — Matplotlibを使用した可視化ツール。
* **`logger.py`** — タイムスタンプとソースコンテキストを含む、色付きのANSIコンソールログ。

### サンプルデータ (`datas/`)

テスト用の構築済みJSONモデル：`simple_simulation_model.json`、`multi_train_simulation_model.json`、`simple_simulation_model_with_timetable.json`。JSONのスキーマは [document/SIMULATION_JSON_FORMAT.md](document/SIMULATION_JSON_FORMAT.md) に記載されています。

## 主要なデザインパターン

* **ストラテジーパターン (Strategy Pattern):** 運転決定（Driving Decision）に使用されます。シミュレーションのコア部分に手を加えることなく、`DrivingDecision` のサブクラスを差し替えるだけで列車の挙動を変更できます。
* **抽象基底クラス (Abstract Base Classes):** `BlockSystem` や `DrivingDecision` は、拡張性のためにインターフェースの契約を強制します。
* **JSON モデル I/O:** Webアプリケーションとの統合における主要なパスです。Pythonでモデルを構築し、JSONにシリアライズして保存し、後で `SimulationModelDecoder` を介してリロードすることができます。

## 言語に関する注意点

ドキュメント、コメント、および変数名には、頻繁に日本語が使用されています（例：ダイヤ制御 = timetable control）。ドメイン用語（専門用語）は日本の鉄道慣習に従っています。
