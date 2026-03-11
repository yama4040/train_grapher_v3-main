# Train Grapher v3 - Copilot Instructions

このドキュメントは、train_grapher_v3リポジトリで作業するコーディングエージェントのためのオンボーディングガイドです。

## リポジトリ概要

**プロジェクト名**: train-grapher-v3  
**バージョン**: 0.1.0  
**言語**: Python 3.12.3  
**目的**: 鉄道列車運行シミュレーションシステム。固定閉塞・移動閉塞システムでの列車の運転をシミュレートし、列車の位置、速度、加速度を時系列で管理・可視化します。train grapher v2のリファクタリングバージョンであり、webバックエンドとも共通化を目指しています。

**主要な依存関係**:
- matplotlib (3.9.2+): グラフ描画
- openpyxl (3.1.4+): Excel操作
- uv: Pythonパッケージマネージャー

**プロジェクトタイプ**: Pythonライブラリパッケージ（ビルドバックエンド: hatchling）

## 重要: 必須の環境セットアップ手順

### 初期セットアップ（必須）

**前提条件**: uv（最新版）がインストールされていること。

1. **依存関係のインストール（必須）**: リポジトリをクローンした後、または`pyproject.toml`を変更した後は、**必ず**以下を実行してください：
   ```powershell
   uv sync
   ```
   これにより`.venv`仮想環境が作成され、すべての依存関係がインストールされます。

2. **Pythonバージョン**: Python 3.12.3が必要です（`.python-version`で指定）。ローカルに3.12系が無い場合はuvで取得するか、適切なPythonを用意してください。

### コード実行方法

**すべてのPythonコマンドは `uv run` プレフィックスを付けて実行してください**。

正しい例:
```powershell
uv run python sample/simple_simulation_example.py
uv run python tests/lib_train.py
```

誤った例（**使用しないこと**）:
```powershell
python sample/simple_simulation_example.py  # uvの仮想環境が使われない
.venv/Scripts/python.exe tests/lib_train.py  # 直接実行は推奨されない
```

### コードフォーマット・リント

このプロジェクトでは**Ruff**を使用してコードスタイルの統一、リント、フォーマットを行っています。

#### コード編集前に理解すべきこと

- **Ruff**: 120文字行長制限でコードをフォーマット、リント、コード修正を実行（Black・isort・その他のツール機能を統合）
- VS Code設定（`.vscode/settings.json`）でファイル保存時に自動フォーマット・修正が有効

#### フォーマット適用（コミット前に必須）

```powershell
# Ruffでフォーマット・リント修正を適用（src/、tests/、sample/に対して実行）
uv run ruff format src/ tests/ sample/

# Ruffでリント修正を適用（安全な修正は自動実行）
uv run ruff check --fix src/ tests/ sample/
```

**注意**: `archive/`ディレクトリは古いコードで、フォーマットエラーが多数あります。このディレクトリは無視してください。

#### フォーマット・リントチェック（変更前の検証）

```powershell
# Ruffチェック（修正なしで問題を確認）
uv run ruff check src/ tests/ sample/
```

**重要**: エラーが出た場合は上記の適用コマンドでフォーマットを修正してから編集を行ってください。

## プロジェクト構造とアーキテクチャ

### ディレクトリ構造

```
train_grapher_v3/
├── .github/                    # GitHub設定・CI設定（現在未設定）
├── .vscode/                    # VS Code設定
│   ├── settings.json          # Python・Ruff設定
│   └── launch.json            # デバッグ設定
├── archive/                    # 古いコード（train grapher v2等）- 無視すること
├── config/                     # 設定ファイル
│   └── database.config.json   # データベースパス設定
├── datas/                      # シミュレーションデータ（gitignore対象）
├── docs/                       # ドキュメント
│   ├── usage.md               # 主要なAPIドキュメント
│   └── プログラム仕様書/        # 設計仕様書
├── logs/                       # ログファイル（gitignore対象）
├── sample/                     # サンプルコード
│   ├── simple_simulation_example.py      # 基本シミュレーション
│   ├── simulation_sample.py              # 複雑な例
│   └── moving_block_simulation_example.py # 移動閉塞の例
├── src/train_grapher_v3/       # **メインソースコード**
│   ├── __init__.py
│   ├── core/                   # コアシミュレーションロジック
│   │   ├── block_system.py    # 閉塞システム管理
│   │   ├── line_shape.py      # 路線形状・Node・Edge・Station定義
│   │   ├── line.py            # 路線全体管理
│   │   ├── runningdata.py     # 列車走行データ管理
│   │   ├── simulation.py      # シミュレーション実行エンジン
│   │   ├── station.py         # 駅管理（別実装）
│   │   ├── status.py          # 列車状態Enum定義
│   │   ├── train_parameter.py # 列車性能パラメータ
│   │   └── train.py           # 列車クラス
│   └── util/                   # ユーティリティ
│       ├── logger.py          # カラーロガー
│       └── simple_viewer.py   # グラフ表示
├── tests/                      # テストコード（pytest未使用）
│   ├── lib_*.py               # 各モジュールのテスト
│   └── ...
├── pyproject.toml              # **プロジェクト設定・依存関係**
└── README.md                   # セットアップガイド
```

### 主要なアーキテクチャコンポーネント

#### 1. **Simulation** (`simulation.py`)
シミュレーションのエントリーポイント。`Line`を受け取り、指定ステップ数だけシミュレーションを実行します。

#### 2. **Line** (`line.py`)
路線全体を管理。列車リスト、路線形状、閉塞システムを統合し、各ステップで全列車の状態を更新します。

#### 3. **LineShape** (`line_shape.py`)
路線の物理的構造を定義：
- **Node**: 分岐点・路線端点
- **Edge**: ノード間の線路区間（長さ、勾配、曲線、駅、閉塞リスト含む）
- **Position**: Edge上の特定位置
- **Route**: 列車が辿るEdgeのシーケンス
- **Station**: 駅（ID、位置、名前）
- **Block**: 閉塞区間（開始位置、速度制限リスト）
- **Grade** / **Curve**: 勾配・曲線データ

#### 4. **BlockSystem** (`block_system.py`)
固定閉塞システムの実装。先行列車との距離、駅停車、速度制限に基づいて各列車の状態（加速・減速・惰行・停車）を決定します。
- 継承可能な設計（`MovingBlockSystem`等のカスタム実装が可能）

#### 5. **Train** (`train.py`)
個別列車を表現。走行経路、性能パラメータ、停車時間、走行データ（`RunningData`）を管理します。

#### 6. **TrainParameter** (`train_parameter.py`)
列車の物理特性（重量、慣性係数、加減速度、マージン等）を定義するデータクラス。

#### 7. **RunningData** (`runningdata.py`)
シミュレーション中の列車の時系列データ（状態、速度、加速度、位置）を格納します。

#### 8. **Status** (`status.py`)
列車の運転状態を表すEnum（加速、減速、惰行、停車等）。

#### 9. **Logger** (`util/logger.py`)
カラーコンソールログ出力ユーティリティ。ログレベルごとに色分け表示。

#### 10. **SimpleViewer** (`util/simple_viewer.py`)
matplotlibを使用してシミュレーション結果をグラフ表示。

### データフロー

1. **LineShape**でノード・エッジ・駅・閉塞を定義
2. **Train**に走行経路（Route）と性能パラメータ（TrainParameter）を設定
3. **BlockSystem**と**Line**を初期化
4. **Simulation**で指定ステップ数実行
   - 各ステップで**Line**が**BlockSystem**を呼び出し各列車の状態を決定
   - 各**Train**が状態に基づいて位置・速度を更新し**RunningData**に記録
5. **SimpleViewer**で結果を可視化

## テスト実行方法

このプロジェクトは**pytestを使用していません**。テストは直接Pythonスクリプトとして実行します。

### テストの実行

```powershell
# 個別のテストファイルを実行
uv run python tests/lib_train.py
uv run python tests/lib_block_system.py
uv run python tests/lib_line_shape.py
uv run python tests/lib_simulation.py  # 注意: 一部のテストにインポートエラーがある可能性
```

### サンプル実行

```powershell
# 基本的なシミュレーション例
uv run python sample/simple_simulation_example.py

# より複雑なシミュレーション
uv run python sample/simulation_sample.py

# 移動閉塞システムの例
uv run python sample/moving_block_simulation_example.py
```

**期待される出力**: シミュレーション進捗メッセージと結果の数値が表示されます。一部のサンプルはmatplotlibウィンドウを開く場合があります。

### 既知の問題

- `tests/lib_simulation.py`は`train_grapher_v3.lib.read_schedule`をインポートしようとしますが、このモジュールは存在しません（未実装機能）。このテストは現在実行できません。

## コーディングガイドライン

### コードスタイル

- **行長**: 120文字（Blackで強制）
- **インポート順序**: isortで自動整理（blackプロファイル）
- **フォーマット**: VS Codeでファイル保存時に自動適用
- **型ヒント**: 推奨（既存コードで広く使用）

### 命名規則

- **クラス**: PascalCase（例: `BlockSystem`, `LineShape`）
- **関数・メソッド**: snake_case（例: `calculate_step`, `get_edge_by_id`）
- **プライベートメンバー**: アンダースコアプレフィックス（例: `_trains`, `_get_status`）
- **定数**: UPPER_CASE（例: プロジェクト内では少ない）

### ドキュメント

- **APIドキュメント**: `docs/usage.md`に詳細なクラス・メソッド説明があります
- **設計仕様**: `docs/プログラム仕様書/index.md`にMermaidクラス図があります
- **Docstring**: 既存コードでは日本語のdocstringが使用されています

### TODOとコメント

プロジェクト内にいくつかのTODOコメントが存在します（例: `src/train_grapher_v3/core/block_system.py`）。これらは将来の改善点を示しています。

## よくある作業パターン

### 新しい閉塞システムを実装する

1. `src/train_grapher_v3/core/block_system.py`の`BlockSystem`クラスを継承
2. `calculate_step`メソッドと必要な内部メソッドをオーバーライド
3. `sample/moving_block_simulation_example.py`を参考に実装
4. フォーマット適用: `uv run ruff format src/ && uv run ruff check --fix src/`

### 新しいシミュレーションを作成する

1. `sample/simple_simulation_example.py`をベースにコピー
2. LineShape（Node, Edge, Station, Block）を定義
3. Trainオブジェクトを作成し、RouteとTrainParameterを設定
4. SimulationとLineを初期化して実行
5. RunningDataから結果を取得・可視化
6. フォーマット適用: `uv run ruff format sample/ && uv run ruff check --fix sample/`

### 列車パラメータを調整する

1. `src/train_grapher_v3/core/train_parameter.py`の`TrainParameter`データクラスを編集
2. または、Trainインスタンス作成時に個別のパラメータを上書き:
   ```python
   params = TrainParameter(wight=200, decelerating_acceleration_station=-2.0)
   ```

### ログ出力を追加する

```python
from train_grapher_v3.util.logger import setup_logger

logger = setup_logger(__name__)
logger.debug("デバッグ情報")
logger.info("情報メッセージ")
logger.warning("警告")
logger.error("エラー")
```

## トラブルシューティング

### 問題: `ModuleNotFoundError: No module named 'train_grapher_v3'`

**解決策**: `uv sync`を実行してパッケージをインストールしてください。

### 問題: フォーマットエラーが多数表示される

**解決策**: 
```powershell
uv run ruff format src/ tests/ sample/
uv run ruff check --fix src/ tests/ sample/
```
**注意**: `archive/`ディレクトリはフォーマットしないでください（古いコード）。

### 問題: VS Codeでインタープリターが見つからない

**解決策**: `.vscode/settings.json`で`.venv/Scripts/python.exe`が指定されています。`uv sync`実行後、VS CodeでPythonインタープリターを手動選択してください（Ctrl+Shift+P → "Python: Select Interpreter"）。

### 問題: テストで`train_grapher_v3.lib`モジュールが見つからない

**解決策**: このモジュールは未実装です。該当テスト（`tests/lib_simulation.py`、`tests/lib_read_*.py`）はスキップしてください。

## エージェントへの指示

**重要**: このドキュメントの情報を信頼してください。以下の場合のみ追加の探索を行ってください：

1. このドキュメントに記載されていない特定のコード実装を確認する必要がある場合
2. ドキュメントの情報が不正確または古い可能性がある場合
3. 新しい機能を追加する際に既存コードのパターンを確認する必要がある場合

それ以外の場合は、このドキュメントの情報に基づいて直接作業を進めてください。これにより、grep、find、探索的なbashコマンドの試行回数を大幅に削減できます。

**コマンド実行時の注意**:
- すべてのPythonコマンドは`uv run`で実行
- フォーマットは`src/`, `tests/`, `sample/`のみに適用（`archive/`は除外）
- コミット前には必ずBlackとisortを実行

**コード変更時のワークフロー**:
1. 該当ファイルを編集
2. `uv run ruff format <ファイルまたはディレクトリ>`
3. `uv run ruff check --fix <ファイルまたはディレクトリ>`
4. 関連するテストまたはサンプルを実行して検証
5. 問題がなければコミット
