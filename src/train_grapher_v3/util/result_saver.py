"""シミュレーション結果の保存機能

このモジュールはシミュレーション実行後の結果を以下の形式で保存します：
- 初期条件JSON: シミュレーションモデルの設定
- サマリーJSON: 各列車の集計結果
- 列車ごとのCSV: 各ステップの状態（train_data/ ディレクトリ以下）
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

from train_grapher_v3.core.line_shape import LineShape
from train_grapher_v3.core.status import Status
from train_grapher_v3.core.train import Train
from train_grapher_v3.util.logger import setup_logger
from train_grapher_v3.util.simulation_model_io import SimulationModelEncoder

logger = setup_logger(__name__)

_STATUS_NAMES: dict[int, str] = {
    Status.NONE_SIMURATION: "NONE_SIMURATION",
    Status.OUT_OF_SERVICE: "OUT_OF_SERVICE",
    Status.STOPPING_STATION: "STOPPING_STATION",
    Status.BRAKE_STATION: "BRAKE_STATION",
    Status.BRAKE_TRAIN: "BRAKE_TRAIN",
    Status.BRAKE_OVERSPEED: "BRAKE_OVERSPEED",
    Status.CONSTANT_SPEED: "CONSTANT_SPEED",
    Status.COASTING: "COASTING",
    Status.POWER_RUN: "POWER_RUN",
    Status.CHECK_SAFETY: "CHECK_SAFETY",
}


class SimulationResultSaver:
    """シミュレーション結果を保存するクラス"""

    def __init__(
        self,
        trains: list[Train],
        step_size: float,
        total_steps: int,
        line_shape: LineShape,
        block_system_type: str = "fixed",
    ):
        """初期化

        Args:
            trains: シミュレーション済みの列車リスト
            step_size: ステップサイズ（秒）
            total_steps: 総ステップ数
            line_shape: 路線形状
            block_system_type: 閉塞システムの種類 ("fixed" または "moving")
        """
        self._trains = trains
        self._step_size = step_size
        self._total_steps = total_steps
        self._line_shape = line_shape
        self._block_system_type = block_system_type

    def save(
        self,
        output_dir: str,
        name: str = "",
        description: str = "",
        author: str = "",
    ) -> None:
        """シミュレーション結果を指定ディレクトリに保存

        以下の構成でファイルを出力します：

            output_dir/
            ├── initial_conditions.json
            ├── results_summary.json
            └── train_data/
                ├── <列車名>.csv
                └── ...

        Args:
            output_dir: 出力ディレクトリのパス
            name: モデル名（initial_conditions.jsonのメタデータに使用）
            description: モデルの説明
            author: 作成者名
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._save_initial_conditions(output_path, name, description, author)
        self._save_results_summary(output_path)
        self._save_train_csvs(output_path)

        logger.info(f"シミュレーション結果を保存しました: {output_path}")

    def _save_initial_conditions(
        self, output_path: Path, name: str, description: str, author: str
    ) -> None:
        """初期条件をJSONで保存"""
        encoder = SimulationModelEncoder(
            self._line_shape,
            self._trains,
            self._step_size,
            self._total_steps,
        )
        json_data = encoder.encode(name, description, author, self._block_system_type)

        file_path = output_path / "initial_conditions.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        logger.info(f"  初期条件を保存しました: {file_path}")

    def _save_results_summary(self, output_path: Path) -> None:
        """サマリーをJSONで保存"""
        train_summaries = [self._calc_train_summary(t) for t in self._trains]

        summary = {
            "saved_at": datetime.now().isoformat(),
            "simulation_config": {
                "step_size": self._step_size,
                "total_steps": self._total_steps,
                "total_time_s": self._total_steps * self._step_size,
            },
            "trains": train_summaries,
        }

        file_path = output_path / "results_summary.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"  サマリーを保存しました: {file_path}")

    def _calc_train_summary(self, train: Train) -> dict:
        """列車ごとのサマリーを計算"""
        rd = train.get_running_data()
        statuses = rd.get_status_all()
        velocities = rd.get_velocity_all()
        positions = rd.get_position_value_all()
        edge_ids = rd.get_edge_id_all()

        total_recorded_steps = len(statuses)

        # アクティブステップ（NONE_SIMURATION以外）
        active_steps = [
            i
            for i, s in enumerate(statuses)
            if s is not None and s != Status.NONE_SIMURATION
        ]

        start_step = active_steps[0] if active_steps else None
        end_step = active_steps[-1] if active_steps else None

        # 走行中の速度（NONE_SIMURATION と STOPPING_STATION を除く）
        running_velocities = [
            v
            for s, v in zip(statuses, velocities)
            if s not in (Status.NONE_SIMURATION, Status.STOPPING_STATION)
            and v is not None
        ]

        max_velocity = max(running_velocities) if running_velocities else None
        avg_velocity = (
            sum(running_velocities) / len(running_velocities)
            if running_velocities
            else None
        )

        last_position = positions[-1] if positions else None
        last_edge_id = edge_ids[-1] if edge_ids else None

        return {
            "name": train.name,
            "total_recorded_steps": total_recorded_steps,
            "start_step": start_step,
            "start_time_s": (
                round(start_step * self._step_size, 3)
                if start_step is not None
                else None
            ),
            "end_step": end_step,
            "end_time_s": (
                round(end_step * self._step_size, 3) if end_step is not None else None
            ),
            "max_velocity_kmh": (
                round(max_velocity, 3) if max_velocity is not None else None
            ),
            "average_velocity_kmh": (
                round(avg_velocity, 3) if avg_velocity is not None else None
            ),
            "final_position_km": (
                round(last_position, 6) if last_position is not None else None
            ),
            "final_edge_id": last_edge_id,
        }

    def _save_train_csvs(self, output_path: Path) -> None:
        """列車ごとのCSVを train_data/ ディレクトリ以下に保存"""
        train_data_dir = output_path / "train_data"
        train_data_dir.mkdir(exist_ok=True)

        for train in self._trains:
            self._save_train_csv(train, train_data_dir)

    @staticmethod
    def _build_route_offsets(train: Train) -> dict[str, float]:
        """ルート上の各エッジの累積オフセットを計算

        ルートの先頭エッジを 0km 基準とし、各エッジの開始位置（km）を返します。

        例: edge1(10km) → edge2(8km) → edge3(5km) の場合
            {"edge1": 0.0, "edge2": 10.0, "edge3": 18.0}
        """
        offsets: dict[str, float] = {}
        cumulative = 0.0
        for edge in train.get_route()._edges:
            offsets[edge.id] = cumulative
            cumulative += edge.length
        return offsets

    def _save_train_csv(self, train: Train, train_data_dir: Path) -> None:
        """1列車分のCSVを保存"""
        rd = train.get_running_data()
        statuses = rd.get_status_all()
        velocities = rd.get_velocity_all()
        accelerations = rd.get_acceleration_all()
        edge_ids = rd.get_edge_id_all()
        positions = rd.get_position_value_all()

        total_steps = len(statuses)

        # ファイル名に使用できない文字を置換
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", train.name)
        file_path = train_data_dir / f"{safe_name}.csv"

        route_offsets = self._build_route_offsets(train)

        def _get(lst: list, i: int):
            return lst[i] if i < len(lst) else None

        def _cum_pos(edge_id: str | None, position: float | None) -> float | None:
            if edge_id is None or position is None:
                return None
            offset = route_offsets.get(edge_id)
            return None if offset is None else position + offset

        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "time_s",
                    "status",
                    "status_name",
                    "edge_id",
                    "position_km",
                    "cumulative_position_km",
                    "velocity_kmh",
                    "acceleration_ms2",
                ]
            )

            for step in range(total_steps):
                status = _get(statuses, step)
                position = _get(positions, step)
                edge_id = _get(edge_ids, step)
                velocity = _get(velocities, step)
                acceleration = _get(accelerations, step)
                cum_position = _cum_pos(edge_id, position)

                writer.writerow(
                    [
                        step,
                        round(step * self._step_size, 6),
                        status if status is not None else "",
                        _STATUS_NAMES.get(status, "") if status is not None else "",
                        edge_id or "",
                        position if position is not None else "",
                        cum_position if cum_position is not None else "",
                        velocity if velocity is not None else "",
                        acceleration if acceleration is not None else "",
                    ]
                )

        logger.info(f"  列車CSVを保存しました: {file_path} ({total_steps} ステップ)")


def save_simulation_results(
    output_dir: str,
    trains: list[Train],
    step_size: float,
    total_steps: int,
    line_shape: LineShape,
    block_system_type: str = "fixed",
    name: str = "",
    description: str = "",
    author: str = "",
) -> None:
    """シミュレーション結果を指定ディレクトリに保存する便利関数

    以下の構成でファイルを出力します：

        output_dir/
        ├── initial_conditions.json
        ├── results_summary.json
        └── train_data/
            ├── <列車名>.csv
            └── ...

    Args:
        output_dir: 出力ディレクトリのパス
        trains: シミュレーション済みの列車リスト
        step_size: ステップサイズ（秒）
        total_steps: 総ステップ数
        line_shape: 路線形状
        block_system_type: 閉塞システムの種類 ("fixed" または "moving")
        name: モデル名
        description: モデルの説明
        author: 作成者名
    """
    saver = SimulationResultSaver(
        trains, step_size, total_steps, line_shape, block_system_type
    )
    saver.save(output_dir, name, description, author)
