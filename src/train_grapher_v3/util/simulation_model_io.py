"""シミュレーションモデルのJSON保存・読込機能

このモジュールはシミュレーションモデルをJSON形式で保存・読込する機能を提供します。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from train_grapher_v3.core.line_shape import (
    Block,
    Curve,
    Edge,
    Grade,
    LineShape,
    Node,
    Station,
)
from train_grapher_v3.core.train import StationStopTime, Train
from train_grapher_v3.core.train_parameter import TrainParameter
from train_grapher_v3.util.logger import setup_logger

logger = setup_logger(__name__)


class SimulationModelEncoder:
    """シミュレーションモデルをJSON形式にエンコードするクラス"""

    def __init__(
        self,
        line_shape: LineShape,
        trains: list[Train],
        step_size: float,
        total_steps: int,
    ):
        """初期化

        Args:
            line_shape: 路線形状
            trains: 列車リスト
            step_size: ステップサイズ（秒）
            total_steps: 総ステップ数
        """
        self.line_shape = line_shape
        self.trains = trains
        self.step_size = step_size
        self.total_steps = total_steps

    def encode(
        self,
        name: str,
        description: str = "",
        author: str = "",
        block_system_type: str = "fixed",
    ) -> dict[str, Any]:
        """シミュレーションモデルをJSON形式の辞書にエンコード

        Args:
            name: モデルの名前
            description: モデルの説明
            author: 作成者名
            block_system_type: 閉塞システムの種類 ("fixed" または "moving")

        Returns:
            JSON形式の辞書
        """
        now = datetime.now().isoformat()

        return {
            "metadata": {
                "version": "1.0",
                "name": name,
                "description": description,
                "author": author,
                "created_at": now,
                "updated_at": now,
            },
            "simulation_config": {
                "step_size": self.step_size,
                "total_steps": self.total_steps,
                "block_system_type": block_system_type,
            },
            "line_shape": self._encode_line_shape(),
            "trains": self._encode_trains(),
        }

    def _encode_line_shape(self) -> dict[str, Any]:
        """路線形状をエンコード"""
        return {
            "nodes": self._encode_nodes(),
            "edges": self._encode_edges(),
        }

    def _encode_nodes(self) -> list[dict[str, Any]]:
        """ノードをエンコード"""
        nodes = []
        for node in self.line_shape._nodes:
            nodes.append(
                {
                    "id": node._id,
                    "offset": node.offset if hasattr(node, "offset") else 0.0,
                }
            )
        return nodes

    def _encode_edges(self) -> list[dict[str, Any]]:
        """エッジをエンコード"""
        edges = []
        for edge in self.line_shape._edges:
            edges.append(
                {
                    "id": edge.id,
                    "length": edge.length,
                    "start_node_id": edge._start_node._id,
                    "end_node_id": edge._end_node._id,
                    "grades": [
                        {"start": g.start, "end": g.end, "grade": g.grade}
                        for g in edge._grade
                    ],
                    "curves": [
                        {"start": c.start, "end": c.end, "curve": c.curve}
                        for c in edge._curve
                    ],
                    "stations": [
                        {"id": s.id, "value": s._value, "name": s.name}
                        for s in edge.get_stations()
                    ],
                    "blocks": [
                        {"start": b.start, "speed_limits": b.speed_limits}
                        for b in edge.get_block_list()
                    ],
                }
            )
        return edges

    def _encode_trains(self) -> list[dict[str, Any]]:
        """列車をエンコード"""
        trains = []
        for train in self.trains:
            # ルートのエッジIDを取得
            route = train.get_route()
            route_edge_ids = [edge.id for edge in route._edges]

            # 駅停車時間を取得
            station_stop_times = []
            for sst in train._station_stop_times:
                station_stop_times.append(
                    {
                        "station_id": sst.station_id,
                        "default_value": sst.default_value,
                        "departure_time": sst.departure_time,
                    }
                )

            # スタート条件を取得
            start_condition = None
            if train._start_step is not None or train._start_position is not None:
                start_condition = {}
                if train._start_step is not None:
                    start_condition["step"] = train._start_step
                if train._start_position is not None:
                    pos, edge = train._start_position.get_position()
                    start_condition["position"] = pos
                    start_condition["edge_id"] = edge.id

            # エンド条件を取得
            end_condition = None
            if train._end_position is not None:
                pos, edge = train._end_position.get_position()
                end_condition = {
                    "position": pos,
                    "edge_id": edge.id,
                }

            train_data = {
                "id": train.name,  # nameをIDとして使用
                "name": train.name,
                "route_edge_ids": route_edge_ids,
                "initial_position": 0.0,  # Train クラスに初期位置情報がないため0.0
                "initial_velocity": 0.0,  # Train クラスに初期速度情報がないため0.0
                "use_timetable": train._use_timetable,
                "end_edge_id": train._end_edge_id,
                "end_position_value": train._end_position_value,
                "parameters": {
                    "wight": train._train_parameter.wight,
                    "factor_of_inertia": train._train_parameter.factor_of_inertia,
                    "decelerating_acceleration": train._train_parameter.decelerating_acceleration,
                    "decelerating_acceleration_station": train._train_parameter.decelerating_acceleration_station,
                    "fast_margine": train._train_parameter.fast_margine,
                    "slow_margine": train._train_parameter.slow_margine,
                    "length": train._train_parameter.length,
                },
                "station_stop_times": station_stop_times,
            }

            # start_conditionとend_conditionを追加（Noneでない場合のみ）
            if start_condition is not None:
                train_data["start_condition"] = start_condition
            if end_condition is not None:
                train_data["end_condition"] = end_condition

            trains.append(train_data)
        return trains


class SimulationModelDecoder:
    """JSON形式からシミュレーションモデルをデコードするクラス"""

    def __init__(self, json_data: dict[str, Any]):
        """初期化

        Args:
            json_data: JSON形式の辞書
        """
        self.json_data = json_data
        self._validate()

    def decode(self) -> tuple[LineShape, list[Train], float, int, str]:
        """JSON形式のデータをデコード

        Returns:
            (line_shape, trains, step_size, total_steps, block_system_type) のタプル
        """
        line_shape = self._decode_line_shape()
        trains = self._decode_trains(line_shape)
        step_size = self.json_data["simulation_config"]["step_size"]
        total_steps = self.json_data["simulation_config"]["total_steps"]
        block_system_type = self.json_data["simulation_config"]["block_system_type"]

        return line_shape, trains, step_size, total_steps, block_system_type

    def _validate(self) -> None:
        """JSON形式のバリデーション"""
        # 必須トップレベルフィールド
        required_top = ["metadata", "simulation_config", "line_shape", "trains"]
        for field in required_top:
            if field not in self.json_data:
                raise ValueError(f"必須フィールド '{field}' がありません")

        # メタデータの検証
        metadata = self.json_data["metadata"]
        if "version" not in metadata or metadata["version"] != "1.0":
            raise ValueError("JSON形式のバージョンが一致しません（1.0が必須）")

        # simulation_configの検証
        config = self.json_data["simulation_config"]
        if (
            "step_size" not in config
            or "total_steps" not in config
            or "block_system_type" not in config
        ):
            raise ValueError("simulation_configに必須フィールドがありません")

        # line_shapeの検証
        line_shape = self.json_data["line_shape"]
        if "nodes" not in line_shape or "edges" not in line_shape:
            raise ValueError("line_shapeに必須フィールド (nodes, edges) がありません")

        # ノードIDとエッジの相互参照検証
        node_ids = {n["id"] for n in line_shape["nodes"]}
        edge_ids = {e["id"] for e in line_shape["edges"]}

        for edge in line_shape["edges"]:
            if edge["start_node_id"] not in node_ids:
                raise ValueError(
                    f"エッジ '{edge['id']}' の開始ノード '{edge['start_node_id']}' が見つかりません"
                )
            if edge["end_node_id"] not in node_ids:
                raise ValueError(
                    f"エッジ '{edge['id']}' の終了ノード '{edge['end_node_id']}' が見つかりません"
                )

        # 列車のエッジ参照検証
        for train in self.json_data["trains"]:
            for edge_id in train.get("route_edge_ids", []):
                if edge_id not in edge_ids:
                    raise ValueError(
                        f"列車 '{train['id']}' のルートにあるエッジ '{edge_id}' が見つかりません"
                    )

        logger.info("JSON形式のバリデーション完了")

    def _decode_line_shape(self) -> LineShape:
        """路線形状をデコード"""
        line_shape_data = self.json_data["line_shape"]

        # ノードをデコード
        nodes = {}
        for node_data in line_shape_data["nodes"]:
            node = Node(node_data["id"], offset=node_data.get("offset", 0.0))
            nodes[node_data["id"]] = node

        # エッジをデコード
        edges = []
        for edge_data in line_shape_data["edges"]:
            # 勾配をデコード
            grades = [
                Grade(start=g["start"], end=g["end"], grade=g["grade"])
                for g in edge_data.get("grades", [])
            ]

            # 曲線をデコード
            curves = [
                Curve(start=c["start"], end=c["end"], curve=c["curve"])
                for c in edge_data.get("curves", [])
            ]

            # 駅をデコード
            stations = [
                Station(id=s["id"], value=s["value"], name=s["name"])
                for s in edge_data.get("stations", [])
            ]

            # ブロックをデコード
            blocks = [
                Block(
                    start=b["start"],
                    speed_limits=[
                        float(sl) if isinstance(sl, str) else sl
                        for sl in b["speed_limits"]
                    ],
                )
                for b in edge_data.get("blocks", [])
            ]

            # エッジを作成
            edge = Edge(
                id=edge_data["id"],
                length=edge_data["length"],
                start_node=nodes[edge_data["start_node_id"]],
                end_node=nodes[edge_data["end_node_id"]],
                grade=grades,
                curve=curves,
                stations=stations,
                block_list=blocks,
            )
            edges.append(edge)

        return LineShape(nodes=list(nodes.values()), edges=edges)

    def _decode_trains(self, line_shape: LineShape) -> list[Train]:
        """列車をデコード"""
        trains = []
        for train_data in self.json_data["trains"]:
            # ルートを取得
            route = line_shape.get_route(train_data["route_edge_ids"])

            # 列車性能パラメータをデコード
            param_data = train_data.get("parameters", {})
            parameters = TrainParameter(
                wight=param_data.get("wight", 389.85),
                factor_of_inertia=param_data.get("factor_of_inertia", 24.54),
                decelerating_acceleration=param_data.get(
                    "decelerating_acceleration", -3.0
                ),
                decelerating_acceleration_station=param_data.get(
                    "decelerating_acceleration_station", -4.0
                ),
                fast_margine=param_data.get("fast_margine", 6.0),
                slow_margine=param_data.get("slow_margine", 15.0),
                length=param_data.get("length", 0.2),
            )

            # 駅停車時間をデコード
            station_stop_times = [
                StationStopTime(
                    station_id=sst["station_id"],
                    default_value=sst["default_value"],
                    departure_time=sst.get("departure_time", None),
                )
                for sst in train_data.get("station_stop_times", [])
            ]

            # ダイヤ制御フラグを取得
            use_timetable = train_data.get("use_timetable", False)

            # 終了位置をデコード
            end_edge_id = train_data.get("end_edge_id", None)
            end_position_value = train_data.get("end_position_value", None)

            # 列車を作成
            train = Train(
                name=train_data.get("name", train_data["id"]),
                line_shape=line_shape,
                route=route,
                train_parameter=parameters,
                station_stop_times=station_stop_times,
                use_timetable=use_timetable,
                end_edge_id=end_edge_id,
                end_position_value=end_position_value,
            )

            # スタート条件を設定
            start_condition = train_data.get("start_condition")
            if start_condition is not None:
                start_step = start_condition.get("step")
                start_position = None
                if "position" in start_condition and "edge_id" in start_condition:
                    edge = line_shape.get_edge_by_id(start_condition["edge_id"])
                    from train_grapher_v3.core.line_shape import Position

                    start_position = Position(start_condition["position"], edge)
                train.set_start_condition(start_step, start_position)
            else:
                # デフォルト値を設定
                train.set_start_condition(None, None)

            # エンド条件を設定
            end_condition = train_data.get("end_condition")
            if end_condition is not None:
                edge = line_shape.get_edge_by_id(end_condition["edge_id"])
                from train_grapher_v3.core.line_shape import Position

                end_position = Position(end_condition["position"], edge)
                train.set_end_condition(end_position)
            else:
                # デフォルト値（ルート終端）を設定
                train.set_end_condition(None)

            trains.append(train)

        return trains


def save_simulation_model(
    file_path: str,
    line_shape: LineShape,
    trains: list[Train],
    step_size: float,
    total_steps: int,
    name: str,
    description: str = "",
    author: str = "",
    block_system_type: str = "fixed",
) -> None:
    """シミュレーションモデルをJSON形式で保存

    Args:
        file_path: 保存ファイルのパス
        line_shape: 路線形状
        trains: 列車リスト
        step_size: ステップサイズ（秒）
        total_steps: 総ステップ数
        name: モデルの名前
        description: モデルの説明
        author: 作成者名
        block_system_type: 閉塞システムの種類
    """
    encoder = SimulationModelEncoder(line_shape, trains, step_size, total_steps)
    json_data = encoder.encode(name, description, author, block_system_type)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"シミュレーションモデルを保存しました: {file_path}")


def load_simulation_model(
    file_path: str,
) -> tuple[LineShape, list[Train], float, int, str]:
    """JSON形式のシミュレーションモデルを読込

    Args:
        file_path: 読込ファイルのパス

    Returns:
        (line_shape, trains, step_size, total_steps, block_system_type) のタプル
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    decoder = SimulationModelDecoder(json_data)
    line_shape, trains, step_size, total_steps, block_system_type = decoder.decode()

    logger.info(f"シミュレーションモデルを読込しました: {file_path}")
    logger.info(f"  - 列車数: {len(trains)}")
    logger.info(f"  - エッジ数: {len(line_shape._edges)}")
    logger.info(f"  - ステップサイズ: {step_size} 秒")
    logger.info(f"  - 総ステップ数: {total_steps}")

    return line_shape, trains, step_size, total_steps, block_system_type
