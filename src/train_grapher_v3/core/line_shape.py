"""路線管理モジュール

* 路線の形状、長さ
* 勾配、カーブ
"""

from __future__ import annotations

from dataclasses import dataclass

from train_grapher_v3.util.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Block:
    start: float
    speed_limits: list[float]


@dataclass
class Grade:
    start: float = 0
    end: float = 0
    grade: float = 0


@dataclass
class Curve:
    start: float = 0
    end: float = 0
    curve: float = 0


class Node:
    """分岐点のクラス"""

    def __init__(self, id: str, *, offset: float = None) -> None:
        """コンストラクタ

        Args:
            id (str): 分岐点id 任意の一意のIDを振る
        """
        self._id = id

        # 始点のある辺
        self._start_edge: list[Edge] = []
        self.offset: float = offset


class Edge:
    """分岐点をつなぐ路線のクラス"""

    def __init__(
        self,
        id: str,
        length: float,
        start_node: Node,
        end_node: Node,
        grade: list[Grade],
        curve: list[Curve],
        *,
        stations: list[Station] = [],
        block_list: list[Block] = [],
    ) -> None:
        """コンストラクタ

        Args:
            id (str): 路線id 任意の一意のIDを振る
            length (float): 路線長(km)
            start_node (Node): スタートの分岐点
            end_node (Node): 終了の分岐点
            grade (list[float]): 勾配情報
        """
        self._id = id
        self._length = length
        self._start_node = start_node
        self._end_node = end_node
        self._grade = grade
        self._curve = curve
        self._statios = stations
        self._block_list = block_list

    @property
    def id(self) -> str:
        return self._id

    @property
    def length(self) -> float:
        return self._length

    @id.setter
    def id(self, value) -> None:
        self._id = value

    @length.setter
    def length(self, value) -> None:
        self._length = value

    def get_curve(self, value) -> float:
        for curve in self._curve:
            if curve.start <= value <= curve.end:
                return curve.curve

        return 0.0

    def get_grade(self, value) -> float:
        for grade in self._grade:
            if grade.start <= value <= grade.end:
                return grade.grade

        return 0.0

    def get_length(self) -> float:
        return self._length

    def set_stations(self, stations: list[Station]) -> None:
        self._statios = stations

    def get_stations(self) -> list[Station]:
        return self._statios

    def get_block_num(self) -> int:
        return len(self._block_list)

    def get_block_list(self) -> list[Block]:
        return self._block_list


class Route:
    """路線形状の一部のクラス。ルートを示すのに使う"""

    def __init__(self, edges: list[Edge]):
        self._edges = edges

    def __len__(self):
        """ルートの長さを返す"""
        return len(self._edges)

    def __getitem__(self, index) -> Edge:
        """インデックスによるエッジへのアクセスを提供"""
        return self._edges[index]

    def get_index_by_edge_id(self, id: str) -> int:
        """ルートの何番目に指定したIDのEdgeがあるかを取得する関数。

        Args:
            id (str): 指定するID。

        Returns:
            int: 見つかったID。見つからない場合-1を返す。
        """
        for index, edge in enumerate(self._edges):
            if edge.id == id:
                return index
        return -1

    def get_next_edge(self, current_edge: Edge) -> Edge | None:
        """登録されているルートで次のedgeを取得する

        Args:
            current_edge_id (str): 現在のedgeのID

        Returns:
            Edge: 次のedge。ない場合はNone
        """
        # 現在のedgeのインデックス
        current_edge_index = self.get_index_by_edge_id(current_edge.id)
        # 次のedgeのインデックス
        next_edge_index = current_edge_index + 1

        # はみ出していないかチェックする
        if next_edge_index >= len(self._edges):
            # はみ出していた場合、Noneを返す
            return None

        # 次のedgeを返す。
        return self._edges[next_edge_index]

    def get_distance(self, position1: Position, position2: Position) -> float | None:
        """ルート上の距離を求める。

        Args:
            position1 (Position): 先行列車の位置
            position2 (Position): 後続列車の位置

        Returns:
            float: _description_
        """
        index1 = self.get_index_by_edge_id(position1.edge_id)
        index2 = self.get_index_by_edge_id(position2.edge_id)

        if index1 == -1 or index2 == -1:
            return None

        # おんなじEdgeにいるとき
        if index1 == index2:
            return position1.value - position2.value

        # 先行列車が前にいた場合
        elif index2 < index1:
            midway_length = 0
            for index in range(index2 + 1, index1):
                midway_length += self._edges[index].length

            return (
                position1.value
                + midway_length
                + position2.edge.length
                - position2.value
            )

        # 先行列車が後にいた場合
        elif index2 > index1:
            midway_length = 0
            for index in range(index1 + 1, index2):
                midway_length += self._edges[index].length

            return -(position2.value + midway_length + position1.edge.length)

    def get_block_diff(self, position1: Position, position2: Position) -> int | None:
        """閉塞が何個離れているか

        Args:
            position1 (Position): 先行列車の位置
            position2 (Position): 後続列車の位置

        Returns:
            int | None: 離れている個数。ルート上に列車がいない場合はNone。後続列車が追い越している場合もNone。
        """
        index1 = self.get_index_by_edge_id(position1.edge_id)
        index2 = self.get_index_by_edge_id(position2.edge_id)

        if index1 == -1 or index2 == -1:
            return None

        block_index1 = position1.get_block_index()
        block_index2 = position2.get_block_index()

        if index1 == index2:
            diff = block_index1 - block_index2
            return None if diff < 0 else diff

        elif index2 < index1:
            midway_count = 0
            for index in range(index2 + 1, index1):
                midway_count += self._edges[index].get_block_num()
            return (
                block_index1
                + midway_count
                + (position2.edge.get_block_num() - block_index2)
            )

        else:
            return None

    def get_start_position(self) -> Position:
        """このルートの始点

        Returns:
            Positon: 始点のPosition
        """
        edge = self[0]
        value = 0
        return Position(value, edge)

    def get_end_position(self) -> Position:
        """このルートの終点

        Returns:
            Positon: 終点のPosition
        """
        edge = self[-1]
        value = edge.length
        return Position(value, edge)

    def get_next_station(self, postion: Position) -> Station | None:
        current_index = self.get_index_by_edge_id(postion.edge_id)
        if current_index is None:
            return None

        min_distance_station = None
        min_distance = float("inf")

        for index in range(current_index, len(self._edges)):
            edge = self._edges[index]
            stations = edge.get_stations()

            for station in stations:
                distance = self.get_distance(station.get_position(edge), postion)
                if (
                    (distance is not None)
                    and (distance > 0)
                    and (distance < min_distance)
                    and ()
                ):
                    min_distance_station = station
                    min_distance = distance

            if min_distance_station is not None:
                break

        return min_distance_station

    def get_next_block_position(self, position: Position) -> Position | None:
        current_block_index = position.get_block_index()

        if position.edge.get_block_num() <= current_block_index + 1:
            next_edge = self.get_next_edge(position.edge)
            if next_edge is None:
                return None

            return Position(next_edge.get_block_list()[0].start, next_edge)

        return Position(
            position.edge.get_block_list()[current_block_index + 1].start, position.edge
        )

    def get_next_block(self, position: Position) -> Block | None:
        current_block_index = position.get_block_index()

        if position.edge.get_block_num() <= current_block_index + 1:
            next_edge = self.get_next_edge(position.edge)
            if next_edge is None:
                return None

            return next_edge.get_block_list()[0]

        return position.edge.get_block_list()[current_block_index + 1]


class Position:
    """路線形状の位置の管理クラス"""

    def __init__(self, position: float, edge: Edge) -> None:
        """位置を管理、制御

        Args:
            position (float): その路線の視点からの距離[km].
            route (Edge): どの路線かを管理する変数.
        """
        self._edge = edge
        self._position = position

    def get_position(self) -> tuple[float, Edge]:
        """現在の位置の取得

        Returns:
            tuple[float, Node | None]: _description_
        """
        return (self._position, self._edge)

    def update_position(self, value: float, route: Route) -> tuple[float, Edge]:
        """位置の更新

        Args:
            value (float): 増加量、マイナス値を入れると動かない
            route (Route): 通るルート

        Returns:
            tuple[float, Edge]: 更新した位置とEdge。ルートの最後をはみ出た場合は最後の位置になる。
        """
        if value < 0:
            return (self._position, self._edge)

        # 距離が増加してもedgeが変わらない場合
        if self._edge.length >= (self._position + value):
            # 距離を足す edgeは変化させない
            self._position = self._position + value
            return (self._position, self._edge)

        # 距離が増加してedgeが変わる場合
        # 距離の増加分
        delta_value = value

        # 現在のedgeidとvalue
        current_edge = self._edge
        current_value = self._position

        # 更新結果がedgeの長さを超えたら
        while current_edge.length < current_value + delta_value:
            # つぎのedge
            next_edge = route.get_next_edge(self._edge)

            # つぎのedgeがない場合
            if next_edge is None:
                # 最後のedgeの最後の距離を返す
                self._position = current_edge.length
                self._edge = current_edge
                return (self._position, self._edge)

            # はみ出た分をdelta_valueにする
            delta_value = delta_value - (current_edge.length - current_value)
            current_value = 0
            # 現在のedgeを更新
            current_edge = next_edge

        # はみ出た距離を現在の距離にする
        self._position = delta_value
        # 現在のedgeを更新
        self._edge = current_edge

        return (self._position, self._edge)

    @property
    def edge_id(self) -> str:
        return self._edge.id

    @property
    def edge(self) -> Edge:
        return self._edge

    @property
    def value(self) -> float:
        return self._position

    def get_grade(self) -> float:
        return self._edge.get_grade(self.value)

    def get_curve(self) -> float:
        return self._edge.get_curve(self.value)

    def get_block_index(self) -> int:
        """今いるエッジで何番目の閉塞にいるかを返す

        Returns:
            int: _description_
        """
        block_num = self.edge.get_block_num()
        if block_num <= 1:
            return 0

        block_list = self.edge.get_block_list()

        for index in range(block_num - 1):
            if block_list[index].start <= self._position <= block_list[index + 1].start:
                return index

        return block_num - 1

    def get_block(self) -> Block | None:
        block_num = self.edge.get_block_num()
        if block_num == 0:
            return None

        block_list = self.edge.get_block_list()

        if block_num == 1:
            return block_list[0]

        for index in range(block_num - 1):
            if block_list[index].start <= self._position <= block_list[index + 1].start:
                return block_list[index]

        return block_list[block_num - 1]


class LineShape:
    """路線形状のクラス"""

    def __init__(self, *, nodes: list[Node] = [], edges: list[Edge] = []) -> None:
        """コンストラクタ

        Args:
            nodes (list[Node], optional): 分岐点(始点、終点も含む). Defaults to [].
            edges (list[Edge], optional): 分岐点をつなぐ路線. Defaults to [].
        """
        self._nodes = nodes
        self._edges = edges

    def add_edge(self, edge: Edge) -> None:
        self._edges.append(edge)

    def _add_node(self, node: Node) -> None:
        self._nodes.append(node)

    def get_edge_by_id(self, id: str) -> Edge | None:
        for edge in self._edges:
            if edge.id == id:
                return edge
        return None

    def get_node_by_id(self, id: str) -> Node:
        for node in self._nodes:
            if node.id == id:
                return node
        return None

    def get_position(self, edge_id: str, value: float) -> Position:
        edge = self.get_edge_by_id(edge_id)
        return Position(value, edge)

    def get_route(self, edge_ids: list[str]) -> Route:
        edges = []

        for edge_id in edge_ids:
            edge = self.get_edge_by_id(edge_id)

            if edge is None:
                logger.error("No edges matching the entered edge_id were found.")
                raise ValueError("No edges matching the entered edge_id were found.")

            edges.append(edge)

        return Route(edges)


class Station:
    def __init__(self, id: str, value: float, *, name: str = None) -> None:
        self._value = value
        self._id = id
        self._name = id if name is None else name

    def get_position(self, edge: Edge) -> Position:
        return Position(self._value, edge)

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name
