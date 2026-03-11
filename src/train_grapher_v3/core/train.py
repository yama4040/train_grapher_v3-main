"""列車管理モジュール
* traingrapherv3用trainクラスの作成
* 列車の動作(物理演算)
* データの管理
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from train_grapher_v3.core.driving_decision import (
    DefaultDrivingDecision,
    DrivingDecision,
)
from train_grapher_v3.core.line_shape import Edge, LineShape, Position, Route, Station
from train_grapher_v3.core.runningdata import RunningData
from train_grapher_v3.core.status import Status
from train_grapher_v3.core.train_parameter import TrainParameter
from train_grapher_v3.util.logger import setup_logger

if TYPE_CHECKING:
    from train_grapher_v3.core.block_system import SignalInstruction

logger = setup_logger(__name__)


@dataclass
class StationStopTime:
    """駅停車時間管理

    ダイヤ制御に対応。ダイヤ出発時刻を設定すると、停車時間を満たしつつ、
    ダイヤ出発時刻より早く出発しない制御が可能。
    """

    station_id: str = 0
    default_value: float = 60
    count: float = field(init=False)
    departure_time: float | None = (
        None  # ダイヤ出発時刻[s]（Noneの場合はダイヤ制御なし）
    )
    arrival_time: float | None = None  # 実到着時刻[s]（シミュレーション中に設定）
    use_timetable: bool = False  # ダイヤ制御を使用するか

    def __post_init__(self):
        self.count = self.default_value

    def is_stop(self, step: int, step_size: float) -> bool:
        """停車すべきか判定

        Args:
            step: 現在のステップ
            step_size: ステップサイズ[s]

        Returns:
            bool: 停車中の場合はTrue
        """
        current_time = step * step_size

        # ダイヤ制御が無効の場合は従来通り
        if not self.use_timetable or self.departure_time is None:
            return self.count > 0

        # ダイヤ制御が有効の場合：
        # 1. 停車時間がまだ残っている、または
        # 2. ダイヤ出発時刻に達していない
        return (self.count > 0) or (current_time < self.departure_time)

    def decrement_stop_count(self, step_size: float) -> None:
        """停車時間をデクリメント"""
        self.count -= step_size

    def set_arrival_time(self, arrival_time: float) -> None:
        """実到着時刻を設定"""
        self.arrival_time = arrival_time


class Train:
    """列車の管理クラス"""

    def __init__(
        self,
        name: str,
        line_shape: LineShape,
        route: Route,
        train_parameter: TrainParameter,
        *,
        station_stop_times: list[StationStopTime] = [],
        driving_decision: DrivingDecision | None = None,
        use_timetable: bool = False,
        end_edge_id: str | None = None,
        end_position_value: float | None = None,
    ) -> None:
        """コンストラクタ

        Args:
            name (str): 列車名
            line_shape (LineShape): 路線形状
            route (Route): ルート
            train_parameter (TrainParameter): 列車性能パラメータ
            station_stop_times (list[StationStopTime]): 駅停車時間リスト
            driving_decision (DrivingDecision | None): 運転判断クラスのインスタンス。Noneの場合はDefaultDrivingDecisionを使用。
            use_timetable (bool): ダイヤ制御を使用するか（デフォルト: False）
            end_edge_id (str | None): 終了位置のエッジID（Noneの場合はルート終了位置）
            end_position_value (float | None): 終了位置のエッジ上の位置(km)
        """
        self._name = name
        self._line_shape = line_shape
        self._route = route
        self._train_parameter = train_parameter
        self._station_stop_times = station_stop_times
        self._running_data = RunningData()
        self._start_step: int | None = None
        self._start_position: Position | None = None
        self._end_position: Position | None = route.get_end_position()
        self._end_flag: bool = False
        self._use_timetable = use_timetable
        self._end_edge_id = end_edge_id
        self._end_position_value = end_position_value

        # 終了位置が指定されている場合は設定
        if end_edge_id is not None and end_position_value is not None:
            end_pos = line_shape.get_position(end_edge_id, end_position_value)
            if end_pos is not None:
                self._end_position = end_pos

        # 駅停車時間にダイヤ制御フラグを設定
        for sst in self._station_stop_times:
            sst.use_timetable = use_timetable

        # 運転判断クラスを設定
        self._driving_decision = driving_decision or DefaultDrivingDecision()

    def set_start_condition(self, step: int | None, position: Position | None) -> None:
        self._start_step = step
        self._start_position = position

    def get_end_flag(self) -> bool:
        return self._end_flag

    def set_end_condition(self, position: Position | None) -> None:
        if position is None:
            self._end_position = self._route.get_end_position()
        else:
            self._end_position = position

    def calculate_step(
        self, step: int, step_size: float, signal_instruction: "SignalInstruction"
    ) -> None:
        """1ステップ分を計算する

        信号システムからの指示を受け取り、運転手として運転判断を行う。

        Args:
            step (int): 計算を行うステップ
            step_size (float): ステップサイズ[s]
            signal_instruction (SignalInstruction): 信号システムからの指示
        """
        # 運転判断を行ってステータスを決定
        status = self._driving_decision.decide(self, step, signal_instruction)

        # ステータスから加速度を取得
        acceleration = self._calc_acceleration(step, status)

        # 位置と速度を計算
        position, velocity = self._physics_calculation(step, step_size, acceleration)

        # 誤差のため、停車判定の時は完全に停車させる
        if (
            status == Status.STOPPING_STATION
            or status == Status.OUT_OF_SERVICE
            or status == Status.CHECK_SAFETY
        ):
            acceleration = 0
            velocity = 0

        # 走行データの更新
        self._update_running_data(step, status, acceleration, velocity, position)

        if self._is_end(position):
            self._end_flag = True

    def _is_end(self, position: Position | None) -> bool:
        if position is None:
            return False

        distance = self._route.get_distance(position, self._end_position)

        if distance is None:
            return False

        if distance >= 0:
            return True

        return False

    def _get_run_resistance(self, velocity: float) -> float:
        """走行抵抗

        Args:
            velocity (float): 速度

        Returns:
            float: 走行抵抗
        """
        # return (2.089 + 0.0394 * velocity + 0.000675 * velocity**2) / self._train_parameter.wight
        return (
            (2.089 + 0.0394 * velocity + 0.000675 * velocity**2)
            / self._train_parameter.wight
            * 150
        )

    def _calc_resistance(self, edge: Edge, value: float) -> float:
        """路線の抵抗を計算する

        抵抗 = 勾配＋曲線

        Args:
            position (Position): 計算を行う位置
        Returns:
            float: 路線の抵抗
        """

        # 勾配抵抗はパーミルそのまま
        grade_resistance = edge.get_grade(value)

        # 曲線抵抗
        curve = edge.get_curve(value)
        curve_resistance = 800 / curve if curve != 0 else 0

        # return (grade_resistance + curve_resistance)
        return (grade_resistance + curve_resistance) * 1

    def _calc_tractive_effort(self, velocity: float) -> float:
        """引張力の計算 走行時は走行抵抗を引く

        Args:
            velocity (float): 速度

        Returns:
            float: 引張力
        """
        if velocity <= 50:
            result = 374752

        else:
            result = 76.513 * velocity**2.0 - 16401.0 * velocity + 949827.0

        return result / 9.8 / self._train_parameter.wight

    def _calc_acceleration(self, step: int, status: int) -> float | None:
        """ステータスから、加速度を計算する
        カーブや勾配、車体性能、モータ性能から加速度を求める
        減速する場合はマイナス
        前ステップのデータが欠損していた場合、0を返す

        Args:
            step (int): 計算を行うステップ
            status (int): 列車のステータス

        Returns:
            float: 加速度
        """
        # データに欠損があるか確かめるフラグ
        is_none = False

        # ステップ0は前の値が欠損してる判定
        if step == 0:
            is_none = True

        # 前ステップの情報取得
        # 欠損値があった場合はフラグを立てる
        if not is_none:
            edge_id = self._running_data.get_edge_id(step - 1)
            if edge_id is None:
                is_none = True

        if not is_none:
            value = self._running_data.get_position_value(step - 1)
            if value is None:
                is_none = True

        if not is_none:
            position = self._line_shape.get_position(edge_id, value)

        if not is_none:
            velocity = self._running_data.get_velocity(step - 1)
            if velocity is None:
                is_none = True

        # シミュレーション実施外
        if status == Status.NONE_SIMURATION:
            acceleration = None

        # 力行
        elif status == Status.POWER_RUN:
            if is_none:
                acceleration = 0
            else:
                tractive_effort = self._calc_tractive_effort(velocity)
                resistance = self._calc_resistance(position.edge, value)
                run_resistance = self._get_run_resistance(velocity)
                acceleration = (
                    tractive_effort - resistance - run_resistance
                ) / self._train_parameter.factor_of_inertia

        # 等速
        elif status == Status.STOPPING_STATION or status == Status.CONSTANT_SPEED:
            acceleration = 0.0

        # 安全確認中（駅間停車からの再発車前）
        elif status == Status.CHECK_SAFETY:
            acceleration = 0.0

        # 惰行
        elif status == Status.COASTING:
            if is_none:
                acceleration = 0
            else:
                tractive_effort = 0
                resistance = self._calc_resistance(position.edge, value)
                run_resistance = self._get_run_resistance(velocity)
                acceleration = (
                    tractive_effort - resistance - run_resistance - run_resistance
                ) / self._train_parameter.factor_of_inertia

        # 制限
        elif status == Status.BRAKE_OVERSPEED:
            if is_none:
                acceleration = 0
            else:
                acceleration = self._train_parameter.decelerating_acceleration

        elif status == Status.BRAKE_TRAIN or Status.BRAKE_STATION:
            if is_none:
                acceleration = 0
            else:
                acceleration = self._train_parameter.decelerating_acceleration_station

        return acceleration

    def _onestep_trapezoidal_rule(
        self, prev_val: float, curr_val: float, step_size: float
    ) -> float:
        """
        1ステップに台形則を用いて積分を行う。

        Parameters:
            prev_val (float): 前の値。
            curr_val (float): 現在の値。
            step_size (float): ステップサイズ。

        Returns:
            float: 積分された区間の値。
        """
        # integral = (prev_val + curr_val) * step_size * 0.5
        # return integral
        return curr_val * step_size

    def _physics_calculation(
        self, step: int, step_size: float, acceleration: float | None
    ) -> tuple[Position, float]:
        """列車の物理計算を行う

        Args:
            step (int): 計算を行うステップ
            step_size (float): _description_
            acceleration (float): _description_

        Returns:
            tuple[Position, float]: _description_
        """
        # 加速度がない場合はNoneを返す
        # シミュレーション実行外の時の処理
        if acceleration is None:
            return (None, None)

        # データに欠損があるか確かめるフラグ
        is_none = False
        if step == 0:
            is_none = True

        # 前ステップの情報の取得
        if not is_none:
            edge_id = self._running_data.get_edge_id(step - 1)
            if edge_id is None:
                is_none = True

        if not is_none:
            value = self._running_data.get_position_value(step - 1)
            if value is None:
                is_none = True

        if not is_none:
            previous_acceleration = self._running_data.get_acceleration(step - 1)
            if previous_acceleration is None:
                is_none = True

        if not is_none:
            previous_velocity = self._running_data.get_velocity(step - 1)
            if previous_velocity is None:
                is_none = True

        if not is_none:
            new_position = self._line_shape.get_position(edge_id, value)

        # データが欠損していた場合、初期位置からにする
        if is_none:
            if self._start_position is None:
                new_position = self._route.get_start_position()
            else:
                new_position = self._start_position
            previous_acceleration = 0
            previous_velocity = 0

        if previous_velocity < 0:
            previous_velocity = 0

        # 新しい位置と速度の計算
        new_velocity = previous_velocity + self._onestep_trapezoidal_rule(
            previous_acceleration, acceleration, step_size
        )

        delta_value = (
            self._onestep_trapezoidal_rule(previous_velocity, new_velocity, step_size)
            / 3600
        )
        new_position.update_position(delta_value, self._route)

        return (new_position, new_velocity)

    def _update_running_data(
        self,
        step: int,
        status: int | None,
        acceleration: float | None,
        velocity: float | None,
        position: Position | None,
    ) -> None:
        """走行データの格納

        Args:
            step (int): 格納するステップ
            status (int): ステータス
            acceleration (float): 加速度
            velocity (float): 速度
            position (Position): 位置
        """
        self._running_data.set_status(status, step)
        self._running_data.set_acceleration(acceleration, step)
        self._running_data.set_velocity(velocity, step)
        if position is None:
            self._running_data.set_edge_id(None, step)
            self._running_data.set_position_value(None, step)
        else:
            self._running_data.set_edge_id(position.edge_id, step)
            self._running_data.set_position_value(position.value, step)

    def get_position(self, step) -> Position | None:
        edge_id = self._running_data.get_edge_id(step)
        if edge_id is None:
            return None

        position_value = self._running_data.get_position_value(step)
        if position_value is None:
            return None

        return self._line_shape.get_position(edge_id, position_value)

    def get_route(self) -> Route:
        return self._route

    def get_running_data(self) -> RunningData:
        return self._running_data

    @property
    def name(self) -> str:
        return self._name

    def get_station_stop_time(self, stetion_id: str) -> StationStopTime | None:
        for station_stop_time in self._station_stop_times:
            if station_stop_time.station_id == stetion_id:
                return station_stop_time

        return None

    def get_before_station_info(self, step: int) -> tuple[Station | None, float | None]:
        if step == 0:
            return None, None

        # 前ステップの位置
        position = self.get_position(step - 1)
        if position is None:
            return None, None

        current_index = self._route.get_index_by_edge_id(position.edge_id)
        if current_index == -1:
            return None, None

        min_distance_station = None
        min_distance = float("inf")

        # 一番近い駅の検索
        for index in range(current_index, -1, -1):
            edge = self._route[index]
            stations = edge.get_stations()

            for station in stations:
                distance = self._route.get_distance(
                    position, station.get_position(edge)
                )
                stop_count = self.get_station_stop_time(station.id)

                if (
                    (distance is not None)
                    and (distance >= 0)
                    and (distance < min_distance)
                    and (stop_count is not None)
                    and (
                        stop_count.default_value > 0
                    )  # 停車時間が0より大きい駅のみ対象
                ):
                    min_distance_station = station
                    min_distance = distance

            if min_distance_station is not None:
                break

        if min_distance_station is None:
            return None, None

        return min_distance_station, min_distance

    def get_next_station_info(self, step: int) -> tuple[Station | None, float | None]:
        if step == 0:
            return None, None

        # 前ステップの位置
        position = self.get_position(step - 1)
        if position is None:
            return None, None

        current_index = self._route.get_index_by_edge_id(position.edge_id)
        if current_index == -1:
            return None, None

        min_distance_station = None
        min_distance = float("inf")

        # 一番近い駅の検索
        for index in range(current_index, len(self._route)):
            edge = self._route[index]
            stations = edge.get_stations()

            for station in stations:
                distance = self._route.get_distance(
                    station.get_position(edge), position
                )
                stop_count = self.get_station_stop_time(station.id)
                if (
                    (distance is not None)
                    and (distance > 0)
                    and (distance < min_distance)
                    and (stop_count is not None)
                    and (
                        stop_count.default_value > 0
                    )  # 停車時間が0より大きい駅のみ対象
                ):
                    min_distance_station = station
                    min_distance = distance

            if min_distance_station is not None:
                break

        if min_distance_station is None:
            return None, None

        return min_distance_station, min_distance
