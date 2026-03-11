from abc import ABC, abstractmethod
from dataclasses import dataclass

from train_grapher_v3.core.line_shape import (
    Block,
    LineShape,
    Station,
)
from train_grapher_v3.core.train import Train
from train_grapher_v3.util.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SignalInstruction:
    """信号システムからの指示情報

    信号システムは衝突防止のために必要な情報のみを提供する。
    運転判断は列車（運転手）が行う。
    """

    instruction_speed: float | None  # 指示速度[km/h] Noneの場合は制限なし
    is_station_stopping: bool  # 駅停車中かどうか
    must_stop_at_next_station: bool  # 次駅で停車しなければならないか
    next_block_distance: float | None  # 次の閉塞までの距離[km]
    next_block: Block | None  # 次の閉塞


class BlockSystem(ABC):
    """閉塞システムの抽象基底クラス

    信号システムとして、衝突防止に必要な情報を提供する。
    具体的な速度制御ロジックはサブクラスで実装する。
    """

    def __init__(self, line_shape: LineShape) -> None:
        """コンストラクタ

        Args:
            line_shape (LineShape): 路線形状
        """
        self._line_shape = line_shape

    def calculate_step(
        self, step: int, trains: list[Train], step_size: float
    ) -> list[SignalInstruction]:
        """1ステップの計算

        信号システムとして、衝突防止に必要な情報のみを提供する。

        Args:
            step (int): 計算するステップ
            trains (list[Train]): 計算する列車のリスト
            step_size (float): 1ステップのサイズ[s]

        Returns:
            list[SignalInstruction]: 各列車への指示情報のリスト
        """
        instruction_list = []

        for train in trains:
            preceding_train = self._get_preceding_train(step, trains, train)
            before_station, before_station_distance = train.get_before_station_info(
                step
            )
            next_station, next_station_distance = train.get_next_station_info(step)

            # 信号システムとしての指示を作成
            instruction = self._get_signal_instruction(
                step,
                step_size,
                train,
                preceding_train,
                before_station,
                before_station_distance,
                next_station,
                next_station_distance,
            )
            instruction_list.append(instruction)

        return instruction_list

    def _get_preceding_train(
        self, step: int, trains: list[Train], train: Train
    ) -> Train | None:
        """先行列車の取得

        Args:
            step (int): 計算を行うステップ
            trains (list[Train]): 走っている列車のリスト
            train (Train): 求めたい列車

        Returns:
            Train | None: 先行列車。見つからない場合はNoneを返す。
        """
        # データに欠損があるか確かめるフラグ
        is_none = False

        # ステップ0は前の値が欠損してる判定
        if step == 0:
            is_none = True

        # 1ステップ前の位置を取得
        if not is_none:
            position = train.get_position(step - 1)
            if position is None:
                is_none = True

        # 欠損値があった場合、Noneを返す
        if is_none:
            return None

        # ルートを取得
        route = train.get_route()

        # ルート上で最短距離の列車を探索
        min_distance_train = None
        min_distance = float("inf")

        for look_train in trains:
            # 自分自身はスキップ
            if train.name == look_train.name:
                continue

            look_train_position = look_train.get_position(step - 1)
            # データがない場合はスキップ
            if look_train_position is None:
                continue

            distance = route.get_distance(look_train_position, position)
            # ルート上にいない or 後ろにいる場合はスキップ
            if distance is None or distance <= 0:
                continue

            if distance < min_distance:
                min_distance_train = look_train
                min_distance = distance

        return min_distance_train

    def get_decelerate_distance(
        self, velocity: float, decelerate: float, target_speed: float
    ) -> float:
        """目標スピードまで落とすために必要な距離

        Args:
            velocity (float): 現在の速度
            decelerate (float): 減速の加速度[km/h/s]
            target_speed (float): 目標速度[km/h]

        Raises:
            ValueError: 加速度が負でない
            ValueError: 目標速度は初期速度より小さくない

        Returns:
            float: 距離[km]
        """
        if decelerate >= 0:
            raise ValueError("加速度は負でなければなりません。")

        if velocity < target_speed:
            raise ValueError("目標速度は初期速度より小さくなければなりません。")

        # 速度が0になる時間 [s]
        t = (target_speed - velocity) / decelerate

        velocity = velocity / 3.6
        decelerate = decelerate / 3.6

        # 停止するまでの距離 [km]
        distance = (velocity * t + 0.5 * decelerate * t**2) / 1000
        return distance

    def _get_signal_instruction(
        self,
        step: int,
        step_size: float,
        train: Train,
        preceding_train: Train | None,
        before_station: Station | None,
        before_station_distance: float | None,
        next_station: Station | None,
        next_station_distance: float | None,
    ) -> SignalInstruction:
        """信号システムからの指示情報を取得する

        信号システムは衝突防止のために必要な情報のみを提供する。
        運転判断は列車（運転手）が行う。

        Args:
            step (int): 求めたいステップ
            step_size (float): ステップサイズ
            train (Train): 求めたい列車
            preceding_train (Train | None): 先行列車。ない場合はNone。
            before_station (Station | None): 通過駅。ない場合はNone。
            before_station_distance (float | None): 通過駅までの距離(プラスで表現)。ない場合はNone。
            next_station (Station | None): 次駅。ない場合はNone。
            next_station_distance (float | None): 次駅までの距離。ない場合はNone。

        Returns:
            SignalInstruction: 信号システムからの指示情報
        """
        # 駅停車中の判定
        is_station_stopping = self._is_station_stop(
            step, train, before_station, step_size
        )

        # 次駅停車判定（停止距離の計算）
        must_stop_at_next_station = False
        if step > 0:
            velocity = train.get_running_data().get_velocity(step - 1)
            if (velocity is not None) and (velocity > 0):
                stop_distance = self.get_decelerate_distance(
                    velocity,
                    train._train_parameter.decelerating_acceleration_station,
                    0,
                )
                must_stop_at_next_station = not self._can_stop_at_next_station(
                    next_station, next_station_distance, stop_distance
                )

        # 指示速度の取得
        instruction_speed = self._get_instruction_speed(step, train, preceding_train)

        # 次の閉塞情報の取得
        next_block_distance, next_block = self._get_next_block_distance(step, train)

        return SignalInstruction(
            instruction_speed=instruction_speed,
            is_station_stopping=is_station_stopping,
            must_stop_at_next_station=must_stop_at_next_station,
            next_block_distance=next_block_distance,
            next_block=next_block,
        )

    def _get_next_block_distance(
        self, step: int, train: Train
    ) -> tuple[float | None, Block | None]:
        """次の閉塞までの距離を求める

        Args:
            step (int): 求めたいステップ
            train (Train): 求めたい列車

        Returns:
            float|None: 距離。閉塞が見つからない場合はNone
        """
        position = train.get_position(step - 1)
        if position is None:
            return None, None

        route = train.get_route()
        next_block_position = route.get_next_block_position(position)
        if next_block_position is None:
            return None, None

        distance = route.get_distance(next_block_position, position)
        block = route.get_next_block(position)
        return distance, block

    def _can_stop_at_next_station(
        self,
        next_station: Station | None,
        next_station_distance: float | None,
        stop_distance: float,
    ) -> bool:
        """次駅に停車できるか判定する関数

        Args:
            next_station (Station | None): 次駅。ない場合はNone
            next_station_distance (float | None): 次駅までの距離。ない場合はNone
            stop_distance (float): 停止距離

        Returns:
            bool: 停車できる場合はTrueを返す。できない場合はFalseを返す。次駅がない場合はTrueを返す。
        """
        # 駅がない場合
        if next_station is None:
            return True

        # 停車距離が超えた場合
        # todo: ステップサイズの誤差を許容したif文にする
        if next_station_distance <= stop_distance:
            return False

        # それ以外
        return True

    def _is_station_stop(
        self, step: int, train: Train, brefore_station: Station | None, step_size: float
    ) -> bool:
        """駅停車中か
        停止中の判定が出た場合、stop_countを減らす。

        Args:
            train (Train): 列車
            brefore_station (Station | None): 通過or今いる駅
            step_size (float): ステップサイズ

        Returns:
            bool: 停止中の場合、True。停止中ではなかったらFalse。
        """
        if brefore_station is None:
            return False

        station_stop_time = train.get_station_stop_time(brefore_station.id)
        if station_stop_time.is_stop(step, step_size):
            station_stop_time.decrement_stop_count(step_size)
            return True

        return False

    @abstractmethod
    def _get_instruction_speed(
        self, step: int, train: Train, preceding_train: Train | None
    ) -> float | None:
        """指示速度の取得（抽象メソッド）

        サブクラスで具体的な速度制御ロジックを実装する。

        Args:
            step (int): 取得したいステップ
            train (Train): 取得したい列車
            preceding_train (Train | None): 先行列車。ない場合はNone

        Returns:
            float|None: 指示速度。制限なしの場合はNoneを返す。
        """
        pass


class FixedBlockSystem(BlockSystem):
    """固定閉塞システム

    先行列車との間の閉塞数に基づいて段階的に速度を制限する従来の閉塞方式。
    """

    def _get_instruction_speed(
        self, step: int, train: Train, preceding_train: Train | None
    ) -> float | None:
        """指示速度の取得

        先行列車との間の閉塞数をカウントして、指示速度を返す。

        Args:
            step (int): 取得したいステップ
            train (Train): 取得したい列車
            preceding_train (Train | None): 先行列車。ない場合はNone

        Returns:
            float|None: 指示速度。制限なしの場合はNoneを返す。
        """
        if step == 0:
            return None

        position = train.get_position(step - 1)
        if position is None:
            # positionが欠損している場合は制限なし
            return None

        # 現在いる閉塞
        block = position.get_block()
        if block is None:
            # 閉塞が取得できない場合は制限なし
            # logger.debug(f"none block {position.edge_id} {position.value}")
            return None

        # 先行列車がいない場合は一番緩い制限を返す
        if preceding_train is None:
            return block.speed_limits[-1]

        preceding_train_position = preceding_train.get_position(step - 1)
        if preceding_train_position is None:
            # preceding_train_positionが欠損している場合は一番緩い制限を返す
            return block.speed_limits[-1]

        # ルートから先行列車間の閉塞数を取得
        route = train.get_route()
        block_diff = route.get_block_diff(preceding_train_position, position)
        if block_diff is None:
            return block.speed_limits[-1]

        if block_diff >= len(block.speed_limits):
            return block.speed_limits[-1]

        return block.speed_limits[block_diff]


class MovingBlockSystem(BlockSystem):
    """移動閉塞システム

    先行列車との距離に基づいて連続的に速度を制御する閉塞方式。
    固定閉塞と異なり、先行列車の位置をリアルタイムで追跡し、
    安全な車間距離を保ちながら最大速度での運転を可能にする。
    """

    def __init__(
        self, line_shape: LineShape, *, min_safe_distance: float = 0.03
    ) -> None:
        """コンストラクタ

        Args:
            line_shape (LineShape): 路線形状
            min_safe_distance (float): 最小安全車間距離[km]。デフォルトは0.03km（30m）
        """
        super().__init__(line_shape)
        self._min_safe_distance = min_safe_distance

    def _get_instruction_speed(
        self, step: int, train: Train, preceding_train: Train | None
    ) -> float | None:
        """指示速度の取得（移動閉塞版）

        先行列車との距離に基づいて指示速度を計算する。
        基本的には路線の最大速度を許可するが、先行列車との安全距離を
        保てなくなる場合は減速を指示する。

        Args:
            step (int): 取得したいステップ
            train (Train): 取得したい列車
            preceding_train (Train | None): 先行列車。ない場合はNone

        Returns:
            float|None: 指示速度[km/h]。制限なしの場合はNoneを返す。
        """
        if step == 0:
            return None

        position = train.get_position(step - 1)
        if position is None:
            return None

        # 現在いる閉塞から路線の最大速度を取得
        block = position.get_block()
        if block is None:
            return None

        # 路線の最大速度（閉塞の速度制限リストの最後の値）
        max_speed = block.speed_limits[-1]

        # 先行列車がいない場合は最大速度を許可
        if preceding_train is None:
            return max_speed

        preceding_train_position = preceding_train.get_position(step - 1)
        if preceding_train_position is None:
            return max_speed

        # 先行列車との距離を計算（後続列車先頭〜先行列車先頭）
        route = train.get_route()
        distance_to_preceding = route.get_distance(preceding_train_position, position)

        # 距離が計算できない、または後ろにいる場合は最大速度
        if distance_to_preceding is None or distance_to_preceding <= 0:
            return max_speed

        # 先行列車の車体長を考慮した実際の車間距離（後続列車先頭〜先行列車末尾）
        preceding_train_length = preceding_train._train_parameter.length
        actual_gap = distance_to_preceding - preceding_train_length

        # 停止すべき位置までの利用可能距離
        # （先行列車末尾位置 - 安全距離）
        available_distance = actual_gap - self._min_safe_distance

        # 安全距離を確保できていない場合は緊急停止
        if available_distance <= 0:
            logger.warning(
                f"{train.name}: 安全距離不足 (車間距離: {actual_gap * 1000:.1f}m, "
                f"最小: {self._min_safe_distance * 1000:.1f}m)"
            )
            return 0

        # 現在の速度を取得
        velocity = train.get_running_data().get_velocity(step - 1)
        if velocity is None or velocity <= 0:
            return max_speed

        # 現在の速度から停止するのに必要な距離を計算
        try:
            stop_distance = self.get_decelerate_distance(
                velocity, train._train_parameter.decelerating_acceleration, 0
            )
        except ValueError:
            # 速度が負の場合などエラーが出た場合は最大速度を返す
            return max_speed

        # 停止距離が利用可能距離を超えている場合、減速が必要
        if stop_distance > available_distance:
            # 利用可能距離内で停止できる最大速度を計算
            # 等加速度運動の式: v^2 = 2 * a * d から
            # v = sqrt(2 * |a| * d)
            # ただし単位変換が必要（a: km/h/s, d: km）
            decel_abs = abs(train._train_parameter.decelerating_acceleration)

            # v^2 = 2 * |a| * d * 1000 * 3.6
            # （物理式から導出: get_decelerate_distanceの逆算）
            allowable_speed = (2 * decel_abs * available_distance * 1000 * 3.6) ** 0.5

            # 路線最大速度を超えないように制限
            return min(allowable_speed, max_speed)

        # 停止距離が十分に短い場合は最大速度を許可
        return max_speed
