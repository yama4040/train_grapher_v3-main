"""運転判断モジュール

列車の運転判断ロジックを定義する抽象基底クラスと具体的な実装を提供する。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from train_grapher_v3.core.block_system import SignalInstruction
    from train_grapher_v3.core.train import Train

from train_grapher_v3.core.status import Status


class DrivingDecision(ABC):
    """運転判断の抽象基底クラス

    列車の運転判断ロジックを実装するための基底クラス。
    継承してdecideメソッドをオーバーライドすることで、
    異なる運転スタイルを実装できる。
    """

    @abstractmethod
    def decide(
        self, train: "Train", step: int, signal_instruction: "SignalInstruction"
    ) -> int:
        """運転判断を行う

        信号システムからの指示と列車の現在状態を元に、
        次のステップでの運転状態を決定する。

        Args:
            train: 列車インスタンス（現在の速度、位置などにアクセス可能）
            step: 現在のステップ
            signal_instruction: 信号システムからの指示

        Returns:
            Status値（Status.POWER_RUN, Status.COASTING等）
        """
        pass


class DefaultDrivingDecision(DrivingDecision):
    """デフォルトの運転判断ロジック

    標準的な運転スタイル。安全性を重視しつつ、
    効率的な運転を行う。
    """

    def decide(
        self, train: "Train", step: int, signal_instruction: "SignalInstruction"
    ) -> int:
        """デフォルトの運転判断ロジック"""
        # シミュレーション範囲内かの判定
        if train._start_step is not None and train._start_step > step:
            return Status.NONE_SIMURATION
        
        if train._end_flag:
            return Status.NONE_SIMURATION

        # 最初のステップはとりあえず等速運転
        if step == 0:
            return Status.CONSTANT_SPEED

        # 駅停車中の判定（信号システムからの指示）
        if signal_instruction.is_station_stopping:
            return Status.STOPPING_STATION

        # 次駅に停車できるかの判定（信号システムからの指示）
        if signal_instruction.must_stop_at_next_station:
            return Status.BRAKE_STATION

        # 前ステップの情報を取得
        velocity = train._running_data.get_velocity(step - 1)
        if velocity is None or velocity <= 0:
            velocity = 0

        status = train._running_data.get_status(step - 1)
        instruction_speed = signal_instruction.instruction_speed

        # 指示速度を超えていた場合の判定
        if (instruction_speed is not None) and (velocity > instruction_speed):
            return Status.BRAKE_OVERSPEED

        # マージンまでブレーキをかけ続ける（運転手の判断）
        if (
            (status is not None)
            and (instruction_speed is not None)
            and (status == Status.BRAKE_OVERSPEED)
            and velocity > instruction_speed - train._train_parameter.fast_margine
        ):
            return Status.BRAKE_OVERSPEED

        # 急な坂では等速運転（運転手の判断）
        position = train.get_position(step - 1)
        if position is not None:
            grade = position.edge.get_grade(position.value)
            if (
                (instruction_speed is not None)
                and (grade <= -10)
                and (velocity > instruction_speed - train._train_parameter.slow_margine)
            ):
                return Status.CONSTANT_SPEED

        # 速度がslow_margineを超えていて、1ステップ前惰行だった場合（運転手の判断）
        # todo : 指示速度でマージンが変わるので要修正
        if (
            (instruction_speed is not None)
            and (
                velocity
                > instruction_speed
                - (
                    train._train_parameter.slow_margine
                    if instruction_speed == 45
                    else 10
                )
            )
            and (status is not None)
            and (status == Status.COASTING)
        ):
            return Status.COASTING

        # 速度がfast_margineを超えていた場合（運転手の判断）
        if (instruction_speed is not None) and (
            velocity > instruction_speed - train._train_parameter.fast_margine
        ):
            return Status.COASTING

        # 指示速度が変わる手前の場合（運転手の判断）
        next_block_distance = signal_instruction.next_block_distance
        next_block = signal_instruction.next_block
        if next_block_distance is not None and next_block is not None:
            if (next_block.speed_limits[-1] > velocity) and (
                next_block_distance < (velocity * 5 / 3600)
            ):
                return Status.COASTING

        # 上記以外は加速
        return Status.POWER_RUN
