"""基準モデル用運転判断モジュール

移動閉塞環境で駅間停車後の再発車時に15秒間の安全確認（CHECK_SAFETY）を挟む
運転判断クラスを提供する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from train_grapher_v3.core.driving_decision import DefaultDrivingDecision
from train_grapher_v3.core.status import Status

if TYPE_CHECKING:
    from train_grapher_v3.core.block_system import SignalInstruction
    from train_grapher_v3.core.train import Train

# 駅間停車と見なさないステータス集合
# CHECK_SAFETY を含めることで、安全確認終了直後の再トリガーを防止する
_STATION_AND_OUT_OF_SIM: frozenset[int] = frozenset(
    {
        Status.NONE_SIMURATION,
        Status.STOPPING_STATION,
        Status.OUT_OF_SERVICE,
        Status.CHECK_SAFETY,
    }
)


class BaseModelDrivingDecision(DefaultDrivingDecision):
    """基準モデル運転判断

    移動閉塞（MovingBlockSystem）環境で、先行列車との安全距離不足により
    駅間停車が発生した後、再発車する際に15秒間の安全確認（CHECK_SAFETY）を挟む。

    状態遷移:
        通常走行
          ↓ MovingBlockSystem が instruction_speed=0 を返す
        駅間停車中（velocity=0, !is_station_stopping）
          ↓ instruction_speed > 0 に変化（閉塞解除）
        CHECK_SAFETY（15秒間）
          ↓ 経過後
        通常走行（DefaultDrivingDecision へフォールスルー）
    """

    def __init__(
        self,
        step_size: float,
        safety_check_time: float = 15.0,
    ) -> None:
        """
        Args:
            step_size: シミュレーションのステップサイズ[s]
            safety_check_time: 安全確認時間[s]（デフォルト15秒）
        """
        super().__init__()
        self._step_size = step_size
        self._safety_check_time = safety_check_time
        self._safety_check_start_step: int | None = None
        # 一度でも走行したかのフラグ（走行前の速度=0 を駅間停車と誤検出しないため）
        self._has_ever_moved: bool = False

    def decide(
        self,
        train: "Train",
        step: int,
        signal_instruction: "SignalInstruction",
    ) -> int:
        """安全確認ロジック付き運転判断"""
        # シミュレーション範囲外チェック
        if train._start_step is not None and train._start_step > step:
            return Status.NONE_SIMURATION

        if train._end_flag:
            return Status.NONE_SIMURATION

        # 最初のステップは等速
        if step == 0:
            return Status.CONSTANT_SPEED

        # 駅停車中：安全確認をリセットして停車処理
        if signal_instruction.is_station_stopping:
            self._safety_check_start_step = None
            return Status.STOPPING_STATION

        # 前ステップの速度を取得し、走行実績フラグを更新
        prev_velocity = train._running_data.get_velocity(step - 1) or 0.0
        if prev_velocity >= 0.5:
            self._has_ever_moved = True

        # 安全確認継続中の判定
        if self._safety_check_start_step is not None:
            elapsed = (step - self._safety_check_start_step) * self._step_size
            if elapsed < self._safety_check_time:
                return Status.CHECK_SAFETY
            # 安全確認完了 → リセット
            self._safety_check_start_step = None

        # 駅間停車 → 再発車の検出（一度も走行していない場合はスキップ）
        if self._has_ever_moved and self._is_restarting_after_interstation_stop(
            step, prev_velocity, signal_instruction, train
        ):
            self._safety_check_start_step = step
            return Status.CHECK_SAFETY

        # 上記以外は DefaultDrivingDecision に委譲
        return super().decide(train, step, signal_instruction)

    def _is_restarting_after_interstation_stop(
        self,
        step: int,
        prev_velocity: float,
        signal_instruction: "SignalInstruction",
        train: "Train",
    ) -> bool:
        """駅間停車後の再発車かどうかを判定する

        以下の条件をすべて満たす場合に True を返す:
        1. 現ステップで instruction_speed > 0（閉塞が解除された）
        2. 前ステップの速度がほぼ 0（< 0.01 km/h）
        3. 前ステップのステータスが駅関連・シミュ外ステータスではない
        """
        # 条件1: 閉塞が解除されていること
        instruction_speed = signal_instruction.instruction_speed
        if instruction_speed is None or instruction_speed <= 0:
            return False

        # 条件2: 前ステップの速度がほぼ 0（呼び出し元から渡す）
        if prev_velocity >= 0.01:
            return False

        # 条件3: 前ステップのステータスが駅関連・シミュ外ではない
        prev_status = train._running_data.get_status(step - 1)
        if prev_status is None or prev_status in _STATION_AND_OUT_OF_SIM:
            return False

        return True
