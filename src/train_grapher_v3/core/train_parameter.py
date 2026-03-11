from dataclasses import dataclass


@dataclass
class TrainParameter:
    """列車の性能とか"""

    wight: float = 389.85
    factor_of_inertia: float = 28.35 * (1 - 0.085)  # 慣性係数
    decelerating_acceleration: float = -3.0  # 制限時の加速度
    decelerating_acceleration_station: float = -4.0  # 駅停車のための減速の加速度
    fast_margine: float = 6.0
    slow_margine: float = 15.0
    length: float = 0.2  # 列車長[km]（デフォルト200m）


# accelerating_acceleration: float = 2.0  # 加速時の加速度
# constant_acceleration: float = 1.0  # 等速時の加速度
# coasting_acceleration: float = 0.0  # 惰行時の加速度
# reso: float = 0.1  # 0.1 #resolution
# decelerate: float = 3
# res: float = 0  # 0 #抵抗　要素を設定する
# run_res: float = 0  # 0 #走行抵抗のみ　要素を設定する
# run_res: float = (
#     0  # Rr = (0.00662*speed**2/params.TRAIN_WEIGHT + 0.38612*speed + 20.4722)/9.8 if speed != 0 else 30/9.8
# )
