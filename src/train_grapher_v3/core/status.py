from typing import Final


class Status:
    NONE_SIMURATION: Final[int] = 0
    """シミュレーション実施外
    """

    OUT_OF_SERVICE: Final[int] = 1
    """営業外
    """

    STOPPING_STATION: Final[int] = 2
    """駅停車中
    """

    BRAKE_STATION: Final[int] = 3
    """駅停車のため減速
    """

    BRAKE_TRAIN: Final[int] = 4
    """列車と衝突回避のため減速
    """

    BRAKE_OVERSPEED: Final[int] = 5
    """速度超過のため減速
    """

    CONSTANT_SPEED: Final[int] = 6
    """等速運転
    """

    COASTING: Final[int] = 7
    """惰行運転
    """

    POWER_RUN: Final[int] = 8
    """力行
    """

    CHECK_SAFETY: Final[int] = 9
    """安全確認
    """
