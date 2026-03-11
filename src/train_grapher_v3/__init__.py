"""Train Grapher v3 - 鉄道列車運行シミュレーションシステム"""

from train_grapher_v3.core.block_system import (
    BlockSystem,
    FixedBlockSystem,
    MovingBlockSystem,
)
from train_grapher_v3.core.driving_decision import (
    DefaultDrivingDecision,
    DrivingDecision,
)

__all__ = [
    "BlockSystem",
    "FixedBlockSystem",
    "MovingBlockSystem",
    "DrivingDecision",
    "DefaultDrivingDecision",
]


def hello() -> str:
    return "Hello from train-grapher-v3!"
