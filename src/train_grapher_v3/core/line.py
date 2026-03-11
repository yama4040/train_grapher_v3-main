from train_grapher_v3.core.block_system import BlockSystem
from train_grapher_v3.core.line_shape import LineShape
from train_grapher_v3.core.train import Train


class Line:
    def __init__(
        self, trains: list[Train], line_shape: LineShape, block_system: BlockSystem
    ) -> None:
        self._trains = trains
        self._line_shape = line_shape
        self._block_system = block_system

    def calculate_step(self, step: int, step_size: float) -> None:
        # 信号システムから各列車への指示を取得
        instruction_list = self._block_system.calculate_step(
            step, self._trains, step_size
        )

        # 列車が信号指示を受けて運転判断を行い、位置を計算
        for train, instruction in zip(self._trains, instruction_list):
            train.calculate_step(step, step_size, instruction)
