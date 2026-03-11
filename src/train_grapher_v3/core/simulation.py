from train_grapher_v3.core.line import Line


class Simulation:
    def __init__(self, *, step_size: float = 0.1, line: Line) -> None:
        self._line = line
        self._step_size = step_size

    def set_line(self, line: Line) -> None:
        self._line = line

    def execution(self, end_step: int, step_size: float, callback=None) -> int:
        for step in range(end_step):
            self._line.calculate_step(step, step_size)

            if step % int(end_step / 100) == 0:
                if callback is not None:
                    callback(step, step / end_step)

        return step
