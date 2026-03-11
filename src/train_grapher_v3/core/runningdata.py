"""走行データ管理用モジュール"""


class RunningData:
    def __init__(self):
        """コンストラクタ"""
        self._status: list[int] = []
        self._velocity: list[float] = []
        self._acceleration: list[float] = []
        self._edge_id: list[str] = []
        self._position_value: list[float] = []

    def _set_value(self, array: list, value: any, step: int) -> list:
        """配列への値の挿入

        Args:
            array (np.ndarray): 配列
            value (float): 値
            step (int): 挿入位置

        Returns:
            np.ndarray: 挿入後の配列
        """
        if step < 0 or len(array) == step:
            # stepが-1のとき、または、
            # 配列の長さとstepが同じとき
            # 配列の最後に値を挿入
            array.append(value)
            return
        else:
            if len(array) > step:
                # 配列長が足りていればそこに挿入
                array[step] = value
                return
            else:
                # 長さが足りない場合、長さを追加(0埋めで)
                diff_length = step - len(array)
                array.extend([None] * diff_length)
                # そこに挿入
                array.append(value)
                return

    def _get_value(self, array: list, step: int) -> any:
        """値の取得

        Args:
            array (list): 配列
            step (int): 取得位置

        Returns:
            any: 取得位置の値
        """
        if step < 0:
            if not array:  # List is empty
                return None
            try:
                # For step = -1, it gets the last element if list is not empty.
                # For other negative steps, it works if index is valid.
                return array[step]
            except IndexError:  # e.g. step = -2 on a list with 1 element
                return None
        else:  # step >= 0
            if len(array) > step:
                return array[step]
            else:
                return None

    def set_status(self, value: int, step: int = -1) -> None:
        self._set_value(self._status, value, step)

    def set_velocity(self, value: float, step: int = -1) -> None:
        self._set_value(self._velocity, value, step)

    def set_acceleration(self, value: float, step: int = -1) -> None:
        self._set_value(self._acceleration, value, step)

    def set_edge_id(self, value: str, step: int = -1) -> None:
        self._set_value(self._edge_id, value, step)

    def set_position_value(self, value: float, step: int = -1) -> None:
        self._set_value(self._position_value, value, step)

    def get_status(self, step: int = -1) -> int | None:
        return self._get_value(self._status, step)

    def get_velocity(self, step: int = -1) -> float | None:
        return self._get_value(self._velocity, step)

    def get_acceleration(self, step: int = -1) -> float | None:
        return self._get_value(self._acceleration, step)

    def get_edge_id(self, step: int = -1) -> str | None:
        return self._get_value(self._edge_id, step)

    def get_position_value(self, step: int = -1) -> float | None:
        return self._get_value(self._position_value, step)

    def get_status_all(self) -> list[int]:
        return self._status

    def get_velocity_all(self) -> list[float]:
        return self._velocity

    def get_acceleration_all(self) -> list[float]:
        return self._acceleration

    def get_edge_id_all(self) -> list[str]:
        return self._edge_id

    def get_position_value_all(self) -> list[float]:
        return self._position_value
