from train_grapher_v3.core.driving_decision import DrivingDecision
from train_grapher_v3.core.status import Status

class RLDrivingDecision(DrivingDecision):
    def __init__(self):
        super().__init__()
        self.current_action = 0
        # 追加：最新の制限速度(km/h)を保持する変数
        self.last_signal_speed = 100.0 

    def set_action(self, action: int):
        self.current_action = action

    def decide(self, train, step: int, signal_instruction) -> int:
        # 追加：シミュレータから渡された最新の制限速度を保存
        if signal_instruction and signal_instruction.instruction_speed is not None:
            self.last_signal_speed = signal_instruction.instruction_speed
        else:
            self.last_signal_speed = 100.0 # 制限なしの場合はデフォルト100km/h
        
        # 定義した5つの行動（0~4）をシミュレータのStatusにマッピングする
        if self.current_action == 0:
            return Status.POWER_RUN         # 力行
        elif self.current_action == 1:
            return Status.COASTING          # 惰行
        elif self.current_action == 2:
            return Status.BRAKE_STATION     # 減速（駅停車用）
        elif self.current_action == 3:
            return Status.CONSTANT_SPEED    # 定速
        elif self.current_action == 4:
            return Status.BRAKE_OVERSPEED   # 減速（常用最大/保安装置）
        else:
            return Status.COASTING          # フェイルセーフ