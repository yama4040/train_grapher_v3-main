from train_grapher_v3.core.driving_decision import DrivingDecision
from train_grapher_v3.core.status import Status

class RLDrivingDecision(DrivingDecision):
    """強化学習エージェントの行動をシミュレータに伝えるためのクラス"""

    def __init__(self):
        super().__init__()
        # エージェントが選択した最新のアクションを保持する
        self.current_action = 0

    def set_action(self, action: int):
        """Envのstep()関数からアクションをセットするためのメソッド"""
        self.current_action = action

    def decide(self, train, step: int, signal_instruction) -> int:
        """シミュレータから毎ステップ呼ばれ、Statusを返す"""
        
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