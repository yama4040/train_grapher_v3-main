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
        # 1. シミュレータから渡された最新の制限速度を保存
        if signal_instruction and signal_instruction.instruction_speed is not None:
            self.last_signal_speed = float(signal_instruction.instruction_speed)
        else:
            self.last_signal_speed = 100.0 # 制限なしの場合はデフォルト100km/h
            
        # RLエージェントが出力した行動を一時保存（reset直後などで未定義の場合は惰行(1)とする）
        intended_action = getattr(self, 'current_action', 1)
        
        # 2. 自列車の現在の速度を取得（★ここが修正のメインです）
        current_speed = train.get_running_data().get_velocity(step)
        if current_speed is None:
            current_speed = 0.0  # Noneの場合は0.0km/hとして扱う
        
        # ==========================================
        # 安全装置（禁止行動のルールベース・オーバーライド）
        # ==========================================
        
        # 速度超過時の処理（加速と惰行を禁止 -> 強制的に減速へ）
        if current_speed > self.last_signal_speed:
            if intended_action in [0, 1]:  # 力行(0) または 惰行(1) を選んだ場合
                return 2 # 強制的に減速(Status.BRAKE_STATION 相当)に上書き
                
        # 先行列車追い越しの処理
        if hasattr(self, 'block_system') and self.block_system is not None:
            preceding_train = self.block_system._get_preceding_train(step, self.trains, train)
            if preceding_train:
                preceding_pos = preceding_train.get_position(step)
                current_pos = train.get_position(step)
                if preceding_pos is not None and current_pos is not None:
                    distance_to_preceding = train.get_route().get_distance(preceding_pos, current_pos)
                    
                    # 位置を超えている（または追突状態）場合
                    if distance_to_preceding is not None and distance_to_preceding <= 0:
                        if intended_action in [0, 2]: # 力行(0) または 減速(2) を選んだ場合
                            return 1 # 強制的に惰行(Status.COASTING 相当)に上書き
        
        # ==========================================
        # 通常の行動マッピング（安全チェック通過後）
        # ==========================================
        # ※ 環境側でStatus Enumの代わりに直接数値を求めている場合はこれでもOKです
        if intended_action == 0:
            return 1  # Status.POWER_RUN
        elif intended_action == 1:
            return 0  # Status.COASTING (環境の定義に合わせて適宜変更してください)
        elif intended_action == 2:
            return -1 # Status.BRAKE_STATION (環境の定義に合わせて適宜変更してください)
        else:
            return 0  # フェイルセーフ（惰行）