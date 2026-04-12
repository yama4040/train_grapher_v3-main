from train_grapher_v3.core.driving_decision import DrivingDecision
from train_grapher_v3.core.status import Status

class RLDrivingDecision(DrivingDecision):
    def __init__(self):
        super().__init__()
        self.current_action = 1  # 初期は惰行(1)
        self.last_signal_speed = 100.0 

    def set_action(self, action: int):
        self.current_action = action

    def decide(self, train, step: int, signal_instruction) -> Status:
        # 1. 制限速度の保存
        if signal_instruction and signal_instruction.instruction_speed is not None:
            self.last_signal_speed = float(signal_instruction.instruction_speed)
        else:
            self.last_signal_speed = 100.0 
            
        intended_action = getattr(self, 'current_action', 1)
        
        # 2. 速度の取得
        current_speed = train.get_running_data().get_velocity(step)
        if current_speed is None:
            current_speed = 0.0  
        
        # ==========================================
        # 安全装置（禁止行動のルールベース・オーバーライド）
        # ==========================================
        # 速度超過時
        if current_speed > self.last_signal_speed:
            if intended_action in [0, 1]:  
                return Status.BRAKE_STATION # 強制減速
                
        # 先行列車追い越し時
        if hasattr(self, 'block_system') and self.block_system is not None:
            preceding_train = self.block_system._get_preceding_train(step, self.trains, train)
            if preceding_train:
                preceding_pos = preceding_train.get_position(step)
                current_pos = train.get_position(step)
                if preceding_pos is not None and current_pos is not None:
                    distance_to_preceding = train.get_route().get_distance(preceding_pos, current_pos)
                    if distance_to_preceding is not None and distance_to_preceding <= 0:
                        if intended_action in [0, 2]: 
                            return Status.COASTING # 強制惰行
        
        # ==========================================
        # 通常の行動マッピング（AIの出力をシミュレータ言語に翻訳）
        # ==========================================
        if intended_action == 0:
            return Status.POWER_RUN      # 力行（加速）
        elif intended_action == 1:
            return Status.COASTING       # 惰行
        elif intended_action == 2:
            return Status.BRAKE_STATION  # 減速（制動）
        else:
            return Status.COASTING       # フェイルセーフ