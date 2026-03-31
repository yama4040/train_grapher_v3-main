import gymnasium as gym
from gymnasium import spaces
import numpy as np

from train_grapher_v3.core.line import Line
from train_grapher_v3.core.block_system import MovingBlockSystem, FixedBlockSystem
from train_grapher_v3.util.simulation_model_io import load_simulation_model
from experiments.rl_env.rl_driving_decision import RLDrivingDecision

class TrainEnv(gym.Env):
    """列車制御のための強化学習環境 (Gymnasium準拠)"""

    def __init__(self, json_path: str, target_train_name: str, reward_fn=None):
        super().__init__()
        self.json_path = json_path
        self.target_train_name = target_train_name # 学習対象の列車名
        
        # 行動空間: 離散値 5パターン
        # 0: 力行, 1: 惰行, 2: 減速(駅), 3: 定速, 4: 減速(最大)
        self.action_space = spaces.Discrete(5)

        # 観測空間: 連続値 4次元
        # [現在の速度(m/s), 制限速度(m/s), 次の駅までの残距離(m), 目標到着時刻までの残り時間(s)]
        high = np.array([100.0, 100.0, 50000.0, 3600.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # LLM生成の報酬関数をセット（Noneの場合はデフォルトを使用）
        self.reward_fn = reward_fn or self.default_reward_fn

        self.current_step = 0
        self.step_size = 0.1
        self.total_steps = 0
        self.line = None
        self.target_train = None
        self.rl_decision = RLDrivingDecision()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # ★追加：一度止まった駅のIDを記録するリスト
        self.visited_stations = set()
        self.current_station_id = None
        
        # ★追加：前回のノッチ操作を記憶する変数
        self.previous_action = None

        line_shape, trains, self.step_size, self.total_steps, block_system_type = load_simulation_model(self.json_path)
        
        self.target_train = next(t for t in trains if t.name == self.target_train_name)
        self.target_train._driving_decision = self.rl_decision

        if block_system_type == "moving":
            block_system = MovingBlockSystem(line_shape)
        else:
            block_system = FixedBlockSystem(line_shape)
            
        self.line = Line(line_shape=line_shape, trains=trains, block_system=block_system)
        self.current_step = 0

        # ==================================================
        # ★修正：確実な早送り処理（普通列車2なら強制的に1200ステップ進める）
        # ==================================================
        start_step = 1200 if self.target_train.name == "普通列車2" else 0
        self.rl_decision.set_action(2) 

        while self.current_step < start_step:
            self.line.calculate_step(self.current_step, self.step_size)
            self.current_step += 1

        while self.current_step < self.total_steps:
            self.line.calculate_step(self.current_step, self.step_size)
            if self.rl_decision.last_signal_speed > 0:
                break
            self.current_step += 1
        # ==================================================

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        # 1. エージェントの行動をセット
        self.rl_decision.set_action(action)

        # 2. シミュレータを1ステップ進める (Simulationクラスの代わり)
        self.line.calculate_step(self.current_step, self.step_size)
        self.current_step += 1

        # 3. 観測、終了判定、情報の取得
        obs = self._get_obs()
        done = self.current_step >= self.total_steps or self.target_train.get_end_flag()
        info = self._get_info()

        # 4. 報酬の計算 (LLM生成の関数を呼ぶ想定)
        reward = self.reward_fn(obs, action, info, done)

        truncated = False # タイムアウトなどの打ち切りフラグ（今回は不使用）
        return obs, reward, done, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self.current_step == 0:
            return np.zeros(4, dtype=np.float32)

        velocity = self.target_train.get_running_data().get_velocity(self.current_step - 1) or 0.0
        speed_limit = self.rl_decision.last_signal_speed / 3.6 

        next_station, distance_to_station = self.target_train.get_next_station_info(self.current_step)
        distance = distance_to_station if distance_to_station is not None else 0.0

        # ★追加：現在目指している駅のIDを保存しておく（報酬計算で使うため）
        if next_station is not None:
            self.current_station_id = next_station.id
        else:
            self.current_station_id = "end"

        time_left = 0.0
        if next_station is not None:
            stop_time_info = self.target_train.get_station_stop_time(next_station.id)
            if stop_time_info and stop_time_info.departure_time:
                target_arrival = stop_time_info.departure_time - stop_time_info.default_value
                current_time = self.current_step * self.step_size
                time_left = target_arrival - current_time

        return np.array([velocity, speed_limit, distance, time_left], dtype=np.float32)

    def _get_info(self) -> dict:
        """デバッグや報酬計算に使える追加情報"""
        return {
            "step": self.current_step,
            "time": self.current_step * self.step_size,
            "train_position": self.target_train.get_position(self.current_step - 1)
        }

    def default_reward_fn(self, obs, action, info, done):
        import math # (ファイルの先頭に書いてもOKです)
        velocity, speed_limit, distance, time_left = obs
        reward = 0.0
        
        # ==========================================
        # 1. 安全な「目標速度」の計算（ATOブレーキパターン）
        # ==========================================
        # 駅の5m手前に、減速度 2.0 m/s^2 で滑らかに止まるための理想速度を計算
        brake_distance = max(0.0, distance - 5.0)
        # 物理公式: v = sqrt(2 * a * x)
        safe_approach_speed = math.sqrt(2.0 * 2.0 * brake_distance) 
        
        # 「信号の制限速度」と「駅へのブレーキ速度」のうち、厳しい方を目標にする
        target_speed = min(speed_limit, safe_approach_speed)

        # ==========================================
        # 2. 速度の評価（目標速度への追従）
        # ==========================================
        if velocity > speed_limit:
            reward -= 10.0  # 信号無視・速度超過は一発アウト級の減点
        elif velocity > target_speed + 2.0:
            reward -= 5.0   # 駅をオーバーランしそうなスピードも強く減点
        else:
            # 目標速度に対して、近いほど加点、遠いほど減点
            speed_diff = abs(target_speed - velocity)
            reward += max(0.0, 1.0 - (speed_diff / 5.0)) 
            
            # ダラダラ這いずるのを防ぐ（最低限のスピードを要求）
            if velocity < 1.0 and target_speed > 5.0:
                reward -= 0.5

        # ==========================================
        # 3. 乗り心地と省エネ（ガチャガチャ運転の禁止）
        # ==========================================
        if self.previous_action is not None:
            if action != self.previous_action:
                # ノッチを切り替えるたびに減点（= 無駄なエネルギーと揺れへの罰則）
                reward -= 0.2
        self.previous_action = action # 次のステップのために記憶を更新

        # ==========================================
        # 4. 駅の停車と終点到達のボーナス
        # ==========================================
        if distance < 10.0 and velocity < 1.0:
            if self.current_station_id not in self.visited_stations:
                reward += 300.0  
                self.visited_stations.add(self.current_station_id)

        if done:
            current_position = info["train_position"]
            pos_km = current_position.value if hasattr(current_position, "value") else 0.0
            if pos_km > 6.5:
                reward += 1000.0
            else:
                reward -= 500.0

        return float(reward)