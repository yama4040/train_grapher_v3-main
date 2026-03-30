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
        
        # 1. JSONモデルの読み込み
        line_shape, trains, self.step_size, self.total_steps, block_system_type = load_simulation_model(self.json_path)
        
        # 2. 対象の列車にRLDrivingDecisionをセット
        self.target_train = next(t for t in trains if t.name == self.target_train_name)
        self.target_train._driving_decision = self.rl_decision

        # 3. 閉塞システムと路線の初期化
        if block_system_type == "moving":
            block_system = MovingBlockSystem(line_shape)
        else:
            block_system = FixedBlockSystem(line_shape)
            
        self.line = Line(line_shape=line_shape, trains=trains, block_system=block_system)
        self.current_step = 0

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
        """観測値の取得"""
        if self.current_step == 0:
            return np.zeros(4, dtype=np.float32)

        # 現在の速度
        velocity = self.target_train.get_running_data().get_velocity(self.current_step - 1) or 0.0
        
        # 制限速度（信号情報などから取得。実装はシミュレータの仕様に合わせる必要があります）
        #signal = self.line._block_system._get_signal_instruction(self.target_train, self.current_step - 1)
        speed_limit = self.rl_decision.last_signal_speed / 3.6 # km/h → m/s に変換

        # 次の駅までの情報
        next_station, distance_to_station = self.target_train.get_next_station_info(self.current_step)
        distance = distance_to_station if distance_to_station is not None else 0.0

        # 目標到着時刻までの残り時間（簡易計算）
        time_left = 0.0
        if next_station is not None:
            stop_time_info = self.target_train.get_station_stop_time(next_station.id)
            if stop_time_info and stop_time_info.departure_time:
                # 出発時刻 - (停車時間) - 現在時刻 = 到着目標時刻までの残り時間
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
        velocity, speed_limit, distance, time_left = obs
        reward = 0.0
        
        # 1. 制限速度超過ペナルティ（絶対ルール）
        if velocity > speed_limit:
            reward -= 10.0
            
        # 2. 前進ボーナス（安全に走っていることへのご褒美）
        if velocity > 0.5 and velocity <= speed_limit:
            reward += 0.1 

        # 3. 【新・引きこもり対策】青信号でのサボりペナルティ
        # 制限速度が5km/h以上（発車OK）なのに、速度が1km/h未満で止まっている場合
        if speed_limit > 5.0 and velocity < 1.0:
            reward -= 0.5  # 「青信号だぞ！進め！」という減点

        # 4. 駅到着時の評価
        if distance < 5.0 and velocity < 1.0: 
            reward += 100.0 
            reward -= abs(time_left) * 0.1 

        return float(reward)