import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

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
        
        # 行動空間: 離散値 3パターン
        # 0: 力行, 1: 惰行, 2: 減速(駅)
        self.action_space = spaces.Discrete(3)

        # 観測空間: 連続値 4次元
        # [現在の速度(m/s), 制限速度(m/s), 次の駅までの残距離(m), 目標到着時刻までの残り時間(s)]
        high = np.array([100.0, 100.0, 2000.0, 3600.0], dtype=np.float32)
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
        reward, reward_components = self.reward_fn(obs, action, info, done)
        
        # ==========================================
        # ★ 追加：報酬成分の履歴を保存
        # ==========================================
        if not hasattr(self, 'reward_history'):
            self.reward_history = []
        self.reward_history.append(reward_components)

        # エピソード終了時（done=True）にCSVへ集計・保存する
        if done:
            self._save_reward_to_csv()
            self.reward_history = [] # 次のエピソードに向けてリセット
        
        # infoに成分辞書を入れておくと、後でデバッグやコールバックで使いやすいです
        info["reward_components"] = reward_components

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
    
    def _save_reward_to_csv(self):
        """
        エピソード終了時に呼び出され、報酬履歴を10区間に分割して
        最大・平均・最小値を算出し、CSVに保存する。
        """
        if not hasattr(self, 'reward_history') or len(self.reward_history) == 0:
            return

        import pandas as pd
        import numpy as np

        # 履歴のリストをPandasのDataFrameに変換
        df = pd.DataFrame(self.reward_history)
        
        # 全ステップ数を取得
        total_steps = len(df)
        if total_steps == 0:
            return

        # 10区間に分割するためのラベル（0〜9）を各行に付与
        num_segments = 10
        # 例: 100ステップなら、0~9行目は区間0、10~19行目は区間1...
        df['segment'] = np.floor(df.index / (total_steps / num_segments)).astype(int)
        df['segment'] = df['segment'].clip(upper=num_segments - 1) # 最大値を9に丸める

        # 区間(segment)ごとにグループ化し、各カラムの 最大(max), 平均(mean), 最小(min) を計算
        summary_df = df.groupby('segment').agg(['max', 'mean', 'min'])

        # カラム名をフラットで見やすくする（例: 'distance_penalty_max'）
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]

        # CSVファイルとして保存（LLMが読み込みやすい場所に保存）
        # ※ファイル名は上書きされる仕様にしていますが、エピソードごとに分けたい場合は
        #   ファイル名に self.current_episode などの変数を足してください。
        save_path = "llm_feedback_reward_summary.csv"
        summary_df.to_csv(save_path)
        
        # コンソールへの通知（デバッグ用・不要なら削除可）
        # print(f"=== 報酬の推移データを {save_path} に保存しました ===")

    def default_reward_fn(self, obs, action, info, done):
        """
        階層型強化学習：下位エージェント（ノッチ操作）向けの報酬関数。
        入力変数として提供された obs, action, info, done のみを使用し、
        出力は (合計報酬, 個々の報酬成分の辞書) のタプルとします。
        """
        # 観測値の展開
        velocity, speed_limit, distance, time_left = obs
        
        # 報酬成分の初期化
        distance_penalty = 0.0
        delay_penalty = 0.0
        comfort_penalty = 0.0
        energy_penalty = 0.0
        stopping_bonus = 0.0
        prohibition_penalty = 0.0

        # ==========================================
        # 状態の初期化（新しい入力変数を追加せず self に保持）
        # ==========================================
        if not hasattr(self, 'action_hold_time'):
            self.action_hold_time = 0
        if not hasattr(self, 'previous_action'):
            self.previous_action = action

        # ==========================================
        # 1. 距離ペナルティ（駅からの発車・前進の促進）
        # ==========================================
        # 目標地点（駅や先行列車）から離れている分だけ微小なペナルティ
        distance_penalty = -0.01 * distance

        # ==========================================
        # 2. 遅延ペナルティ
        # ==========================================
        # time_left（残り時間）が0未満（＝予定走行時間を超過）の場合、1秒オーバーするごとにペナルティ
        if time_left < 0:
            delay_penalty = -1.0 * abs(time_left)

        # ==========================================
        # 3. 乗り心地に関するペナルティ
        # ==========================================
        # action: 0(力行), 1(惰行), 2(減速) を想定
        if action == self.previous_action:
            self.action_hold_time += 1
        else:
            # 操作が切り替わった際、保持時間が7秒（ステップ）未満だった場合はペナルティ
            if self.action_hold_time < 7:
                base_comfort_pen = -1.0 * (7 - self.action_hold_time)
                
                # 直接切り替え（加速⇔減速で惰行を挟まない動作）の判定
                is_direct_switch = (self.previous_action == 0 and action == 2) or \
                                (self.previous_action == 2 and action == 0)
                
                if is_direct_switch:
                    base_comfort_pen *= 2.0  # ペナルティを2倍にする
                    
                comfort_penalty += base_comfort_pen
            
            # 新しい操作に切り替わったので保持時間をリセット
            self.action_hold_time = 0

        self.previous_action = action

        # ==========================================
        # 4. 消費エネルギーに関するペナルティ
        # ==========================================
        if action == 0:  # 加速（力行）している間
            energy_penalty = -0.1  # 他と比べて重要度が低いため微小

        # ==========================================
        # 5. 停止位置精度によるボーナス
        # ==========================================
        if velocity < 0.1:  # 速度がほぼ0（停止）と判定
            if distance <= 1.0:
                # 前後1m以内（ドア開扉範囲）は最大ボーナス
                stopping_bonus = 100.0
            elif distance <= 10.0:
                # 手前10m以内の場合、ずれが小さいほど指数関数的に増加
                stopping_bonus = 10.0 * math.exp(-0.5 * distance)

        # ==========================================
        # 6. 禁止行動へのペナルティ
        # ==========================================
        # 速度超過をした場合、加速(0)と惰行(1)を選択したら極大ペナルティ
        if velocity > speed_limit and action in [0, 1]:
            prohibition_penalty += -500.0

        # 前方列車の位置を超えている（追突状態）場合の判定
        # infoに前方列車との距離情報が入っていると仮定。なければ安全マージンとして9999.0
        preceding_distance = info.get("preceding_distance", 9999.0)
        if preceding_distance <= 0 and action in [0, 2]: # 加速(0)と減速(2)を選択したら極大ペナルティ
            prohibition_penalty += -1000.0

        # 終点到達時の処理（エピソード終了時）
        terminal_reward = 0.0
        if done:
            # ご提示いただいたテンプレートの終了時ボーナス/ペナルティを継承
            current_position = info.get("train_position", None)
            pos_km = current_position.value if hasattr(current_position, "value") else 0.0
            if pos_km > 6.5:
                terminal_reward += 1000.0
            else:
                terminal_reward -= 500.0

        # ==========================================
        # 報酬の集計と出力
        # ==========================================
        # 記録用辞書の作成
        reward_components = {
            'distance_penalty': distance_penalty,
            'delay_penalty': delay_penalty,
            'comfort_penalty': comfort_penalty,
            'energy_penalty': energy_penalty,
            'stopping_bonus': stopping_bonus + terminal_reward, # 終点ボーナスをここに統合
            'prohibition_penalty': prohibition_penalty
        }
        
        # 合計報酬の計算
        total_reward = sum(reward_components.values())

        # 指定された通り (合計報酬, 個々の報酬成分の辞書) を返す
        return float(total_reward), reward_components