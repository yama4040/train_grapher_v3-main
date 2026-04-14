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
    """
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
    """
    def step(self, action: int):
        # 1. エージェントの行動をセット（この行動を1秒間維持します）
        self.rl_decision.set_action(action)

        # 1秒間に相当するシミュレーションのステップ数を計算 (step_size=0.1なら10回)
        # ※ step_size が定義されていない場合はデフォルトで10回（1秒分）回します
        skip_frames = int(1.0 / self.step_size) if hasattr(self, 'step_size') else 10

        done = False

        # 2. シミュレータを1秒分（skip_frames回）進めるループ
        for _ in range(skip_frames):
            self.line.calculate_step(self.current_step, self.step_size)
            self.current_step += 1  # シミュレータの内部時間は0.1秒ごとに進む

            # 内部ループでの終了判定
            done = self.current_step >= self.total_steps or self.target_train.get_end_flag()
            
            # 1秒経過する前に駅に到着したり条件を満たした場合、計算を打ち切る
            if done:
                break

        # 3. 1秒進んだ後（またはループを抜けた直後）の観測、情報の取得
        obs = self._get_obs()
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
        num_segments = 50
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

    #obs：シミュレータからの情報，action：エージェントの行動，info：シミュレータからの追加情報，done：エピソード終了フラグ
    def default_reward_fn(self, obs, action, info, done):
        """
        階層型強化学習：フェーズ完全分離（駅減速 vs 追従）＆ 到達時間最適化モデル
        """
        import math
        import json
        
        # 1. 観測値の単位変換 (m/s -> km/h)
        v_ms = float(obs[0])
        limit_ms = float(obs[1])
        velocity_kmh = v_ms * 3.6
        speed_limit_kmh = limit_ms * 3.6
        distance_m = float(obs[2])
        
        if limit_ms > 40.0:  
            velocity_kmh = v_ms
            speed_limit_kmh = limit_ms

        current_position = info.get("train_position", None)
        pos_km = current_position.value if hasattr(current_position, "value") else 0.0
        act = int(action)

        current_step = getattr(self, 'current_step', 0)
        step_size = getattr(self, 'step_size', 0.1)
        current_time_sec = current_step * step_size 

        # ==========================================
        # 環境パラメータの取得
        # ==========================================
        if not hasattr(self, 'target_station_km'):
            self.target_station_km = 2.0
            self.wait_sec = 180.0  
            self.brake_accel = 3.5  
            self.fast_margine = 5.0   
            self.slow_margine = 18.0  
            try:
                with open(getattr(self, 'json_path', ''), 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.target_station_km = float(config["line_shape"]["edges"][0]["stations"][0]["value"])
                    for t in config.get("trains", []):
                        if t.get("name") == getattr(self, 'target_train_name', ''):
                            wait_steps = int(t.get("start_condition", {}).get("step", 1800))
                            self.wait_sec = wait_steps * step_size
                            params = t.get("parameters", {})
                            self.brake_accel = abs(float(params.get("decelerating_acceleration_station", -3.5)))
                            self.fast_margine = float(params.get("fast_margine", 5.0))
                            self.slow_margine = float(params.get("slow_margine", 18.0))
            except Exception:
                pass

        if current_step <= 1 or not hasattr(self, 'prev_distance_m'):
            self.action_hold_time = 0
            self.previous_action = act
            self.visited_stations = set()
            self.prev_distance_m = distance_m
            self.start_pos_km = pos_km

        preceding_distance = info.get("preceding_distance", 9999.0)

        tve_reward = 0.0
        sl_penalty = 0.0
        ic_penalty = 0.0
        ef_penalty = 0.0
        tce_penalty = 0.0

        is_waiting_period = (current_time_sec < self.wait_sec)

        # ==========================================
        # 物理限界とフェーズの完全分離
        # ==========================================
        safe_v_station = math.sqrt(7.2 * self.brake_accel * max(distance_m, 0.0))
        safe_v_preced = math.sqrt(7.2 * 2.5 * max(preceding_distance - 50.0, 0.0))
        target_v_kmh = min(speed_limit_kmh, safe_v_station, safe_v_preced)

        margin = target_v_kmh - velocity_kmh

        # ★ 今回のブレイクスルー：今は「駅への減速中」なのかを明示的に判定
        is_station_phase = (target_v_kmh == safe_v_station) and (target_v_kmh < speed_limit_kmh - 5.0)

        # ==========================================
        # 1. 致命的な反則の即時罰金
        # ==========================================
        if velocity_kmh < -0.5: sl_penalty -= 500.0
        if preceding_distance <= 0: sl_penalty -= 1000.0

        # ==========================================
        # 2. 待機とフェーズ別速度制御
        # ==========================================
        if is_waiting_period:
            if velocity_kmh > 0.5: sl_penalty -= 100.0  
            if act == 2: tve_reward += 2.0  
            else: ef_penalty -= 5.0
        else:
            # 前進報酬（距離を稼ぐモチベーション）
            progress_m = self.prev_distance_m - distance_m
            if progress_m > 0:
                tve_reward += progress_m * 10.0 

            # Anti-Strike機構（発車の強制）
            if velocity_kmh < 1.0 and target_v_kmh > 5.0:
                if act == 0: tve_reward += 30.0
                else: tce_penalty -= 20.0

            # --- フェーズ制御 ---
            if margin < -2.0:
                # 【緊急ブレーキゾーン】速度超過
                sl_penalty -= abs(margin) * 5.0
                if act != 2: sl_penalty -= 50.0 

            elif margin <= self.fast_margine: 
                # 【上限到達ゾーン】目標速度に追いついた！
                if is_station_phase:
                    # 駅が近い -> ブレーキでカーブをなぞるのが大正解
                    if act == 2: tve_reward += 15.0
                    elif act == 1: tve_reward += 5.0
                    elif act == 0: ef_penalty -= 30.0
                else:
                    # 巡航中・先行列車追従中 -> 惰行(1)で距離を保つのが大正解（ブレーキは踏まない！）
                    if act == 1: tve_reward += 15.0
                    elif act == 0: ef_penalty -= 30.0
                    elif act == 2: ef_penalty -= 10.0 # 無駄なブレーキは罰金

            elif margin <= self.slow_margine: 
                # 【エコバンド・予備惰行ゾーン】
                if is_station_phase:
                    # 駅が近い -> 早めにアクセルを離して予備惰行
                    if act == 1: tve_reward += 15.0
                    elif act == 0: ef_penalty -= 30.0
                else:
                    # 巡航中・先行列車追従中 -> 加速か惰行をキープ
                    if act == 0 or act == 1: 
                        tve_reward += 10.0
                    if act == self.previous_action: 
                        tve_reward += 5.0 # フラフラせずキープしたらボーナス

            else:
                # 【全力加速ゾーン】
                if act == 0: tve_reward += 15.0
                elif act == 2: ef_penalty -= 20.0

        self.prev_distance_m = distance_m

        # ==========================================
        # 3. 操作保持（チャタリング防止）
        # ==========================================
        if not is_waiting_period:
            if act == self.previous_action:
                self.action_hold_time += 1
                ic_penalty += 1.0
            else:
                # 駅の減速カーブ中のみ3秒、それ以外は7秒のホールド
                min_hold = 3 if is_station_phase else 7
                if velocity_kmh < 1.0 or margin < -2.0:
                    min_hold = 0 # 発車時・緊急時は無罪

                if self.action_hold_time < min_hold:
                    ic_penalty -= 30.0  
                self.action_hold_time = 0
        self.previous_action = act

        # ==========================================
        # 4. 終端報酬
        # ==========================================
        terminal_reward = 0.0
        info["arrival_time_sec"] = current_time_sec

        if done:
            error_m = abs(pos_km - self.target_station_km) * 1000.0
            if error_m <= 50.0 and velocity_kmh < 1.0:
                # 停止成功
                precision_bonus = 3000.0 / math.sqrt(max(error_m, 0.1))
                terminal_reward += precision_bonus
                
                target_run_time = 120.0 
                actual_run_time = current_time_sec - self.wait_sec
                info["actual_running_time"] = actual_run_time
                time_diff = abs(target_run_time - actual_run_time)
                
                if time_diff <= 5.0: terminal_reward += 2000.0
                else: terminal_reward -= time_diff * 10.0
                
            elif pos_km > self.target_station_km + 0.05:
                # オーバーラン
                terminal_reward -= 5000.0
            else:
                # 途中終了（タイムアウトなど）：容赦ない罰金で「もっと早く走れ」と促す
                terminal_reward -= (distance_m ** (1/3)) * 200.0

        reward_components = {
            'tve_reward': tve_reward, 'tce_penalty': tce_penalty,
            'ic_penalty': ic_penalty, 'ef_penalty': ef_penalty,
            'sl_penalty': sl_penalty, 'de_bonus': terminal_reward
        }
        return float(sum(reward_components.values())), reward_components