import math
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from experiments.rl_env.rl_driving_decision import RLDrivingDecision
from train_grapher_v3.core.block_system import FixedBlockSystem, MovingBlockSystem
from train_grapher_v3.core.line import Line
from train_grapher_v3.util.simulation_model_io import load_simulation_model


class TrainEnv(gym.Env):
    """列車制御のための強化学習環境 (Gymnasium準拠)"""

    def __init__(self, json_path: str, target_train_name: str, save_dir: str = None, reward_fn=None):
        super().__init__()
        self.json_path = json_path
        self.target_train_name = target_train_name
        self.save_dir = save_dir

        # エピソード記録用
        self.episode_count = 1
        self.step_history = []
        self.log_interval = 50  # ★ 50エピソードに1回だけログを取る

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
        self.current_step = 0
        self.step_history = []  # ★ 新しいエピソードの開始時に履歴をリセット

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
        skip_frames = int(1.0 / self.step_size) if hasattr(self, "step_size") else 10

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
        # ★ ログ取得判定：50エピソードごと、または初回のみ記録
        # ==========================================
        is_logging_episode = (self.episode_count % self.log_interval == 0) or (self.episode_count == 1)

        # 記録対象のエピソードのみ履歴を保存する（メモリ節約）
        if self.save_dir is not None and is_logging_episode:
            v_ms = float(obs[0])
            velocity_kmh = v_ms * 3.6
            pos_km = info.get("train_position").value if info.get("train_position") else 0.0

            row = {
                "step": self.current_step,
                "position_km": pos_km,
                "velocity_kmh": velocity_kmh,
                "action": action,
                "tve_reward": reward_components.get("tve_reward", 0.0),
                "tce_penalty": reward_components.get("tce_penalty", 0.0),
                "ic_penalty": reward_components.get("ic_penalty", 0.0),
                "ef_penalty": reward_components.get("ef_penalty", 0.0),
                "sl_penalty": reward_components.get("sl_penalty", 0.0),
                "de_bonus": reward_components.get("de_bonus", 0.0),
                "step_total_reward": reward,
            }
            self.step_history.append(row)

        # ==========================================
        # エピソード終了時の処理
        # ==========================================
        if done:
            if self.save_dir is not None and is_logging_episode and len(self.step_history) > 0:
                import pandas as pd

                df = pd.DataFrame(self.step_history)

                # ① 人間用のフルログ（全ステップ保存）
                full_filename = f"episode_{self.episode_count:04d}_full.csv"
                df.to_csv(os.path.join(self.save_dir, full_filename), index=False, encoding="utf-8-sig")

                # ==========================================
                # ② LLM用の要約ログ（フェーズ圧縮方式）
                # ==========================================
                phase_history = []
                start_idx = 0

                for i in range(1, len(df) + 1):
                    is_last = i == len(df)

                    if not is_last:
                        # 区切る条件：アクションが変わった、または特大ペナルティを食らった
                        action_changed = df.loc[i, "action"] != df.loc[i - 1, "action"]
                        huge_penalty = (
                            df.loc[i, "ic_penalty"] <= -20
                            or df.loc[i, "ef_penalty"] <= -20
                            or df.loc[i, "sl_penalty"] <= -10
                        )
                        should_break = action_changed or huge_penalty
                    else:
                        should_break = True

                    # ブロックの区切りが来たら、1行に圧縮して記録
                    if should_break:
                        end_idx = i - 1
                        segment = df.iloc[start_idx : end_idx + 1]

                        # 行動（アクション）をLLMが分かりやすい文字列に変換
                        act_val = int(segment.iloc[0]["action"])
                        act_str = {0: "0_加速", 1: "1_惰行", 2: "2_ブレーキ"}.get(act_val, str(act_val))

                        phase_row = {
                            "step_range": f"{int(segment.iloc[0]['step'])} -> {int(segment.iloc[-1]['step'])}",
                            "duration_sec": round(len(segment) * 0.1, 1),
                            "action": act_str,  # ★行動を明記！
                            "vel_kmh_change": f"{segment.iloc[0]['velocity_kmh']:.1f} -> {segment.iloc[-1]['velocity_kmh']:.1f}",
                            "pos_km_change": f"{segment.iloc[0]['position_km']:.3f} -> {segment.iloc[-1]['position_km']:.3f}",
                            "sum_tve_reward": round(segment["tve_reward"].sum(), 1),
                            "sum_ic_penalty": round(segment["ic_penalty"].sum(), 1),
                            "sum_ef_penalty": round(segment["ef_penalty"].sum(), 1),
                            "sum_sl_penalty": round(segment["sl_penalty"].sum(), 1),
                            "total_reward": round(segment["step_total_reward"].sum(), 1),
                        }
                        phase_history.append(phase_row)
                        start_idx = i  # 次のブロックの開始位置を更新

                # 圧縮した要約版を保存
                df_llm = pd.DataFrame(phase_history)
                llm_filename = f"episode_{self.episode_count:04d}_llm_summary.csv"
                df_llm.to_csv(os.path.join(self.save_dir, llm_filename), index=False, encoding="utf-8-sig")

            # 内部変数のリセット
            self.step_history = []
            self.episode_count += 1

        info["reward_components"] = reward_components
        truncated = False
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
            "train_position": self.target_train.get_position(self.current_step - 1),
        }

    def _save_reward_to_csv(self):
        """
        エピソード終了時に呼び出され、報酬履歴を10区間に分割して
        最大・平均・最小値を算出し、CSVに保存する。
        """
        if not hasattr(self, "reward_history") or len(self.reward_history) == 0:
            return

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
        df["segment"] = np.floor(df.index / (total_steps / num_segments)).astype(int)
        df["segment"] = df["segment"].clip(upper=num_segments - 1)  # 最大値を9に丸める

        # 区間(segment)ごとにグループ化し、各カラムの 最大(max), 平均(mean), 最小(min) を計算
        summary_df = df.groupby("segment").agg(["max", "mean", "min"])

        # カラム名をフラットで見やすくする（例: 'distance_penalty_max'）
        summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]

        # CSVファイルとして保存（LLMが読み込みやすい場所に保存）
        # ※ファイル名は上書きされる仕様にしていますが、エピソードごとに分けたい場合は
        #   ファイル名に self.current_episode などの変数を足してください。
        save_path = "llm_feedback_reward_summary.csv"
        summary_df.to_csv(save_path)

        # コンソールへの通知（デバッグ用・不要なら削除可）
        # print(f"=== 報酬の推移データを {save_path} に保存しました ===")

    # obs：シミュレータからの情報，action：エージェントの行動，info：シミュレータからの追加情報，done：エピソード終了フラグ
    def default_reward_fn(self, obs, action, info, done):
        """
        階層型強化学習：鉄道の定刻運行・省エネ・安全性を考慮した総合的な報酬関数
        修正版：スケール不一致の解消、Bang-Bang制御の撲滅、惰行（エコドライブ）の明確なボーナス化
        """
        import math
        
        # ==========================================
        # 1. 状態の取得と変数の初期化
        # ==========================================
        velocity_kmh, speed_limit_kmh, _, time_left = obs
        velocity_ms = velocity_kmh / 3.6
        
        # 制限速度のハードキャップ (100km/h想定)
        speed_limit_ms = min(speed_limit_kmh / 3.6, 27.78)
        
        current_position = info.get("train_position", None)
        pos_km = current_position.value if hasattr(current_position, "value") else 0.0
        
        # actionの定義: 0=加速(力行), 1=惰行, 2=ブレーキ
        act = int(action)
        current_step = getattr(self, "current_step", 0)
        train = getattr(self, "train", None)
        
        tve_reward = 0.0   
        tce_penalty = 0.0  
        ic_penalty = 0.0   
        ef_penalty = 0.0   
        sl_penalty = 0.0   
        de_bonus = 0.0     

        # 初期化処理
        if current_step <= 1 or not hasattr(self, "prev_distance_m"):
            self.action_hold_time = 0
            self.previous_action = act
            self.visited_stations = set()
            
            self.target_station_km = 2.0
            self.wait_steps = 1800
            if train is not None:
                next_station_info = train.get_next_station_info(current_step)
                if next_station_info and next_station_info[0]:
                    self.target_station_km = next_station_info[0].position.value
                    self.current_station_id = next_station_info[0].id
            
            self.prev_distance_m = abs(self.target_station_km - pos_km) * 1000.0
            self.start_pos_km = pos_km

        distance_m = abs(self.target_station_km - pos_km) * 1000.0
        preceding_distance = info.get("preceding_distance", 9999.0)
        is_waiting_period = current_step < self.wait_steps

        # ==========================================
        # 2. 待機期間中 (フライング防止)
        # ==========================================
        if is_waiting_period:
            if velocity_ms < 0.1 and act == 2: 
                tve_reward += 1.0  
            elif velocity_ms > 0.1:
                sl_penalty -= (10.0 + velocity_ms * 10.0)

        # ==========================================
        # 3. 走行期間中 (速度管理と進行)
        # ==========================================
        if not is_waiting_period:
            # 機外停車（立ち往生）の厳罰化。0.8km地点などで止まるのを防ぐ
            if velocity_ms < 0.1 and distance_m > 50.0:
                tce_penalty -= 10.0  

            # 理想の目標上限速度の計算
            v_brake = math.sqrt(2.0 * 0.7 * max(distance_m - 5.0, 0.0))
            min_safe_distance = 100.0
            # 先行列車に対する減速度も0.7m/s^2を想定して計算を精緻化
            v_preceding = math.sqrt(2.0 * 0.7 * max(preceding_distance - min_safe_distance, 0.0))
            target_v_max = min(speed_limit_ms, v_brake, v_preceding)

            # 速度超過とオーバーラン
            is_overspeed = velocity_ms > target_v_max
            if is_overspeed:
                over_speed = velocity_ms - target_v_max
                # 軽微な超過は線形ペナルティにし、フルブレーキの連打を防ぐ
                if over_speed < 2.0:
                    sl_penalty -= over_speed * 5.0
                else:
                    sl_penalty -= (over_speed ** 2) * 2.0 
                
            if pos_km > self.target_station_km:
                over_run_m = (pos_km - self.target_station_km) * 1000.0
                sl_penalty -= (10.0 + over_run_m * 2.0) 

            # 進行報酬とエコドライブボーナス
            progress_m = self.prev_distance_m - distance_m
            if pos_km <= self.target_station_km and progress_m > 0:
                if not is_overspeed:
                    # 進行報酬のスケールを下げ、停止ボーナスに価値を移す
                    tve_reward += progress_m * 1.0 
                    
                    # 【新設】エコドライブボーナス：目標速度付近で「惰行(act=1)」していると稼げる
                    if (target_v_max - 5.0) <= velocity_ms <= target_v_max and velocity_ms > 2.0:
                        if act == 1:
                            tve_reward += 2.0 

            # 力行(act=0)時のペナルティ（惰行への誘導）
            if act == 0: 
                ef_penalty -= 0.5 
                # 目標速度に近づいているのに加速すると強めの罰
                if velocity_ms > max(target_v_max - 3.0, 0.0):
                    ef_penalty -= 5.0 

        self.prev_distance_m = distance_m

        # ==========================================
        # 4. 乗り心地 (バンバン制御の赤字化)
        # ==========================================
        if not is_waiting_period:
            if act == self.previous_action:
                self.action_hold_time += 1
            else:
                min_hold_steps = 70 
                if self.action_hold_time < min_hold_steps:
                    # 切り替え罰を大幅強化し、頻繁なノッチ操作を割に合わなくする
                    penalty_val = -20.0 
                    if abs(self.previous_action - act) == 2:
                        penalty_val -= 50.0  # 加速⇔ブレーキの直接切り替えは特大ペナルティ
                    ic_penalty += penalty_val
                self.action_hold_time = 0
        self.previous_action = act

        # ==========================================
        # 5. 停止位置精度ボーナス (ゴールへの超・引力)
        # ==========================================
        if not is_waiting_period and pos_km <= self.target_station_km + 0.02:
            if distance_m <= 100.0:
                de_bonus += 5.0 * math.exp(-distance_m / 20.0)
                if velocity_ms < 1.0:
                    station_id = getattr(self, "current_station_id", "station_target")
                    if station_id not in self.visited_stations:
                        if distance_m <= 10.0:
                            de_bonus += 500.0
                            if distance_m <= 1.0: 
                                de_bonus += 2000.0
                            self.visited_stations.add(station_id)

        terminal_reward = 0.0
        if done and distance_m <= 5.0 and velocity_ms < 0.5:
            # 駅に停まるモチベーションを最大化するため、ボーナスを5000に引き上げ
            terminal_reward += 5000.0

        # ==========================================
        # 6. 合計報酬と辞書の構築
        # ==========================================
        reward_components = {
            "tve_reward": tve_reward,
            "tce_penalty": tce_penalty,
            "ic_penalty": ic_penalty,
            "ef_penalty": ef_penalty,
            "sl_penalty": sl_penalty,
            "de_bonus": de_bonus + terminal_reward,
        }
        
        total_reward = float(sum(reward_components.values()))
        
        return total_reward, reward_components