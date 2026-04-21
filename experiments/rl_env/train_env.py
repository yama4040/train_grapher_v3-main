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
        # [現在の速度(km/s), 制限速度(km/s), 次の駅までの残距離(km), 目標到着時刻までの残り時間(s)]
        high = np.array([150.0, 150.0, 3.0, 3600.0], dtype=np.float32)
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
            velocity_kmh = float(obs[0])
            
            #v_ms = float(obs[0])
            #velocity_kmh = v_ms * 3.6
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
                import numpy as np
                df = pd.DataFrame(self.step_history)
                
                # ==========================================
                # ① 人間用のフルログ（全ステップ保存）
                # ==========================================
                full_filename = f"episode_{self.episode_count:04d}_full.csv"
                df.to_csv(os.path.join(self.save_dir, full_filename), index=False, encoding='utf-8-sig')
                
                # ==========================================
                # ② Eureka方式：50区間の統計データ（マクロ分析用）
                # ==========================================
                num_segments = min(50, len(df))
                
                summary_data = []
                target_cols = ['tve_reward', 'tce_penalty', 'ic_penalty', 'ef_penalty', 'sl_penalty', 'de_bonus']
                
                # ★ NumPyに頼らず、Pandasの標準機能(iloc)で確実に分割する
                chunk_size = max(1, len(df) // num_segments)
                
                for i in range(num_segments):
                    start_idx = i * chunk_size
                    # 最後の区間は、余った端数の行まで全て含める
                    end_idx = len(df) if i == num_segments - 1 else (i + 1) * chunk_size
                    
                    segment = df.iloc[start_idx:end_idx]
                    
                    if len(segment) == 0: continue
                    
                    row_data = {"segment": i}
                    for col in target_cols:
                        row_data[f"{col}_max"] = round(segment[col].max(), 2)
                        row_data[f"{col}_mean"] = round(segment[col].mean(), 2)
                        row_data[f"{col}_min"] = round(segment[col].min(), 2)
                    summary_data.append(row_data)
                
                df_summary = pd.DataFrame(summary_data)
                summary_filename = f"episode_{self.episode_count:04d}_summary.csv"
                df_summary.to_csv(os.path.join(self.save_dir, summary_filename), index=False, encoding='utf-8-sig')

                # ==========================================
                # ③ LLM用プロンプト：各成分の致命的シーン抽出（ミクロ分析用）
                # ==========================================
                report = []
                report.append(f"■ エピソード {self.episode_count} の学習フィードバックレポート ■")
                report.append(f"・終了時の位置: {df.iloc[-1]['position_km']:.3f} km / {self.target_station_km} km")
                report.append(f"・合計報酬: {df['step_total_reward'].sum():.1f} 点\n")
                
                # 分析したいペナルティ成分のリスト
                penalty_targets = {
                    "ic_penalty": "チャタリング・保持違反",
                    "ef_penalty": "エネルギー非効率・サボり",
                    "sl_penalty": "速度超過・追突・逆走"
                }

                report.append("▼ 各ペナルティの致命的な勘違いシーン（最悪の瞬間とその前後）▼")
                
                for col, name in penalty_targets.items():
                    min_val = df[col].min()
                    # 罰金が発生していない（0以上）ならスキップ
                    if min_val >= -0.1:
                        report.append(f"\n【{name} ({col})】: 致命的な罰金はありませんでした。")
                        continue
                    
                    # 最も悪い値を出したステップのインデックスを取得
                    worst_idx = df[col].idxmin()
                    start_idx = max(0, worst_idx - 3)
                    end_idx = min(len(df), worst_idx + 4)
                    worst_scene = df.iloc[start_idx:end_idx]
                    
                    report.append(f"\n【{name} ({col})】 ワースト記録: {min_val:.1f} 点")
                    report.append(" Step | 速 度 | Action | 前進tve | 保持ic | 効率ef | 超過sl | サボtce")
                    report.append("-" * 75)
                    
                    for _, row in worst_scene.iterrows():
                        act_str = {0: "0(加速)", 1: "1(惰行)", 2: "2(減速)"}.get(int(row["action"]), str(int(row["action"])))
                        # ワーストステップには矢印をつけて目立たせる
                        marker = "=>" if int(row['step']) == int(df.iloc[worst_idx]['step']) else "  "
                        
                        step_info = (f"{marker}{int(row['step']):>4} | {row['velocity_kmh']:>4.1f} | "
                                     f"{act_str:>6} | "
                                     f"{row['tve_reward']:>6.1f} | {row['ic_penalty']:>6.1f} | "
                                     f"{row['ef_penalty']:>6.1f} | {row['sl_penalty']:>6.1f} | {row['tce_penalty']:>6.1f}")
                        report.append(step_info)

                # テキストファイルとして保存
                txt_filename = f"episode_{self.episode_count:04d}_llm_prompt.txt"
                with open(os.path.join(self.save_dir, txt_filename), "w", encoding="utf-8") as f:
                    f.write("\n".join(report))
                
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
        speed_limit = self.rl_decision.last_signal_speed

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

    def default_reward_fn(self, obs, action, info, done):
        """
        階層型強化学習：鉄道の定刻運行・省エネ・安全性を考慮した総合的な報酬関数
        ユーザー提案反映・究極版：「本物の惰行」の強制と、ブレーキの完全解放
        """
        import math
        
        # ==========================================
        # 1. 状態の取得と変数の初期化
        # ==========================================
        velocity_kmh, speed_limit_kmh, _, time_left = obs
        speed_limit_kmh = min(speed_limit_kmh, 100.0) 
        
        current_position = info.get("train_position", None)
        pos_km = current_position.value if hasattr(current_position, "value") else 0.0
        act = int(action)
        current_step = getattr(self, "current_step", 0)
        train = getattr(self, "train", None)
        
        tve_reward = 0.0   
        tce_penalty = 0.0  
        ic_penalty = 0.0   
        ef_penalty = 0.0   
        sl_penalty = 0.0   
        de_bonus = 0.0     

        if current_step <= 1 or not hasattr(self, "prev_distance_km"):
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
            
            self.prev_distance_km = abs(self.target_station_km - pos_km)
            self.start_pos_km = pos_km

        distance_km = abs(self.target_station_km - pos_km)
        preceding_distance_km = info.get("preceding_distance", 9999.0)
        is_waiting_period = current_step < self.wait_steps

        # ==========================================
        # 2. 待機期間中
        # ==========================================
        if is_waiting_period:
            if velocity_kmh < 0.5 and act == 2:
                tve_reward += 1.0  
            elif velocity_kmh > 0.5:
                sl_penalty -= 5.0  

        # ==========================================
        # 3. 走行期間中 (速度管理と進行)
        # ==========================================
        if not is_waiting_period:
            tce_penalty -= 0.5  

            if pos_km > self.target_station_km:
                target_v_max_kmh = 0.0
            else:
                v_brake_kmh = math.sqrt(7200.0 * 1.5 * max(distance_km - 0.002, 0.0))
                min_safe_distance_km = 0.100 
                v_preceding_kmh = math.sqrt(7200.0 * 1.5 * max(preceding_distance_km - min_safe_distance_km, 0.0))
                target_v_max_kmh = min(speed_limit_kmh, v_brake_kmh, v_preceding_kmh)

            # パニック防止の遊び（マージン）
            allowable_v_max = target_v_max_kmh + 3.0
            is_overspeed = velocity_kmh > allowable_v_max
            
            if is_overspeed:
                over_speed_kmh = velocity_kmh - allowable_v_max
                sl_penalty -= min(over_speed_kmh * 2.0, 30.0) 
                
            if velocity_kmh > 126.0: 
                sl_penalty -= 100.0 

            # 進行報酬
            progress_km = self.prev_distance_km - distance_km
            if pos_km <= self.target_station_km and progress_km > 0:
                if not is_overspeed:
                    tve_reward += progress_km * 10000.0  
                else:
                    max_allowed_progress_km = target_v_max_kmh / 36000.0
                    capped_progress_km = min(progress_km, max_allowed_progress_km)
                    tve_reward += capped_progress_km * 10000.0  
                
                # 【改善1】駅接近時（500m以内）の厳格な行動誘導
                if distance_km <= 0.500:
                    if act == 2 and velocity_kmh > 1.8 and not is_overspeed: 
                        tve_reward += 5.0  
                    elif act == 0 and velocity_kmh > 0.5: 
                        # 波状運転（尺取り虫）を防ぐため、駅構内での再加速を完全禁止
                        tce_penalty -= 20.0  
                        
            if act == 0 and velocity_kmh > 3.6 and distance_km > 0.500: 
                ef_penalty -= 0.5 

        self.prev_distance_km = distance_km

        # ==========================================
        # 4. 乗り心地（ユーザー提案の完成形：本物の惰行とブレーキ解放）
        # ==========================================
        if not is_waiting_period:
            if act != self.previous_action:
                # 【改善2】0⇔2の直接切り替えは「絶対悪」
                if abs(self.previous_action - act) == 2:
                    ic_penalty -= 20.0  
                else:
                    # 【改善3】1(惰行)から2(減速)へ切り替える時は、保持ルールを完全に免除！
                    # これにより、AIはいつでも好きなタイミングで一発ブレーキを踏める
                    if self.previous_action == 1 and act == 2:
                        ic_penalty -= 0.0  
                    else:
                        # 【改善4】「本物の惰行」の強制
                        if self.previous_action == 1:
                            # 1(惰行)から0(加速)に戻す場合、最低3秒(30step)は惰行を維持させる
                            # これで「一瞬の惰行」を使ったノコギリ運転が不可能になる
                            if self.action_hold_time < 30:
                                ic_penalty -= 10.0
                        else:
                            # 0(加速)から1(惰行)へ切り替える時は、入りやすくする
                            if self.action_hold_time < 30:
                                ic_penalty -= 2.0
                            else:
                                ic_penalty -= 0.5
                self.action_hold_time = 0
            else:
                self.action_hold_time += 1
        self.previous_action = act

        # ==========================================
        # 5. 停止位置精度と終了判定
        # ==========================================
        terminal_reward = 0.0
        if done:
            if velocity_kmh < 1.8: 
                if distance_km <= 0.005: 
                    terminal_reward += 100000.0  
                else:
                    if pos_km > self.target_station_km:
                        terminal_reward += 50000.0 * math.exp(-distance_km / 0.005) 
                    else:
                        terminal_reward += 50000.0 * math.exp(-distance_km / 0.020) 
            else:
                terminal_reward -= 5000.0 
        else:
            if not is_waiting_period and pos_km <= self.target_station_km:
                if distance_km <= 0.500: 
                    de_bonus += 5.0 * math.exp(-distance_km / 0.100) 

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