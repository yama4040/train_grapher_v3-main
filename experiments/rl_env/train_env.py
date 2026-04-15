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
        self.target_train_name = target_train_name  # 学習対象の列車名
        self.save_dir = save_dir  # CSV保存先のディレクトリ

        # ★ エピソード記録用の変数を準備
        self.episode_count = 1
        self.step_history = []

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
        # ★ 追加：1ステップごとの詳細データを記録
        # ==========================================
        if self.save_dir is not None:
            # 分析しやすいように、観測値なども一緒に記録しておきます
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
        # ★ 追加：エピソード終了時にCSVとして保存
        # ==========================================
        if done:
            if self.save_dir is not None and len(self.step_history) > 0:
                import pandas as pd

                df = pd.DataFrame(self.step_history)

                # ① 人間用のフルログ（全ステップ保存）
                full_filename = f"episode_{self.episode_count:04d}_full.csv"
                df.to_csv(os.path.join(self.save_dir, full_filename), index=False, encoding="utf-8-sig")

                # ==========================================
                # ② LLMフィードバック用の要約ログ（ハイライト抽出）
                # ==========================================
                trigger_indices = set()

                for i in range(1, len(df)):
                    # トリガーA: アクション（ノッチ）が変化した瞬間
                    if df.loc[i, "action"] != df.loc[i - 1, "action"]:
                        trigger_indices.add(i)

                    # トリガーB: 特大ペナルティ（-20以下）が発生した瞬間
                    if (
                        df.loc[i, "ic_penalty"] <= -20
                        or df.loc[i, "ef_penalty"] <= -20
                        or df.loc[i, "sl_penalty"] <= -10
                    ):
                        trigger_indices.add(i)

                # トリガーC: 最後の5ステップ（終了間際）
                for i in range(max(0, len(df) - 5), len(df)):
                    trigger_indices.add(i)

                # 各トリガーの「前後5ステップ」を抽出対象にする
                window_size = 5
                extract_indices = set()
                for idx in trigger_indices:
                    for j in range(max(0, idx - window_size), min(len(df), idx + window_size + 1)):
                        extract_indices.add(j)

                # インデックスをソートして抽出
                sorted_indices = sorted(list(extract_indices))
                df_llm = df.iloc[sorted_indices].copy()

                # 抽出した要約版を保存
                llm_filename = f"episode_{self.episode_count:04d}_llm_summary.csv"
                df_llm.to_csv(os.path.join(self.save_dir, llm_filename), index=False, encoding="utf-8-sig")

                self.episode_count += 1

            self.step_history = []  # 次のエピソードに向けてクリア

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
        後続列車の走行パターン最適化（CBTC、機外停車回避、乗り心地重視）
        """

        # ==========================================
        # 1. 状態の取得と変数の初期化
        # ==========================================
        velocity_kmh, speed_limit_kmh, _, time_left = obs
        velocity_ms = velocity_kmh / 3.6
        speed_limit_ms = speed_limit_kmh / 3.6

        current_position = info.get("train_position", None)
        pos_km = current_position.value if hasattr(current_position, "value") else 0.0
        act = int(action)

        current_step = getattr(self, "current_step", 0)
        train = getattr(self, "train", None)

        # 報酬成分の初期化
        progress_reward = 0.0  # 進行報酬
        delay_penalty = 0.0  # 遅延ペナルティ
        comfort_penalty = 0.0  # 乗り心地（ジャーク・スイッチング）ペナルティ
        energy_penalty = 0.0  # 消費エネルギーペナルティ
        safety_penalty = 0.0  # 制限速度・機外停車・接近ペナルティ
        stop_bonus = 0.0  # 停止位置精度ボーナス

        # 初期化処理（エピソード開始時）
        if current_step <= 1 or not hasattr(self, "prev_distance_m"):
            self.action_hold_time = 0
            self.previous_action = act
            self.visited_stations = set()

            # 目標駅の取得
            self.target_station_km = 2.0
            self.wait_steps = 1800
            if train is not None:
                next_station_info = train.get_next_station_info(current_step)
                if next_station_info and next_station_info[0]:
                    self.target_station_km = next_station_info[0].position.value
                    self.current_station_id = next_station_info[0].id

            self.prev_distance_m = abs(self.target_station_km - pos_km) * 1000.0
            self.start_pos_km = pos_km

        # 更新される距離と先行列車の情報
        distance_m = abs(self.target_station_km - pos_km) * 1000.0
        preceding_distance = info.get("preceding_distance", 9999.0)

        # 待機状態の判定
        is_waiting_period = current_step < self.wait_steps

        # ==========================================
        # 2. 安全性に関するペナルティ (Safety Penalty)
        # ==========================================
        # (A) フライングペナルティ
        if is_waiting_period and velocity_ms > 0.1:
            safety_penalty -= 500.0

        if not is_waiting_period:
            # (B) 逆走・オーバーラン・衝突
            if velocity_kmh < -0.5:
                safety_penalty -= 2000.0
            if pos_km > self.target_station_km + 0.05:
                safety_penalty -= 2000.0
            if preceding_distance <= 0:
                safety_penalty -= 5000.0

            # (C) 禁止行動への強力なペナルティ（速度超過時の加速・惰行など）
            # ※本来はアクションマスクで防ぐべきですが、報酬でも学習させます
            is_overspeed = velocity_ms > speed_limit_ms
            if is_overspeed:
                safety_penalty -= (velocity_ms - speed_limit_ms) * 50.0  # 速度超過ペナルティ
                if act in [1, 2]:  # 加速(1)や惰行(2)を禁止（※アクション定義に依存）
                    safety_penalty -= 1000.0

            # 先行列車接近時の加速禁止
            min_safe_distance = 100.0  # CBTCの安全距離に基づく仮の値
            if preceding_distance < min_safe_distance and act == 1:
                safety_penalty -= 1000.0

            # (D) 機外停車ペナルティ
            # 速度0かつ駅停車中でない場合は重いペナルティ
            is_station_stopping = info.get("is_station_stopping", False)
            if velocity_ms < 0.1 and not is_station_stopping and distance_m > 50.0:
                safety_penalty -= 500.0

        # ==========================================
        # 3. 進行・追従に関する報酬 (Progress & Tracking)
        # ==========================================
        if not is_waiting_period:
            # 駅停車のための安全速度と先行列車との間隔維持
            safe_v_station = math.sqrt(3.0 * max(distance_m, 0.0))
            safe_v_preceding = math.sqrt(2.0 * max(preceding_distance - min_safe_distance, 0.0))
            target_velocity_ms = min(speed_limit_ms, safe_v_station, safe_v_preceding)

            progress_m = self.prev_distance_m - distance_m

            # 目標地点に近づいていることに対する報酬
            if progress_m > 0:
                progress_reward += progress_m * 2.0

            # 目標速度への追従評価
            if velocity_ms <= target_velocity_ms + 1.0:
                progress_reward += velocity_ms * 0.2
            else:
                over_speed = velocity_ms - target_velocity_ms
                progress_reward -= (over_speed**2) * 1.5

        self.prev_distance_m = distance_m

        # ==========================================
        # 4. 乗り心地ペナルティ (Comfort Penalty)
        # ==========================================
        if not is_waiting_period:
            if act == self.previous_action:
                self.action_hold_time += 1
            else:
                # 最低7秒（約70ステップ想定、1step=0.1sの場合）保持の確認
                # ここではプロンプトの「最低7秒間」に基づき、1step=0.1sとして70を設定（適宜調整）
                min_hold_steps = 70
                if self.action_hold_time < min_hold_steps:
                    penalty_val = -2.0 * (min_hold_steps - self.action_hold_time)

                    # 加速⇔減速の直接切り替え（惰行を挟まない場合）はペナルティ2倍
                    # アクションの定義（例: 0=ブレーキ, 1=惰行, 2=力行）に応じて判定
                    if abs(self.previous_action - act) >= 2:
                        penalty_val *= 2.0

                    comfort_penalty += penalty_val

                self.action_hold_time = 0
        self.previous_action = act

        # ==========================================
        # 5. 遅延ペナルティ & 消費エネルギー (Time & Energy)
        # ==========================================
        if not is_waiting_period:
            # 遅延ペナルティ：予定走行時間をオーバーした場合（time_leftが負になった場合など）
            if time_left < 0:
                delay_penalty -= 1.0  # 1秒ごとにペナルティを加算

            # 消費エネルギーペナルティ：加速（力行）時に微小なペナルティ
            if act == 2:  # 2を力行とした場合
                energy_penalty -= 0.05 * velocity_ms

        # ==========================================
        # 6. 停止位置精度ボーナス (Stop Accuracy Bonus)
        # ==========================================
        terminal_reward = 0.0
        if not is_waiting_period and distance_m <= 50.0:
            # 駅に接近するにつれてTD誤差を意識した優先度的な引力ボーナス
            stop_bonus += 10.0 * math.exp(-distance_m / 10.0)

            if velocity_ms < 0.5:
                station_id = getattr(self, "current_station_id", "station_target")
                if station_id not in self.visited_stations:
                    if distance_m <= 10.0:
                        # 10m以内でボーナス（距離が近いほど指数関数的に増加）
                        stop_bonus += 500.0 * math.exp(-distance_m)

                        # 1m以内の場合はドア開扉可能条件を満たすため特大ボーナス
                        if distance_m <= 1.0:
                            stop_bonus += 2000.0

                        self.visited_stations.add(station_id)

        if done and distance_m <= 10.0:
            terminal_reward += 1000.0

        # ==========================================
        # 7. 合計報酬と辞書の構築
        # ==========================================
        reward_components = {
            "progress_reward": progress_reward,
            "delay_penalty": delay_penalty,
            "comfort_penalty": comfort_penalty,
            "energy_penalty": energy_penalty,
            "safety_penalty": safety_penalty,
            "stop_bonus": stop_bonus + terminal_reward,
        }

        total_reward = float(sum(reward_components.values()))

        return total_reward, reward_components
