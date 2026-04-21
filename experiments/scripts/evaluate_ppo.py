import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# パスの追加
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO

from experiments.rl_env.train_env import TrainEnv
from train_grapher_v3.util.logger import setup_logger

logger = setup_logger(__name__)


def main():
    # ==========================================
    # 1. 環境とモデルの読み込み
    # ==========================================
    json_path = "datas/multi_train_simulation_model_generated.json"
    target_train_name = "普通列車2"  # ★修正済みの列車名

    # モデルのパスを明示的に指定（学習後のモデルをロードするため）
    model_path = "PPO_datas/20260421165141/ppo_final_model.zip"

    if not os.path.exists(model_path):
        logger.error(f"モデルが見つかりません。先に学習を実行してください: {model_path}")
        return

    logger.info("環境とモデルを読み込み中...")
    env = TrainEnv(json_path=json_path, target_train_name=target_train_name)
    model = PPO.load(model_path)

    # ==========================================
    # 2. AIによるテスト走行（推論）の実行
    # ==========================================
    logger.info("AIによるテスト走行を開始します...")

    # === (テスト走行開始前) ===
    obs, info = env.reset()

    # ★ JSONから駅の距離を引用
    import json

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            target_pos_km = float(config["line_shape"]["edges"][0]["stations"][0]["value"])
    except Exception:
        target_pos_km = 2.0  # 読み込めなかった時の保険

    done = False

    times = []
    positions = []
    velocities = []
    actions_taken = []
    speed_limits = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        current_time = info["time"]
        velocity = obs[0]
        speed_limit = obs[1]

        # ★追加: Positionオブジェクトから「距離(km)」の数値だけを安全に抽出
        pos_obj = info["train_position"]
        if hasattr(pos_obj, "value"):
            pos_km = pos_obj.value
        else:
            pos_km = 0.0  # エラー回避用のフォールバック

        times.append(current_time)
        positions.append(pos_km)
        velocities.append(velocity)
        actions_taken.append(action)
        speed_limits.append(speed_limit)

    logger.info(f"テスト走行完了！ (総時間: {times[-1]:.1f} 秒, 到達距離: {positions[-1]:.2f} km)")

    # ==========================================
    # 3. 走行結果のグラフ化（2つのウィンドウを表示）
    # ==========================================
    logger.info("走行結果のグラフを描画します...")

    # 【グラフ1】 距離ベース（鉄道本来のランカーブ）
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(positions, speed_limits, color="red", linestyle="--", alpha=0.7, label="Speed Limit (km/h)")
    ax1.plot(positions, velocities, color="blue", linewidth=2, label="Train Velocity (km/h)")

    # ==========================================
    # ★ ここに追加：駅の停止位置（目標位置）に太い黒線を引く
    # ==========================================
    # シミュレーション終了時（ループを抜けた後）のAIの最終データから逆算します。
    # obs は [速度, 制限速度, 残り距離(m), 残り時間] なので、obs[2] が残り距離です。
    final_distance_km = float(obs[2]) / 1000.0

    # 最終的な列車の位置(km) ＋ 残り距離(km) ＝ 目標の駅の位置(km)
    # === evaluate_ppo.py のグラフ描画部分 ===
    target_pos_km = 2.0  # ★絶対に 2.0 をハードコードする

    ax1.axvline(x=target_pos_km, color="black", linewidth=3, linestyle="-", label=f"Station ({target_pos_km}km)")
    ax1.set_xlim(left=0.0, right=2.5)  # ★横幅も 2.5km まで完全固定する
    # ==========================================

    ax1.set_xlabel("Distance (km)")
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title("AI Driving Result: Run Curve (Distance-based)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    fig1.tight_layout()

    # 【グラフ2】 時間ベース（速度とノッチ操作の履歴）
    fig2, (ax2_speed, ax2_action) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax2_speed.plot(times, speed_limits, color="red", linestyle="--", alpha=0.7, label="Speed Limit (km/h)")
    ax2_speed.plot(times, velocities, color="blue", linewidth=2, label="Train Velocity (km/h)")
    ax2_speed.set_ylabel("Speed (km/h)")
    ax2_speed.set_title("AI Driving Result: Run Curve & Notch (Time-based)")
    ax2_speed.grid(True, linestyle=":", alpha=0.6)
    ax2_speed.legend()

    ax2_action.step(times, actions_taken, color="green", where="post", label="Action (Notch)")
    ax2_action.set_yticks([0, 1, 2])
    ax2_action.set_yticklabels(["Power (0)", "Coast (1)", "Brake (2)"])
    ax2_action.set_xlabel("Time (s)")
    ax2_action.set_ylabel("Action")
    ax2_action.grid(True, linestyle=":", alpha=0.6)
    fig2.tight_layout()

    print("\n📊 距離ベースと時間ベース、2つのグラフを描画しました。ウィンドウを閉じるとプログラムが終了します。")
    plt.show()


if __name__ == "__main__":
    main()
