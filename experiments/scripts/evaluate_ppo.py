import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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
    target_train_name = "普通列車2"  # ※JSONに合わせる
    model_path = "./output/models/ppo_baseline_model.zip"

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
    
    obs, info = env.reset()
    done = False
    
    # グラフ描画用のデータ保存リスト
    times = []
    positions = []
    velocities = []
    actions_taken = []
    speed_limits = []

    while not done:
        # AIが現在の観測(obs)から最適な行動を推論
        # deterministic=True にすることで、ランダムな探索をせず最も自信のある行動を選ばせます
        action, _states = model.predict(obs, deterministic=True)
        
        # 行動を実行して1ステップ進める
        obs, reward, done, truncated, info = env.step(action)
        
        # 記録用にデータを抽出
        current_step = info["step"]
        current_time = info["time"]
        position = info["train_position"]
        # obs = [velocity, speed_limit, distance, time_left] (train_env.py の定義より)
        velocity = obs[0] * 3.6     # m/s -> km/h に変換して見やすくする
        speed_limit = obs[1] * 3.6  # m/s -> km/h に変換
        
        times.append(current_time)
        positions.append(position)
        velocities.append(velocity)
        actions_taken.append(action)
        speed_limits.append(speed_limit)

    # 修正： positions[-1] の :.2f を削除し、そのまま文字として表示させる
    logger.info(f"テスト走行完了！ (総時間: {times[-1]:.1f} 秒, 最終位置: {positions[-1]})")

    # ==========================================
    # 3. 走行結果のグラフ化（ランカーブ）
    # ==========================================
    logger.info("走行結果のグラフを描画します...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # 上段：速度と制限速度（★横軸を times に変更）
    ax1.plot(times, speed_limits, color='red', linestyle='--', alpha=0.7, label='Speed Limit (km/h)')
    ax1.plot(times, velocities, color='blue', linewidth=2, label='Train Velocity (km/h)')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title('AI Driving Result: Run Curve & Notch Operations (Time-based)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # 下段：ノッチ操作（★横軸を times に変更）
    ax2.step(times, actions_taken, color='green', where='post', label='Action (Notch)')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['Power (0)', 'Coast (1)', 'Station Brk (2)', 'Hold Spd (3)', 'Max Brk (4)'])
    ax2.set_xlabel('Time (s)') # ★横軸のラベルを時間に変更
    ax2.set_ylabel('Action')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    print("\n📊 ランカーブを描画しました。ウィンドウを閉じるとプログラムが終了します。")
    plt.show()

if __name__ == "__main__":
    main()