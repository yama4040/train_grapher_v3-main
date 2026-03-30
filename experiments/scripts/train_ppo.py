import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# パスの追加
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from experiments.rl_env.train_env import TrainEnv
from train_grapher_v3.util.logger import setup_logger

logger = setup_logger(__name__)

# ==========================================
# カスタムコールバック（ターミナル出力用）
# ==========================================
class TerminalOutputCallback(BaseCallback):
    """エピソードが完了するごとにターミナルに結果を出力するコールバック"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # 環境から返ってくる info 辞書の中に "episode" があればエピソード完了
        # (Monitor ラッパーを付けると自動的に追加されます)
        info = self.locals.get("infos")[0]
        if "episode" in info:
            self.episode_count += 1
            reward = info["episode"]["r"]  # 獲得した合計報酬
            length = info["episode"]["l"]  # かかったステップ数
            
            # ターミナルに分かりやすく出力
            print(f"🏁 [エピソード {self.episode_count}] "
                  f"トータルステップ: {self.num_timesteps} | "
                  f"獲得報酬: {reward:8.2f} | "
                  f"エピソード長: {length} steps")
        return True

# ==========================================
# 学習曲線のプロット関数
# ==========================================
def plot_learning_curve(log_folder: str, title: str = "Learning Curve"):
    """Monitorが生成したCSVから学習曲線をMatplotlibで描画する"""
    csv_path = os.path.join(log_folder, "monitor.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"CSVファイルが見つかりません: {csv_path}")
        return

    # MonitorのCSVは1行目がメタデータ（コメント）なので skiprows=1 にする
    df = pd.read_csv(csv_path, skiprows=1)
    
    # ノイズが多いので、直近10エピソードの移動平均も計算して滑らかな線を作る
    window = min(10, len(df))
    df['rolling_reward'] = df['r'].rolling(window=window, min_periods=1).mean()

    # グラフの描画設定
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['r'], alpha=0.3, color='blue', label='Episode Reward (Raw)')
    plt.plot(df.index, df['rolling_reward'], color='red', linewidth=2, label=f'Moving Average ({window} ep)')
    
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 学習完了後にポップアップで表示
    print("\n📊 グラフを描画しました。ウィンドウを閉じるとプログラムが終了します。")
    plt.show()

# ==========================================
# メイン処理
# ==========================================
def main():
    json_path = "datas/multi_train_simulation_model_generated.json"
    target_train_name = "普通列車2" # ※JSON側の名前に合わせてください
    
    # ログとモデルの保存先
    log_dir = "./output/rl_logs/ppo_baseline/"
    os.makedirs(log_dir, exist_ok=True)

    logger.info(f"環境を初期化します: {json_path}")
    
    # 環境の作成と Monitor ラッパーの適用
    env = TrainEnv(json_path=json_path, target_train_name=target_train_name)
    env = Monitor(env, log_dir) # これを付けることで monitor.csv が出力される

    # 環境チェック
    check_env(env)
    
    # PPOモデルの初期化
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0,  # SB3標準の長ったらしいテーブル出力をオフにする
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64
    )

    logger.info("🚀 学習を開始します...")
    
    # コールバックを渡して学習実行
    callback = TerminalOutputCallback()
    total_timesteps = 500000 
    
    # ★ progress_bar=True を追加
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # モデルの保存
    save_dir = "./output/models/"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "ppo_baseline_model")
    model.save(model_path)
    logger.info(f"💾 学習済みモデルを保存しました: {model_path}.zip")

    # 学習曲線の表示
    plot_learning_curve(log_dir, title="PPO Baseline Learning Curve")

if __name__ == "__main__":
    main()