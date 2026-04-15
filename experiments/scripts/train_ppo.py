import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# パスの追加
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,  # ← CheckpointCallbackを追加
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

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
            print(
                f"🏁 [エピソード {self.episode_count}] "
                f"トータルステップ: {self.num_timesteps} | "
                f"獲得報酬: {reward:8.2f} | "
                f"エピソード長: {length} steps"
            )
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
    df["rolling_reward"] = df["r"].rolling(window=window, min_periods=1).mean()

    # グラフの描画設定
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["r"], alpha=0.3, color="blue", label="Episode Reward (Raw)")
    plt.plot(df.index, df["rolling_reward"], color="red", linewidth=2, label=f"Moving Average ({window} ep)")

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # 学習完了後にポップアップで表示
    print("\n📊 グラフを描画しました。ウィンドウを閉じるとプログラムが終了します。")
    plt.show()


# ==========================================
# メイン処理
# ==========================================
def main():
    json_path = "datas/multi_train_simulation_model_generated.json"
    target_train_name = "普通列車2"

    # ==========================================
    # 1. 保存用フォルダ（PPO_datas\YYYYMMDDhhmmss\）の作成
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = os.path.join("PPO_datas", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Monitor用のログフォルダ（テンポラリ）
    log_dir = os.path.join(save_dir, "rl_logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.info(f"環境を初期化します: {json_path}")
    logger.info(f"データ保存先: {save_dir}")

    # ★ Envに保存先のパス(save_dir)を渡す
    env = TrainEnv(json_path=json_path, target_train_name=target_train_name, save_dir=save_dir)
    env = Monitor(env, log_dir)

    check_env(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log=os.path.join(save_dir, "tensorboard_logs"),
    )

    logger.info("🚀 学習を開始します...")

    total_timesteps = 1000000

    # ==========================================
    # ★追加：定期セーブ（チェックポイント）の設定
    # ==========================================
    # 10万ステップ（全体の10%）ごとにモデルをバックアップ保存する
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, save_path=os.path.join(save_dir, "checkpoints"), name_prefix="ppo_model"
    )

    # もし自作のTerminalOutputCallbackも使っている場合は、リストにして両方渡します
    # callbacks = [TerminalOutputCallback(total_timesteps), checkpoint_callback]

    # 学習実行（コールバックを渡す）
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True)

    # ==========================================
    # ★追加：最終モデルの保存
    # ==========================================
    final_model_path = os.path.join(save_dir, "ppo_final_model")
    model.save(final_model_path)
    logger.info(f"🎉 最終モデルを保存しました: {final_model_path}.zip")

    # ==========================================
    # 学習終了後：学習曲線の出力・保存（既存のコード）
    # ==========================================
    logger.info("学習曲線を生成・保存しています...")
    # ... (以降は既存のCSV・グラフ出力コード) ...
    monitor_file = os.path.join(log_dir, "monitor.csv")

    if os.path.exists(monitor_file):
        # 1行目はSB3のメタデータなのでスキップして読み込む
        df_monitor = pd.read_csv(monitor_file, skiprows=1)

        plt.figure(figsize=(10, 5))
        # 生の報酬推移（薄い線）
        plt.plot(df_monitor["r"], alpha=0.3, label="Episode Reward (Raw)", color="blue")
        # 10エピソードの移動平均（濃い線：トレンドが見やすい）
        plt.plot(
            df_monitor["r"].rolling(window=10).mean(), label="10-Episode Moving Avg", color="darkblue", linewidth=2
        )

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)

        # グラフ画像とCSVの保存
        plt.savefig(os.path.join(save_dir, "learning_curve.png"))
        df_monitor.to_csv(os.path.join(save_dir, "learning_curve_data.csv"), index=False)
        logger.info(f"学習曲線を {save_dir} に保存しました。")


if __name__ == "__main__":
    main()
