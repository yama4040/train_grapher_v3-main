"""実験実行スクリプト

使い方:
    uv run python experiments/scripts/run.py <実験名>
    uv run python experiments/scripts/run.py exp001_example
    uv run python experiments/scripts/run.py exp001_example --no-save

実行結果は experiments/results/<実験名>/<タイムスタンプ>/ に保存されます。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from train_grapher_v3.core.block_system import FixedBlockSystem, MovingBlockSystem
from train_grapher_v3.core.line import Line
from train_grapher_v3.core.simulation import Simulation
from train_grapher_v3.util.logger import setup_logger
from train_grapher_v3.util.result_saver import save_simulation_results
from train_grapher_v3.util.simulation_model_io import load_simulation_model

import os
from stable_baselines3.common.monitor import Monitor

logger = setup_logger(__name__)

CONFIGS_DIR = Path("experiments/configs")
RESULTS_DIR = Path("experiments/results")


def run_experiment(experiment_name: str, save: bool = True) -> Path | None:
    """実験を実行し、結果を保存する。

    Args:
        experiment_name: configs/ 以下のディレクトリ名（例: exp001_example）
        save: 結果をファイルに保存するか

    Returns:
        結果の保存先ディレクトリ（save=False の場合は None）
    """
    config_dir = CONFIGS_DIR / experiment_name
    model_path = config_dir / "model.json"
    meta_path = config_dir / "meta.json"

    if not model_path.exists():
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        logger.error(f"  configs/{experiment_name}/model.json を作成してください。")
        sys.exit(1)

    # メタデータ読込
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    logger.info("=" * 60)
    logger.info(f"実験: {experiment_name}")
    if meta.get("name"):
        logger.info(f"名前: {meta['name']}")
    if meta.get("description"):
        logger.info(f"概要: {meta['description']}")
    logger.info("=" * 60)

    # モデル読込
    line_shape, trains, step_size, total_steps, block_system_type = (
        load_simulation_model(str(model_path))
    )
    logger.info(f"ステップサイズ: {step_size} 秒 / 総ステップ数: {total_steps}")
    logger.info(f"閉塞システム: {block_system_type} / 列車数: {len(trains)}")

    # 閉塞システム生成
    if block_system_type == "moving":
        block_system = MovingBlockSystem(line_shape)
    else:
        block_system = FixedBlockSystem(line_shape)

    # シミュレーション実行
    line = Line(line_shape=line_shape, trains=trains, block_system=block_system)
    simulation = Simulation(line=line, step_size=step_size)

    logger.info("シミュレーション実行中...")
    simulation.execution(total_steps, step_size)
    logger.info("シミュレーション完了")

    if not save:
        return None

    # 結果保存先: results/<実験名>/<タイムスタンプ>/
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / experiment_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # シミュレーション結果を保存（CSV・サマリーJSON）
    save_simulation_results(
        output_dir=str(output_dir),
        trains=trains,
        step_size=step_size,
        total_steps=total_steps,
        line_shape=line_shape,
        block_system_type=block_system_type,
        name=meta.get("name", experiment_name),
        description=meta.get("description", ""),
        author=meta.get("author", ""),
    )

    # 実行情報を run_info.json として保存（再現・追跡用）
    run_info = {
        "experiment": experiment_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_file": str(model_path),
        "meta": meta,
        "simulation": {
            "step_size": step_size,
            "total_steps": total_steps,
            "block_system_type": block_system_type,
            "train_count": len(trains),
        },
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"保存先: {output_dir.resolve()}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="実験を実行して results/ に保存します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  uv run python experiments/scripts/run.py exp001_example
  uv run python experiments/scripts/run.py exp002_moving_block --no-save
""",
    )
    parser.add_argument(
        "experiment",
        help="実験名（experiments/configs/ 以下のディレクトリ名）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="結果をファイルに保存しない（動作確認用）",
    )
    args = parser.parse_args()
    run_experiment(args.experiment, save=not args.no_save)


if __name__ == "__main__":
    main()
