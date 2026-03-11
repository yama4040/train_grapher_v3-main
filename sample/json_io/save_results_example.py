"""シミュレーション結果の保存サンプル

JSONモデルを読み込んでシミュレーションを実行し、結果を以下の構成で保存します：

    output/
    ├── initial_conditions.json   # 初期条件（シミュレーションモデル）
    ├── results_summary.json      # 列車ごとの集計結果
    └── train_data/
        ├── <列車名>.csv          # ステップごとの詳細データ
        └── ...
"""

from pathlib import Path

from train_grapher_v3.core.block_system import FixedBlockSystem
from train_grapher_v3.core.line import Line
from train_grapher_v3.core.simulation import Simulation
from train_grapher_v3.util.logger import setup_logger
from train_grapher_v3.util.result_saver import save_simulation_results
from train_grapher_v3.util.simulation_model_io import load_simulation_model

logger = setup_logger(__name__)

# JSONモデルのパス（適宜変更してください）
JSON_MODEL_PATH = "datas/simple_simulation_model.json"

# 結果の出力ディレクトリ
OUTPUT_DIR = "output/simulation_results"


def main():
    # モデルを読込
    line_shape, trains, step_size, total_steps, block_system_type = (
        load_simulation_model(JSON_MODEL_PATH)
    )

    # シミュレーションを実行
    block_system = FixedBlockSystem(line_shape)
    line = Line(line_shape=line_shape, trains=trains, block_system=block_system)
    simulation = Simulation(line=line, step_size=step_size)
    simulation.execution(total_steps, step_size)

    # 結果を保存
    save_simulation_results(
        output_dir=OUTPUT_DIR,
        trains=trains,
        step_size=step_size,
        total_steps=total_steps,
        line_shape=line_shape,
        block_system_type=block_system_type,
        name="サンプルシミュレーション",
    )

    logger.info(f"保存先: {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
