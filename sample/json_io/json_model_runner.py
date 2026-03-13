"""JSON形式のシミュレーションモデルを読込・実行するスクリプト

このスクリプトは以下の処理を行います：
1. JSON形式のシミュレーションモデルを読込
2. FixedBlockSystemでシミュレーションを実行
3. 結果をグラフ表示
"""

from pathlib import Path

from train_grapher_v3.core.block_system import FixedBlockSystem
from train_grapher_v3.core.line import Line
from train_grapher_v3.core.simulation import Simulation
from train_grapher_v3.util.logger import setup_logger
from train_grapher_v3.util.simple_viewer import display_graph_all
from train_grapher_v3.util.simulation_model_io import load_simulation_model

logger = setup_logger(__name__)


def run_simulation_from_json(json_file: str, output_graph: bool = True) -> Simulation:
    """JSON形式のシミュレーションモデルを読込・実行

    Args:
        json_file: JSON形式のモデルファイルパス
        output_graph: 実行後にグラフを表示するか

    Returns:
        実行済みのSimulationオブジェクト
    """
    logger.info("=" * 60)
    logger.info("JSON形式のシミュレーションモデルを読込中...")
    logger.info("=" * 60)

    # JSONモデルを読込
    line_shape, trains, step_size, total_steps, block_system_type = load_simulation_model(json_file)

    logger.info("読込完了:")
    logger.info(f"  - ステップサイズ: {step_size} 秒")
    logger.info(f"  - 総ステップ数: {total_steps}")
    logger.info(f"  - シミュレーション時間: {step_size * total_steps} 秒 ({step_size * total_steps / 60:.1f} 分)")
    logger.info(f"  - ブロックシステム: {block_system_type}")
    logger.info(f"  - 列車数: {len(trains)}")
    for train in trains:
        logger.info(f"    * {train.name}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("シミュレーション実行中...")
    logger.info("=" * 60)

    # ブロックシステムを作成
    block_system = FixedBlockSystem(line_shape)

    # Lineを作成
    line = Line(line_shape=line_shape, trains=trains, block_system=block_system)

    # Simulationを実行
    simulation = Simulation(line=line, step_size=step_size)

    # 1000ステップごとにログ出力するコールバック
    def _progress_callback(step: int, fraction: float) -> None:
        if step % 1 == 0:
            logger.info(f"進捗: step={step}/{total_steps} ({fraction * 100:.2f}%)")

    simulation.execution(total_steps, step_size, callback=_progress_callback)

    logger.info("")
    logger.info("=" * 60)
    logger.info("シミュレーション実行完了")
    logger.info("=" * 60)

    # 結果のサマリーを表示
    logger.info("結果:")
    for train in trains:
        running_data = train.get_running_data()
        position_values = running_data.get_position_value_all()
        velocity_values = running_data.get_velocity_all()
        # リストが空でなく、かつ最後の値がNoneでない場合は値を使用、それ以外は0.0
        final_position = 0.0
        final_velocity = 0.0
        if position_values and len(position_values) > 0:
            last_pos = position_values[-1]
            if last_pos is not None:
                final_position = last_pos
        if velocity_values and len(velocity_values) > 0:
            last_vel = velocity_values[-1]
            if last_vel is not None:
                final_velocity = last_vel
        logger.info(f"  - {train.name}: 最終位置={final_position:.2f}km, 最終速度={final_velocity:.2f}km/h")

    # グラフ表示
    if output_graph:
        logger.info("")
        logger.info("グラフを表示中...")
        display_graph_all(trains)

    return simulation


def main():
    """メイン処理"""
    # サンプルJSONファイル
    # sample_json = Path("sample/json_io/sim.json")
    sample_json = Path("datas/multi_train_simulation_model_generated.json")

    if sample_json.exists():
        logger.info(f"サンプルモデルを実行: {sample_json}")
        run_simulation_from_json(str(sample_json))
    else:
        logger.error(f"ファイルが見つかりません: {sample_json}")


if __name__ == "__main__":
    main()
