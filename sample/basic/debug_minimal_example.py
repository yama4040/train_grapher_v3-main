"""デバッグ用の最小シミュレーションサンプル

最小限の設定で動作を確認するためのサンプル
"""

from train_grapher_v3.core.block_system import FixedBlockSystem
from train_grapher_v3.core.line import Line
from train_grapher_v3.core.line_shape import (
    Block,
    Edge,
    Grade,
    LineShape,
    Node,
)
from train_grapher_v3.core.simulation import Simulation
from train_grapher_v3.core.train import Train
from train_grapher_v3.core.train_parameter import TrainParameter
from train_grapher_v3.util.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """メイン処理"""
    logger.info("=== デバッグ用最小シミュレーション開始 ===")

    # ステップサイズ
    step_size = 0.1
    total_steps = 100  # 10秒だけ

    # 路線形状の作成
    node_start = Node("start", offset=0.0)
    node_end = Node("end", offset=5.0)

    grades = [Grade(start=0.0, end=5.0, grade=0.0)]

    blocks = [
        Block(start=0.0, speed_limits=[25, 45, 80, 120]),
        Block(start=2.5, speed_limits=[25, 45, 80, 120]),
    ]

    edge1 = Edge(
        id="edge1",
        length=5.0,
        start_node=node_start,
        end_node=node_end,
        grade=grades,
        curve=[],
        stations=[],
        block_list=blocks,
    )

    line_shape = LineShape(nodes=[node_start, node_end], edges=[edge1])

    # 列車の作成
    route = line_shape.get_route(["edge1"])
    train_param = TrainParameter()

    train1 = Train(
        name="テスト列車",
        line_shape=line_shape,
        route=route,
        train_parameter=train_param,
        station_stop_times=[],
    )

    # 開始条件を設定
    start_position = line_shape.get_position("edge1", 0.0)
    logger.info(
        f"開始位置: edge={start_position.edge_id}, value={start_position.value}"
    )
    train1.set_start_condition(step=0, position=start_position)

    # 開始条件の確認
    logger.info(f"_start_step: {train1._start_step}")
    logger.info(f"_start_position: {train1._start_position}")

    # シミュレーション実行
    block_system = FixedBlockSystem(line_shape)
    line = Line(trains=[train1], line_shape=line_shape, block_system=block_system)
    simulation = Simulation(step_size=step_size, line=line)

    logger.info("シミュレーション実行中...")
    simulation.execution(total_steps, step_size)

    # 結果確認
    running_data = train1.get_running_data()

    logger.info("\n=== 結果 ===")
    logger.info(f"総ステップ数: {len(running_data.get_position_value_all())}")

    # 最初の10ステップを表示
    for step in range(min(10, total_steps)):
        pos_val = running_data.get_position_value(step)
        edge_id = running_data.get_edge_id(step)
        velocity = running_data.get_velocity(step)
        status = running_data.get_status(step)

        logger.info(
            f"Step {step}: edge={edge_id}, pos={pos_val}, vel={velocity}, status={status}"
        )

    # 最後のステップを表示
    final_pos = running_data.get_position_value_all()[-1]
    final_vel = running_data.get_velocity_all()[-1]
    logger.info(f"\n最終位置: {final_pos}")
    logger.info(f"最終速度: {final_vel}")

    logger.info("=== 完了 ===")


if __name__ == "__main__":
    main()
