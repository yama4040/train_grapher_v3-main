"""シンプルなシミュレーション実行サンプル

単純な路線形状で1両の列車を走らせるシミュレーションの例
"""

from train_grapher_v3.core.block_system import FixedBlockSystem
from train_grapher_v3.core.line import Line
from train_grapher_v3.core.line_shape import (
    Block,
    Edge,
    Grade,
    LineShape,
    Node,
    Station,
)
from train_grapher_v3.core.simulation import Simulation
from train_grapher_v3.core.train import StationStopTime, Train
from train_grapher_v3.core.train_parameter import TrainParameter
from train_grapher_v3.util.logger import setup_logger
from train_grapher_v3.util.simple_viewer import display_graph

logger = setup_logger(__name__)


def main():
    """メイン処理"""
    logger.info("=== シンプルシミュレーション開始 ===")

    # ステップサイズ（秒）
    step_size = 0.1
    # 総ステップ数（3000ステップ = 300秒 = 5分）
    total_steps = 3000

    # ========================================
    # 1. 路線形状の作成
    # ========================================
    logger.info("路線形状を作成中...")

    # ノードの作成（始点と終点）
    node_start = Node("node_start", offset=0.0)
    node_end = Node("node_end", offset=10.0)

    # 勾配情報（フラット）
    grades = [Grade(start=0.0, end=10.0, grade=0.0)]

    # 駅の作成
    station1 = Station(id="station1", value=2.0, name="第一駅")
    station2 = Station(id="station2", value=5.0, name="第二駅")
    station3 = Station(id="station3", value=8.0, name="第三駅")

    # 閉塞の作成（固定閉塞システム）
    # 各閉塞の速度制限リスト: [0個前, 1個前, 2個前, ...]
    blocks = [
        Block(start=0.0, speed_limits=[25, 45, 80, 120]),
        Block(start=2.5, speed_limits=[25, 45, 80, 120]),
        Block(start=5.0, speed_limits=[25, 45, 80, 120]),
        Block(start=7.5, speed_limits=[25, 45, 80, 120]),
    ]

    # エッジの作成（10kmの直線路）
    edge1 = Edge(
        id="edge1",
        length=10.0,
        start_node=node_start,
        end_node=node_end,
        grade=grades,
        curve=[],
        stations=[station1, station2, station3],
        block_list=blocks,
    )

    # 路線形状の作成
    line_shape = LineShape(nodes=[node_start, node_end], edges=[edge1])

    # ========================================
    # 2. 列車の作成
    # ========================================
    logger.info("列車を作成中...")

    # ルートの作成
    route = line_shape.get_route(["edge1"])

    # 列車性能パラメータ
    train_param = TrainParameter(
        wight=389.85,
        factor_of_inertia=28.35 * (1 - 0.085),
        decelerating_acceleration=-3.0,
        decelerating_acceleration_station=-4.0,
        fast_margine=6.0,
        slow_margine=15.0,
    )

    # 駅停車時間の設定（各駅30秒停車）
    station_stop_times = [
        StationStopTime(station_id="station1", default_value=30.0),
        StationStopTime(station_id="station2", default_value=30.0),
        StationStopTime(station_id="station3", default_value=30.0),
    ]

    # 列車の作成
    train1 = Train(
        name="普通列車",
        line_shape=line_shape,
        route=route,
        train_parameter=train_param,
        station_stop_times=station_stop_times,
    )

    # 列車の開始位置を設定
    start_position = line_shape.get_position("edge1", 0.0)
    train1.set_start_condition(step=0, position=start_position)

    # ========================================
    # 3. シミュレーションの実行
    # ========================================
    logger.info("シミュレーションを実行中...")

    # 閉塞システムの作成
    block_system = FixedBlockSystem(line_shape)

    # 路線の作成
    line = Line(trains=[train1], line_shape=line_shape, block_system=block_system)

    # シミュレーションの作成と実行
    simulation = Simulation(step_size=step_size, line=line)

    def progress_callback(step: int, progress: float):
        """進捗表示コールバック"""
        print(f"\r進捗: {progress * 100:.1f}% (ステップ {step}/{total_steps})", end="")

    simulation.execution(total_steps, step_size, callback=progress_callback)
    print()  # 改行

    logger.info("シミュレーション完了")

    # ========================================
    # 4. 結果の表示
    # ========================================
    logger.info("結果を表示中...")

    # 列車のデータを取得
    running_data = train1.get_running_data()

    # 基本情報の表示
    print("\n=== シミュレーション結果 ===")
    print(f"総ステップ数: {len(running_data.get_position_value_all())}")

    final_position = running_data.get_position_value_all()[-1]
    final_velocity = running_data.get_velocity_all()[-1]

    if final_position is not None:
        print(f"最終位置: {final_position:.3f} km")
    else:
        print("最終位置: データなし")

    if final_velocity is not None:
        print(f"最終速度: {final_velocity:.1f} km/h")
    else:
        print("最終速度: データなし")

    # グラフ表示
    try:
        display_graph(train1)
        logger.info("グラフ表示完了")
    except Exception as e:
        logger.error(f"グラフ表示エラー: {e}")

    logger.info("=== プログラム終了 ===")


if __name__ == "__main__":
    main()
