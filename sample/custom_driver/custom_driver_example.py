"""カスタム運転判断ロジックの例

運転判断ロジックをカスタマイズして、異なる運転スタイルを実装する例。
クラスベースのDrivingDecisionを使用。
"""

from train_grapher_v3.core.block_system import FixedBlockSystem
from train_grapher_v3.core.driving_decision import (
    AggressiveDrivingDecision,
    ConservativeDrivingDecision,
    DefaultDrivingDecision,
    EcoFriendlyDrivingDecision,
)
from train_grapher_v3.core.line import Line
from train_grapher_v3.core.line_shape import Block, Edge, LineShape, Node, Station
from train_grapher_v3.core.simulation import Simulation
from train_grapher_v3.core.train import StationStopTime, Train
from train_grapher_v3.core.train_parameter import TrainParameter
from train_grapher_v3.util.logger import setup_logger
from train_grapher_v3.util.simple_viewer import display_graph

logger = setup_logger(__name__)


def main():
    logger.info("=== カスタム運転判断ロジックの例 ===")

    # 路線形状の作成
    logger.info("路線形状を作成中...")
    node1 = Node("node1")
    node2 = Node("node2")

    block1 = Block(start=0.0, speed_limits=[0, 45, 75, 110])
    block2 = Block(start=2.0, speed_limits=[0, 45, 75, 110])
    block3 = Block(start=4.0, speed_limits=[0, 45, 75, 110])

    station1 = Station(id="station1", value=1.0, name="駅A")
    station2 = Station(id="station2", value=3.0, name="駅B")

    edge = Edge(
        id="edge1",
        start_node=node1,
        end_node=node2,
        length=5.0,
        grade=[],
        curve=[],
        block_list=[block1, block2, block3],
        stations=[station1, station2],
    )

    line_shape = LineShape(nodes=[node1, node2], edges=[edge])
    route = line_shape.get_route(["edge1"])

    # 列車パラメータ
    train_param = TrainParameter(
        wight=150,
        factor_of_inertia=1.1,
        decelerating_acceleration=-3.0,
        decelerating_acceleration_station=-2.5,
        fast_margine=5.0,
        slow_margine=10.0,
    )

    # 4つの異なる運転スタイルの列車を作成
    logger.info("異なる運転スタイルの列車を作成...")

    # 1. デフォルト運転手
    train_default = Train(
        name="通常列車",
        line_shape=line_shape,
        route=route,
        train_parameter=train_param,
        station_stop_times=[
            StationStopTime(station_id="station1", default_value=30),
            StationStopTime(station_id="station2", default_value=30),
        ],
        driving_decision=DefaultDrivingDecision(),  # クラスインスタンスを渡す
    )
    train_default.set_start_condition(step=0, position=route.get_start_position())

    # 2. アグレッシブ運転手（マージン50%削減）
    train_aggressive = Train(
        name="アグレッシブ列車",
        line_shape=line_shape,
        route=route,
        train_parameter=train_param,
        station_stop_times=[
            StationStopTime(station_id="station1", default_value=30),
            StationStopTime(station_id="station2", default_value=30),
        ],
        driving_decision=AggressiveDrivingDecision(margin_reduction=0.5),
    )
    train_aggressive.set_start_condition(step=0, position=route.get_start_position())

    # 3. 保守的運転手（マージン1.5倍）
    train_conservative = Train(
        name="保守的列車",
        line_shape=line_shape,
        route=route,
        train_parameter=train_param,
        station_stop_times=[
            StationStopTime(station_id="station1", default_value=30),
            StationStopTime(station_id="station2", default_value=30),
        ],
        driving_decision=ConservativeDrivingDecision(safety_margin_multiplier=1.5),
    )
    train_conservative.set_start_condition(step=0, position=route.get_start_position())

    # 4. エコドライブ運転手
    train_eco = Train(
        name="エコ列車",
        line_shape=line_shape,
        route=route,
        train_parameter=train_param,
        station_stop_times=[
            StationStopTime(station_id="station1", default_value=30),
            StationStopTime(station_id="station2", default_value=30),
        ],
        driving_decision=EcoFriendlyDrivingDecision(coasting_preference=0.8),
    )
    train_eco.set_start_condition(step=0, position=route.get_start_position())

    # 各列車を個別にシミュレーション
    total_steps = 3000
    step_size = 1.0

    def run_simulation(train: Train, name: str):
        logger.info(f"{name}のシミュレーションを実行中...")
        block_system = FixedBlockSystem(line_shape)
        line = Line([train], line_shape, block_system)
        simulation = Simulation(line=line)

        def progress_callback(step: int, progress: float):
            if step % 300 == 0:
                print(
                    f"\r{name} - 進捗: {progress * 100:.1f}% (ステップ {step}/{total_steps})",
                    end="",
                )

        simulation.execution(total_steps, step_size, callback=progress_callback)
        print()  # 改行
        return train

    # 各列車をシミュレーション実行
    train_default = run_simulation(train_default, "通常列車")
    train_aggressive = run_simulation(train_aggressive, "アグレッシブ列車")
    train_conservative = run_simulation(train_conservative, "保守的列車")
    train_eco = run_simulation(train_eco, "エコ列車")

    logger.info("シミュレーション完了")

    # 結果表示
    logger.info("結果を比較中...")
    print("\n=== シミュレーション結果比較 ===\n")

    def print_train_result(train: Train, name: str):
        running_data = train.get_running_data()
        velocities = running_data.get_velocity_all()
        max_velocity = max(v for v in velocities if v is not None)

        # 加速回数をカウント
        statuses = running_data.get_status_all()
        from train_grapher_v3.core.status import Status

        power_run_count = sum(1 for s in statuses if s == Status.POWER_RUN)
        coasting_count = sum(1 for s in statuses if s == Status.COASTING)
        brake_count = sum(
            1 for s in statuses if s in [Status.BRAKE_OVERSPEED, Status.BRAKE_STATION]
        )

        print(f"{name}:")
        print(f"  最高速度: {max_velocity:.1f} km/h")
        print(f"  力行ステップ数: {power_run_count}")
        print(f"  惰行ステップ数: {coasting_count}")
        print(f"  ブレーキステップ数: {brake_count}")
        print()

    print_train_result(train_default, "通常列車")
    print_train_result(train_aggressive, "アグレッシブ列車")
    print_train_result(train_conservative, "保守的列車")
    print_train_result(train_eco, "エコ列車")

    # グラフ表示
    logger.info("グラフを表示中...")
    display_graph(train_default)
    display_graph(train_aggressive)
    display_graph(train_conservative)
    display_graph(train_eco)
    logger.info("グラフ表示完了")

    logger.info("\n=== 使用例の説明 ===")
    logger.info(
        "カスタム運転判断クラスを使用するには、DrivingDecisionを継承したクラスを作成します。"
    )
    logger.info("例: train = Train(..., driving_decision=AggressiveDrivingDecision())")
    logger.info("")
    logger.info("利用可能な運転判断クラス:")
    logger.info("  - DefaultDrivingDecision: 標準的な運転スタイル")
    logger.info("  - AggressiveDrivingDecision: 積極的な加速、最小限のマージン")
    logger.info("  - ConservativeDrivingDecision: 安全重視、広いマージン")
    logger.info("  - EcoFriendlyDrivingDecision: 惰行を多用、エネルギー効率重視")
    logger.info("")
    logger.info("各クラスはパラメータでカスタマイズ可能:")
    logger.info("  AggressiveDrivingDecision(margin_reduction=0.3)")
    logger.info("  ConservativeDrivingDecision(safety_margin_multiplier=2.0)")
    logger.info("  EcoFriendlyDrivingDecision(coasting_preference=0.9)")

    logger.info("\n=== プログラム終了 ===")


if __name__ == "__main__":
    main()
