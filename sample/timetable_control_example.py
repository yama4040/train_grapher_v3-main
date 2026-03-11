"""ダイヤ制御の例

列車がダイヤ情報を持ち、停車時間を満たしながる、
かつダイヤ出発時刻より早出ししない制御を実演するサンプル。
"""

from train_grapher_v3.core.block_system import FixedBlockSystem
from train_grapher_v3.core.line import Line
from train_grapher_v3.core.line_shape import (
    Block,
    Curve,
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

logger = setup_logger(__name__)


def create_line_shape() -> LineShape:
    """単純な路線形状を作成"""
    # ノードを作成
    node_start = Node("node_start", offset=0.0)
    node_end = Node("node_end", offset=10.0)

    # エッジを作成（10km、フラット）
    grades = [Grade(start=0.0, end=10.0, grade=0.0)]
    curves = [Curve(start=0.0, end=10.0, curve=0.0)]

    # 駅を作成
    stations = [
        Station(id="station_A", value=2.0, name="A駅"),
        Station(id="station_B", value=5.0, name="B駅"),
    ]

    # ブロックを作成（固定閉塞）
    blocks = [
        Block(start=0.0, speed_limits=[25, 45, 80, 120]),
    ]

    edge1 = Edge(
        id="edge1",
        length=10.0,
        start_node=node_start,
        end_node=node_end,
        grade=grades,
        curve=curves,
        stations=stations,
        block_list=blocks,
    )

    line_shape = LineShape(nodes=[node_start, node_end], edges=[edge1])
    return line_shape


def create_timetable_train(line_shape: LineShape) -> Train:
    """ダイヤ制御を使った列車を作成

    例：
    - A駅到着（予定）: 50秒、ダイヤ出発: 110秒（60秒停車）
    - B駅到着（予定）: 200秒、ダイヤ出発: 260秒（60秒停車）
    """
    route = line_shape.get_route(["edge1"])

    params = TrainParameter(
        wight=389.85,
        factor_of_inertia=24.54,
        decelerating_acceleration=-3.0,
        decelerating_acceleration_station=-2.0,
        fast_margine=6.0,
        slow_margine=15.0,
    )

    # ダイヤ情報を含む駅停車時間を設定
    # use_timetable=Trueなので、departure_timeが有効になる
    station_stop_times = [
        StationStopTime(
            station_id="station_A",
            default_value=60.0,  # 最小停車時間: 60秒
            departure_time=110.0,  # ダイヤ出発時刻: 110秒
        ),
        StationStopTime(
            station_id="station_B",
            default_value=60.0,  # 最小停車時間: 60秒
            departure_time=260.0,  # ダイヤ出発時刻: 260秒
        ),
    ]

    train = Train(
        name="ダイヤ制御列車",
        line_shape=line_shape,
        route=route,
        train_parameter=params,
        station_stop_times=station_stop_times,
        use_timetable=True,  # ダイヤ制御を有効化
    )

    return train


def create_normal_train(line_shape: LineShape) -> Train:
    """通常の停車時間制御のみの列車を作成（比較用）"""
    route = line_shape.get_route(["edge1"])

    params = TrainParameter(
        wight=389.85,
        factor_of_inertia=24.54,
        decelerating_acceleration=-3.0,
        decelerating_acceleration_station=-2.0,
        fast_margine=6.0,
        slow_margine=15.0,
    )

    station_stop_times = [
        StationStopTime(station_id="station_A", default_value=60.0),
        StationStopTime(station_id="station_B", default_value=60.0),
    ]

    train = Train(
        name="通常列車",
        line_shape=line_shape,
        route=route,
        train_parameter=params,
        station_stop_times=station_stop_times,
        use_timetable=False,  # ダイヤ制御を無効化
    )

    return train


def main():
    """メイン処理"""
    logger.info("ダイヤ制御シミュレーション開始")

    # 路線形状を作成
    line_shape = create_line_shape()

    # ダイヤ制御列車と通常列車を作成
    train_timetable = create_timetable_train(line_shape)
    train_normal = create_normal_train(line_shape)

    # 列車の初期位置を設定
    edge = line_shape._edges[0]
    start_position = line_shape.get_position(edge.id, 0.0)

    train_timetable.set_start_condition(step=0, position=start_position)
    train_normal.set_start_condition(step=0, position=start_position)

    # ブロックシステムを作成
    block_system = FixedBlockSystem(line_shape)

    # Lineを作成
    line = Line(
        line_shape=line_shape,
        trains=[train_timetable, train_normal],
        block_system=block_system,
    )

    # シミュレーションを実行
    step_size = 0.1
    simulation = Simulation(line=line, step_size=step_size)
    total_steps = 4000

    logger.info(f"シミュレーション実行: {total_steps}ステップ")
    simulation.execution(total_steps, step_size)

    logger.info("シミュレーション完了")

    # 結果の確認
    logger.info("\n" + "=" * 60)
    logger.info("【ダイヤ制御列車の走行データ】")
    logger.info("=" * 60)

    running_data_timetable = train_timetable.get_running_data()
    max_step = len(running_data_timetable._velocity)
    logger.info(f"最終ステップ: {max_step - 1}")
    logger.info(f"最終時刻: {(max_step - 1) * step_size:.1f}秒")

    # A駅通過周辺のデータを表示
    logger.info("\nA駅周辺（50秒付近）の走行データ:")
    for i in range(max(0, 400), min(max_step, 1300), 100):
        time = i * step_size
        status = running_data_timetable.get_status(i)
        velocity = running_data_timetable.get_velocity(i)
        logger.info(f"  時刻 {time:6.1f}秒: 状態={status}, 速度={velocity:6.1f} km/h")

    logger.info("\nB駅周辺（200秒付近）の走行データ:")
    for i in range(max(0, 1850), min(max_step, 2850), 100):
        time = i * step_size
        status = running_data_timetable.get_status(i)
        velocity = running_data_timetable.get_velocity(i)
        logger.info(f"  時刻 {time:6.1f}秒: 状態={status}, 速度={velocity:6.1f} km/h")

    logger.info("\n" + "=" * 60)
    logger.info("【通常列車の走行データ】")
    logger.info("=" * 60)

    running_data_normal = train_normal.get_running_data()
    max_step_normal = len(running_data_normal._velocity)
    logger.info(f"最終ステップ: {max_step_normal - 1}")
    logger.info(f"最終時刻: {(max_step_normal - 1) * step_size:.1f}秒")

    logger.info("\n【ダイヤ制御と通常制御の比較】")
    logger.info(f"ダイヤ制御列車の最終時刻: {(max_step - 1) * step_size:.1f}秒")
    logger.info(f"通常列車の最終時刻: {(max_step_normal - 1) * step_size:.1f}秒")

    logger.info("\nポイント:")
    logger.info("- ダイヤ制御列車: A駅で110秒、B駅で260秒に出発（ダイヤに従う）")
    logger.info("- 通常列車: 60秒の停車時間後、可能な限り早く出発")
    logger.info("- ダイヤ制御は停車時間とダイヤ出発時刻の両立を実現")
    logger.info("- JSON形式で departure_time と use_timetable で制御可能")


if __name__ == "__main__":
    main()
