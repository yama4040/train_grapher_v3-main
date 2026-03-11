"""既存のシミュレーションモデルをJSON形式で保存するスクリプト

このスクリプトは以下の処理を行います：
1. 既存のサンプルコードと同じシミュレーション構造を構築
2. それをJSON形式で保存
3. 保存したJSONを読込んで検証
"""

from train_grapher_v3.core.line_shape import Block, Edge, Grade, LineShape, Node, Station
from train_grapher_v3.core.train import StationStopTime, Train
from train_grapher_v3.core.train_parameter import TrainParameter
from train_grapher_v3.util.logger import setup_logger
from train_grapher_v3.util.simulation_model_io import load_simulation_model, save_simulation_model

logger = setup_logger(__name__)


def save_simple_simulation_model():
    """シンプルシミュレーションモデルをJSON形式で保存"""
    logger.info("=" * 60)
    logger.info("シンプルシミュレーションモデルを作成・保存中...")
    logger.info("=" * 60)

    # ========================================
    # 路線形状の作成
    # ========================================
    logger.info("路線形状を作成中...")

    # ノードの作成
    node_start = Node("node_start", offset=0.0)
    node_end = Node("node_end", offset=10.0)

    # 勾配情報（フラット）
    grades = [Grade(start=0.0, end=10.0, grade=0.0)]

    # 駅の作成
    station1 = Station(id="station1", value=2.0, name="第一駅")
    station2 = Station(id="station2", value=5.0, name="第二駅")
    station3 = Station(id="station3", value=8.0, name="第三駅")

    # 閉塞の作成（固定閉塞システム）
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
    # 列車の作成
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
    train = Train(
        name="列車1号",
        line_shape=line_shape,
        route=route,
        train_parameter=train_param,
        station_stop_times=station_stop_times,
    )

    # ========================================
    # JSON形式で保存
    # ========================================
    logger.info("JSON形式で保存中...")

    output_file = "datas/simple_simulation_model_generated.json"
    save_simulation_model(
        file_path=output_file,
        line_shape=line_shape,
        trains=[train],
        step_size=0.1,
        total_steps=3000,
        name="シンプルシミュレーション（自動生成）",
        description="既存のsimple_simulation_exampleをJSON形式で保存したもの",
        author="train_grapher_v3",
        block_system_type="fixed",
    )

    logger.info(f"保存完了: {output_file}")

    # ========================================
    # JSON形式を読込んで検証
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("保存したJSONを読込んで検証中...")
    logger.info("=" * 60)

    loaded_line_shape, loaded_trains, step_size, total_steps, block_system_type = load_simulation_model(
        output_file
    )

    logger.info("検証結果:")
    logger.info(f"  - ステップサイズ: {step_size} 秒")
    logger.info(f"  - 総ステップ数: {total_steps}")
    logger.info(f"  - ブロックシステム: {block_system_type}")
    logger.info(f"  - 列車数: {len(loaded_trains)}")
    logger.info(f"  - エッジ数: {len(loaded_line_shape._edges)}")

    for train in loaded_trains:
        logger.info(f"  列車: {train.name}")
        logger.info(f"    - 重量: {train._train_parameter.wight} ton")
        logger.info(f"    - 駅停車数: {len(train._station_stop_times)}")


def save_multi_train_simulation_model():
    """複数列車シミュレーションモデルをJSON形式で保存"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("複数列車シミュレーションモデルを作成・保存中...")
    logger.info("=" * 60)

    # ========================================
    # 路線形状の作成
    # ========================================
    logger.info("路線形状を作成中...")

    # ノードの作成
    node_start = Node("node_start", offset=0.0)
    node_end = Node("node_end", offset=15.0)

    # 勾配情報（フラット）
    grades = [Grade(start=0.0, end=15.0, grade=0.0)]

    # 駅の作成
    station1 = Station(id="station1", value=3.0, name="駅A")
    station2 = Station(id="station2", value=7.5, name="駅B")
    station3 = Station(id="station3", value=12.0, name="駅C")

    # 閉塞の作成
    blocks = [
        Block(start=0.0, speed_limits=[20, 40, 70, 100]),
        Block(start=3.75, speed_limits=[20, 40, 70, 100]),
        Block(start=7.5, speed_limits=[20, 40, 70, 100]),
        Block(start=11.25, speed_limits=[20, 40, 70, 100]),
    ]

    # エッジの作成（15kmの直線路）
    edge1 = Edge(
        id="edge1",
        length=15.0,
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
    # 列車1の作成（急行）
    # ========================================
    route = line_shape.get_route(["edge1"])

    train_param_1 = TrainParameter(
        wight=389.85,
        factor_of_inertia=24.54,
        decelerating_acceleration=-2.5,
        decelerating_acceleration_station=-3.5,
        fast_margine=8.0,
        slow_margine=12.0,
    )

    station_stop_times_1 = [
        StationStopTime(station_id="station1", default_value=20.0),
        StationStopTime(station_id="station2", default_value=20.0),
        StationStopTime(station_id="station3", default_value=20.0),
    ]

    train1 = Train(
        name="急行列車",
        line_shape=line_shape,
        route=route,
        train_parameter=train_param_1,
        station_stop_times=station_stop_times_1,
    )

    # ========================================
    # 列車2の作成（普通）
    # ========================================
    train_param_2 = TrainParameter(
        wight=320.5,
        factor_of_inertia=22.0,
        decelerating_acceleration=-2.0,
        decelerating_acceleration_station=-3.0,
        fast_margine=5.0,
        slow_margine=18.0,
    )

    station_stop_times_2 = [
        StationStopTime(station_id="station1", default_value=40.0),
        StationStopTime(station_id="station2", default_value=40.0),
        StationStopTime(station_id="station3", default_value=40.0),
    ]

    train2 = Train(
        name="普通列車",
        line_shape=line_shape,
        route=route,
        train_parameter=train_param_2,
        station_stop_times=station_stop_times_2,
    )

    # ========================================
    # JSON形式で保存
    # ========================================
    logger.info("JSON形式で保存中...")

    output_file = "datas/multi_train_simulation_model_generated.json"
    save_simulation_model(
        file_path=output_file,
        line_shape=line_shape,
        trains=[train1, train2],
        step_size=0.1,
        total_steps=4000,
        name="複数列車シミュレーション（自動生成）",
        description="2両の列車が同じ路線を走行するシミュレーション",
        author="train_grapher_v3",
        block_system_type="fixed",
    )

    logger.info(f"保存完了: {output_file}")

    # ========================================
    # JSON形式を読込んで検証
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("保存したJSONを読込んで検証中...")
    logger.info("=" * 60)

    loaded_line_shape, loaded_trains, step_size, total_steps, block_system_type = load_simulation_model(
        output_file
    )

    logger.info("検証結果:")
    logger.info(f"  - 列車数: {len(loaded_trains)}")
    for train in loaded_trains:
        logger.info(f"  列車: {train.name}")
        logger.info(f"    - 初期位置: 0.0 km")
        logger.info(f"    - 加速度: {train._train_parameter.decelerating_acceleration} m/s²")


def main():
    """メイン処理"""
    logger.info("")
    logger.info("シミュレーションモデルのJSON形式保存デモンストレーション")
    logger.info("")

    # シンプルシミュレーション
    save_simple_simulation_model()

    # 複数列車シミュレーション
    save_multi_train_simulation_model()

    logger.info("")
    logger.info("=" * 60)
    logger.info("すべての処理が完了しました")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
