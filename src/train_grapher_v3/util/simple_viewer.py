import matplotlib.pyplot as plt

from train_grapher_v3.core.train import Train


def display_graph_all(trains: list[Train]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for train in trains:
        position_values: list[float] = train.get_running_data().get_position_value_all()
        edge_ids: list[str] = train.get_running_data().get_edge_id_all()

        cumulative_position_values = []
        cumulative_distance = 0.0

        for i, position in enumerate(position_values):
            if edge_ids[i] is not None:
                edge = train._line_shape.get_edge_by_id(edge_ids[i])
                cumulative_distance = edge._start_node.offset or 0.0

            if position is None:
                cumulative_position_values.append(None)
            else:
                cumulative_position_values.append(position + cumulative_distance)

        step = range(len(cumulative_position_values))
        ax.plot(step, cumulative_position_values)

    ax.set_xlabel("step")
    ax.set_ylabel("position [km]")
    plt.show()


def display_graph(train: Train):
    """列車単体のグラフを表示

    Args:
        train (Train): 表示したい列車
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    position_values: list[float] = train.get_running_data().get_position_value_all()
    edge_ids: list[str] = train.get_running_data().get_edge_id_all()
    accelerations: list[float] = train.get_running_data().get_acceleration_all()
    velocitys: list[float] = train.get_running_data().get_velocity_all()

    cumulative_position_values = []
    cumulative_distance = 0.0
    last_edge_id = edge_ids[0]

    # グラフに表示するエッジのIDとその位置を記録
    displayed_edge_ids = []
    edge_change_indices = [0]

    for i, position in enumerate(position_values):
        if (edge_ids[i] != last_edge_id) and (edge_ids[i] is not None):
            # 路線が切り替わったら、前の路線の距離を累積に加算
            if last_edge_id is not None:
                cumulative_distance += train._line_shape.get_edge_by_id(
                    last_edge_id
                ).length

            # 現在のエッジの終了位置を記録
            edge_change_indices.append(i - 1)

            # 次のエッジの開始位置を記録
            edge_change_indices.append(i)
            displayed_edge_ids.append(last_edge_id)

            last_edge_id = edge_ids[i]

        if position is None:
            cumulative_position_values.append(None)
        else:
            cumulative_position_values.append(position + cumulative_distance)

    # 最後のエッジの終了位置を記録
    edge_change_indices.append(len(position_values) - 1)
    displayed_edge_ids.append(last_edge_id)

    step = range(len(cumulative_position_values))

    ax1.plot(step, cumulative_position_values)
    ax2.plot(step, velocitys)
    ax3.plot(step, accelerations)

    colors = [
        [190 / 255, 229 / 255, 237 / 255],
        [234 / 255, 193 / 255, 237 / 255],
        [173 / 255, 237 / 255, 183 / 255],
    ]

    # 背景色の塗りつぶしとエッジIDの表示
    for j in range(0, len(edge_change_indices), 2):
        start = edge_change_indices[j]
        end = edge_change_indices[j + 1]

        ax1.axvspan(start, end, color=colors[int(j / 2) % len(colors)], alpha=0.3)
        ax2.axvspan(start, end, color=colors[int(j / 2) % len(colors)], alpha=0.3)
        ax3.axvspan(start, end, color=colors[int(j / 2) % len(colors)], alpha=0.3)

        # エッジIDをエッジの中央に表示
        ax1.text(
            (start + end) / 2,
            max([i for i in cumulative_position_values if i is not None]) * 0.95,
            displayed_edge_ids[j // 2],
            ha="center",
            va="top",
            fontsize=10,
            color="black",
            rotation=90,
        )

    ax3.set_xlabel("step")

    ax1.set_ylabel("position [km]")
    ax2.set_ylabel("velocity [km/h]")
    ax3.set_ylabel("acceleration [km/h/s]")

    plt.show()
