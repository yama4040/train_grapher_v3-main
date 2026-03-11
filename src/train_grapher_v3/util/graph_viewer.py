"""シミュレーション結果のグラフ描画機能

save_simulation_results() で保存したディレクトリから結果を読み込み、
以下のグラフを生成します：
- ダイヤ図: 時間-位置グラフ（複数シミュレーション結果の重ね合わせ可）
- ランカーブ: 位置-速度グラフ（複数列車・複数シミュレーションの重ね合わせ可）

使用例::

    from train_grapher_v3.util.graph_viewer import SimulationResult, plot_diagram, plot_running_curve

    result1 = SimulationResult.load("output/sim1", label="ケースA")
    result2 = SimulationResult.load("output/sim2", label="ケースB")

    # ダイヤ図（全列車、全時刻）
    plot_diagram([result1, result2])

    # ダイヤ図（時間・位置範囲を指定）
    plot_diagram([result1, result2], time_range=(0, 300), position_range=(0, 10))

    # ランカーブ（複数列車の比較）
    plot_running_curve(
        [(result1, "列車A"), (result2, "列車A")],
        position_range=(0, 5),
    )
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

_NONE_SIMURATION = 0  # Status.NONE_SIMURATION

# 日本語フォントの設定
_JP_FONT_CANDIDATES = ["Meiryo", "Yu Gothic", "MS Gothic", "BIZ UDGothic"]


def _setup_japanese_font() -> None:
    """日本語対応フォントを matplotlib に設定する。

    候補フォントを順に探し、最初に見つかったものを使用します。
    見つからない場合は何も変更しません。
    """
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in _JP_FONT_CANDIDATES:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            return


_setup_japanese_font()


@dataclass
class TrainData:
    """1列車のステップデータ"""

    name: str
    times_s: list[float] = field(default_factory=list)
    statuses: list[int] = field(default_factory=list)
    positions_km: list[float | None] = field(default_factory=list)
    cum_positions_km: list[float | None] = field(default_factory=list)
    velocities_kmh: list[float | None] = field(default_factory=list)
    accelerations_ms2: list[float | None] = field(default_factory=list)


@dataclass
class SimulationResult:
    """保存済みシミュレーション結果

    save_simulation_results() で保存したディレクトリから読み込みます。

    Attributes:
        output_dir: 読み込んだディレクトリのパス
        label: 凡例に表示するラベル
        step_size: ステップサイズ（秒）
        total_steps: 総ステップ数
        trains: 列車名 -> TrainData のマップ
    """

    output_dir: Path
    label: str
    step_size: float
    total_steps: int
    trains: dict[str, TrainData]

    @classmethod
    def load(cls, output_dir: str, label: str = "") -> "SimulationResult":
        """保存済みの結果ディレクトリから読み込み

        Args:
            output_dir: save_simulation_results() で保存したディレクトリのパス
            label: 凡例に表示するラベル（省略時はディレクトリ名）

        Returns:
            SimulationResult オブジェクト

        Raises:
            FileNotFoundError: results_summary.json または CSV が存在しない場合
        """
        output_path = Path(output_dir)
        summary_path = output_path / "results_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        step_size = summary["simulation_config"]["step_size"]
        total_steps = summary["simulation_config"]["total_steps"]

        trains: dict[str, TrainData] = {}
        train_data_dir = output_path / "train_data"
        for train_info in summary["trains"]:
            train_name = train_info["name"]
            safe_name = re.sub(r'[\\/:*?"<>|]', "_", train_name)
            csv_file = train_data_dir / f"{safe_name}.csv"
            if csv_file.exists():
                trains[train_name] = _load_train_csv(csv_file, train_name)

        return cls(
            output_dir=output_path,
            label=label or output_path.name,
            step_size=step_size,
            total_steps=total_steps,
            trains=trains,
        )

    @property
    def train_names(self) -> list[str]:
        """列車名のリスト"""
        return list(self.trains.keys())


# ---- 内部ヘルパー --------------------------------------------------------


def _load_train_csv(csv_file: Path, actual_name: str) -> TrainData:
    """CSVファイルから列車データを読み込み"""
    data = TrainData(name=actual_name)

    def _to_float(s: str) -> float | None:
        return float(s) if s else None

    with open(csv_file, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        has_cum = "cumulative_position_km" in (reader.fieldnames or [])
        for row in reader:
            data.times_s.append(float(row["time_s"]))
            status_str = row["status"]
            data.statuses.append(int(status_str) if status_str else _NONE_SIMURATION)
            data.positions_km.append(_to_float(row["position_km"]))
            # cumulative_position_km がない古い形式は position_km で代替
            data.cum_positions_km.append(
                _to_float(row["cumulative_position_km"])
                if has_cum
                else _to_float(row["position_km"])
            )
            data.velocities_kmh.append(_to_float(row["velocity_kmh"]))
            data.accelerations_ms2.append(_to_float(row["acceleration_ms2"]))

    return data


def _label_for(result: SimulationResult, train_name: str, multi_result: bool) -> str:
    """凡例ラベルを生成"""
    if multi_result:
        return f"{result.label}: {train_name}"
    return train_name


def _get_color_cycle() -> list[str]:
    return [p["color"] for p in plt.rcParams["axes.prop_cycle"]]


def _build_xy_with_gaps(
    xs_raw: list,
    ys_raw: list,
    statuses: list,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
) -> tuple[list, list]:
    """アクティブな (x, y) ペアを抽出し、非連続区間に None を挿入

    NONE_SIMURATION ステータスの行および指定範囲外の行を除外します。
    除外によって区間が途切れる場合は None を挿入して線を分断します。
    """
    xs: list = []
    ys: list = []
    prev_active = False

    for x, s, y in zip(xs_raw, statuses, ys_raw):
        if s == _NONE_SIMURATION or x is None or y is None:
            if prev_active:
                xs.append(None)
                ys.append(None)
            prev_active = False
            continue
        if x_range is not None and not (x_range[0] <= x <= x_range[1]):
            if prev_active:
                xs.append(None)
                ys.append(None)
            prev_active = False
            continue
        if y_range is not None and not (y_range[0] <= y <= y_range[1]):
            if prev_active:
                xs.append(None)
                ys.append(None)
            prev_active = False
            continue
        xs.append(x)
        ys.append(y)
        prev_active = True

    return xs, ys


# ---- 公開 API ------------------------------------------------------------


def plot_diagram(
    results: "SimulationResult | list[SimulationResult]",
    *,
    time_range: tuple[float, float] | None = None,
    position_range: tuple[float, float] | None = None,
    train_names: list[str] | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> Figure:
    """ダイヤ図（時間-位置グラフ）を描画

    Args:
        results: SimulationResult または SimulationResult のリスト。
            複数渡すと重ね合わせて表示します。
        time_range: 表示する時間範囲 ``(start_s, end_s)``。``None`` で全範囲。
        position_range: 表示する位置範囲 ``(start_km, end_km)``。``None`` で全範囲。
        train_names: 表示する列車名のリスト。``None`` で全列車を表示。
        ax: 描画先の Axes。``None`` の場合は新規作成。
        show: ``True`` の場合 ``plt.show()`` を呼ぶ。

    Returns:
        matplotlib Figure オブジェクト

    Example::

        result = SimulationResult.load("output/sim1", label="ケースA")
        plot_diagram(result, time_range=(0, 300), position_range=(0, 10))

        # 複数結果の重ね合わせ
        plot_diagram([result1, result2])
    """
    if isinstance(results, SimulationResult):
        results = [results]

    multi_result = len(results) > 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    colors = _get_color_cycle()
    color_idx = 0

    for result in results:
        names = train_names if train_names is not None else result.train_names
        for name in names:
            if name not in result.trains:
                continue
            data = result.trains[name]
            label = _label_for(result, name, multi_result)
            color = colors[color_idx % len(colors)]
            color_idx += 1

            xs, ys = _build_xy_with_gaps(
                data.times_s,
                data.cum_positions_km,
                data.statuses,
                x_range=time_range,
                y_range=position_range,
            )
            ax.plot(xs, ys, label=label, color=color)

    ax.set_xlabel("時間 [s]")
    ax.set_ylabel("位置 [km]")
    ax.set_title("ダイヤ図")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if time_range is not None:
        ax.set_xlim(time_range)
    if position_range is not None:
        ax.set_ylim(position_range)

    if show:
        plt.show()

    return fig


# TrainSpec: (result, train_name) または (result, train_name, custom_label)
type TrainSpec = tuple[SimulationResult, str] | tuple[SimulationResult, str, str]


def plot_running_curve(
    train_specs: list[TrainSpec],
    *,
    position_range: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> Figure:
    """ランカーブ（位置-速度グラフ）を描画

    Args:
        train_specs: 描画する列車の指定リスト。各要素は以下のいずれか：

            - ``(result, train_name)`` — 結果と列車名
            - ``(result, train_name, label)`` — カスタムラベルを指定

        position_range: 表示する位置範囲 ``(start_km, end_km)``。``None`` で全範囲。
        ax: 描画先の Axes。``None`` の場合は新規作成。
        show: ``True`` の場合 ``plt.show()`` を呼ぶ。

    Returns:
        matplotlib Figure オブジェクト

    Example::

        result1 = SimulationResult.load("output/sim1", label="ケースA")
        result2 = SimulationResult.load("output/sim2", label="ケースB")

        # 同名列車を2つのシミュレーション結果から比較
        plot_running_curve(
            [(result1, "列車A"), (result2, "列車A")],
            position_range=(0, 5),
        )

        # カスタムラベル付き
        plot_running_curve([
            (result1, "列車A", "ケースA 列車A"),
            (result2, "列車B", "ケースB 列車B"),
        ])
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    colors = _get_color_cycle()
    unique_results = {id(spec[0]) for spec in train_specs}
    multi_result = len(unique_results) > 1

    for i, spec in enumerate(train_specs):
        result: SimulationResult = spec[0]
        train_name: str = spec[1]
        custom_label: str | None = spec[2] if len(spec) >= 3 else None  # type: ignore[misc]

        if train_name not in result.trains:
            continue

        data = result.trains[train_name]
        label = custom_label or _label_for(result, train_name, multi_result)
        color = colors[i % len(colors)]

        # X: 累積位置、Y: 速度（NONE_SIMURATION は除外）
        xs, ys = _build_xy_with_gaps(
            data.cum_positions_km,
            data.velocities_kmh,
            data.statuses,
            x_range=position_range,  # X（位置）に対して位置フィルタを適用
            y_range=None,
        )
        ax.plot(xs, ys, label=label, color=color)

    ax.set_xlabel("位置 [km]")
    ax.set_ylabel("速度 [km/h]")
    ax.set_title("ランカーブ")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if position_range is not None:
        ax.set_xlim(position_range)

    if show:
        plt.show()

    return fig
