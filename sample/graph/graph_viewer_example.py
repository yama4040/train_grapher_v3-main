"""ダイヤ図・ランカーブの描画サンプル

このスクリプトは graph_viewer モジュールの使い方を示します。
事前に save_simulation_results() で結果を保存しておく必要があります。

実行手順:
    1. まず結果を保存する（このスクリプト内で実行）
    2. 保存した結果からグラフを描画する

使い方:
    uv run python sample/graph/graph_viewer_example.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from train_grapher_v3.core.block_system import FixedBlockSystem
from train_grapher_v3.core.line import Line
from train_grapher_v3.core.simulation import Simulation
from train_grapher_v3.util.graph_viewer import SimulationResult, plot_diagram, plot_running_curve
from train_grapher_v3.util.result_saver import save_simulation_results
from train_grapher_v3.util.simulation_model_io import load_simulation_model

# ---- 設定 ----------------------------------------------------------------

# 使用するJSONモデル
# SIMPLE_MODEL = "datas/simple_simulation_model.json"
SIMPLE_MODEL = "datas/simple_simulation_model_generated.json"  # 生成したモデルを使用する場合はこちら
# MULTI_MODEL = "datas/multi_train_simulation_model.json"
MULTI_MODEL = "datas/multi_train_simulation_model_generated.json"  # 生成したモデルを使用する場合はこちら

# 結果の保存先
OUTPUT_SIMPLE = "output/graph_example/simple"
OUTPUT_MULTI = "output/graph_example/multi"


# ---- シミュレーション実行 ------------------------------------------------


def run_and_save(json_path: str, output_dir: str, label: str) -> None:
    """JSONモデルを実行して結果を保存"""
    line_shape, trains, step_size, total_steps, block_system_type = load_simulation_model(json_path)
    block_system = FixedBlockSystem(line_shape)
    line = Line(line_shape=line_shape, trains=trains, block_system=block_system)
    simulation = Simulation(line=line, step_size=step_size)
    simulation.execution(total_steps, step_size)

    save_simulation_results(
        output_dir=output_dir,
        trains=trains,
        step_size=step_size,
        total_steps=total_steps,
        line_shape=line_shape,
        block_system_type=block_system_type,
        name=label,
    )
    print(f"保存完了: {Path(output_dir).resolve()}")


# ---- グラフ描画 ----------------------------------------------------------


def example_diagram_single() -> None:
    """ダイヤ図：単一シミュレーション結果"""
    result = SimulationResult.load(OUTPUT_MULTI, label="マルチ列車")

    print(f"列車一覧: {result.train_names}")

    plot_diagram(result, show=False)
    plt.suptitle("ダイヤ図（全時刻・全区間）")
    plt.tight_layout()
    plt.show()


def example_diagram_range() -> None:
    """ダイヤ図：時間帯・距離範囲を指定"""
    result = SimulationResult.load(OUTPUT_MULTI, label="マルチ列車")

    plot_diagram(
        result,
        time_range=(0, 200),  # 0〜200秒
        position_range=(0, 5),  # 0〜5km
        show=False,
    )
    plt.suptitle("ダイヤ図（時間帯・区間指定）")
    plt.tight_layout()
    plt.show()


def example_diagram_multi_result() -> None:
    """ダイヤ図：複数シミュレーション結果の重ね合わせ"""
    result_simple = SimulationResult.load(OUTPUT_SIMPLE, label="シンプル")
    result_multi = SimulationResult.load(OUTPUT_MULTI, label="マルチ")

    plot_diagram(
        [result_simple, result_multi],
        show=False,
    )
    plt.suptitle("ダイヤ図（複数シミュレーション重ね合わせ）")
    plt.tight_layout()
    plt.show()


def example_running_curve_single() -> None:
    """ランカーブ：単一列車"""
    result = SimulationResult.load(OUTPUT_SIMPLE, label="シンプル")

    train_name = result.train_names[0]
    plot_running_curve(
        [(result, train_name)],
        show=False,
    )
    plt.suptitle(f"ランカーブ（{train_name}）")
    plt.tight_layout()
    plt.show()


def example_running_curve_multi() -> None:
    """ランカーブ：複数列車の重ね合わせ（シミュレーションをまたぐ）"""
    result_simple = SimulationResult.load(OUTPUT_SIMPLE, label="シンプル")
    result_multi = SimulationResult.load(OUTPUT_MULTI, label="マルチ")

    # シンプルの列車1本 ＋ マルチの全列車を重ね合わせ
    specs = [(result_simple, result_simple.train_names[0])]
    for name in result_multi.train_names:
        specs.append((result_multi, name))

    plot_running_curve(
        specs,
        position_range=(0, 5),  # 0〜5km の区間
        show=False,
    )
    plt.suptitle("ランカーブ（複数列車・複数シミュレーション重ね合わせ）")
    plt.tight_layout()
    plt.show()


def example_subplot() -> None:
    """ダイヤ図とランカーブを subplot で並べて表示"""
    result = SimulationResult.load(OUTPUT_MULTI, label="マルチ列車")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    plot_diagram(result, ax=ax1, show=False)
    ax1.set_title("ダイヤ図")

    specs = [(result, name) for name in result.train_names]
    plot_running_curve(specs, ax=ax2, show=False)
    ax2.set_title("ランカーブ")

    plt.suptitle("ダイヤ図 ＋ ランカーブ")
    plt.tight_layout()
    plt.show()


# ---- メイン --------------------------------------------------------------


def main() -> None:
    # シミュレーションを実行して結果を保存
    print("=== シミュレーション実行・保存 ===")
    run_and_save(SIMPLE_MODEL, OUTPUT_SIMPLE, "シンプル")
    run_and_save(MULTI_MODEL, OUTPUT_MULTI, "マルチ列車")

    print("\n=== グラフ描画 ===")

    # ダイヤ図の例
    example_diagram_single()
    example_diagram_range()
    example_diagram_multi_result()

    # ランカーブの例
    example_running_curve_single()
    example_running_curve_multi()

    # subplot での組み合わせ例
    example_subplot()


if __name__ == "__main__":
    main()
