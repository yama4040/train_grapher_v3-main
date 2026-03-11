"""実験結果の一覧を表示するスクリプト

使い方:
    uv run python experiments/scripts/list_results.py
    uv run python experiments/scripts/list_results.py exp001_example
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("experiments/results")


def _load_run_info(run_dir: Path) -> dict:
    run_info_path = run_dir / "run_info.json"
    if run_info_path.exists():
        return json.loads(run_info_path.read_text(encoding="utf-8"))
    return {}


def _load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "results_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def list_all() -> None:
    """全実験の実行一覧を表示"""
    if not RESULTS_DIR.exists() or not any(RESULTS_DIR.iterdir()):
        print("結果がありません。まず実験を実行してください:")
        print("  uv run python experiments/scripts/run.py <実験名>")
        return

    experiments = sorted(RESULTS_DIR.iterdir())
    print(f"{'実験名':<35} {'実行数':>5}  最終実行")
    print("-" * 65)

    for exp_dir in experiments:
        if not exp_dir.is_dir():
            continue
        runs = sorted(exp_dir.iterdir())
        run_count = len(runs)
        latest = runs[-1].name if runs else "-"
        print(f"{exp_dir.name:<35} {run_count:>5}  {latest}")


def list_experiment(experiment_name: str) -> None:
    """指定実験の全実行を詳細表示"""
    exp_dir = RESULTS_DIR / experiment_name
    if not exp_dir.exists():
        print(f"実験 '{experiment_name}' の結果が見つかりません。")
        return

    runs = sorted(exp_dir.iterdir())
    if not runs:
        print(f"実験 '{experiment_name}' の実行記録がありません。")
        return

    print(f"実験: {experiment_name}  ({len(runs)} 件)")
    print("=" * 65)

    for run_dir in runs:
        if not run_dir.is_dir():
            continue
        info = _load_run_info(run_dir)
        summary = _load_summary(run_dir)

        print(f"\n  実行ID: {run_dir.name}")
        if info.get("timestamp"):
            print(f"  日時:   {info['timestamp']}")

        sim = info.get("simulation", {})
        if sim:
            total_time = sim.get("step_size", 0) * sim.get("total_steps", 0)
            print(
                f"  設定:   {sim.get('block_system_type', '-')} | "
                f"列車 {sim.get('train_count', '-')} 本 | "
                f"シミュ時間 {total_time:.0f} 秒"
            )

        trains = summary.get("trains", [])
        if trains:
            print("  列車結果:")
            for t in trains:
                max_v = t.get("max_velocity_kmh")
                end_t = t.get("end_time_s")
                print(
                    f"    - {t['name']}: "
                    f"最高速度 {max_v} km/h, "
                    f"終了時刻 {end_t} 秒"
                )

        print(f"  パス:   {run_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="実験結果の一覧を表示します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  uv run python experiments/scripts/list_results.py              # 全実験を一覧表示
  uv run python experiments/scripts/list_results.py exp001_example  # 特定実験の詳細
""",
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        help="実験名（省略時は全実験を一覧表示）",
    )
    args = parser.parse_args()

    if args.experiment:
        list_experiment(args.experiment)
    else:
        list_all()


if __name__ == "__main__":
    main()
