"""基準モデル バッチ実行スクリプト

移動閉塞環境で15秒安全確認を行う基準モデルを、
experiments/datas/ 内の JSON ファイルに対して実行する。

使い方:
    # 動作確認（ランダム 1 ファイル）
    uv run python experiments/base_model/runner.py --test

    # 全ファイル実行（4並列）
    uv run python experiments/base_model/runner.py --select all --workers 4

    # 最初の 5 ファイルのみ
    uv run python experiments/base_model/runner.py --select first:5

    # ランダムに 3 ファイル
    uv run python experiments/base_model/runner.py --select random:3

    # インデックス指定
    uv run python experiments/base_model/runner.py --select index:0,1,5

    # データディレクトリを明示指定
    uv run python experiments/base_model/runner.py --datas path/to/datas --select all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# プロジェクトルートと base_model ディレクトリを sys.path に追加
# Windows の spawn 方式でも確実に動作するよう、モジュールレベルで実行する
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BASE_MODEL_DIR = Path(__file__).parent
_DEFAULT_DATAS_DIR = _BASE_MODEL_DIR.parent / "datas"
_DEFAULT_RESULTS_DIR = _BASE_MODEL_DIR.parent / "results" / "base_model"

for _p in (str(_PROJECT_ROOT / "src"), str(_BASE_MODEL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from driving_decision import BaseModelDrivingDecision  # noqa: E402
from train_grapher_v3.core.block_system import MovingBlockSystem  # noqa: E402
from train_grapher_v3.core.line import Line  # noqa: E402
from train_grapher_v3.core.simulation import Simulation  # noqa: E402
from train_grapher_v3.core.status import Status  # noqa: E402
from train_grapher_v3.util.logger import setup_logger  # noqa: E402
from train_grapher_v3.util.result_saver import save_simulation_results  # noqa: E402
from train_grapher_v3.util.simulation_model_io import load_simulation_model  # noqa: E402

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# トップレベル関数（ProcessPoolExecutor で pickle される）
# ---------------------------------------------------------------------------


def run_single_file(args: tuple[str, str]) -> dict:
    """1 ファイルのシミュレーションを実行する。

    ProcessPoolExecutor に渡すため、トップレベル関数として定義する。
    Windows の spawn 方式では別プロセスで実行されるため、
    このモジュールが再インポートされ sys.path の設定も再実行される。

    Args:
        args: (model_path, output_dir) の文字列タプル

    Returns:
        実行結果の辞書:
          success, model_path, output_dir, check_safety_count, train_count, error
    """
    model_path = Path(args[0])
    output_dir = Path(args[1])

    try:
        # モデル読込
        line_shape, trains, step_size, total_steps, _ = load_simulation_model(
            str(model_path)
        )

        # 移動閉塞を強制使用（実験の目的）
        block_system = MovingBlockSystem(line_shape)

        # 各列車の運転判断を BaseModelDrivingDecision に差し替え
        for train in trains:
            train._driving_decision = BaseModelDrivingDecision(step_size)

        # シミュレーション実行
        line = Line(line_shape=line_shape, trains=trains, block_system=block_system)
        simulation = Simulation(line=line, step_size=step_size)
        simulation.execution(total_steps, step_size)

        # 結果保存
        output_dir.mkdir(parents=True, exist_ok=True)
        save_simulation_results(
            output_dir=str(output_dir),
            trains=trains,
            step_size=step_size,
            total_steps=total_steps,
            line_shape=line_shape,
            block_system_type="moving",
            name=model_path.stem,
            description="base_model: 15秒安全確認付き移動閉塞シミュレーション",
            author="base_model_runner",
        )

        # CHECK_SAFETY ステータスのステップ数カウント
        check_safety_count = sum(
            1
            for train in trains
            for s in train.get_running_data().get_status_all()
            if s == Status.CHECK_SAFETY
        )

        # run_info.json 保存
        run_info = {
            "model_file": str(model_path),
            "output_dir": str(output_dir),
            "timestamp": datetime.now().isoformat(),
            "block_system_type": "moving",
            "driving_decision": "BaseModelDrivingDecision",
            "safety_check_time_s": 15.0,
            "simulation": {
                "step_size": step_size,
                "total_steps": total_steps,
                "train_count": len(trains),
            },
            "results": {
                "check_safety_total_steps": check_safety_count,
            },
        }
        (output_dir / "run_info.json").write_text(
            json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return {
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "success": True,
            "error": None,
            "check_safety_count": check_safety_count,
            "train_count": len(trains),
        }

    except Exception as e:
        return {
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "success": False,
            "error": str(e),
            "check_safety_count": 0,
            "train_count": 0,
        }


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def select_files(all_files: list[Path], select: str) -> list[Path]:
    """--select オプションに従ってファイルを選択する。

    Args:
        all_files: 候補ファイルのリスト（ソート済み）
        select: "all" / "first:N" / "last:N" / "random:N" / "index:0,1,2"

    Returns:
        選択されたファイルのリスト
    """
    if select == "all":
        return list(all_files)

    if select.startswith("first:"):
        n = int(select.split(":")[1])
        return all_files[:n]

    if select.startswith("last:"):
        n = int(select.split(":")[1])
        return all_files[-n:]

    if select.startswith("random:"):
        n = int(select.split(":")[1])
        return random.sample(all_files, min(n, len(all_files)))

    if select.startswith("index:"):
        indices = [int(i) for i in select.split(":")[1].split(",")]
        return [all_files[i] for i in indices if 0 <= i < len(all_files)]

    raise ValueError(f"不正な --select 値: '{select}'")


# ---------------------------------------------------------------------------
# バッチ実行
# ---------------------------------------------------------------------------


def run_batch(
    datas_dir: Path,
    output_base_dir: Path,
    select: str = "all",
    max_workers: int = 1,
) -> list[dict]:
    """バッチ実行。

    Args:
        datas_dir: JSON ファイルのディレクトリ
        output_base_dir: 結果保存のベースディレクトリ
        select: ファイル選択オプション
        max_workers: 並列ワーカー数

    Returns:
        各ファイルの実行結果リスト
    """
    all_files = sorted(datas_dir.glob("*.json"))
    if not all_files:
        logger.error(f"JSON ファイルが見つかりません: {datas_dir}")
        return []

    selected_files = select_files(all_files, select)
    logger.info(f"対象ファイル数: {len(selected_files)} / {len(all_files)}")

    # バッチ単位で結果をまとめるタイムスタンプディレクトリ
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = output_base_dir / batch_ts

    job_args = [
        (str(f), str(batch_output_dir / f.stem)) for f in selected_files
    ]

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(run_single_file, arg): arg[0] for arg in job_args
        }
        for future in as_completed(future_to_path):
            result = future.result()
            results.append(result)
            mark = "OK" if result["success"] else "NG"
            error_msg = f"  ERROR: {result['error']}" if not result["success"] else ""
            logger.info(
                f"[{mark}] {Path(result['model_path']).name}"
                f"  CHECK_SAFETY={result['check_safety_count']} steps"
                + error_msg
            )

    return results


# ---------------------------------------------------------------------------
# テスト実行
# ---------------------------------------------------------------------------


def run_test(datas_dir: Path, output_base_dir: Path) -> bool:
    """テスト実行（--test フラグ用）。

    datas_dir からランダムに 1 ファイル選択して実行し、結果を検証する。

    Returns:
        テスト成功の場合 True
    """
    all_files = sorted(datas_dir.glob("*.json"))
    if not all_files:
        logger.error(f"テスト用 JSON ファイルが見つかりません: {datas_dir}")
        return False

    selected = random.choice(all_files)
    logger.info(f"テスト対象: {selected.name}")

    test_output = output_base_dir / "test" / datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_single_file((str(selected), str(test_output)))

    if not result["success"]:
        logger.error(f"実行エラー: {result['error']}")
        return False

    # 結果ファイルの存在確認
    success = True
    output_dir = Path(result["output_dir"])
    required_files = [
        "run_info.json",
        "initial_conditions.json",
        "results_summary.json",
    ]
    for fname in required_files:
        fpath = output_dir / fname
        if not fpath.exists():
            logger.error(f"結果ファイルが存在しません: {fpath}")
            success = False

    # CHECK_SAFETY の記録確認（警告のみ）
    check_count = result["check_safety_count"]
    if check_count > 0:
        logger.info(f"CHECK_SAFETY 記録ステップ数: {check_count} steps")
    else:
        logger.warning(
            "CHECK_SAFETY が記録されませんでした。"
            "移動閉塞による駅間停車が発生しなかった可能性があります。"
        )

    logger.info("=" * 55)
    logger.info("テスト結果サマリー")
    logger.info(f"  対象ファイル            : {selected.name}")
    logger.info(f"  列車数                  : {result['train_count']}")
    logger.info(f"  CHECK_SAFETY ステップ数 : {check_count}")
    logger.info(f"  結果保存先              : {output_dir}")
    logger.info(f"  判定                    : {'PASS' if success else 'FAIL'}")
    logger.info("=" * 55)

    return success


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基準モデル（15秒安全確認）バッチ実行スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 動作確認（1 ファイルランダム実行）
  uv run python experiments/base_model/runner.py --test

  # 全ファイル実行（4 並列）
  uv run python experiments/base_model/runner.py --select all --workers 4

  # 最初の 5 ファイルのみ
  uv run python experiments/base_model/runner.py --select first:5

  # ランダムに 3 ファイル
  uv run python experiments/base_model/runner.py --select random:3

  # インデックス指定（0-based）
  uv run python experiments/base_model/runner.py --select index:0,2,4
""",
    )
    parser.add_argument(
        "--select",
        default="all",
        help=(
            "実行ファイルの選択: 'all', 'first:N', 'last:N', 'random:N', 'index:0,1,2'"
            " (デフォルト: all)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="並列ワーカー数 (デフォルト: 1)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="テストモード: ランダム 1 ファイルを実行して検証する",
    )
    parser.add_argument(
        "--datas",
        default=str(_DEFAULT_DATAS_DIR),
        help=f"データディレクトリ (デフォルト: {_DEFAULT_DATAS_DIR})",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    datas_dir = Path(args.datas)

    if not datas_dir.exists():
        logger.error(f"データディレクトリが存在しません: {datas_dir}")
        sys.exit(1)

    if args.test:
        success = run_test(datas_dir, _DEFAULT_RESULTS_DIR)
        sys.exit(0 if success else 1)

    results = run_batch(
        datas_dir=datas_dir,
        output_base_dir=_DEFAULT_RESULTS_DIR,
        select=args.select,
        max_workers=args.workers,
    )

    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    failed_count = total - success_count
    total_check_safety = sum(r["check_safety_count"] for r in results)

    logger.info("=" * 60)
    logger.info("バッチ実行サマリー")
    logger.info(f"  総ファイル数               : {total}")
    logger.info(f"  成功                       : {success_count}")
    logger.info(f"  失敗                       : {failed_count}")
    logger.info(f"  CHECK_SAFETY 合計ステップ数: {total_check_safety}")
    logger.info("=" * 60)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
