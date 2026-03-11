import logging
import os  # Added import for os module
from datetime import datetime
from typing import Optional

# ANSIエスケープシーケンスを使って色を定義するための定数
COLORS = {
    "DEBUG": "\033[94m",  # 青色
    "INFO": "\033[96m",  # 水色
    "WARNING": "\033[93m",  # 黄色
    "ERROR": "\033[91m",  # 赤色
    "CRITICAL": "\033[91m",  # 赤色
    "RESET": "\033[0m",  # リセット
}


class MultilineFormatter(logging.Formatter):
    def format(self, record):
        # superでフォーマット
        original_message = super().format(record)

        # 各行の先頭に情報を付加
        lines = original_message.split("\n")
        timestamp = self.formatTime(record, self.datefmt)
        header = f"{timestamp}.{int(record.msecs)} {record.name}:{record.lineno} {record.funcName} [{record.levelname}]"

        # すべての行を結合
        formatted_message = "\n".join(f"{header} {line}" for line in lines)

        return formatted_message


class ColoredConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            # ログレベル名の色を設定する
            levelname = record.levelname
            if (levelname in COLORS) and (levelname != "CRITICAL"):
                # CRITICALのとき以外ログレベル名に色を付ける
                formatted_levelname = (
                    f"{COLORS[levelname]}{levelname:^8}{COLORS['RESET']}"
                )
                record.levelname = formatted_levelname

            # ログメッセージを通常の方法で処理する
            super().emit(record)

            # ログレベル名を元に戻す
            record.levelname = levelname

        except Exception:
            self.handleError(record)

    def format(self, record):
        # CRITICALのとき背景色を変更
        message = super().format(record)
        if record.levelname == "CRITICAL":
            # 赤色の背景にするANSIエスケープシーケンス
            message = f"\033[41m{message.replace(COLORS['RESET'], '')}\033[0m"
        return message


def setup_logger(
    name: str, *, log_file: Optional[str] = None, level: int = logging.DEBUG
) -> logging.Logger:
    """
    指定された名前、ログファイル、およびログレベルでロガーを設定します。

    コンソールとファイルにログを記録する。ファイルには詳細事項を、コンソールには見やすいように表示する。
    設定したロガーで記録できる。

    Args:
        name (str): ロガーの名前(基本的に__name__を指定)
        log_file (Optional[str], optional): ログを書き込むファイル Noneの場合時刻入りのファイル名になる
        level (int, optional): ログレベル（例：logging.INFO、logging.DEBUG）

    Returns:
        logging.Logger: 設定されたロガー

    Examples:
        使い方

        >>> logger = setup_logger(__name__) # ロガーの作成
        >>>
        >>> logger.debug("debug") # デバッグに使うような詳細な情報
        >>> logger.info("info") # 想定通りに動作しているときのログ
        >>> logger.warning("warning") # 予期しないことが発生 or 近い将来に問題が発生する
        >>> logger.error("error") # エラーが発生してコードの一部が実行不可能
        >>> logger.critical("critical") # 重大なエラーが発生してコードすべてが実行不可能

    """
    # カスタムロガーを作成
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 現在の日時を取得してファイル名に組み込む
    if log_file is None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = "./logs"  # Define log directory
        log_file = os.path.join(log_dir, f"log_{current_time}.log")

        # log_dirが存在しない場合は作成
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # ハンドラを作成
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    console_handler = ColoredConsoleHandler()
    console_handler.setLevel(level)

    # フォーマッタを作成し、ハンドラに追加
    f_format = MultilineFormatter("%(message)s", "%Y-%m-%d %H:%M:%S")
    c_format = MultilineFormatter("%(message)s", "%Y-%m-%d %H:%M:%S")

    # ハンドラにフォーマッタを登録
    file_handler.setFormatter(f_format)
    console_handler.setFormatter(c_format)

    # ハンドラをロガーに追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
