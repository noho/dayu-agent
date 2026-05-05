"""MinerU 云 API 配额本地跟踪器。

提供每日页数配额的消耗与查询，支持文件持久化以跨进程共享状态。

设计目标：

- 为 MinerU 云 API 提供本地页数配额计量，防止每日上限（默认 5000 页）被意外突破。
- 配额状态持久化到 ``~/.dayu/quota_state.json``，使 CLI 多次调用之间
  能共享同一天的消耗记录。
- 日期变更时自动重置计数。
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from dayu.log import Log

_MODULE = __name__

_DEFAULT_DAILY_LIMIT: int = 5000
_DEFAULT_STATE_FILE: str = "~/.dayu/quota_state.json"


class QuotaExhaustedError(RuntimeError):
    """MinerU 配额不足异常。

    当当日可用页数不足以覆盖本次请求时抛出。
    """

    def __init__(self, used: int, requested: int, limit: int) -> None:
        self.used = used
        self.requested = requested
        self.limit = limit
        super().__init__(
            f"MinerU 配额不足: 已用={used}, 请求={requested}, 上限={limit}"
        )


class QuotaTracker:
    """MinerU 云 API 每日页数配额跟踪器。

    配额状态持久化到 JSON 文件，跨进程共享同一天的消耗记录。
    日期变更时自动重置。

    ⚠️ 当前实现不是线程安全的。CLI 单进程场景足够；
    如果 dayu-agent 变成多线程服务，需要加锁（threading.Lock）。

    Attributes:
        daily_limit: 每日页数上限。
    """

    def __init__(
        self,
        daily_limit: int = _DEFAULT_DAILY_LIMIT,
        state_file: str = _DEFAULT_STATE_FILE,
    ) -> None:
        """初始化配额跟踪器。

        Args:
            daily_limit: 每日页数上限，默认 5000。
            state_file: 状态持久化文件路径，支持 ``~`` 展开。
        """
        self.daily_limit: int = daily_limit
        self._state_file: Path = Path(state_file).expanduser()
        self._used_today: int = 0
        self._date: str = date.today().isoformat()
        self._load_state()

    def _load_state(self) -> None:
        """从持久化文件加载配额状态。"""
        if not self._state_file.exists():
            self._reset()
            return
        try:
            raw_text = self._state_file.read_text(encoding="utf-8")
            state = json.loads(raw_text)
            stored_date = str(state.get("date", ""))
            if stored_date == date.today().isoformat():
                self._used_today = int(state.get("used_today", 0))
                self._date = stored_date
            else:
                self._reset()
        except Exception as exc:
            Log.warn(
                f"配额状态文件加载失败，已重置: {exc}",
                module=_MODULE,
            )
            self._reset()

    def _reset(self) -> None:
        """重置配额计数到当日初始状态。"""
        self._used_today = 0
        self._date = date.today().isoformat()

    def _save_state(self) -> None:
        """持久化当前配额状态到文件。"""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"date": self._date, "used_today": self._used_today},
                ensure_ascii=False,
            )
            self._state_file.write_text(payload, encoding="utf-8")
        except Exception as exc:
            Log.warn(f"配额状态文件写入失败: {exc}", module=_MODULE)

    def _ensure_current_date(self) -> None:
        """确保内部日期为今天，不是则重置。"""
        today = date.today().isoformat()
        if today != self._date:
            self._reset()

    def check_and_consume(self, pages: int) -> bool:
        """检查并消费配额。

        Args:
            pages: 本次请求的页数。

        Returns:
            ``True`` 表示配额充足且已消费；``False`` 表示配额不足，未消费。
        """
        self._ensure_current_date()
        if self._used_today + pages > self.daily_limit:
            Log.warn(
                (
                    "MinerU 配额不足: "
                    f"已用={self._used_today}, 请求={pages}, "
                    f"上限={self.daily_limit}"
                ),
                module=_MODULE,
            )
            return False
        self._used_today += pages
        self._save_state()
        return True

    def check_and_consume_or_raise(self, pages: int) -> None:
        """检查并消费配额，不足时抛出异常。

        Args:
            pages: 本次请求的页数。

        Raises:
            QuotaExhaustedError: 配额不足时抛出。
        """
        self._ensure_current_date()
        if self._used_today + pages > self.daily_limit:
            raise QuotaExhaustedError(
                used=self._used_today,
                requested=pages,
                limit=self.daily_limit,
            )
        self._used_today += pages
        self._save_state()

    def get_remaining(self) -> int:
        """查询当日剩余配额页数。

        Returns:
            剩余可用页数；若日期已变更则返回全量上限。
        """
        self._ensure_current_date()
        return self.daily_limit - self._used_today

    def get_used(self) -> int:
        """查询当日已消耗页数。

        Returns:
            已消耗的页数。
        """
        self._ensure_current_date()
        return self._used_today


__all__ = [
    "QuotaExhaustedError",
    "QuotaTracker",
]
