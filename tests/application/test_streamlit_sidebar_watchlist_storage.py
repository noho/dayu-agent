"""自选股侧栏持久化路径测试。"""

from __future__ import annotations

import json
from pathlib import Path

from dayu.web.streamlit.components.sidebar import WatchlistItem, load_watchlist_items, save_watchlist_items


def _storage_path(workspace_root: Path) -> Path:
    """构造自选股存储路径。

    Args:
        workspace_root: 工作区目录。

    Returns:
        自选股 JSON 路径。

    Raises:
        无。
    """

    return workspace_root / ".dayu" / "streamlit" / "watchlist.json"


def test_save_watchlist_items_writes_to_new_nested_path(tmp_path: Path) -> None:
    """保存时写入新路径并自动创建目录。"""

    items = [
        WatchlistItem(
            ticker="AAPL",
            company_name="Apple",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    ]
    save_watchlist_items(tmp_path, items)

    output_path = _storage_path(tmp_path)
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["items"][0]["ticker"] == "AAPL"


def test_load_watchlist_items_returns_empty_when_file_missing(tmp_path: Path) -> None:
    """读取时若文件不存在，应返回空列表。"""

    assert load_watchlist_items(tmp_path) == []
