"""Streamlit 自选股组件基础单元测试。"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

watchlist_module = importlib.import_module("dayu.web.streamlit.components.watchlist")


@pytest.mark.unit
def test_watchlist_storage_path_uses_workspace_dayu_streamlit(tmp_path: Path) -> None:
    """验证自选股存储路径固定落在 `workspace/.dayu/streamlit/watchlist.json`。"""

    storage_path = watchlist_module._watchlist_storage_path(tmp_path)

    assert storage_path == tmp_path / ".dayu" / "streamlit" / "watchlist.json"


@pytest.mark.unit
def test_load_watchlist_items_returns_empty_when_file_missing(tmp_path: Path) -> None:
    """验证存储文件不存在时返回空列表。"""

    loaded = watchlist_module.load_watchlist_items(tmp_path)

    assert loaded == []


@pytest.mark.unit
def test_save_and_load_watchlist_items_round_trip(tmp_path: Path) -> None:
    """验证自选股持久化后可正确读回。"""

    items = [
        watchlist_module.WatchlistItem(
            ticker="AAPL",
            company_name="Apple",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        ),
        watchlist_module.WatchlistItem(
            ticker="MSFT",
            company_name="Microsoft",
            created_at="2026-01-02T00:00:00+00:00",
            updated_at="2026-01-02T00:00:00+00:00",
        ),
    ]

    watchlist_module.save_watchlist_items(tmp_path, items)
    loaded = watchlist_module.load_watchlist_items(tmp_path)

    assert loaded == items


@pytest.mark.unit
def test_load_watchlist_items_returns_empty_on_invalid_json(tmp_path: Path) -> None:
    """验证 JSON 解析失败时返回空列表而不是抛异常。"""

    storage_path = watchlist_module._watchlist_storage_path(tmp_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_text("{not-json}", encoding="utf-8")

    loaded = watchlist_module.load_watchlist_items(tmp_path)

    assert loaded == []


@pytest.mark.unit
def test_series_cell_str_trims_and_handles_empty_cells() -> None:
    """验证表格单元格字符串规整逻辑。"""

    row = pd.Series({"股票代码": "  aapl  ", "公司名称": pd.NA})

    assert watchlist_module._series_cell_str(row, "股票代码") == "aapl"
    assert watchlist_module._series_cell_str(row, "公司名称") == ""
    assert watchlist_module._series_cell_str(row, "不存在列") == ""


@pytest.mark.unit
def test_build_final_dataframe_applies_delete_edit_and_add() -> None:
    """验证删除、编辑、新增三类变更都会合并到最终表格。"""

    original = pd.DataFrame(
        [
            {"股票代码": "AAPL", "公司名称": "Apple"},
            {"股票代码": "TSLA", "公司名称": "Tesla"},
        ]
    )
    editor_state = {
        "deleted_rows": [0],
        "edited_rows": {"0": {"公司名称": "Tesla Motors"}},
        "added_rows": [{"股票代码": "MSFT", "公司名称": "Microsoft"}],
    }

    final_df = watchlist_module._build_final_dataframe(original, editor_state)

    assert final_df.to_dict(orient="records") == [
        {"股票代码": "TSLA", "公司名称": "Tesla Motors"},
        {"股票代码": "MSFT", "公司名称": "Microsoft"},
    ]


@pytest.mark.unit
def test_apply_table_to_storage_rejects_duplicate_ticker(tmp_path: Path) -> None:
    """验证保存时会拒绝重复股票代码。"""

    original = pd.DataFrame([{"股票代码": "AAPL", "公司名称": "Apple"}])
    editor_state = {
        "edited_rows": {},
        "deleted_rows": [],
        "added_rows": [{"股票代码": "aapl", "公司名称": "Apple Inc."}],
    }

    ok, message = watchlist_module._apply_table_to_storage(
        workspace_root=tmp_path,
        original=original,
        editor_state=editor_state,
        previous=[],
    )

    assert ok is False
    assert "重复" in message
