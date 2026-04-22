"""Streamlit 自选股管理对话框组件。

在表格内完成自选股的添加、删除、编辑，保存后写入本地 JSON。
存储路径：workspace/streamlit_watchlist.json。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from dayu.web.streamlit.components.sidebar import (
    WatchlistItem,
    load_watchlist_items,
    save_watchlist_items,
)


def _now_iso8601() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串。

    Returns:
        不含微秒的 ISO8601 字符串。

    Raises:
        无。
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_cell_empty(val: object) -> bool:
    """判断表格单元格是否为空（含 NaN、pd.NA）。

    Args:
        val: 单元格原始值。

    Returns:
        视为空则为 True。

    Raises:
        无。
    """

    if val is None:
        return True
    try:
        if val is pd.NA:
            return True
    except (TypeError, AttributeError):
        pass
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


def _series_cell_str(series_row: pd.Series, key: str) -> str:
    """从 DataFrame 行读取单元格为去空白字符串；空或缺失视为空串。

    Args:
        series_row: 一行数据。
        key: 列名。

    Returns:
        去首尾空白后的字符串，空单元格返回空串。

    Raises:
        无。
    """

    if key not in series_row.index:
        return ""
    val: object = series_row[key]
    if _is_cell_empty(val):
        return ""
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def _items_to_dataframe(items: list[WatchlistItem]) -> pd.DataFrame:
    """将自选股列表转为可编辑表格 DataFrame。

    Args:
        items: 自选股条目列表。

    Returns:
        列为「股票代码」「公司名称」的 DataFrame。

    Raises:
        无。
    """

    if not items:
        return pd.DataFrame(columns=["股票代码", "公司名称"])
    rows = [{"股票代码": i.ticker, "公司名称": i.company_name} for i in items]
    return pd.DataFrame(rows)


def _build_final_dataframe(
    original: pd.DataFrame,
    editor_state: dict,
) -> pd.DataFrame:
    """根据 data_editor 的 session state 构建最终的 DataFrame。

    处理删除、编辑、新增三种操作，返回反映用户所有修改后的 DataFrame。

    Args:
        original: 原始 DataFrame（即传给 data_editor 的数据）。
        editor_state: st.session_state[editor_key]，包含 deleted_rows、edited_rows、added_rows。

    Returns:
        处理后的最终 DataFrame。

    Raises:
        无。
    """

    # 复制原始数据
    df = original.copy()

    # 处理删除：按索引删除行（从大到小删除避免索引变化）
    deleted_indices = editor_state.get("deleted_rows", [])
    if deleted_indices:
        # 从大到小排序，确保删除时索引不偏移
        for idx in sorted(deleted_indices, reverse=True):
            if idx < len(df):
                df = df.drop(df.index[idx])
        df = df.reset_index(drop=True)

    # 处理编辑：更新指定行的数据
    edited_rows = editor_state.get("edited_rows", {})
    for row_idx, changes in edited_rows.items():
        row_idx = int(row_idx)
        if row_idx < len(df):
            for col_name, new_value in changes.items():
                df.at[df.index[row_idx], col_name] = new_value

    # 处理新增：添加新行
    added_rows = editor_state.get("added_rows", [])
    for added in added_rows:
        new_row = {"股票代码": added.get("股票代码", ""), "公司名称": added.get("公司名称", "")}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df


def _apply_table_to_storage(
    workspace_root: Path,
    original: pd.DataFrame,
    editor_state: dict,
    previous: list[WatchlistItem],
) -> tuple[bool, str]:
    """根据编辑后的表格原子写回存储。

    Args:
        workspace_root: 工作区根目录。
        original: 原始 DataFrame（传给 data_editor 的数据）。
        editor_state: st.session_state[editor_key]，包含删除/编辑/新增信息。
        previous: 保存前的列表，用于保留已有条目的 created_at。

    Returns:
        (是否成功, 说明信息)。

    Raises:
        无：校验失败时返回 (False, 错误说明)，不向调用方抛出。
    """

    # 构建最终 DataFrame（应用删除、编辑、新增）
    final_df = _build_final_dataframe(original, editor_state)

    prev_by_ticker = {p.ticker.upper(): p for p in previous}

    rows_out: list[WatchlistItem] = []
    seen_tickers: set[str] = set()
    now = _now_iso8601()

    for _, series_row in final_df.iterrows():
        ticker = _series_cell_str(series_row, "股票代码").upper()
        name = _series_cell_str(series_row, "公司名称")

        if not ticker and not name:
            continue
        if not ticker:
            return False, "存在未填写股票代码的行，请补全或删除空行。"
        if not name:
            return False, f"股票代码 {ticker} 未填写公司名称。"
        if ticker in seen_tickers:
            return False, f"股票代码 {ticker} 重复，请保留唯一一行。"
        seen_tickers.add(ticker)

        prev = prev_by_ticker.get(ticker)
        if prev is not None:
            if name == prev.company_name:
                rows_out.append(prev)
            else:
                rows_out.append(
                    WatchlistItem(
                        ticker=prev.ticker,
                        company_name=name,
                        created_at=prev.created_at,
                        updated_at=now,
                    )
                )
        else:
            rows_out.append(
                WatchlistItem(
                    ticker=ticker,
                    company_name=name,
                    created_at=now,
                    updated_at=now,
                )
            )

    save_watchlist_items(workspace_root, rows_out)
    return True, f"已保存 {len(rows_out)} 条自选股。"


@st.dialog("自选股管理", width="large")
def render_watchlist_manager(workspace_root: Path) -> None:
    """弹出对话框：在表格内添加、删除、编辑自选股，保存后写回文件。

    Args:
        workspace_root: 工作区根目录。

    Raises:
        无：加载失败时在界面提示，不向调用方抛出。
    """

    editor_key = "watchlist_table_editor"

    # 检查是否有保存后的刷新标记，用于重新加载数据
    refresh_key = "watchlist_needs_refresh"
    if st.session_state.get(refresh_key, False):
        st.session_state[refresh_key] = False
        # 清除 data_editor 状态以重新加载最新数据
        if editor_key in st.session_state:
            del st.session_state[editor_key]

    st.markdown("在表格中编辑；底部可新增行；删除行即移除该自选股。编辑后检查代码与名称是否正确，然后点击 **保存**。")

    try:
        previous = load_watchlist_items(workspace_root)
    except Exception as exc:
        st.error(f"加载自选股失败: {exc}")
        previous = []

    df = _items_to_dataframe(previous)

    # 注意：不要在这里删除 session_state[editor_key]，否则会丢失删除/编辑记录
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        key=editor_key,
        column_config={
            "股票代码": st.column_config.TextColumn(
                "股票代码",
                help="如 AAPL、MSFT；新增行填写代码",
            ),
            "公司名称": st.column_config.TextColumn(
                "公司名称",
                help="公司显示名称",
            ),
        },
        hide_index=True,
        width="stretch",
    )

    _, btn_col = st.columns([4, 1])
    with btn_col:
        save_clicked = st.button("保存", type="secondary", key="watchlist_save_btn", width="stretch")

    if save_clicked:
        # 从 session_state 获取编辑器的完整状态（包含删除、编辑、新增记录）
        editor_state = st.session_state.get(editor_key, {})
        ok, msg = _apply_table_to_storage(workspace_root, df, editor_state, previous)
        if ok:
            st.success(msg)
            # 设置刷新标记，下次渲染时会清除 data_editor 状态并重新加载数据
            st.session_state["watchlist_needs_refresh"] = True
        else:
            st.error(msg)