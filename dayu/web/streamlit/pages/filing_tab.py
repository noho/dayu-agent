"""Streamlit 财报管理 Tab 页面入口。"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd
import streamlit as st

from dayu.services.protocols import FinsServiceProtocol
from dayu.web.streamlit.components.watchlist import WatchlistItem
from dayu.web.streamlit.pages.filing.download_panel import (
    render_download_section,
)

_DATAFRAME_ROW_HEIGHT_PX = 35
_DATAFRAME_HEADER_HEIGHT_PX = 38


class _FilingInfo(TypedDict):
    """单条财报文件展示信息。"""

    document_id: str
    file_name: str
    file_path: str
    form_type: str
    filing_date: str
    report_date: str
    fiscal_year: str
    fiscal_period: str
    status: str


def _get_filing_list(
    workspace_root: Path,
    ticker: str,
    fins_service: FinsServiceProtocol | None,
) -> list[_FilingInfo]:
    """获取指定股票的已下载财报列表。

    参数:
        workspace_root: 工作区根目录。
        ticker: 股票代码。
        fins_service: 财报服务协议实例；为 None 时返回空列表。

    返回值:
        财报文件信息列表。

    异常:
        OSError: 底层读取财报文件失败时抛出。
        ValueError: 财报服务返回非法数据时抛出。
    """

    if fins_service is None:
        return []

    summaries = fins_service.list_filings(ticker)

    resolved_root = workspace_root.resolve()
    filings: list[_FilingInfo] = []
    for s in summaries:
        file_name = s.primary_file_name or "未知"
        if s.primary_file_path:
            # 两端同步 resolve，避免在 Windows 上一端带盘符、一端不带导致
            # relative_to 误判 drive 不匹配；输出固定 POSIX 风格，跨平台稳定。
            try:
                resolved_file = Path(s.primary_file_path).resolve()
                relative_path = resolved_file.relative_to(resolved_root)
                file_path = relative_path.as_posix()
            except ValueError:
                file_path = s.primary_file_path
        else:
            file_path = "未知"

        filing_info: _FilingInfo = {
            "document_id": s.document_id,
            "file_name": file_name,
            "file_path": file_path,
            "form_type": s.form_type or "未知",
            "filing_date": s.filing_date or "未知",
            "report_date": s.report_date or "未知",
            "fiscal_year": str(s.fiscal_year) if s.fiscal_year is not None else "未知",
            "fiscal_period": s.fiscal_period or "未知",
            "status": "可用" if not s.is_deleted else "已删除",
        }
        filings.append(filing_info)

    return filings


def _render_filing_table(filings: list[_FilingInfo]) -> None:
    """渲染财报列表表格。"""

    df_data = []
    for f in filings:
        df_data.append({
            "文件名称": f["file_name"],
            "文件路径": f["file_path"],
            "表单类型": f["form_type"],
            "申报日期": f["filing_date"],
            "报告日期": f["report_date"],
            "财年": f["fiscal_year"],
            "财期": f["fiscal_period"],
            "状态": f["status"],
        })

    if df_data:
        df = pd.DataFrame(df_data)
        table_height = _calculate_dataframe_height(len(df_data))
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            height=table_height,
            on_select="ignore",
            selection_mode="single-row",
            key="filing_table",
            column_config={
                "文件名称": st.column_config.TextColumn("文件名称", width="medium"),
                "文件路径": st.column_config.TextColumn("文件路径", width="large"),
                "表单类型": st.column_config.TextColumn("表单类型", width="small"),
                "申报日期": st.column_config.TextColumn("申报日期", width="small"),
                "报告日期": st.column_config.TextColumn("报告日期", width="small"),
                "财年": st.column_config.TextColumn("财年", width="small"),
                "财期": st.column_config.TextColumn("财期", width="small"),
                "状态": st.column_config.TextColumn("状态", width="small"),
            },
        )
    else:
        st.info("暂无有效财报数据")


def _calculate_dataframe_height(visible_rows: int) -> int:
    """按可见行数计算 DataFrame 组件高度（像素）。

    参数:
        visible_rows: 目标可见数据行数，必须 >= 1。

    返回值:
        DataFrame 组件高度（像素）。

    异常:
        ValueError: 当 visible_rows 小于 1 时抛出。
    """

    if visible_rows < 1:
        raise ValueError("visible_rows 必须大于等于 1")
    return _DATAFRAME_HEADER_HEIGHT_PX + visible_rows * _DATAFRAME_ROW_HEIGHT_PX


def render_filing_tab(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    fins_service: FinsServiceProtocol | None,
) -> None:
    """渲染财报管理 Tab。

    参数:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        fins_service: 财报服务协议实例；为 None 时下载功能不可用。

    返回值:
        无。

    异常:
        无。
    """

    show_download_settings = render_download_section(selected_stock, fins_service)

    try:
        filings = _get_filing_list(workspace_root, selected_stock.ticker, fins_service)
    except (OSError, ValueError):
        st.error("读取财报列表失败，请确认工作区路径是否正确")
        filings = []

    st.markdown("---")

    if filings:
        _render_filing_table(filings)
    else:
        if not show_download_settings:
            st.info("暂无财报，请点击「下载财报」按钮获取")
