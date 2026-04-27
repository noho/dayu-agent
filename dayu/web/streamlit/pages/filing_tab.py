"""Streamlit 财报管理 Tab 页面。

展示已下载财报列表，提供下载新财报功能。
"""

from __future__ import annotations

import asyncio
import datetime
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import pandas as pd
import streamlit as st

from dayu.contracts.fins import (
    DownloadCommandPayload,
    DownloadProgressPayload,
    FinsCommand,
    FinsCommandName,
    FinsEvent,
    FinsEventType,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins.score_sec_ci import FORM_PROFILES
from dayu.fins.storage import FsSourceDocumentRepository
from dayu.services.contracts import FinsSubmitRequest
from dayu.services.protocols import FinsServiceProtocol
from dayu.web.streamlit.components.filing_download_progress import (
    DOWNLOAD_STATUS_COMPLETED,
    DOWNLOAD_STATUS_FAILED,
    DownloadTaskState,
    add_active_download,
    build_download_logs_html,
    get_ticker_active_download,
    init_download_state,
    mark_download_completed,
    render_download_progress_area,
    update_download_progress,
)
from dayu.web.streamlit.components.watchlist import WatchlistItem

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


def _get_filing_list(workspace_root: Path, ticker: str) -> list[_FilingInfo]:
    """获取指定股票的已下载财报列表。

    参数:
        workspace_root: 工作区根目录。
        ticker: 股票代码。

    返回值:
        财报文件信息列表。
    """

    try:
        # 使用 FsSourceDocumentRepository 获取文档列表
        source_repo = FsSourceDocumentRepository(
            workspace_root,
            create_directories=False,
        )

        # 获取 filing 文档 ID 列表
        document_ids = source_repo.list_source_document_ids(ticker, SourceKind.FILING)

        filings = []
        for doc_id in document_ids:
            try:
                # 读取文档元数据
                meta = source_repo.get_source_meta(ticker, doc_id, SourceKind.FILING)
                file_name, file_path = _resolve_primary_file_display(
                    source_repo=source_repo,
                    workspace_root=workspace_root,
                    ticker=ticker,
                    document_id=doc_id,
                )

                # 提取关键信息
                filing_info: _FilingInfo = {
                    "document_id": doc_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "form_type": meta.get("form_type", "未知"),
                    "filing_date": meta.get("filing_date", "未知"),
                    "report_date": meta.get("report_date", "未知"),
                    "fiscal_year": meta.get("fiscal_year", "未知"),
                    "fiscal_period": meta.get("fiscal_period", "未知"),
                    "status": "可用" if not meta.get("is_deleted", False) else "已删除",
                }
                filings.append(filing_info)
            except (OSError, ValueError, KeyError):
                # 读取单个文档失败时跳过，不阻塞整个列表
                continue

        # 按申报日期排序（最新的在前）
        filings.sort(key=lambda x: x.get("filing_date", ""), reverse=True)
        return filings

    except (OSError, ValueError) as e:
        st.error("读取财报列表失败，请确认工作区路径是否正确")
        return []


def _resolve_primary_file_display(
    source_repo: FsSourceDocumentRepository,
    workspace_root: Path,
    ticker: str,
    document_id: str,
) -> tuple[str, str]:
    """解析源文档主文件的展示名称、路径。

    参数:
        source_repo: 源文档仓储实例。
        workspace_root: 工作区根目录。
        ticker: 股票代码。
        document_id: 文档 ID。

    返回值:
        二元组 `(文件名, 文件路径展示值)`。

    异常:
        无：解析失败时返回“未知”占位，不向调用方抛出异常。
    """

    try:
        primary_source = source_repo.get_primary_source(ticker, document_id, SourceKind.FILING)
        materialized_path = primary_source.materialize().resolve()
        filename = materialized_path.name or "未知"
        try:
            relative_path = materialized_path.relative_to(workspace_root.resolve())
            file_path = str(relative_path)
        except ValueError:
            file_path = str(materialized_path)
        return filename, file_path
    except (OSError, ValueError):
        return "未知", "未知"


def _render_filing_table(filings: list[_FilingInfo]) -> None:
    """渲染财报列表表格。

    参数:
        filings: 财报文件信息列表。

    返回值:
        无。

    异常:
        无。
    """

    # 准备表格数据
    df_data = []
    for f in filings:
        df_data.append({
            "文件名称": f.get("file_name", "未知"),
            "文件路径": f.get("file_path", "未知"),
            "表单类型": f.get("form_type", "未知"),
            "申报日期": f.get("filing_date", "未知"),
            "报告日期": f.get("report_date", "未知"),
            "财年": f.get("fiscal_year", "未知"),
            "财期": f.get("fiscal_period", "未知"),
            "状态": f.get("status", "未知"),
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

    调用方保证 `visible_rows >= 1`；不满足时抛出 ``ValueError``。

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


def _init_download_settings_state(selected_stock: WatchlistItem) -> None:
    """初始化下载设置会话状态。

    参数:
        selected_stock: 当前选中的自选股。

    返回值:
        无。

    异常:
        无。
    """
    if "show_download_settings" not in st.session_state:
        st.session_state.show_download_settings = False
    if "download_settings_ticker" not in st.session_state:
        st.session_state.download_settings_ticker = selected_stock.ticker


def _render_filing_header_actions(selected_stock: WatchlistItem) -> None:
    """渲染财报页头部操作按钮。

    参数:
        selected_stock: 当前选中的自选股。

    返回值:
        无。

    异常:
        无。
    """

    _, spacer_column = st.columns([1, 1])
    button_text = "❌ 关闭下载" if _should_show_download_settings_for_ticker(selected_stock.ticker) else "📥 下载财报"
    with spacer_column:
        if st.button(button_text, width="stretch", type="secondary", key=f"toggle_download_settings_{selected_stock.ticker}"):
            _toggle_download_settings(selected_stock)
            st.rerun()


def _should_show_download_settings_for_ticker(ticker: str) -> bool:
    """判断当前股票是否应展示下载设置区域。

    参数:
        ticker: 股票代码。

    返回值:
        `True` 表示当前页面应展示该股票的下载设置区域，否则返回 `False`。

    异常:
        无。
    """

    return st.session_state.get("show_download_settings", False) and st.session_state.get("download_settings_ticker") == ticker


def _toggle_download_settings(selected_stock: WatchlistItem) -> None:
    """切换下载设置区域的显示/隐藏。

    参数:
        selected_stock: 当前选中的自选股。

    返回值:
        无。

    异常:
        无。
    """
    _init_download_settings_state(selected_stock)
    # 如果当前是隐藏状态，或者切换到不同股票时，显示设置区域
    if (
        not st.session_state.show_download_settings
        or st.session_state.download_settings_ticker != selected_stock.ticker
    ):
        st.session_state.show_download_settings = True
        st.session_state.download_settings_ticker = selected_stock.ticker
    else:
        st.session_state.show_download_settings = False


def _render_download_settings(
    selected_stock: WatchlistItem,
    fins_service: FinsServiceProtocol | None,
) -> None:
    """在当前页面渲染下载任务设置区域。

    使用 expander 展开/收起下载设置表单，提交后实时展示下载进度。

    参数:
        selected_stock: 当前选中的自选股。
        fins_service: 财报服务实例；为 None 时仅显示提示信息。

    返回值:
        无。

    异常:
        无。
    """
    _init_download_settings_state(selected_stock)
    ticker = selected_stock.ticker

    # 检查是否有正在进行的下载任务
    existing_task = get_ticker_active_download(ticker)
    if existing_task:
        # 如果有正在进行的任务，显示提示，不提供新的下载设置
        st.info(f"📥 已有正在进行的下载任务（会话 ID: {existing_task.session_id}），请等待完成")
        return

    if fins_service is None:
        st.warning("财报服务不可用，无法进行下载操作")
        return

    with st.container():
        st.markdown("**📥 下载财报设置**")

        form_types = st.multiselect(
            "选择要下载的财报表单类型",
            options=FORM_PROFILES.keys(),
            default=["10-K", "10-Q"],
            help="选择需要下载的 SEC 表单类型",
            key=f"download_form_types_{ticker}",
        )

        today = datetime.date.today()
        three_years_ago = today.replace(year=today.year - 3)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=three_years_ago,
                help="可选，默认三年前，留空表示不限制开始日期",
                key=f"download_start_date_{ticker}",
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=today,
                help="可选，默认今天，留空表示不限制结束日期",
                key=f"download_end_date_{ticker}",
            )

        overwrite = st.checkbox(
            "覆盖已有文件",
            value=False,
            help="如果文件已存在，是否重新下载",
            key=f"download_overwrite_{ticker}",
        )

        if st.button("开始下载", width="stretch", type="primary", key=f"download_start_btn_{ticker}"):
            if not form_types:
                st.error("请至少选择一种表单类型")
            else:
                start_date_str = start_date.isoformat() if start_date else None
                end_date_str = end_date.isoformat() if end_date else None

                submission = None
                try:
                    submission = fins_service.submit(
                        FinsSubmitRequest(
                            command=FinsCommand(
                                name=FinsCommandName.DOWNLOAD,
                                payload=DownloadCommandPayload(
                                    ticker=ticker,
                                    form_type=tuple(form_types),
                                    start_date=start_date_str,
                                    end_date=end_date_str,
                                    overwrite=overwrite,
                                ),
                                stream=True,
                            ),
                        ),
                    )

                    # 添加到活跃下载任务，并隐藏下载设置
                    add_active_download(submission.session_id, ticker)

                    # 创建状态容器用于实时更新
                    status_container = st.status("开始下载财报...", expanded=True)
                    progress_bar = status_container.progress(0.0)
                    status_logs_placeholder = status_container.empty()

                    task_data = st.session_state.active_downloads.get(submission.session_id, {})
                    current_task = DownloadTaskState.from_dict(task_data)
                    status_logs_placeholder.markdown(
                        build_download_logs_html(current_task.logs),
                        unsafe_allow_html=True,
                    )

                    # 消费流式事件
                    async def consume_stream():
                        execution = submission.execution
                        if not isinstance(execution, AsyncIterator):
                            # 同步结果，直接标记完成
                            mark_download_completed(
                                submission.session_id,
                                success=True,
                                message="下载完成（同步模式）",
                            )
                            return

                        async for event in execution:
                            if not isinstance(event, FinsEvent):
                                continue

                            if event.type == FinsEventType.PROGRESS:
                                payload = event.payload
                                if isinstance(payload, DownloadProgressPayload):
                                    # 更新任务状态
                                    update_download_progress(submission.session_id, payload)
                                    # 获取最新状态
                                    task_data = st.session_state.active_downloads.get(
                                        submission.session_id, {}
                                    )
                                    current_task = DownloadTaskState.from_dict(task_data)

                                    # 更新 UI
                                    progress_bar.progress(current_task.progress / 100.0)
                                    status_logs_placeholder.markdown(
                                        build_download_logs_html(current_task.logs),
                                        unsafe_allow_html=True,
                                    )

                            elif event.type == FinsEventType.RESULT:
                                # 最终结果，任务完成
                                mark_download_completed(
                                    submission.session_id, success=True, message="下载完成"
                                )
                                break

                    # 安全运行异步协程，兼容已有事件循环的运行环境
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        asyncio.run(consume_stream())
                    else:
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            pool.submit(asyncio.run, consume_stream()).result()

                    # 获取最终状态
                    final_task_data = st.session_state.active_downloads.get(
                        submission.session_id, {}
                    )
                    final_task = DownloadTaskState.from_dict(final_task_data)
                    status_logs_placeholder.markdown(
                        build_download_logs_html(final_task.logs),
                        unsafe_allow_html=True,
                    )

                    # 更新最终状态
                    if final_task.status == DOWNLOAD_STATUS_COMPLETED:
                        status_container.update(
                            label="✅ 下载完成！", state="complete", expanded=True
                        )
                        st.success(f"已成功下载 {final_task.downloaded_filing_count} 个财报，{final_task.downloaded_count} 个文件")

                        # 显示错误汇总（如果有）
                        if final_task.errors:
                            st.warning(f"下载过程中出现 {len(final_task.errors)} 个错误")

                    elif final_task.status == DOWNLOAD_STATUS_FAILED:
                        status_container.update(label="❌ 下载失败", state="error", expanded=True)
                        if final_task.errors:
                            for error in final_task.errors:
                                st.error(error)

                except Exception:
                    st.error("下载任务失败，请稍后重试")
                    if submission is not None:
                        mark_download_completed(
                            submission.session_id, success=False, message="下载任务执行异常"
                        )

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

    # 初始化下载状态
    init_download_state()
    _init_download_settings_state(selected_stock)

    title_column, actions_column = st.columns([4, 1], gap="small", vertical_alignment="center")
    with title_column:
        st.subheader(f"{selected_stock.company_name} ({selected_stock.ticker}) - 财报管理")
    with actions_column:
        if fins_service is not None:
            _render_filing_header_actions(selected_stock)

    # 展示活跃下载任务进度（如果有）
    render_download_progress_area(selected_stock.ticker)

    # 下载设置区域（展开/收起）
    if _should_show_download_settings_for_ticker(selected_stock.ticker):
        _render_download_settings(selected_stock, fins_service)

    # 获取已下载财报列表
    filings = _get_filing_list(workspace_root, selected_stock.ticker)

    st.markdown("---")

    # 展示财报列表
    if filings:
        _render_filing_table(filings)
    else:
        if not _should_show_download_settings_for_ticker(selected_stock.ticker):
            
            st.info("暂无财报，请点击「下载财报」按钮获取")