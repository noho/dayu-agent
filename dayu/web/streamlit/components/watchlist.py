"""Streamlit 自选股管理对话框组件。

在表格内完成自选股的添加、删除、编辑，保存后写入本地 JSON。
存储路径：workspace/.dayu/streamlit/watchlist.json。

支持从工作区 ``portfolio/`` 目录扫描公司文件夹，经 ``dayu.fins.storage``
盘点元数据后合并进自选股列表（仅新增缺失条目，并可在存在有效
``meta.json`` 时更新公司名称）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from dayu.fins.domain.document_models import CompanyMetaInventoryEntry
from dayu.fins.storage.fs_company_meta_repository import FsCompanyMetaRepository
from dayu.fins.ticker_normalization import try_normalize_ticker

_PORTFOLIO_SYNC_INVALID_META_NAME_SUFFIX = "（公司元数据无效）"


@dataclass(frozen=True)
class _PortfolioWatchlistCandidate:
    """从 portfolio 盘点得到的单条自选股候选。

    属性:
        ticker: 写入自选股的股票代码（优先取公司 meta 中的规范 ticker）。
        company_name: 展示用公司名称。
        official: 是否来自有效 ``meta.json``（为 True 时可覆盖已有条目的名称）。
    """

    ticker: str
    company_name: str
    official: bool


@dataclass(frozen=True)
class WatchlistItem:
    """自选股条目（Streamlit内部使用）。

    属性:
        ticker: 股票代码，如 AAPL。
        company_name: 公司名称，如 苹果。
        created_at: 创建时间 ISO8601 格式。
        updated_at: 更新时间 ISO8601 格式。
    """

    ticker: str
    company_name: str
    created_at: str
    updated_at: str


def _watchlist_storage_path(workspace_root: Path) -> Path:
    """返回自选股持久化文件路径。

    参数:
        workspace_root: 工作区根目录。

    返回值:
        自选股 JSON 文件绝对路径。

    异常:
        无。
    """

    return workspace_root / ".dayu" / "streamlit" / "watchlist.json"


def load_watchlist_items(workspace_root: Path) -> list[WatchlistItem]:
    """从本地 JSON 文件加载自选股列表。

    参数:
        workspace_root: 工作区根目录。

    返回值:
        自选股条目列表；文件不存在时返回空列表。

    异常:
        json.JSONDecodeError: JSON 格式损坏时抛出（与文件不存在区分）。
        KeyError: JSON 结构缺少必要字段时抛出。
    """

    storage_path = _watchlist_storage_path(workspace_root)
    if not storage_path.exists():
        return []

    with open(storage_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: list[WatchlistItem] = []
    for item_data in data.get("items", []):
        items.append(
            WatchlistItem(
                ticker=str(item_data["ticker"]),
                company_name=str(item_data["company_name"]),
                created_at=str(item_data["created_at"]),
                updated_at=str(item_data["updated_at"]),
            )
        )
    return items


def save_watchlist_items(workspace_root: Path, items: list[WatchlistItem]) -> None:
    """将自选股列表写入本地 JSON 文件。

    参数:
        workspace_root: 工作区根目录。
        items: 要持久化的条目列表。

    异常:
        OSError: 无法创建目录或写入文件时抛出。
    """

    storage_path = _watchlist_storage_path(workspace_root)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "items": [
            {
                "ticker": item.ticker,
                "company_name": item.company_name,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in items
        ],
        "version": "1.0",
    }
    with open(storage_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _merge_key_for_watchlist(ticker: str) -> str:
    """将 ticker 转为自选股合并用键（归一化后大写，便于跨写法去重）。

    参数:
        ticker: 原始股票代码或目录名。

    返回值:
        非空合并键；输入全空白时返回空串。

    异常:
        无。
    """

    cleaned = ticker.strip()
    if not cleaned:
        return ""
    normalized = try_normalize_ticker(cleaned)
    if normalized is not None:
        return normalized.canonical.upper()
    return cleaned.upper()


def _apply_inventory_entry_to_discovery(
    discovery: dict[str, _PortfolioWatchlistCandidate],
    entry: CompanyMetaInventoryEntry,
) -> None:
    """将单条盘点结果写入 discovery 映射（官方 meta 优先于缺失/无效目录）。

    参数:
        discovery: 合并键到候选条目的可变映射。
        entry: 仓储返回的单条盘点记录。

    返回值:
        无。

    异常:
        无。
    """

    if entry.status == "hidden_directory":
        return
    if entry.status == "available" and entry.company_meta is not None:
        meta = entry.company_meta
        ticker = meta.ticker.strip()
        if not ticker:
            return
        merge_key = _merge_key_for_watchlist(ticker)
        if not merge_key:
            return
        name = meta.company_name.strip()
        discovery[merge_key] = _PortfolioWatchlistCandidate(
            ticker=ticker,
            company_name=name or ticker,
            official=True,
        )
        return

    raw_dir = entry.directory_name.strip()
    if not raw_dir:
        return
    normalized = try_normalize_ticker(raw_dir)
    ticker_raw = normalized.canonical if normalized is not None else raw_dir.strip()
    merge_key = _merge_key_for_watchlist(ticker_raw)
    if not merge_key:
        return
    existing = discovery.get(merge_key)
    if existing is not None and existing.official:
        return
    if entry.status == "missing_meta":
        discovery[merge_key] = _PortfolioWatchlistCandidate(
            ticker=ticker_raw,
            company_name=raw_dir,
            official=False,
        )
        return
    if entry.status == "invalid_meta":
        discovery[merge_key] = _PortfolioWatchlistCandidate(
            ticker=ticker_raw,
            company_name=f"{raw_dir}{_PORTFOLIO_SYNC_INVALID_META_NAME_SUFFIX}",
            official=False,
        )


def merge_watchlist_from_portfolio(workspace_root: Path) -> tuple[bool, str]:
    """从 ``workspace/portfolio`` 扫描公司目录并合并写入自选股 JSON。

    使用 ``FsCompanyMetaRepository.scan_company_meta_inventory()`` 枚举目录与
    公司级 ``meta.json``，不直接拼接路径盲读文件。

    合并规则:
    - 保留原自选股顺序；若盘点到有效 meta 且规范 ticker 或公司名称变化，则更新该条并刷新 ``updated_at``。
    - 盘点中存在而当前列表无相同合并键的条目，在末尾按合并键字典序追加。
    - 无有效 meta 的目录仍可按文件夹名加入列表（名称可能为目录名或带无效后缀）。

    参数:
        workspace_root: 工作区根目录。

    返回值:
        ``(True, 说明文案)``；当前实现不因业务条件返回 ``False``，预留与 UI 一致接口。

    异常:
        OSError: 仓储初始化或扫描失败时抛出。
        json.JSONDecodeError: 自选股 JSON 损坏时由 ``load_watchlist_items`` 抛出。
    """

    workspace = workspace_root.resolve()
    repo = FsCompanyMetaRepository(workspace)
    discovery: dict[str, _PortfolioWatchlistCandidate] = {}
    for inv_entry in repo.scan_company_meta_inventory():
        _apply_inventory_entry_to_discovery(discovery, inv_entry)

    previous = load_watchlist_items(workspace)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    prev_by_key: dict[str, WatchlistItem] = {}
    key_order: list[str] = []
    for item in previous:
        merge_key = _merge_key_for_watchlist(item.ticker)
        if not merge_key:
            continue
        if merge_key not in prev_by_key:
            key_order.append(merge_key)
        prev_by_key[merge_key] = item

    merged: list[WatchlistItem] = []
    meta_synced = 0
    for merge_key in key_order:
        item = prev_by_key[merge_key]
        cand = discovery.get(merge_key)
        if cand is None:
            merged.append(item)
            continue
        if cand.official:
            if cand.ticker != item.ticker or cand.company_name != item.company_name:
                meta_synced += 1
                merged.append(
                    WatchlistItem(
                        ticker=cand.ticker,
                        company_name=cand.company_name,
                        created_at=item.created_at,
                        updated_at=now,
                    )
                )
            else:
                merged.append(item)
        else:
            merged.append(item)

    existing_keys = {_merge_key_for_watchlist(w.ticker) for w in merged if _merge_key_for_watchlist(w.ticker)}
    added = 0
    for merge_key in sorted(discovery.keys()):
        if merge_key in existing_keys:
            continue
        cand = discovery[merge_key]
        merged.append(
            WatchlistItem(
                ticker=cand.ticker,
                company_name=cand.company_name,
                created_at=now,
                updated_at=now,
            )
        )
        existing_keys.add(merge_key)
        added += 1

    save_watchlist_items(workspace, merged)
    parts: list[str] = [f"已写入 {len(merged)} 条自选股。"]
    if added:
        parts.append(f"从 portfolio 新增 {added} 条。")
    if meta_synced:
        parts.append(f"按有效 meta 同步 {meta_synced} 条（代码或名称）。")
    if not added and not meta_synced:
        parts.append("与 portfolio 比对：无新增条目，且无待同步的有效 meta 变更。")
    return True, " ".join(parts)


def reconcile_streamlit_selected_ticker_after_watchlist_change(workspace_root: Path) -> None:
    """在自选股文件变更后，将 ``selected_ticker`` 对齐到同合并键下的规范 ticker。

    参数:
        workspace_root: 工作区根目录。

    返回值:
        无。

    异常:
        无：读盘失败时静默跳过。
    """

    raw = st.session_state.get("selected_ticker")
    if not isinstance(raw, str) or not raw.strip():
        return
    try:
        items = load_watchlist_items(workspace_root)
    except Exception:
        return
    sel_key = _merge_key_for_watchlist(raw)
    if not sel_key:
        return
    for it in items:
        if _merge_key_for_watchlist(it.ticker) == sel_key:
            st.session_state["selected_ticker"] = it.ticker
            return


def _is_cell_empty(val: object) -> bool:
    """判断表格单元格是否为空（含 NaN、pd.NA、None）。

    参数:
        val: 单元格原始值。

    返回值:
        视为空则为 True。

    异常:
        无。
    """

    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def _series_cell_str(series_row: pd.Series, key: str) -> str:
    """从 DataFrame 行读取单元格为去空白字符串；空或缺失视为空串。

    参数:
        series_row: 一行数据。
        key: 列名。

    返回值:
        去首尾空白后的字符串，空单元格返回空串。

    异常:
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

    参数:
        items: 自选股条目列表。

    返回值:
        列为「股票代码」「公司名称」的 DataFrame。

    异常:
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

    参数:
        original: 原始 DataFrame（即传给 data_editor 的数据）。
        editor_state: st.session_state[editor_key]，包含 deleted_rows、edited_rows、added_rows。

    返回值:
        处理后的最终 DataFrame。

    异常:
        无。
    """

    # 复制原始数据
    df = original.copy()

    # 处理删除：按索引删除行（从大到小删除避免索引变化）
    deleted_indices = editor_state.get("deleted_rows", [])
    if deleted_indices:
        valid_deleted = {
            int(raw_idx) for raw_idx in deleted_indices if 0 <= int(raw_idx) < len(df)
        }
        if valid_deleted:
            keep_positions = [pos for pos in range(len(df)) if pos not in valid_deleted]
            df = df.iloc[keep_positions]

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

    参数:
        workspace_root: 工作区根目录。
        original: 原始 DataFrame（传给 data_editor 的数据）。
        editor_state: st.session_state[editor_key]，包含删除/编辑/新增信息。
        previous: 保存前的列表，用于保留已有条目的 created_at。

    返回值:
        (是否成功, 说明信息)。

    异常:
        无：校验失败时返回 (False, 错误说明)，不向调用方抛出。
    """

    # 构建最终 DataFrame（应用删除、编辑、新增）
    final_df = _build_final_dataframe(original, editor_state)

    prev_by_ticker = {p.ticker.upper(): p for p in previous}

    rows_out: list[WatchlistItem] = []
    seen_tickers: set[str] = set()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

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


def _trigger_sidebar_refresh() -> None:
    """标记并触发整页重跑，确保左侧自选股导航栏立即刷新。

    参数:
        无。

    返回值:
        无。

    异常:
        无。`st.rerun()` 的控制流异常由 Streamlit 内部处理。
    """

    st.session_state["watchlist_needs_refresh"] = True
    st.rerun()


@st.dialog("自选股管理", width="large")
def render_watchlist_manager(workspace_root: Path) -> None:
    """弹出对话框：在表格内添加、删除、编辑自选股，保存后写回文件。

    参数:
        workspace_root: 工作区根目录。

    异常:
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

    sync_col, _sync_spacer = st.columns([1, 3])
    with sync_col:
        if st.button(
            "从 portfolio 同步",
            key="watchlist_sync_portfolio_btn",
            help="扫描 workspace/portfolio 下公司目录，合并到自选股（保留已有顺序，可写入新代码并同步有效 meta 中的名称）",
        ):
            try:
                _, sync_msg = merge_watchlist_from_portfolio(workspace_root)
                st.session_state["watchlist_last_sync_message"] = sync_msg
                reconcile_streamlit_selected_ticker_after_watchlist_change(workspace_root)
                if editor_key in st.session_state:
                    del st.session_state[editor_key]
                st.session_state["watchlist_needs_refresh"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"从 portfolio 同步失败: {exc}")

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
            _trigger_sidebar_refresh()
        else:
            st.error(msg)
