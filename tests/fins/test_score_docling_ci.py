"""CN/HK Docling CI scorer 单元测试。"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest

from dayu.fins.domain.document_models import (
    ProcessedCreateRequest,
    ProcessedHandle,
    SourceDocumentUpsertRequest,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins.score_docling_ci import (
    JsonObject,
    PROFILES,
    REPORT_KIND_ANNUAL,
    REPORT_KIND_QUARTERLY,
    REPORT_KIND_SEMIANNUAL,
    ScoreConfig,
    _discover_docling_snapshots,
    _matched_group_labels,
    main,
    parse_args,
    score_batch,
    score_document,
)
from tests.fins.storage_testkit import FsStorageTestContext, build_fs_storage_test_context


def _write_snapshot_json(
    context: FsStorageTestContext,
    *,
    handle: ProcessedHandle,
    file_name: str,
    payload: JsonObject,
) -> None:
    """通过 blob 仓储写入 snapshot JSON。

    Args:
        context: 测试仓储上下文。
        handle: processed 句柄。
        file_name: 文件名。
        payload: JSON payload。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    context.blob_repository.store_file(
        handle,
        file_name,
        BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
        content_type="application/json",
    )


def _prepare_source_and_processed(
    context: FsStorageTestContext,
    *,
    ticker: str,
    document_id: str,
    source_kind: SourceKind,
    form_type: str,
    create_processed: bool = True,
    ingest_complete: bool = True,
    is_deleted: bool = False,
) -> ProcessedHandle:
    """创建 source 文档，并按需创建 processed manifest。

    Args:
        context: 测试仓储上下文。
        ticker: 股票代码。
        document_id: 文档 ID。
        source_kind: 来源类型。
        form_type: 表单类型。
        create_processed: 是否创建 processed 记录。
        ingest_complete: 是否完成入库。
        is_deleted: 是否逻辑删除。

    Returns:
        processed 句柄。

    Raises:
        OSError: 仓储写入失败时抛出。
    """

    context.source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=document_id,
            form_type=form_type,
            primary_document=f"{document_id}_docling.json",
            meta={
                "ticker": ticker,
                "form_type": form_type,
                "fiscal_period": form_type if source_kind == SourceKind.FILING else None,
                "report_kind": form_type if source_kind == SourceKind.FILING else "material",
                "ingest_complete": ingest_complete,
                "is_deleted": is_deleted,
                "document_version": "v1",
                "source_fingerprint": f"{document_id}_fingerprint",
            },
        ),
        source_kind,
    )
    if create_processed:
        context.processed_repository.create_processed(
            ProcessedCreateRequest(
                ticker=ticker,
                document_id=document_id,
                internal_document_id=document_id,
                source_kind=source_kind.value,
                form_type=form_type,
                meta={
                    "form_type": form_type,
                    "source_kind": source_kind.value,
                    "is_deleted": False,
                },
                sections=[],
                tables=[],
                financials=None,
            )
        )
        return context.processed_repository.get_processed_handle(ticker, document_id)
    return ProcessedHandle(ticker=ticker, document_id=document_id)


def _long_context(seed: str) -> str:
    """构造搜索证据上下文。

    Args:
        seed: 关键词。

    Returns:
        足够长的中文上下文。

    Raises:
        无。
    """

    return f"{seed} 是本报告中的核心披露内容，包含经营变化、财务表现、风险因素和管理层解释。" * 5


def _annual_sections() -> list[JsonObject]:
    """构造年报章节列表。"""

    titles = [
        "公司简介",
        "主营业务",
        "財務回顧",
        "主要会计数据和财务指标",
        "公司治理",
        "股东信息",
        "重大事项与风险提示",
        "關鍵審計事項",
        "财务报表",
        "资产负债表",
        "利润表",
        "現⾦流量表",
        "附註",
        "董事会报告",
        "关联交易",
        "内部控制",
        "募集资金",
        "环保信息",
        "社会责任",
        "释义",
    ]
    return [
        {
            "ref": f"s_{index:04d}",
            "title": title,
            "level": 1,
            "parent_ref": None,
            "page_range": [index, index],
        }
        for index, title in enumerate(titles, start=1)
    ]


@pytest.mark.unit
def test_quarterly_key_financials_profile_matches_hk_headings() -> None:
    """季度 profile 应识别港股关键财务数据章节标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    profile = PROFILES[REPORT_KIND_QUARTERLY]
    text = "主要財務衡量指標\n財務表現摘要\n關鍵數據摘要"

    assert "key_financials" in _matched_group_labels(text, profile.key_groups)


@pytest.mark.unit
def test_semiannual_key_financials_profile_matches_hk_headings() -> None:
    """中报 profile 应识别港股财务摘要类章节标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    profile = PROFILES[REPORT_KIND_SEMIANNUAL]
    text = "財務摘要\n財務概要\n財務表現摘要\n業績摘要\n2025年上半年財務表現指標"

    assert "key_financials" in _matched_group_labels(text, profile.key_groups)


@pytest.mark.unit
def test_semiannual_mda_profile_matches_hk_review_headings() -> None:
    """中报 profile 应识别港股业务与财务回顾类章节标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    profile = PROFILES[REPORT_KIND_SEMIANNUAL]
    text = "財務表現\n業務表現\n集團回顧\n財務及營運回顧\n業績綜述"

    assert "mda" in _matched_group_labels(text, profile.key_groups)


@pytest.mark.unit
def test_annual_financial_profile_matches_hk_loss_statement_headings() -> None:
    """年报 profile 应识别港股损益与亏损表标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    profile = PROFILES[REPORT_KIND_ANNUAL]
    text = "Consolidated Income Statement 合併利潤表\n合併綜合虧損表\n綜合收益表"

    assert "income_statement" in _matched_group_labels(text, profile.financial_groups)


@pytest.mark.unit
def test_quarterly_financial_profile_matches_operating_statement_headings() -> None:
    """季报 financial profile 应识别港股经营状况表标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    profile = PROFILES[REPORT_KIND_QUARTERLY]
    text = "未經審計中期簡明合併經營狀況表"

    assert "income_statement" in _matched_group_labels(text, profile.financial_groups)


@pytest.mark.unit
def test_financial_profile_does_not_match_statement_from_line_items_only() -> None:
    """financial profile 不应仅凭财报行项目判定三大报表存在。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    profile = PROFILES[REPORT_KIND_QUARTERLY]
    text = "淨利息收入\n手續費及佣金收入\n一、营业收入\n一、经营活动产生的现金流量"

    labels = _matched_group_labels(text, profile.financial_groups)

    assert "income_statement" not in labels
    assert "cash_flow" not in labels


@pytest.mark.unit
def test_annual_financial_profile_matches_hk_cash_flow_statement_headings() -> None:
    """年报 profile 应识别港股現金流動表标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    profile = PROFILES[REPORT_KIND_ANNUAL]
    text = "綜合現金流動表\n主要業務活動之現金流量"

    assert "cash_flow" in _matched_group_labels(text, profile.financial_groups)


@pytest.mark.unit
def test_annual_profile_matches_cn_audit_and_accounting_notes_headings() -> None:
    """年报 profile 应识别 A 股审计报告与会计报表注释标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    profile = PROFILES[REPORT_KIND_ANNUAL]
    text = "已审财务报表\n审计 意 见\n关 键 审计事项\n注册会计师对财务报表审计的责任\n七 会计报表主要项目注释"
    labels = _matched_group_labels(text, profile.key_groups)

    assert "audit" in labels
    assert "notes" in labels


@pytest.mark.unit
def test_annual_profile_matches_hk_accounting_notes_headings() -> None:
    """年报 profile 应识别港股会计报表注释标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    profile = PROFILES[REPORT_KIND_ANNUAL]
    text = "合併會計報表註釋\n五 合併會計報表主要項目註釋"

    assert "notes" in _matched_group_labels(text, profile.key_groups)


def _read_section_calls(sections: list[JsonObject], *, dangling_placeholder: bool = False) -> list[JsonObject]:
    """构造 read_section calls。

    Args:
        sections: 章节列表。
        dangling_placeholder: 是否写入悬挂表格占位符。

    Returns:
        read_section calls。

    Raises:
        无。
    """

    calls: list[JsonObject] = []
    for section in sections:
        ref = str(section["ref"])
        title = str(section["title"])
        placeholder = " [[t_9999]]" if dangling_placeholder and ref == "s_0009" else ""
        if ref == "s_0009" and not dangling_placeholder:
            placeholder = " [[t_0001]] [[t_0002]] [[t_0003]]"
        calls.append(
            {
                "request": {"ref": ref},
                "response": {
                    "ref": ref,
                    "title": title,
                    "content": f"{title}。本节披露主营业务、营业收入、净利润、现金流和风险提示。{placeholder}" * 20,
                },
            }
        )
    return calls


def _search_calls() -> list[JsonObject]:
    """构造 search_document calls。"""

    queries = ("主营业务", "营业收入", "净利润", "现金流")
    return [
        {
            "request": {
                "query": query,
                "query_id": f"annual_quarter_core40.q{index:03d}",
                "query_text": query,
                "query_intent": query,
                "query_weight": 1.0,
            },
            "response": {
                "matches": [
                    {
                        "section": {"ref": "s_0002", "title": "主营业务"},
                        "evidence": {"context": _long_context(query), "matched_text": query},
                        "is_exact_phrase": True,
                    }
                ],
                "total_matches": 1,
                "diagnostics": {"strategy_hit_counts": {"exact": 1}},
            },
        }
        for index, query in enumerate(queries, start=1)
    ]


def _list_tables_payload() -> JsonObject:
    """构造 list_tables payload。"""

    tables = [
        {
            "table_ref": "t_0001",
            "caption": "合并资产负债表",
            "row_count": 8,
            "col_count": 3,
            "headers": ["项目", "2025年", "2024年"],
            "within_section": {"ref": "s_0010", "title": "资产负债表"},
            "page_no": 10,
            "is_financial": True,
            "table_type": "financial",
        },
        {
            "table_ref": "t_0002",
            "caption": "合并利润表",
            "row_count": 8,
            "col_count": 3,
            "headers": ["项目", "2025年", "2024年"],
            "within_section": {"ref": "s_0011", "title": "利润表"},
            "page_no": 11,
            "is_financial": True,
            "table_type": "financial",
        },
        {
            "table_ref": "t_0003",
            "caption": "合并现金流量表",
            "row_count": 8,
            "col_count": 3,
            "headers": ["项目", "2025年", "2024年"],
            "within_section": {"ref": "s_0012", "title": "现金流量表"},
            "page_no": 12,
            "is_financial": True,
            "table_type": "financial",
        },
    ]
    return {"calls": [{"response": {"tables": tables, "total": len(tables), "financial_count": len(tables)}}]}


def _get_table_payload() -> JsonObject:
    """构造 get_table payload。"""

    calls: list[JsonObject] = []
    captions = {
        "t_0001": ("合并资产负债表", "s_0010"),
        "t_0002": ("合并利润表", "s_0011"),
        "t_0003": ("合并现金流量表", "s_0012"),
    }
    for table_ref, (caption, section_ref) in captions.items():
        calls.append(
            {
                "request": {"table_ref": table_ref},
                "response": {
                    "table_ref": table_ref,
                    "caption": caption,
                    "row_count": 8,
                    "col_count": 3,
                    "headers": ["项目", "2025年", "2024年"],
                    "within_section": {"ref": section_ref, "title": caption},
                    "page_no": int(table_ref[-1]) + 9,
                    "is_financial": True,
                    "table_type": "financial",
                    "data": {
                        "kind": "markdown",
                        "markdown": "| 项目 | 2025年 | 2024年 |\n|---|---:|---:|\n| 营业收入 | 100 | 90 |",
                    },
                },
            }
        )
    return {"calls": calls}


def _page_payload() -> JsonObject:
    """构造 get_page_content payload。"""

    return {
        "calls": [
            {
                "request": {"page_no": page_no},
                "response": {
                    "page_no": page_no,
                    "supported": True,
                    "has_content": True,
                    "sections": [{"ref": f"s_{page_no:04d}"}],
                    "tables": [],
                    "text_preview": "页面内容",
                },
            }
            for page_no in range(1, 21)
        ]
    }


def _meta_payload(
    *,
    ticker: str,
    document_id: str,
    source_kind: SourceKind,
    form_type: str,
    document_type: str,
) -> JsonObject:
    """构造 tool_snapshot_meta payload。"""

    search_queries = ["主营业务", "营业收入", "净利润", "现金流"]
    return {
        "snapshot_type": "snapshot_meta",
        "schema_version": "tool_snapshot_v1.0.0",
        "snapshot_schema_version": "tool_snapshot_v1.0.0",
        "mode": "ci",
        "ticker": ticker,
        "document_id": document_id,
        "source_kind": source_kind.value,
        "market": "CN",
        "form_type": form_type,
        "document_type": document_type,
        "search_queries": list(search_queries),
        "search_query_pack_name": "annual_quarter_core40",
        "search_query_pack_version": "search_query_pack_v1.0.0",
        "search_query_count": len(search_queries),
        "tools": [
            "get_document_sections",
            "read_section",
            "list_tables",
            "get_table",
            "get_page_content",
            "search_document",
        ],
        "written_files": [],
    }


def _write_annual_snapshot(
    context: FsStorageTestContext,
    *,
    ticker: str,
    document_id: str,
    handle: ProcessedHandle,
    dangling_placeholder: bool = False,
    omit_files: tuple[str, ...] = (),
    search_call_limit: int | None = None,
) -> None:
    """写入完整 annual snapshot。"""

    sections = _annual_sections()
    search_calls = _search_calls()
    if search_call_limit is not None:
        search_calls = search_calls[:search_call_limit]
    payloads: dict[str, JsonObject] = {
        "tool_snapshot_meta.json": _meta_payload(
            ticker=ticker,
            document_id=document_id,
            source_kind=SourceKind.FILING,
            form_type="FY",
            document_type="annual_report",
        ),
        "tool_snapshot_get_document_sections.json": {"calls": [{"response": {"sections": sections}}]},
        "tool_snapshot_read_section.json": {"calls": _read_section_calls(sections, dangling_placeholder=dangling_placeholder)},
        "tool_snapshot_search_document.json": {"calls": search_calls},
        "tool_snapshot_list_tables.json": _list_tables_payload(),
        "tool_snapshot_get_table.json": _get_table_payload(),
        "tool_snapshot_get_page_content.json": _page_payload(),
    }
    for file_name, payload in payloads.items():
        if file_name in omit_files:
            continue
        _write_snapshot_json(context, handle=handle, file_name=file_name, payload=payload)


def _write_material_snapshot(
    context: FsStorageTestContext,
    *,
    ticker: str,
    document_id: str,
    handle: ProcessedHandle,
) -> None:
    """写入 material snapshot。"""

    sections: list[JsonObject] = [
        {"ref": "s_0001", "title": "重大交易公告", "level": 1, "parent_ref": None, "page_range": [1, 1]},
        {"ref": "s_0002", "title": "影响和风险提示", "level": 1, "parent_ref": None, "page_range": [2, 2]},
    ]
    payloads: dict[str, JsonObject] = {
        "tool_snapshot_meta.json": _meta_payload(
            ticker=ticker,
            document_id=document_id,
            source_kind=SourceKind.MATERIAL,
            form_type="MATERIAL_OTHER",
            document_type="material",
        ),
        "tool_snapshot_get_document_sections.json": {"calls": [{"response": {"sections": sections}}]},
        "tool_snapshot_read_section.json": {
            "calls": [
                {
                    "request": {"ref": "s_0001"},
                    "response": {"ref": "s_0001", "title": "重大交易公告", "content": _long_context("公告 事项 交易 影响 风险")},
                },
                {
                    "request": {"ref": "s_0002"},
                    "response": {"ref": "s_0002", "title": "影响和风险提示", "content": _long_context("董事会 风险提示")},
                },
            ]
        },
        "tool_snapshot_search_document.json": {"calls": _search_calls()},
        "tool_snapshot_list_tables.json": {"calls": [{"response": {"tables": [], "total": 0, "financial_count": 0}}]},
        "tool_snapshot_get_table.json": {"calls": []},
        "tool_snapshot_get_page_content.json": _page_payload(),
    }
    for file_name, payload in payloads.items():
        _write_snapshot_json(context, handle=handle, file_name=file_name, payload=payload)


@pytest.mark.unit
def test_discover_docling_snapshots_uses_source_and_processed_manifests(tmp_path: Path) -> None:
    """样本发现应通过 source/processed 仓储对齐 active filing/material。"""

    context = build_fs_storage_test_context(tmp_path)
    _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="fil_missing_processed",
        source_kind=SourceKind.FILING,
        form_type="FY",
        create_processed=False,
    )
    handle = _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="mat_001",
        source_kind=SourceKind.MATERIAL,
        form_type="MATERIAL_OTHER",
        create_processed=True,
    )
    _write_material_snapshot(context, ticker="000001", document_id="mat_001", handle=handle)

    result = _discover_docling_snapshots(
        base=str(tmp_path),
        tickers=["000001"],
        report_kind_filter="all",
        source_kind_filter="all",
    )

    assert [snapshot.document_id for snapshot in result.snapshots] == ["mat_001"]
    assert len(result.completeness_failures) == 1
    assert result.completeness_failures[0].document_id == "fil_missing_processed"
    assert result.completeness_failures[0].reason == "缺少 processed 快照"


@pytest.mark.unit
def test_discover_docling_snapshots_skips_inactive_and_deleted_sources(tmp_path: Path) -> None:
    """样本发现应排除未完成入库和逻辑删除的 source。"""

    context = build_fs_storage_test_context(tmp_path)
    _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="fil_inactive",
        source_kind=SourceKind.FILING,
        form_type="FY",
        ingest_complete=False,
    )
    _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="mat_deleted",
        source_kind=SourceKind.MATERIAL,
        form_type="MATERIAL_OTHER",
        is_deleted=True,
    )

    result = _discover_docling_snapshots(
        base=str(tmp_path),
        tickers=["000001"],
        report_kind_filter="all",
        source_kind_filter="all",
    )

    assert result.snapshots == []
    assert result.completeness_failures == []


@pytest.mark.unit
def test_discover_docling_snapshots_resolves_report_kind_filters(tmp_path: Path) -> None:
    """样本发现应识别 annual、semiannual、quarterly 与 material。"""

    context = build_fs_storage_test_context(tmp_path)
    for document_id, form_type, source_kind in (
        ("fil_annual", "FY", SourceKind.FILING),
        ("fil_semi", "H1", SourceKind.FILING),
        ("fil_quarter", "Q3", SourceKind.FILING),
        ("mat_notice", "MATERIAL_OTHER", SourceKind.MATERIAL),
    ):
        _prepare_source_and_processed(
            context,
            ticker="000001",
            document_id=document_id,
            source_kind=source_kind,
            form_type=form_type,
        )

    all_result = _discover_docling_snapshots(
        base=str(tmp_path),
        tickers=["000001"],
        report_kind_filter="all",
        source_kind_filter="all",
    )
    quarterly_result = _discover_docling_snapshots(
        base=str(tmp_path),
        tickers=["000001"],
        report_kind_filter="quarterly",
        source_kind_filter="filing",
    )

    assert {snapshot.report_kind for snapshot in all_result.snapshots} == {
        "annual",
        "semiannual",
        "quarterly",
        "material",
    }
    assert [snapshot.document_id for snapshot in quarterly_result.snapshots] == ["fil_quarter"]


@pytest.mark.unit
def test_bad_snapshot_meta_becomes_completeness_failure(tmp_path: Path) -> None:
    """坏 meta 应进入 completeness hard gate，而不是中断整批评分。"""

    context = build_fs_storage_test_context(tmp_path)
    handle = _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="fil_bad_meta",
        source_kind=SourceKind.FILING,
        form_type="FY",
    )
    _write_snapshot_json(
        context,
        handle=handle,
        file_name="tool_snapshot_meta.json",
        payload={"snapshot_schema_version": "tool_snapshot_v1.0.0", "market": "CN"},
    )

    batch = score_batch(base=str(tmp_path), tickers=["000001"], cfg=ScoreConfig(), report_kind="all", source_kind="all")

    assert batch.documents == []
    assert len(batch.completeness_failures) == 1
    assert "source_kind" in batch.completeness_failures[0].reason


@pytest.mark.unit
def test_table_placeholder_dangling_triggers_hard_gate(tmp_path: Path) -> None:
    """read_section.content 中的悬挂表格占位符应触发 hard gate。"""

    context = build_fs_storage_test_context(tmp_path)
    handle = _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="fil_dangling_table",
        source_kind=SourceKind.FILING,
        form_type="FY",
    )
    _write_annual_snapshot(
        context,
        ticker="000001",
        document_id="fil_dangling_table",
        handle=handle,
        dangling_placeholder=True,
    )
    snapshot = _discover_docling_snapshots(
        base=str(tmp_path),
        tickers=["000001"],
        report_kind_filter="annual",
        source_kind_filter="filing",
    ).snapshots[0]

    doc = score_document(snapshot, context.blob_repository, ScoreConfig())

    assert doc.hard_gate.passed is False
    assert any("悬挂 table placeholder" in reason for reason in doc.hard_gate.reasons)
    assert doc.dimensions["D_table"].details["dangling_placeholders"] == ["t_9999"]


@pytest.mark.unit
def test_table_placeholder_valid_reference_passes_for_active_filing(tmp_path: Path) -> None:
    """正常 annual filing 应能解引用正文表格占位符并通过评分。"""

    context = build_fs_storage_test_context(tmp_path)
    handle = _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="fil_annual_ok",
        source_kind=SourceKind.FILING,
        form_type="FY",
    )
    _write_annual_snapshot(context, ticker="000001", document_id="fil_annual_ok", handle=handle)

    batch = score_batch(base=str(tmp_path), tickers=["000001"], cfg=ScoreConfig(), report_kind="annual", source_kind="filing")

    assert batch.passed is True
    assert len(batch.documents) == 1
    doc = batch.documents[0]
    assert doc.hard_gate.passed is True
    assert doc.dimensions["D_table"].details["dangling_placeholders"] == []
    assert doc.dimensions["D_table"].details["missing_get_table_refs"] == []


@pytest.mark.unit
def test_missing_non_meta_snapshot_only_zeroes_related_dimension(tmp_path: Path) -> None:
    """缺少单个非 meta snapshot 应扣对应维度，不应变成 completeness failure。"""

    context = build_fs_storage_test_context(tmp_path)
    handle = _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="fil_missing_search",
        source_kind=SourceKind.FILING,
        form_type="FY",
    )
    _write_annual_snapshot(
        context,
        ticker="000001",
        document_id="fil_missing_search",
        handle=handle,
        omit_files=("tool_snapshot_search_document.json",),
    )

    batch = score_batch(base=str(tmp_path), tickers=["000001"], cfg=ScoreConfig(), report_kind="annual", source_kind="filing")

    assert batch.completeness_failures == []
    assert len(batch.documents) == 1
    doc = batch.documents[0]
    assert doc.dimensions["C_search"].points == 0.0
    assert doc.dimensions["A_structure"].points > 0.0
    missing_snapshots = doc.dimensions["E_traceability"].details["missing_snapshots"]
    assert isinstance(missing_snapshots, list)
    assert "tool_snapshot_search_document.json" in missing_snapshots


@pytest.mark.unit
def test_search_call_count_mismatch_reduces_search_and_traceability_scores(tmp_path: Path) -> None:
    """search calls 少于 meta query pack 时应在 C/E 维显式扣分。"""

    context = build_fs_storage_test_context(tmp_path)
    handle = _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="fil_partial_search",
        source_kind=SourceKind.FILING,
        form_type="FY",
    )
    _write_annual_snapshot(
        context,
        ticker="000001",
        document_id="fil_partial_search",
        handle=handle,
        search_call_limit=2,
    )
    snapshot = _discover_docling_snapshots(
        base=str(tmp_path),
        tickers=["000001"],
        report_kind_filter="annual",
        source_kind_filter="filing",
    ).snapshots[0]

    doc = score_document(snapshot, context.blob_repository, ScoreConfig())

    assert doc.dimensions["C_search"].details["search_query_count"] == 4
    assert doc.dimensions["C_search"].details["call_count"] == 2
    assert doc.dimensions["C_search"].details["missing_search_call_count"] == 2
    assert doc.dimensions["C_search"].details["search_call_count_matches_meta"] is False
    assert doc.dimensions["E_traceability"].details["search_call_errors"] == ["search_call_count"]


@pytest.mark.unit
def test_score_docling_ci_handles_material_and_writes_reports(tmp_path: Path) -> None:
    """scorer 应支持 material 文档，并输出 JSON/MD 报告。"""

    context = build_fs_storage_test_context(tmp_path)
    handle = _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="mat_report",
        source_kind=SourceKind.MATERIAL,
        form_type="MATERIAL_OTHER",
    )
    _write_material_snapshot(context, ticker="000001", document_id="mat_report", handle=handle)
    output_json = tmp_path / "reports" / "docling.json"
    output_md = tmp_path / "reports" / "docling.md"

    exit_code = main(
        [
            "--base",
            str(tmp_path),
            "--tickers",
            "000001",
            "--source-kind",
            "material",
            "--report-kind",
            "material",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--min-doc-pass",
            "1",
            "--min-doc-warn",
            "1",
            "--min-batch-avg",
            "1",
            "--min-batch-p10",
            "1",
        ]
    )

    assert exit_code == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["profile_id"] == "cn_hk_docling_v1"
    assert report["summary"]["document_count"] == 1
    assert report["documents"][0]["source_kind"] == "material"
    assert report["documents"][0]["report_kind"] == "material"
    assert output_md.read_text(encoding="utf-8").startswith("# CN/HK Docling CI 评分报告")


@pytest.mark.unit
def test_plain_material_without_tables_does_not_fail_default_thresholds(tmp_path: Path) -> None:
    """普通材料无表格时不应被事实上的财务表要求拖垮。"""

    context = build_fs_storage_test_context(tmp_path)
    handle = _prepare_source_and_processed(
        context,
        ticker="000001",
        document_id="mat_plain_notice",
        source_kind=SourceKind.MATERIAL,
        form_type="MATERIAL_OTHER",
    )
    _write_material_snapshot(context, ticker="000001", document_id="mat_plain_notice", handle=handle)

    batch = score_batch(base=str(tmp_path), tickers=["000001"], cfg=ScoreConfig(), report_kind="material", source_kind="material")

    assert batch.passed is True
    assert batch.documents[0].total_score >= 85.0
    assert batch.documents[0].dimensions["D_table"].points == 20.0
    assert batch.documents[0].dimensions["D_table"].details["financial_table_requirement"] == "not_applicable"


@pytest.mark.unit
def test_parse_args_accepts_basic_cli_path() -> None:
    """CLI 参数解析应保留最小 runner 接入路径。"""

    args = parse_args(
        [
            "--base",
            "workspace",
            "--tickers",
            "000001,00700",
            "--source-kind",
            "material",
            "--report-kind",
            "material",
        ]
    )

    assert args.base == "workspace"
    assert args.tickers == "000001,00700"
    assert args.source_kind == "material"
    assert args.report_kind == "material"
