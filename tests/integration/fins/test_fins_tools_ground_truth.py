"""FinsToolService 与 Ground Truth 基线比对集成测试。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dayu.fins.processors.registry import build_fins_processor_registry
from dayu.fins.tools.service import FinsToolService
from tests.fins.storage_testkit import build_fs_storage_test_context

GROUND_TRUTH_BASELINE_RELATIVE = Path("tests/fixtures/fins/ground_truth")
GROUND_TRUTH_MANIFEST_FILE_NAME = "manifest.json"
TOOL_TRUTH_FILE_NAMES = [
    "tool_snapshot_list_documents.json",
    "tool_snapshot_get_document_sections.json",
    "tool_snapshot_read_section.json",
    "tool_snapshot_search_document.json",
    "tool_snapshot_list_tables.json",
    "tool_snapshot_get_table.json",
    "tool_snapshot_get_page_content.json",
    "tool_snapshot_get_financial_statement.json",
    "tool_snapshot_query_xbrl_facts.json",
]
SEARCH_MAX_MATCHES_PER_SECTION = 2
SEARCH_MAX_SNIPPET_CHARS = 360


@dataclass(frozen=True)
class GroundTruthSample:
    """Ground truth 样本定义。"""

    ticker: str
    env_document_id_key: str


GROUND_TRUTH_SAMPLES = [
    GroundTruthSample(ticker="0300", env_document_id_key="FINS_TRUTH_0300_DOC_ID"),
    GroundTruthSample(ticker="AAPL", env_document_id_key="FINS_TRUTH_AAPL_DOC_ID"),
    GroundTruthSample(ticker="TCOM", env_document_id_key="FINS_TRUTH_TCOM_DOC_ID"),
]


@pytest.mark.integration
def test_fins_tool_service_matches_ground_truth_baseline() -> None:
    """验证三只样本 ticker 的工具输出与 ground truth 完全一致。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 任一工具输出与基线不一致时抛出。
    """

    repo_root = _resolve_repo_root()
    workspace_root = _resolve_workspace_root(repo_root=repo_root)
    baseline_root = _resolve_baseline_root(repo_root=repo_root)
    baseline_samples = _load_baseline_samples(baseline_root=baseline_root)
    context = build_fs_storage_test_context(workspace_root)
    processor_registry = build_fins_processor_registry()
    service = FinsToolService(
        company_repository=context.company_repository,
        source_repository=context.source_repository,
        processed_repository=context.processed_repository,
        processor_registry=processor_registry,
    )

    executed_samples = 0
    for sample in GROUND_TRUTH_SAMPLES:
        document_id = _resolve_document_id_for_sample(
            sample=sample,
            baseline_samples=baseline_samples,
        )
        if document_id is None:
            continue
        if not _source_document_exists(
            workspace_root=workspace_root,
            ticker=sample.ticker,
            document_id=document_id,
        ):
            continue
        truth_dir = baseline_root / sample.ticker / document_id
        if not truth_dir.exists():
            continue
        if sample.ticker == "0300":
            _assert_hk_company_meta(workspace_root=workspace_root, ticker=sample.ticker)
        executed_samples += 1
        for file_name in TOOL_TRUTH_FILE_NAMES:
            truth_path = truth_dir / file_name
            if not truth_path.exists():
                raise AssertionError(f"缺少 ground truth 文件: {truth_path}")
            payload = _read_json(truth_path)
            if sample.ticker == "0300" and file_name == "tool_snapshot_search_document.json":
                _assert_hk_search_queries(payload=payload, truth_path=truth_path)
            tool_name = str(payload.get("tool", "")).strip()
            calls = payload.get("calls")
            if not isinstance(calls, list):
                raise AssertionError(f"ground truth calls 非法: {truth_path}")
            for index, call in enumerate(calls):
                request_payload = call.get("request")
                expected_response = call.get("response")
                if not isinstance(request_payload, dict):
                    raise AssertionError(f"ground truth request 非法: {truth_path}#{index}")
                actual_response = _execute_service_tool(
                    service=service,
                    tool_name=tool_name,
                    request_payload=request_payload,
                )
                _assert_independent_contracts(
                    tool_name=tool_name,
                    request_payload=request_payload,
                    response_payload=actual_response,
                )
                assert (
                    actual_response == expected_response
                ), f"工具输出不一致: ticker={sample.ticker} document_id={document_id} tool={tool_name} call={index}"
    if executed_samples == 0:
        pytest.skip(
            "未发现可执行样本：请先运行入库脚本写入 tests/fixtures/fins/ground_truth/manifest.json，"
            "并确保 workspace 中存在对应 source 文档。"
        )


def _assert_hk_company_meta(*, workspace_root: Path, ticker: str) -> None:
    """断言指定 ticker 的公司元数据已归类为 HK。

    Args:
        workspace_root: 工作区根目录。
        ticker: 股票代码。

    Returns:
        无。

    Raises:
        AssertionError: market 非 HK 或元数据缺失时抛出。
    """

    meta_path = workspace_root / "portfolio" / ticker / "meta.json"
    if not meta_path.exists():
        raise AssertionError(f"缺少公司元数据文件: {meta_path}。请先重传并重建 company meta。")
    meta_payload = _read_json(meta_path)
    market = str(meta_payload.get("market", "")).strip().upper()
    assert market == "HK", (
        f"{ticker} 的 market 当前为 {market!r}，预期为 'HK'。"
        "请先重新 upload/process 以重建 company meta 与 ground truth。"
    )


def _assert_hk_search_queries(*, payload: dict[str, Any], truth_path: Path) -> None:
    """断言 HK 样本的搜索词包包含繁体关键词。

    Args:
        payload: `tool_snapshot_search_document.json` 的解析结果。
        truth_path: truth 文件路径。

    Returns:
        无。

    Raises:
        AssertionError: 查询词不包含预期繁体词时抛出。
    """

    calls = payload.get("calls")
    if not isinstance(calls, list):
        raise AssertionError(f"ground truth calls 非法: {truth_path}")
    queries: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        request_payload = call.get("request")
        if not isinstance(request_payload, dict):
            continue
        queries.append(str(request_payload.get("query", "")).strip())
    assert "回購" in queries, f"HK truth 查询词缺少繁体关键词 '回購': {truth_path}"
    assert "營業收入" in queries, f"HK truth 查询词缺少繁体关键词 '營業收入': {truth_path}"


def _assert_independent_contracts(
    *,
    tool_name: str,
    request_payload: dict[str, Any],
    response_payload: Mapping[str, Any],
) -> None:
    """执行与快照无关的独立断言。

    Args:
        tool_name: 工具名称。
        request_payload: 调用请求。
        response_payload: 工具响应。

    Returns:
        无。

    Raises:
        AssertionError: 任一契约断言失败时抛出。
    """

    _assert_common_identity_fields(
        tool_name=tool_name,
        request_payload=request_payload,
        response_payload=response_payload,
    )
    if tool_name == "get_document_sections":
        _assert_sections_contract(response_payload=response_payload)
        return
    if tool_name == "list_tables":
        _assert_tables_contract(response_payload=response_payload)
        return
    if tool_name == "search_document":
        _assert_search_contract(response_payload=response_payload)


def _assert_common_identity_fields(
    *,
    tool_name: str,
    request_payload: dict[str, Any],
    response_payload: Mapping[str, Any],
) -> None:
    """断言通用身份字段。

    Args:
        tool_name: 工具名称。
        request_payload: 调用请求。
        response_payload: 工具响应。

    Returns:
        无。

    Raises:
        AssertionError: 字段缺失或不一致时抛出。
    """

    ticker = request_payload.get("ticker")
    if ticker is not None:
        assert response_payload.get("ticker") == ticker, f"{tool_name} 响应 ticker 不一致"
    document_id = request_payload.get("document_id")
    if document_id is not None:
        assert response_payload.get("document_id") == document_id, f"{tool_name} 响应 document_id 不一致"


def _assert_sections_contract(*, response_payload: Mapping[str, Any]) -> None:
    """断言章节工具返回契约。

    Args:
        response_payload: `get_document_sections` 响应。

    Returns:
        无。

    Raises:
        AssertionError: 契约不满足时抛出。
    """

    assert "has_page_info" not in response_payload, "get_document_sections 不应返回 has_page_info"
    sections = response_payload.get("sections")
    assert isinstance(sections, list), "get_document_sections.sections 必须为 list"
    for section in sections:
        assert isinstance(section, dict), "section 元素必须为 dict"
        assert "ref" in section, "section 必须包含 ref"
        assert "title" in section, "section 必须包含 title"


def _assert_tables_contract(*, response_payload: Mapping[str, Any]) -> None:
    """断言表格列表工具返回契约。

    Args:
        response_payload: `list_tables` 响应。

    Returns:
        无。

    Raises:
        AssertionError: 契约不满足时抛出。
    """

    assert "has_page_info" not in response_payload, "list_tables 不应返回 has_page_info"
    tables = response_payload.get("tables")
    assert isinstance(tables, list), "list_tables.tables 必须为 list"
    for table in tables:
        assert isinstance(table, dict), "table 元素必须为 dict"
        assert "table_ref" in table, "table 必须包含 table_ref"
        assert "row_count" in table, "table 必须包含 row_count"
        assert "col_count" in table, "table 必须包含 col_count"


def _assert_search_contract(*, response_payload: Mapping[str, Any]) -> None:
    """断言搜索工具返回契约。

    Args:
        response_payload: `search_document` 响应。

    Returns:
        无。

    Raises:
        AssertionError: 去重规则或长度上限不满足时抛出。
    """

    assert "raw_matches" not in response_payload, "search_document 不应暴露 raw_matches"
    assert "raw_match_count" not in response_payload, "search_document 不应暴露 raw_match_count"
    matches = response_payload.get("matches")
    assert isinstance(matches, list), "search_document.matches 必须为 list"
    total_matches = response_payload.get("total_matches")
    assert total_matches == len(matches), "search_document.total_matches 必须等于 matches 数量"

    count_by_section: dict[str, int] = {}
    for match in matches:
        assert isinstance(match, dict), "match 元素必须为 dict"
        section_ref = str(match.get("section_ref", ""))
        count_by_section[section_ref] = count_by_section.get(section_ref, 0) + 1
        snippet = str(match.get("snippet", ""))
        assert len(snippet) <= SEARCH_MAX_SNIPPET_CHARS, "search_document.snippet 超过 360 字符上限"

    for section_ref, count in count_by_section.items():
        assert (
            count <= SEARCH_MAX_MATCHES_PER_SECTION
        ), f"search_document section_ref={section_ref!r} 命中超过 {SEARCH_MAX_MATCHES_PER_SECTION} 条"


def _resolve_repo_root() -> Path:
    """解析仓库根目录路径。

    Args:
        无。

    Returns:
        仓库根目录绝对路径。

    Raises:
        AssertionError: 仓库目录不存在时抛出。
    """

    repo_root = Path(__file__).resolve().parents[3]
    if not repo_root.exists():
        raise AssertionError(f"仓库目录不存在: {repo_root}")
    return repo_root


def _resolve_workspace_root(*, repo_root: Path) -> Path:
    """解析仓库内 `workspace` 目录路径。

    Args:
        repo_root: 仓库根目录。

    Returns:
        `workspace` 绝对路径。

    Raises:
        pytest.skip.Exception: 工作区目录不存在时跳过测试。
    """

    workspace_root = repo_root / "workspace"
    if not workspace_root.exists():
        pytest.skip(f"workspace 目录不存在，跳过 ground truth 集成测试: {workspace_root}")
    return workspace_root


def _resolve_baseline_root(*, repo_root: Path) -> Path:
    """解析固定 ground truth 基线目录。

    Args:
        repo_root: 仓库根目录。

    Returns:
        基线目录绝对路径。

    Raises:
        无。
    """

    return repo_root / GROUND_TRUTH_BASELINE_RELATIVE


def _load_baseline_samples(*, baseline_root: Path) -> dict[str, dict[str, Any]]:
    """读取基线 manifest 并按 ticker 建立索引。

    Args:
        baseline_root: 基线目录。

    Returns:
        `ticker -> sample_entry` 映射；manifest 不存在时返回空映射。

    Raises:
        AssertionError: manifest 结构非法时抛出。
    """

    manifest_path = baseline_root / GROUND_TRUTH_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        return {}
    payload = _read_json(manifest_path)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise AssertionError(f"ground truth manifest 非法（samples 不是 list）: {manifest_path}")
    indexed: dict[str, dict[str, Any]] = {}
    for item in samples:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        indexed[ticker] = item
    return indexed


def _resolve_document_id_for_sample(
    *,
    sample: GroundTruthSample,
    baseline_samples: dict[str, dict[str, Any]],
) -> str | None:
    """为样本 ticker 解析 ground truth 文档 ID。

    解析优先级：
    1. 对应环境变量（如 `FINS_TRUTH_AAPL_DOC_ID`）
    2. 固定基线 manifest（`tests/fixtures/fins/ground_truth/manifest.json`）

    Args:
        sample: 样本定义。
        baseline_samples: 基线样本映射。

    Returns:
        文档 ID；无可用样本时返回 `None`。

    Raises:
        无。
    """

    from_env = os.getenv(sample.env_document_id_key)
    if from_env:
        normalized = from_env.strip()
        return normalized or None
    sample_entry = baseline_samples.get(sample.ticker.upper())
    if sample_entry is None:
        return None
    document_id = str(sample_entry.get("document_id", "")).strip()
    return document_id or None


def _source_document_exists(*, workspace_root: Path, ticker: str, document_id: str) -> bool:
    """判断 workspace 中是否存在对应 source 文档。

    Args:
        workspace_root: 工作区根目录。
        ticker: 股票代码。
        document_id: 文档 ID。

    Returns:
        `True` 表示 source 文档可用；否则返回 `False`。

    Raises:
        无。
    """

    filing_meta = workspace_root / "portfolio" / ticker / "filings" / document_id / "meta.json"
    material_meta = workspace_root / "portfolio" / ticker / "materials" / document_id / "meta.json"
    return filing_meta.exists() or material_meta.exists()


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的字典。

    Raises:
        ValueError: JSON 内容非法时抛出。
        OSError: 文件读取失败时抛出。
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _execute_service_tool(
    *,
    service: FinsToolService,
    tool_name: str,
    request_payload: dict[str, Any],
) -> Mapping[str, Any]:
    """执行单次工具调用并返回结果。

    Args:
        service: 财报工具服务实例。
        tool_name: 工具名称。
        request_payload: 工具请求参数。

    Returns:
        工具返回结果。

    Raises:
        ValueError: 工具名不支持时抛出。
    """

    if tool_name == "list_documents":
        return service.list_documents(**request_payload)
    if tool_name == "get_document_sections":
        return service.get_document_sections(**request_payload)
    if tool_name == "read_section":
        return service.read_section(**request_payload)
    if tool_name == "search_document":
        return service.search_document(**request_payload)
    if tool_name == "list_tables":
        return service.list_tables(**request_payload)
    if tool_name == "get_table":
        return service.get_table(**request_payload)
    if tool_name == "get_page_content":
        return service.get_page_content(**request_payload)
    if tool_name == "get_financial_statement":
        return service.get_financial_statement(**request_payload)
    if tool_name == "query_xbrl_facts":
        return service.query_xbrl_facts(**request_payload)
    raise ValueError(f"不支持的工具名: {tool_name}")
