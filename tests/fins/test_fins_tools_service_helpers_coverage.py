"""fins.tools.service 辅助函数补充覆盖测试。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from dayu.engine.processors.source import Source
from dayu.fins.domain.document_models import CompanyMeta, SourceHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.tools import search_engine as _search_engine_module
from dayu.fins.tools import service as _service_module
from dayu.fins.tools import service_helpers as _sh_module
from dayu.fins.tools.service import ToolArgumentError
from tests.fins.legacy_repository_adapters import LegacyCompatibleFinsToolService as FinsToolService


@dataclass
class _ProcessorStub:
    """处理器桩。"""

    def list_sections(self) -> list[dict[str, Any]]:
        return [
            {"ref": "sec_1", "title": "Section 1", "level": 1, "parent_ref": None, "preview": ""},
            {"ref": "sec_2", "title": "Section 2", "level": 1, "parent_ref": None, "preview": ""},
        ]

    def get_section_title(self, ref: str) -> Optional[str]:
        for sec in self.list_sections():
            if sec.get("ref") == ref:
                return sec.get("title")
        return None

    def list_tables(self) -> list[dict[str, Any]]:
        return [
            {
                "table_ref": "tbl_1",
                "caption": "c1",
                "context_before": "ctx",
                "row_count": 1,
                "col_count": 2,
                "is_financial": False,
                "table_type": "layout",
                "headers": ["a"],
                "section_ref": "sec_1",
                "page_no": 1,
            },
            {
                "table_ref": "tbl_2",
                "caption": "c2",
                "context_before": "ctx",
                "row_count": 2,
                "col_count": 2,
                "is_financial": True,
                "table_type": "financial",
                "headers": ["a"],
                "section_ref": "sec_2",
                "page_no": 2,
            },
        ]

    def get_page_content(self, page_no: int) -> dict[str, Any]:
        return {"sections": [1], "tables": [2], "text_preview": f"p{page_no}", "has_content": True, "total_items": 3, "supported": True}


class _RepositoryStub:
    """仓储桩。"""

    def get_company_meta(self, ticker: str) -> CompanyMeta:
        """返回最小公司元数据。"""

        return CompanyMeta(
            company_id="1",
            company_name="Apple Inc.",
            ticker=ticker,
            market="US",
            resolver_version="test",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
        """返回首个可识别的 ticker。"""

        if "AAPL" in candidates:
            return "AAPL"
        return None

    def list_document_ids(self, ticker: str, source_kind: Optional[SourceKind] = None) -> list[str]:
        del ticker
        return ["d1", "d2", "d3"]

    def get_document_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
        del ticker
        if document_id == "d1":
            raise FileNotFoundError(document_id)
        if document_id == "d2":
            return {"is_deleted": True, "ingest_complete": True}
        if document_id == "d3":
            return {"is_deleted": False, "ingest_complete": False}
        if document_id == "d4":
            return {"is_deleted": False, "ingest_complete": True, "form_type": "10-Q", "report_date": "2025-03-29"}
        raise FileNotFoundError(document_id)

    def get_source_handle(self, ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
        raise FileNotFoundError(document_id)

    def get_primary_source(self, ticker: str, document_id: str, source_kind: SourceKind) -> Source:
        raise FileNotFoundError(f"{ticker}:{document_id}:{source_kind.value}")

    def get_processed_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
        del ticker, document_id
        return {}


class _RegistryStub:
    """注册表桩。"""

    def create(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> object:
        """返回固定处理器桩。"""

        _ = (source, form_type, media_type)
        return _ProcessorStub()

    def create_with_fallback(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
        on_fallback: Optional[Any] = None,
    ) -> object:
        """返回固定处理器桩。"""

        _ = on_fallback
        return self.create(source, form_type=form_type, media_type=media_type)


@pytest.mark.unit
def test_service_list_tables_and_page_content_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖 list_tables 过滤分支与 get_page_content 返回分支。"""

    service = FinsToolService(repository=_RepositoryStub(), processor_registry=_RegistryStub())
    processor = _ProcessorStub()
    monkeypatch.setattr(service, "_get_or_create_processor", lambda **kwargs: processor)

    result = service.list_tables(ticker="AAPL", document_id="d", financial_only=True, within_section_ref="sec_2")
    assert result["total"] == 1
    assert result["tables"][0]["table_ref"] == "tbl_2"

    page = service.get_page_content(ticker="AAPL", document_id="d", page_no=1)
    assert "error" not in page
    assert page["text_preview"] == "p1"


@pytest.mark.unit
def test_service_collect_source_docs_and_source_kind_branches() -> None:
    """覆盖 source 文档采集的异常/跳过与 source_kind 未命中分支。"""

    service = FinsToolService(repository=_RepositoryStub(), processor_registry=_RegistryStub())
    docs = service._collect_source_documents_by_kind("AAPL", SourceKind.FILING)
    assert docs == []

    assert service._resolve_document_form_type(ticker="AAPL", document_id="missing") is None

    with pytest.raises(FileNotFoundError, match="Document not found"):
        service._resolve_source_kind(ticker="AAPL", document_id="missing")


@pytest.mark.unit
def test_service_module_does_not_reexport_split_private_helpers() -> None:
    """确保 service 模块不再作为 split helper 的兼容出口。"""

    exported_names = vars(_service_module)
    assert "SEARCH_MODE_EXACT" not in exported_names
    assert "_build_synonym_queries" not in exported_names
    assert "_parse_xbrl_decimals_value" not in exported_names


@pytest.mark.unit
def test_text_and_period_normalizers() -> None:
    """覆盖参数标准化分支。"""

    from dayu.fins._converters import normalize_optional_text, require_non_empty_text
    assert normalize_optional_text("  ") is None
    with pytest.raises(ToolArgumentError):
        require_non_empty_text(None, empty_error=ToolArgumentError("x", "a", None, "Argument must not be empty"))
    with pytest.raises(ToolArgumentError):
        require_non_empty_text("   ", empty_error=ToolArgumentError("x", "a", "   ", "Argument must not be empty"))

    with pytest.raises(ToolArgumentError):
        _sh_module._normalize_periods("Q1")  # type: ignore[arg-type]
    assert _sh_module._normalize_periods([" ", "Q1"]) == ["Q1"]


@pytest.mark.unit
def test_page_range_and_date_inference_helpers() -> None:
    """覆盖页码范围与财期/财年推断分支。"""

    assert _sh_module._extract_page_range({"page_range": [1, 2]}) == [1, 2]
    assert _sh_module._extract_page_range({"page_range": [1, "2"]}) is None

    assert _sh_module._infer_fiscal_period({"form_type": "10-Q", "report_date": "bad"}) is None
    assert _sh_module._infer_fiscal_period({"form_type": "10-Q", "report_date": "2025-02-01"}) is None
    assert _sh_module._infer_fiscal_period({"form_type": "10-K", "report_date": "2025-12-01"}) == "FY"
    assert _sh_module._infer_fiscal_period({"form_type": "20-F"}) == "FY"

    assert _sh_module._infer_fiscal_year({"report_date": "bad"}, None) is None
    assert _sh_module._infer_fiscal_year({"report_date": "2025-12-01"}, "Q4") is None
    assert _sh_module._infer_fiscal_year({"fiscal_year": 2025, "report_date": "2026-12-01"}, "Q4") == 2025
    assert _sh_module._resolve_fiscal_year_with_fallback(None, 2024) == 2024
    assert _sh_module._resolve_fiscal_year_with_fallback("2023", 2024) == 2023
    assert _sh_module._resolve_fiscal_year_with_fallback("bad", 2024) == 2024
    assert _sh_module._resolve_fiscal_period_with_fallback(None, "FY") == "FY"
    assert _sh_module._resolve_fiscal_period_with_fallback(" q1 ", "FY") == "q1"

    assert _sh_module._extract_year("2025") is None
    assert _sh_module._extract_year("xxxx-01-01") is None
    assert _sh_module._extract_year("0-01-01") is None


@pytest.mark.unit
def test_numeric_and_table_payload_helpers() -> None:
    """覆盖数值转换与表格 payload 分支。"""

    assert _sh_module._to_optional_float(" ") is None
    assert _sh_module._to_optional_float("bad") is None
    assert _sh_module._to_optional_float(float("nan")) is None

    markdown = _sh_module._build_table_data_payload({"data_format": "markdown", "data": "|a|\n|---|"})
    assert markdown["kind"] == "markdown"

    raw_text = _sh_module._build_table_data_payload({"data_format": "markdown", "data": "not table"})
    assert raw_text["kind"] == "raw_text"

    records = _sh_module._build_table_data_payload({"data": [{"A": 1}], "columns": ["A", None, "A"]})
    assert records["kind"] == "records"

    assert _sh_module._normalize_table_rows("bad") == []
    rows = _sh_module._normalize_table_rows([{" a ": 1}, ["x", "y"], 3])
    assert rows[0]["a"] == 1
    assert rows[1]["0"] == "x"
    assert rows[2]["value"] == 3

    assert _sh_module._normalize_table_columns([None, " A ", "A"], rows=[{"A": 1}]) == ["A"]
    assert _sh_module._normalize_table_columns(None, rows=[]) == []
    assert _sh_module._normalize_table_columns(None, rows=[{"x": 1}]) == ["x"]

    assert _sh_module._coerce_table_text(None) == ""
    assert _sh_module._coerce_table_text(1) == "1"
    assert _sh_module._looks_like_markdown_table("") is False
    assert _sh_module._looks_like_markdown_table("a\nb") is False
    assert _sh_module._looks_like_markdown_table("|a|\nplain") is False

    assert _sh_module._normalize_table_type("bad") is None


@pytest.mark.unit
def test_taxonomy_and_query_payload_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖 taxonomy 解析、默认 concepts 与 query payload 归一化分支。"""

    class _P1:
        def get_xbrl_taxonomy(self) -> str:
            raise RuntimeError("boom")

    class _P2:
        xbrl_taxonomy = "ifrs-full-2024"

    assert _sh_module._resolve_processor_taxonomy(_P1()) is None
    assert _sh_module._resolve_processor_taxonomy(_P2()) == "ifrs-full"

    assert _sh_module._normalize_taxonomy_name("us-gaap-2024") == "us-gaap"
    assert _sh_module._normalize_taxonomy_name("ifrs") == "ifrs-full"
    assert _sh_module._normalize_taxonomy_name("other") is None

    monkeypatch.setattr(_sh_module, "_DEFAULT_XBRL_CONCEPTS_BY_FORM_TAXONOMY", {("10-K", "us-gaap"): ("A", "B")})
    monkeypatch.setattr(_sh_module, "_DEFAULT_XBRL_CONCEPTS_BY_TAXONOMY", {"us-gaap": ("C",)})
    monkeypatch.setattr(_sh_module, "_GLOBAL_DEFAULT_XBRL_CONCEPTS", ("D",))

    assert _sh_module._resolve_default_xbrl_concepts(form_type="10-K", taxonomy="us-gaap") == ["A", "B"]
    assert _sh_module._resolve_default_xbrl_concepts(form_type="8-K", taxonomy="us-gaap") == ["C"]
    assert _sh_module._resolve_default_xbrl_concepts(form_type=None, taxonomy=None) == ["D"]

    payload = _sh_module._normalize_xbrl_query_payload(payload={"query_params": [], "facts": "bad"}, default_concepts=["X"])
    assert payload["query_params"]["concepts"] == ["X"]
    assert payload["facts"] == []

    assert _sh_module._normalize_concepts_for_query("bad", ["A"]) == ["A"]
    assert _sh_module._normalize_concepts_for_query([" ", "B"], ["A"]) == ["B"]


@pytest.mark.unit
def test_fact_normalization_and_dedup_helpers() -> None:
    """覆盖 fact 规范化、去重、segment 签名与 decimals 分支。"""

    fact_from_value = _sh_module._normalize_single_fact({"concept": "A", "value": "<b>x</b>"})
    assert fact_from_value is not None
    assert fact_from_value["text_value"] == "x"

    assert _sh_module._normalize_single_fact({"concept": "A", "value": None}) is None

    assert _sh_module._clean_fact_text_value("") == ""
    assert _sh_module._looks_like_html_text("") is False

    normalized_pairs = [
        (
            {
                "concept": "us-gaap:Revenue",
                "numeric_value": 1.0,
                "fiscal_period": "Q1",
                "statement_type": "IncomeStatement",
                "unit": "USD",
                "period_end": "2024-09-28",
                "fiscal_year": 2024,
            },
            {"decimals": "-3", "segment": {"x": 1}},
            2,
        ),
        (
            {
                "concept": "Revenue",
                "numeric_value": 1.0,
                "fiscal_period": "Q1",
                "statement_type": "IncomeStatement",
                "unit": "USD",
                "period_end": "2024-09-28",
                "fiscal_year": 2024,
            },
            {"decimals": "-3", "segment": {"x": 1}},
            1,
        ),
    ]
    deduped = _sh_module._deduplicate_xbrl_facts(normalized_pairs)
    assert len(deduped) == 1

    class _Unserializable:
        pass

    signature = _sh_module._build_segment_signature({"bad": _Unserializable()})
    assert "bad" in signature

    assert _sh_module._parse_xbrl_decimals("INF") == 100000
    assert _sh_module._parse_xbrl_decimals("x") == -100000


# ============================================================================
# _build_evidence_matches — ref_to_topic 回退查表
# ============================================================================


@pytest.mark.unit
class TestBuildEvidenceMatchesRefToTopic:
    """_build_evidence_matches ref_to_topic 回退查表测试。"""

    def _make_entry(
        self,
        section_ref: str,
        section_title: str,
        context: str = "some text",
        query: str = "test query",
        strategy: str = "exact",
    ) -> dict:
        return {
            "section_ref": section_ref,
            "section_title": section_title,
            "evidence": {"matched_text": context[:50], "context": context},
            "page_no": None,
            "_priority": 1,
            "_query": query,
            "_strategy": strategy,
        }

    def test_no_ref_to_topic_no_inherit(self) -> None:
        """不传 ref_to_topic 时，子章节 topic 保持 None。"""
        entries = [self._make_entry("sec_x_c01", "A. Subsection Title")]
        matches = _search_engine_module._build_evidence_matches(entries, "10-K", ref_to_topic=None)
        assert matches[0]["section"]["topic"] is None
        # 查询归属字段
        assert matches[0]["matched_query"] == "test query"
        assert matches[0]["is_exact_phrase"] is True

    def test_ref_to_topic_fills_child_topic(self) -> None:
        """传入 ref_to_topic 时，子章节从索引中获取父节 topic。"""
        ref_to_topic: dict = {"sec_x_c01": "mda"}
        entries = [self._make_entry("sec_x_c01", "A. Subsection Title")]
        matches = _search_engine_module._build_evidence_matches(entries, "10-K", ref_to_topic=ref_to_topic)
        assert matches[0]["section"]["topic"] == "mda"

    def test_own_topic_wins_over_ref_to_topic(self) -> None:
        """当 section_title 能自解析出 topic 时，不使用 ref_to_topic 覆盖。"""
        # "Part II - Item 7" 可自解析为 mda
        ref_to_topic: dict = {"sec_007": "risk_factors"}  # 表中胡乱填了 risk_factors
        entries = [self._make_entry("sec_007", "Part II - Item 7")]
        matches = _search_engine_module._build_evidence_matches(entries, "10-K", ref_to_topic=ref_to_topic)
        # 自解析结果 mda 优先，不被 ref_to_topic 中的 risk_factors 覆盖
        assert matches[0]["section"]["topic"] == "mda"

    def test_evidence_passthrough(self) -> None:
        """evidence 结构原样透传。"""
        entries = [self._make_entry("sec_1", "Part I - Item 1A")]
        matches = _search_engine_module._build_evidence_matches(entries, "10-K")
        assert "matched_text" in matches[0]["evidence"]
        assert "context" in matches[0]["evidence"]

    def test_expansion_is_not_exact(self) -> None:
        """expansion 策略的命中 is_exact_phrase 为 False。"""
        entries = [self._make_entry("sec_1", "Item 1", strategy="phrase_variant", query="revenue")]
        matches = _search_engine_module._build_evidence_matches(entries, "10-K")
        assert matches[0]["matched_query"] == "revenue"
        assert matches[0]["is_exact_phrase"] is False

    def test_snippet_fallback_builds_evidence(self) -> None:
        """无 evidence 字典时从 snippet 构建 evidence，matched_text 围绕查询居中。"""
        long_text = "prefix " * 30 + "target revenue data here" + " suffix" * 30
        entry = {
            "section_ref": "sec_1",
            "section_title": "Item 1",
            "snippet": long_text,
            "page_no": 3,
            "_priority": 0,
            "_query": "revenue",
            "_strategy": "exact",
        }
        matches = _search_engine_module._build_evidence_matches([entry], "10-K")
        ev = matches[0]["evidence"]
        # matched_text 应包含 query 而非仅取头部
        assert "revenue" in ev["matched_text"]
        # context 是完整 snippet
        assert ev["context"] == long_text
        assert matches[0]["is_exact_phrase"] is True


# ============================================================================
# _center_matched_text — 围绕查询居中截取
# ============================================================================

from dayu.fins.tools.search_engine import (
    _cap_entries_with_exact_priority,
    _center_matched_text,
)


@pytest.mark.unit
class TestCenterMatchedText:
    """_center_matched_text 围绕查询居中截取测试。"""

    def test_short_snippet_unchanged(self) -> None:
        """snippet 短于 max_chars 时原样返回。"""
        result = _center_matched_text("short text", "short", max_chars=120)
        assert result == "short text"

    def test_empty_snippet(self) -> None:
        """空 snippet 返回空字符串。"""
        assert _center_matched_text("", "query") == ""

    def test_empty_query_fallback_to_head(self) -> None:
        """空 query 回退到截取头部。"""
        long_text = "a" * 200
        result = _center_matched_text(long_text, "", max_chars=50)
        assert result == long_text[:50]

    def test_query_at_beginning(self) -> None:
        """query 在 snippet 开头时截取头部。"""
        text = "revenue growth was strong" + " extra" * 50
        result = _center_matched_text(text, "revenue", max_chars=40)
        assert "revenue" in result
        assert len(result) <= 40

    def test_query_in_middle_centered(self) -> None:
        """query 在 snippet 中部时围绕其居中。"""
        prefix = "a " * 80
        suffix = " b" * 80
        text = prefix + "KEYWORD HERE" + suffix
        result = _center_matched_text(text, "KEYWORD HERE", max_chars=60)
        assert "KEYWORD HERE" in result
        assert len(result) <= 60

    def test_query_at_end(self) -> None:
        """query 在 snippet 尾部时截取尾部。"""
        text = "prefix " * 40 + "revenue"
        result = _center_matched_text(text, "revenue", max_chars=40)
        assert "revenue" in result
        assert len(result) <= 40

    def test_query_not_found_fallback(self) -> None:
        """query 不在 snippet 中时回退到头部截取。"""
        text = "a" * 200
        result = _center_matched_text(text, "zzz_not_here", max_chars=50)
        assert result == text[:50]

    def test_case_insensitive(self) -> None:
        """查找 query 时忽略大小写。"""
        text = "prefix " * 20 + "Total Revenue was 100M" + " suffix" * 20
        result = _center_matched_text(text, "total revenue", max_chars=60)
        assert "Total Revenue" in result


# ============================================================================
# _cap_entries_with_exact_priority — exact 优先限流
# ============================================================================


@pytest.mark.unit
class TestCapEntriesWithExactPriority:
    """_cap_entries_with_exact_priority 限流逻辑测试。"""

    @staticmethod
    def _entry(strategy: str, ref: str = "sec_1") -> dict:
        return {"_strategy": strategy, "section_ref": ref, "_priority": 0 if strategy == "exact" else 2}

    def test_small_list_no_cap(self) -> None:
        """条目不足 _CAP_MIN_TRIGGER 时不裁剪。"""
        entries = [self._entry("exact"), self._entry("synonym")]
        result = _cap_entries_with_exact_priority(entries)
        assert len(result) == 2

    def test_no_exact_no_cap(self) -> None:
        """全部为 expansion 时不裁剪。"""
        entries = [self._entry("synonym", f"s{i}") for i in range(15)]
        result = _cap_entries_with_exact_priority(entries)
        assert len(result) == 15

    def test_exact_present_caps_expansion(self) -> None:
        """exact 存在时 expansion 被裁剪到总量 30%。"""
        exact = [self._entry("exact", f"e{i}") for i in range(5)]
        expansion = [self._entry("synonym", f"x{i}") for i in range(20)]
        entries = exact + expansion
        result = _cap_entries_with_exact_priority(entries)
        # expansion 配额 = max(2, int(25 * 0.3)) = 7
        assert len(result) == 5 + 7

    def test_exact_all_preserved(self) -> None:
        """所有 exact 条目都被保留。"""
        exact = [self._entry("exact", f"e{i}") for i in range(10)]
        expansion = [self._entry("token", f"x{i}") for i in range(10)]
        entries = exact + expansion
        result = _cap_entries_with_exact_priority(entries)
        exact_in_result = [e for e in result if e["_strategy"] == "exact"]
        assert len(exact_in_result) == 10

    def test_expansion_minimum_two(self) -> None:
        """expansion 配额下限为 2 条。"""
        exact = [self._entry("exact", f"e{i}") for i in range(3)]
        expansion = [self._entry("synonym", f"x{i}") for i in range(5)]
        entries = exact + expansion
        result = _cap_entries_with_exact_priority(entries)
        expansion_in_result = [e for e in result if e["_strategy"] != "exact"]
        assert len(expansion_in_result) == 2

    def test_display_budget_caps_total(self) -> None:
        """display_budget 存在时总数不超过 budget。"""
        exact = [self._entry("exact", f"e{i}") for i in range(5)]
        expansion = [self._entry("synonym", f"x{i}") for i in range(20)]
        entries = exact + expansion
        # 无 budget 时: expansion_cap = max(2, int(25*0.3)) = 7, 总数 12
        result_no_budget = _cap_entries_with_exact_priority(entries)
        assert len(result_no_budget) == 12
        # 有 budget=10 时: budget_remaining = 10 - 5 = 5, expansion_cap = min(7, max(2, 5)) = 5
        result_with_budget = _cap_entries_with_exact_priority(entries, display_budget=10)
        assert len(result_with_budget) == 10

    def test_display_budget_expansion_minimum_preserved(self) -> None:
        """display_budget 场景下 expansion 保底 2 条。"""
        exact = [self._entry("exact", f"e{i}") for i in range(9)]
        expansion = [self._entry("synonym", f"x{i}") for i in range(10)]
        entries = exact + expansion
        # budget=10, budget_remaining = 10 - 9 = 1 → max(2, 1) = 2
        result = _cap_entries_with_exact_priority(entries, display_budget=10)
        expansion_in_result = [e for e in result if e["_strategy"] != "exact"]
        assert len(expansion_in_result) == 2

    def test_display_budget_exact_exceeds_budget(self) -> None:
        """exact 数量超过 budget 时不触发 budget 收紧。"""
        exact = [self._entry("exact", f"e{i}") for i in range(15)]
        expansion = [self._entry("synonym", f"x{i}") for i in range(10)]
        entries = exact + expansion
        # exact=15 > budget=10, 走常规 expansion_cap = max(2, int(25*0.3)) = 7
        result = _cap_entries_with_exact_priority(entries, display_budget=10)
        assert len(result) == 15 + 7

    def test_display_budget_none_no_effect(self) -> None:
        """display_budget=None 时行为与旧版一致。"""
        exact = [self._entry("exact", f"e{i}") for i in range(5)]
        expansion = [self._entry("synonym", f"x{i}") for i in range(20)]
        entries = exact + expansion
        result_default = _cap_entries_with_exact_priority(entries)
        result_none = _cap_entries_with_exact_priority(entries, display_budget=None)
        assert len(result_default) == len(result_none)
