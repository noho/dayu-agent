"""6-K active source 主文件修复测试。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dayu.fins.domain.document_models import (
    CompanyMeta,
    SourceDocumentUpsertRequest,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines import sec_6k_primary_document_repair as repair_module
from tests.fins.storage_testkit import FsStorageTestContext, build_fs_storage_test_context


def _create_active_6k_filing(
    context: FsStorageTestContext,
    *,
    ticker: str,
    document_id: str,
    primary_document: str,
    file_payloads: dict[str, bytes],
) -> None:
    """创建带多个 HTML 文件的 active 6-K。

    Args:
        context: 测试仓储上下文。
        ticker: 股票代码。
        document_id: 文档 ID。
        primary_document: 当前主文件名。
        file_payloads: 文件名到文件内容的映射。

    Returns:
        无。

    Raises:
        OSError: 仓储写入失败时抛出。
    """

    context.source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=document_id.replace("fil_", ""),
            form_type="6-K",
            primary_document=primary_document,
            meta={
                "form_type": "6-K",
                "report_date": "2025-09-30",
                "is_deleted": False,
            },
        ),
        source_kind=SourceKind.FILING,
    )
    handle = context.source_repository.get_source_handle(ticker, document_id, SourceKind.FILING)
    file_metas = []
    for filename, payload in file_payloads.items():
        file_metas.append(
            context.blob_repository.store_file(
                handle=handle,
                filename=filename,
                data=BytesIO(payload),
                content_type="text/html",
            )
        )
    context.source_repository.update_source_document(
        SourceDocumentUpsertRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=document_id.replace("fil_", ""),
            form_type="6-K",
            primary_document=primary_document,
            meta={
                "form_type": "6-K",
                "report_date": "2025-09-30",
                "is_deleted": False,
            },
            files=file_metas,
        ),
        source_kind=SourceKind.FILING,
    )


@pytest.mark.unit
def test_repair_active_6k_primary_document_promotes_parseable_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 cover primary 失败且附件可提取核心报表时会修正主文件。"""

    context = build_fs_storage_test_context(tmp_path)
    _create_active_6k_filing(
        context,
        ticker="ALVO",
        document_id="fil_alvo_q1",
        primary_document="form6-k.htm",
        file_payloads={
            "form6-k.htm": b"FORM 6-K cover page",
            "ex99-1.htm": b"EX-99.1 press release",
        },
    )

    assessment_by_filename = {
        "form6-k.htm": repair_module.SixKPrimaryCandidateAssessment(
            filename="form6-k.htm",
            income_row_count=0,
            balance_sheet_row_count=0,
            filename_priority=3,
        ),
        "ex99-1.htm": repair_module.SixKPrimaryCandidateAssessment(
            filename="ex99-1.htm",
            income_row_count=18,
            balance_sheet_row_count=32,
            filename_priority=0,
        ),
    }

    def _fake_assess_active_6k_candidate(
        *,
        source_repository: object,
        ticker: str,
        document_id: str,
        filename: str,
        primary_document: str,
    ) -> repair_module.SixKPrimaryCandidateAssessment:
        """返回固定候选评估结果。"""

        del source_repository, ticker, document_id, primary_document
        return assessment_by_filename[filename]

    monkeypatch.setattr(
        repair_module,
        "_assess_active_6k_candidate",
        _fake_assess_active_6k_candidate,
    )
    reprocess_calls: list[tuple[str, str]] = []

    outcome = repair_module.reconcile_active_6k_primary_document(
        source_repository=context.source_repository,
        ticker="ALVO",
        document_id="fil_alvo_q1",
        mark_processed_reprocess_required=lambda ticker, document_id: reprocess_calls.append(
            (ticker, document_id)
        ),
    )

    assert outcome is not None
    assert outcome.previous_primary_document == "form6-k.htm"
    assert outcome.selected_primary_document == "ex99-1.htm"
    updated_meta = context.source_repository.get_source_meta("ALVO", "fil_alvo_q1", SourceKind.FILING)
    assert updated_meta["primary_document"] == "ex99-1.htm"
    assert reprocess_calls == [("ALVO", "fil_alvo_q1")]


@pytest.mark.unit
def test_reconcile_active_6k_primary_document_updates_non_cover_primary_when_sibling_is_better(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证当前主文件不是 cover 时也会按处理器真源重选主文件。"""

    context = build_fs_storage_test_context(tmp_path)
    _create_active_6k_filing(
        context,
        ticker="BABA",
        document_id="fil_baba_q4",
        primary_document="ex99-2.htm",
        file_payloads={
            "ex99-1.htm": b"EX-99.1 quarterly results",
            "ex99-2.htm": b"EX-99.2 supplementary materials",
        },
    )

    assessment_by_filename = {
        "ex99-1.htm": repair_module.SixKPrimaryCandidateAssessment(
            filename="ex99-1.htm",
            income_row_count=15,
            balance_sheet_row_count=26,
            filename_priority=0,
        ),
        "ex99-2.htm": repair_module.SixKPrimaryCandidateAssessment(
            filename="ex99-2.htm",
            income_row_count=0,
            balance_sheet_row_count=0,
            filename_priority=1,
        ),
    }

    def _fake_assess_active_6k_candidate(
        *,
        source_repository: object,
        ticker: str,
        document_id: str,
        filename: str,
        primary_document: str,
    ) -> repair_module.SixKPrimaryCandidateAssessment:
        """返回固定候选评估结果。"""

        del source_repository, ticker, document_id, primary_document
        return assessment_by_filename[filename]

    monkeypatch.setattr(
        repair_module,
        "_assess_active_6k_candidate",
        _fake_assess_active_6k_candidate,
    )

    outcome = repair_module.reconcile_active_6k_primary_document(
        source_repository=context.source_repository,
        ticker="BABA",
        document_id="fil_baba_q4",
    )

    assert outcome is not None
    assert outcome.previous_primary_document == "ex99-2.htm"
    assert outcome.selected_primary_document == "ex99-1.htm"
    meta = context.source_repository.get_source_meta("BABA", "fil_baba_q4", SourceKind.FILING)
    assert meta["primary_document"] == "ex99-1.htm"


# ---------------------------------------------------------------------------
# reconcile_active_6k_primary_document: 提前返回 None 的分支
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reconcile_returns_none_when_is_deleted(
    tmp_path: Path,
) -> None:
    """验证 is_deleted=True 时直接返回 None。"""

    context = build_fs_storage_test_context(tmp_path)
    context.source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="DEL",
            document_id="fil_del_1",
            internal_document_id="del_1",
            form_type="6-K",
            primary_document="form6-k.htm",
            meta={"form_type": "6-K", "is_deleted": True},
        ),
        source_kind=SourceKind.FILING,
    )

    outcome = repair_module.reconcile_active_6k_primary_document(
        source_repository=context.source_repository,
        ticker="DEL",
        document_id="fil_del_1",
    )
    assert outcome is None


@pytest.mark.unit
def test_reconcile_returns_none_when_form_type_not_6k(
    tmp_path: Path,
) -> None:
    """验证 form_type 不是 6-K 时返回 None。"""

    context = build_fs_storage_test_context(tmp_path)
    context.source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="TENK",
            document_id="fil_10k_1",
            internal_document_id="10k_1",
            form_type="10-K",
            primary_document="form10-k.htm",
            meta={"form_type": "10-K", "is_deleted": False},
        ),
        source_kind=SourceKind.FILING,
    )

    outcome = repair_module.reconcile_active_6k_primary_document(
        source_repository=context.source_repository,
        ticker="TENK",
        document_id="fil_10k_1",
    )
    assert outcome is None


@pytest.mark.unit
def test_reconcile_returns_none_when_no_html_candidates(
    tmp_path: Path,
) -> None:
    """验证 candidate_filenames 为空（无 HTML 文件）时返回 None。"""

    context = build_fs_storage_test_context(tmp_path)
    _create_active_6k_filing(
        context,
        ticker="NOHTML",
        document_id="fil_nohtml_1",
        primary_document="form6-k.htm",
        file_payloads={"report.pdf": b"pdf-bytes"},
    )

    outcome = repair_module.reconcile_active_6k_primary_document(
        source_repository=context.source_repository,
        ticker="NOHTML",
        document_id="fil_nohtml_1",
    )
    assert outcome is None


@pytest.mark.unit
def test_reconcile_returns_none_when_best_assessment_is_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证所有候选均无法提取核心报表时返回 None。"""

    context = build_fs_storage_test_context(tmp_path)
    _create_active_6k_filing(
        context,
        ticker="NOCORE",
        document_id="fil_nocore_1",
        primary_document="form6-k.htm",
        file_payloads={"form6-k.htm": b"cover", "ex99-1.htm": b"no data"},
    )

    def _fake_assess(
        *,
        source_repository: object,
        ticker: str,
        document_id: str,
        filename: str,
        primary_document: str,
    ) -> repair_module.SixKPrimaryCandidateAssessment:
        """返回无核心报表的评估结果。"""
        del source_repository, ticker, document_id, primary_document
        return repair_module.SixKPrimaryCandidateAssessment(
            filename=filename,
            income_row_count=0,
            balance_sheet_row_count=0,
            filename_priority=0,
        )

    monkeypatch.setattr(repair_module, "_assess_active_6k_candidate", _fake_assess)

    outcome = repair_module.reconcile_active_6k_primary_document(
        source_repository=context.source_repository,
        ticker="NOCORE",
        document_id="fil_nocore_1",
    )
    assert outcome is None


@pytest.mark.unit
def test_reconcile_returns_none_when_best_is_current_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证最佳候选即当前主文件时返回 None。"""

    context = build_fs_storage_test_context(tmp_path)
    _create_active_6k_filing(
        context,
        ticker="SAME",
        document_id="fil_same_1",
        primary_document="form6-k.htm",
        file_payloads={"form6-k.htm": b"cover", "ex99-1.htm": b"data"},
    )

    def _fake_assess(
        *,
        source_repository: object,
        ticker: str,
        document_id: str,
        filename: str,
        primary_document: str,
    ) -> repair_module.SixKPrimaryCandidateAssessment:
        """返回固定评估，主文件已是最优。"""
        del source_repository, ticker, document_id, primary_document
        if filename == "form6-k.htm":
            return repair_module.SixKPrimaryCandidateAssessment(
                filename=filename,
                income_row_count=10,
                balance_sheet_row_count=20,
                filename_priority=0,
            )
        return repair_module.SixKPrimaryCandidateAssessment(
            filename=filename,
            income_row_count=5,
            balance_sheet_row_count=5,
            filename_priority=1,
        )

    monkeypatch.setattr(repair_module, "_assess_active_6k_candidate", _fake_assess)

    outcome = repair_module.reconcile_active_6k_primary_document(
        source_repository=context.source_repository,
        ticker="SAME",
        document_id="fil_same_1",
    )
    assert outcome is None


# ---------------------------------------------------------------------------
# _list_candidate_html_filenames: 各分支
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_candidate_raises_when_files_not_list() -> None:
    """验证 meta.files 非 list 时抛出 ValueError。"""

    with pytest.raises(ValueError, match="files 必须为 list"):
        repair_module._list_candidate_html_filenames({"files": "not-a-list"})


@pytest.mark.unit
def test_list_candidate_raises_when_files_missing() -> None:
    """验证 meta 中无 files 键时抛出 ValueError。"""

    with pytest.raises(ValueError, match="files 必须为 list"):
        repair_module._list_candidate_html_filenames({})


@pytest.mark.unit
def test_list_candidate_skips_non_dict_entries() -> None:
    """验证 entry 非 dict 时跳过。"""

    result = repair_module._list_candidate_html_filenames(
        {"files": ["string-entry", 42, {"name": "good.htm"}]}
    )
    assert result == ["good.htm"]


@pytest.mark.unit
def test_list_candidate_skips_empty_name() -> None:
    """验证 name 为空时跳过。"""

    result = repair_module._list_candidate_html_filenames(
        {"files": [{"name": ""}, {"name": "  "}, {"name": "valid.html"}]}
    )
    assert result == ["valid.html"]


@pytest.mark.unit
def test_list_candidate_skips_non_html_extensions() -> None:
    """验证非 htm/html 扩展名时跳过。"""

    result = repair_module._list_candidate_html_filenames(
        {"files": [{"name": "data.pdf"}, {"name": "img.png"}, {"name": "doc.htm"}]}
    )
    assert result == ["doc.htm"]


# ---------------------------------------------------------------------------
# _select_best_primary_candidate: 排序与选择逻辑
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_select_best_returns_none_when_no_extractable() -> None:
    """验证无可提取核心报表的候选时返回 None。"""

    result = repair_module._select_best_primary_candidate(
        primary_document="form6-k.htm",
        candidate_assessments=[
            repair_module.SixKPrimaryCandidateAssessment(
                filename="form6-k.htm",
                income_row_count=0,
                balance_sheet_row_count=0,
                filename_priority=0,
            ),
        ],
    )
    assert result is None


@pytest.mark.unit
def test_select_best_prefers_higher_core_row_count() -> None:
    """验证优先选择核心报表行数最多的候选。"""

    result = repair_module._select_best_primary_candidate(
        primary_document="form6-k.htm",
        candidate_assessments=[
            repair_module.SixKPrimaryCandidateAssessment(
                filename="ex99-1.htm",
                income_row_count=5,
                balance_sheet_row_count=5,
                filename_priority=0,
            ),
            repair_module.SixKPrimaryCandidateAssessment(
                filename="ex99-2.htm",
                income_row_count=20,
                balance_sheet_row_count=30,
                filename_priority=0,
            ),
        ],
    )
    assert result is not None
    assert result.filename == "ex99-2.htm"


@pytest.mark.unit
def test_select_best_breaks_tie_by_filename_priority() -> None:
    """验证行数相同时按 filename_priority 排序。"""

    result = repair_module._select_best_primary_candidate(
        primary_document="form6-k.htm",
        candidate_assessments=[
            repair_module.SixKPrimaryCandidateAssessment(
                filename="ex99-2.htm",
                income_row_count=10,
                balance_sheet_row_count=10,
                filename_priority=5,
            ),
            repair_module.SixKPrimaryCandidateAssessment(
                filename="ex99-1.htm",
                income_row_count=10,
                balance_sheet_row_count=10,
                filename_priority=1,
            ),
        ],
    )
    assert result is not None
    assert result.filename == "ex99-1.htm"


# ---------------------------------------------------------------------------
# _update_active_6k_primary_document: files 非 list 报错
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_raises_when_meta_files_not_list() -> None:
    """验证 _update_active_6k_primary_document 中 files 非 list 时报错。"""

    mock_repo = MagicMock()
    with pytest.raises(ValueError, match="files 必须为 list"):
        repair_module._update_active_6k_primary_document(
            source_repository=mock_repo,
            ticker="TCK",
            document_id="doc_1",
            meta={"files": "bad"},
            selected_primary_document="new.htm",
        )


# ---------------------------------------------------------------------------
# _resolve_target_tickers: ticker 子集去重排序、全量扫描
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_target_tickers_normalizes_subset() -> None:
    """验证 ticker 子集去重、排序并保持首次出现顺序。"""

    mock_repo = MagicMock()
    result = repair_module._resolve_target_tickers(
        company_repository=mock_repo,
        target_tickers=["baba", "BABA", "  alvo  ", "baba"],
    )
    # 去重后保持首次出现顺序：BABA, ALVO
    assert result == ["BABA", "ALVO"]


@pytest.mark.unit
def test_resolve_target_tickers_raises_on_empty_subset() -> None:
    """验证传入全空白 ticker 子集时抛出 ValueError。"""

    mock_repo = MagicMock()
    with pytest.raises(ValueError, match="target_tickers 不能为空"):
        repair_module._resolve_target_tickers(
            company_repository=mock_repo,
            target_tickers=["  ", ""],
        )


@pytest.mark.unit
def test_resolve_target_tickers_full_scan(tmp_path: Path) -> None:
    """验证 target_tickers 为 None 时从仓储全量扫描。"""

    context = build_fs_storage_test_context(tmp_path)
    # 写入两个公司元数据
    context.company_repository.upsert_company_meta(
        CompanyMeta(
            company_id="c1",
            company_name="Alpha Corp",
            ticker="AAA",
            market="us",
            resolver_version="1",
            updated_at="2025-01-01T00:00:00Z",
        )
    )
    context.company_repository.upsert_company_meta(
        CompanyMeta(
            company_id="c2",
            company_name="Beta Corp",
            ticker="BBB",
            market="us",
            resolver_version="1",
            updated_at="2025-01-01T00:00:00Z",
        )
    )

    result = repair_module._resolve_target_tickers(
        company_repository=context.company_repository,
        target_tickers=None,
    )
    assert result == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# _normalize_document_ids: None 返回、集合规范化、空列表报错
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_document_ids_returns_none_when_input_none() -> None:
    """验证传入 None 时返回 None。"""

    result = repair_module._normalize_document_ids(None)
    assert result is None


@pytest.mark.unit
def test_normalize_document_ids_strips_and_deduplicates() -> None:
    """验证去空格和去重。"""

    result = repair_module._normalize_document_ids([" doc1 ", "doc2", " doc1 "])
    assert result is not None
    assert result == {"doc1", "doc2"}


@pytest.mark.unit
def test_normalize_document_ids_raises_on_empty_after_strip() -> None:
    """验证全空白列表清洗后为空时抛出 ValueError。"""

    with pytest.raises(ValueError, match="target_document_ids 不能为空"):
        repair_module._normalize_document_ids(["  ", ""])


# ---------------------------------------------------------------------------
# _mark_processed_for_batch: 正常调用和 FileNotFoundError 捕获
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mark_processed_calls_repository() -> None:
    """验证正常调用 mark_processed_reprocess_required。"""

    mock_repo = MagicMock()
    repair_module._mark_processed_for_batch(
        processed_repository=mock_repo,
        ticker="TCK",
        document_id="doc_1",
    )
    mock_repo.mark_processed_reprocess_required.assert_called_once_with(
        "TCK", "doc_1", True
    )


@pytest.mark.unit
def test_mark_processed_swallows_file_not_found() -> None:
    """验证 FileNotFoundError 被静默捕获。"""

    mock_repo = MagicMock()
    mock_repo.mark_processed_reprocess_required.side_effect = FileNotFoundError

    # 不应抛出异常
    repair_module._mark_processed_for_batch(
        processed_repository=mock_repo,
        ticker="TCK",
        document_id="doc_missing",
    )


# ---------------------------------------------------------------------------
# SixKPrimaryCandidateAssessment: 属性覆盖
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assessment_total_core_row_count() -> None:
    """验证 total_core_row_count 返回 income + balance_sheet。"""

    assessment = repair_module.SixKPrimaryCandidateAssessment(
        filename="test.htm",
        income_row_count=3,
        balance_sheet_row_count=7,
        filename_priority=0,
    )
    assert assessment.total_core_row_count == 10


@pytest.mark.unit
def test_assessment_has_extractable_core_statements_true() -> None:
    """验证两个报表均有行时 has_extractable_core_statements 为 True。"""

    assessment = repair_module.SixKPrimaryCandidateAssessment(
        filename="test.htm",
        income_row_count=1,
        balance_sheet_row_count=1,
        filename_priority=0,
    )
    assert assessment.has_extractable_core_statements is True


@pytest.mark.unit
def test_assessment_has_extractable_core_statements_false_when_one_missing() -> None:
    """验证任一报表行数为 0 时 has_extractable_core_statements 为 False。"""

    assessment = repair_module.SixKPrimaryCandidateAssessment(
        filename="test.htm",
        income_row_count=5,
        balance_sheet_row_count=0,
        filename_priority=0,
    )
    assert assessment.has_extractable_core_statements is False
