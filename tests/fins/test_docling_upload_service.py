"""DoclingUploadService 测试。"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any, cast

import pytest

from dayu.fins.domain.enums import SourceKind
from dayu.fins.storage import DocumentBlobRepositoryProtocol, SourceDocumentRepositoryProtocol
from dayu.fins.pipelines.docling_upload_service import (
    DoclingUploadService,
    _PendingFileAsset,
    _build_upload_source_fingerprint,
    _can_skip_upload,
    _convert_bytes_with_docling,
    _increment_document_version,
    _normalize_ticker,
    _pick_primary_docling_file,
    _resolve_document_version,
    _resolve_upsert_mode,
    _validate_source_files,
    build_cn_filing_ids,
    build_sec_filing_ids,
    build_material_ids,
    derive_report_kind,
    normalize_cn_fiscal_period,
    resolve_upload_action,
    validate_material_upload_ids,
)
from tests.fins.storage_testkit import FsStorageTestContext, build_fs_storage_test_context


def _convert_docling_stub(raw_data: bytes, stream_name: str) -> dict[str, str]:
    """返回固定 Docling 转换结果。

    Args:
        raw_data: 输入原始字节（占位，不使用）。
        stream_name: 输入流名称（即文件名）。

    Returns:
        固定结构化结果。

    Raises:
        无。
    """

    _ = raw_data
    return {"name": stream_name, "source": "docling"}


def _convert_docling_error(_: bytes, __: str) -> dict[str, object]:
    """抛出转换错误。

    Args:
        _: 输入字节。
        __: 输入流名称。

    Returns:
        无。

    Raises:
        RuntimeError: 固定抛出。
    """

    raise RuntimeError("convert failed")


def _build_service_context(tmp_path: Path) -> tuple[FsStorageTestContext, DoclingUploadService]:
    """构建上传服务测试上下文。

    Args:
        tmp_path: 临时工作区目录。

    Returns:
        `(storage_context, service)` 二元组。

    Raises:
        OSError: 底层仓储初始化失败时抛出。
    """

    context = build_fs_storage_test_context(tmp_path)
    service = DoclingUploadService(
        source_repository=context.source_repository,
        blob_repository=context.blob_repository,
        convert_with_docling=_convert_docling_stub,
    )
    return context, service


def test_execute_upload_create_material_success(tmp_path: Path) -> None:
    """验证 create 上传可写入原文件与 docling 文件。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    context, service = _build_service_context(tmp_path)
    sample_file = tmp_path / "deck.pdf"
    sample_file.write_text("hello", encoding="utf-8")

    result = service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        action="create",
        document_id="mat_demo",
        internal_document_id="mat_demo",
        form_type="MATERIAL_OTHER",
        files=[sample_file],
        overwrite=False,
        meta={"material_name": "Deck", "ingest_method": "upload"},
    )

    assert result.status == "uploaded"
    assert result.document_id == "mat_demo"
    assert len(result.file_events) == 3
    assert result.file_events[0].event_type == "conversion_started"
    meta = context.source_repository.get_source_meta("AAPL", "mat_demo", SourceKind.MATERIAL)
    assert str(meta["primary_document"]).endswith("_docling.json")
    assert len(meta["files"]) == 2


def test_execute_upload_create_material_skip_when_fingerprint_same(tmp_path: Path) -> None:
    """验证相同指纹可跳过重复上传。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    _, service = _build_service_context(tmp_path)
    sample_file = tmp_path / "deck.pdf"
    sample_file.write_text("hello", encoding="utf-8")

    first_result = service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        action="create",
        document_id="mat_demo",
        internal_document_id="mat_demo",
        form_type="MATERIAL_OTHER",
        files=[sample_file],
        overwrite=False,
        meta={"material_name": "Deck", "ingest_method": "upload"},
    )
    second_result = service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        action="create",
        document_id="mat_demo",
        internal_document_id="mat_demo",
        form_type="MATERIAL_OTHER",
        files=[sample_file],
        overwrite=False,
        meta={"material_name": "Deck", "ingest_method": "upload"},
    )

    assert first_result.status == "uploaded"
    assert second_result.status == "skipped"
    assert all(event.event_type == "file_skipped" for event in second_result.file_events)


def test_execute_upload_skip_does_not_run_docling_again_for_same_source_file(tmp_path: Path) -> None:
    """验证相同原始文件再次上传时，会在 convert 前直接跳过。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    convert_calls: list[str] = []

    def _counting_convert(raw_data: bytes, stream_name: str) -> dict[str, str]:
        """记录 Docling convert 调用次数。

        Args:
            raw_data: 输入原始字节（占位，不使用）。
            stream_name: 输入流名称（即文件名）。

        Returns:
            固定转换结果。

        Raises:
            无。
        """

        _ = raw_data
        convert_calls.append(stream_name)
        return {"name": stream_name, "source": "docling"}

    context = build_fs_storage_test_context(tmp_path)
    service = DoclingUploadService(
        source_repository=context.source_repository,
        blob_repository=context.blob_repository,
        convert_with_docling=_counting_convert,
    )
    sample_file = tmp_path / "deck.pdf"
    sample_file.write_text("hello", encoding="utf-8")

    first_result = service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        action="create",
        document_id="mat_demo",
        internal_document_id="mat_demo",
        form_type="MATERIAL_OTHER",
        files=[sample_file],
        overwrite=False,
        meta={"material_name": "Deck", "ingest_method": "upload"},
    )
    second_result = service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        action="create",
        document_id="mat_demo",
        internal_document_id="mat_demo",
        form_type="MATERIAL_OTHER",
        files=[sample_file],
        overwrite=False,
        meta={"material_name": "Deck", "ingest_method": "upload"},
    )

    assert first_result.status == "uploaded"
    assert second_result.status == "skipped"
    assert convert_calls == ["deck.pdf"]


def test_execute_upload_delete_material(tmp_path: Path) -> None:
    """验证 delete 动作可执行逻辑删除。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    context, service = _build_service_context(tmp_path)
    sample_file = tmp_path / "deck.pdf"
    sample_file.write_text("hello", encoding="utf-8")
    service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        action="create",
        document_id="mat_demo",
        internal_document_id="mat_demo",
        form_type="MATERIAL_OTHER",
        files=[sample_file],
        overwrite=False,
        meta={"material_name": "Deck", "ingest_method": "upload"},
    )

    result = service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        action="delete",
        document_id="mat_demo",
        internal_document_id="mat_demo",
        form_type="MATERIAL_OTHER",
        files=[],
        overwrite=False,
        meta={},
    )

    assert result.status == "deleted"
    meta = context.source_repository.get_source_meta("AAPL", "mat_demo", SourceKind.MATERIAL)
    assert bool(meta.get("is_deleted", False)) is True


def test_execute_upload_raises_when_docling_convert_failed(tmp_path: Path) -> None:
    """验证 Docling 转换失败会抛出异常。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    context = build_fs_storage_test_context(tmp_path)
    service = DoclingUploadService(
        source_repository=context.source_repository,
        blob_repository=context.blob_repository,
        convert_with_docling=_convert_docling_error,
    )
    sample_file = tmp_path / "deck.pdf"
    sample_file.write_text("hello", encoding="utf-8")

    with pytest.raises(RuntimeError, match="convert failed"):
        service.execute_upload(
            ticker="AAPL",
            source_kind=SourceKind.MATERIAL,
            action="create",
            document_id="mat_demo",
            internal_document_id="mat_demo",
            form_type="MATERIAL_OTHER",
            files=[sample_file],
            overwrite=False,
            meta={"material_name": "Deck", "ingest_method": "upload"},
        )


def test_init_raises_when_source_repository_none() -> None:
    """验证 source_repository 为空时抛异常。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    with pytest.raises(ValueError, match="source_repository 不能为空"):
        DoclingUploadService(
            source_repository=cast(SourceDocumentRepositoryProtocol, None),
            blob_repository=cast(DocumentBlobRepositoryProtocol, object()),
        )


def test_execute_upload_create_and_update_filing(tmp_path: Path) -> None:
    """验证 filing create + update 分支可执行并更新版本。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    context, service = _build_service_context(tmp_path)
    sample_file = tmp_path / "filing.pdf"
    sample_file.write_text("v1", encoding="utf-8")

    create_result = service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.FILING,
        action="create",
        document_id="fil_demo",
        internal_document_id="int_fil_demo",
        form_type="10-K",
        files=[sample_file],
        overwrite=False,
        meta={"ingest_method": "upload"},
    )
    sample_file.write_text("v2", encoding="utf-8")
    update_result = service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.FILING,
        action="update",
        document_id="fil_demo",
        internal_document_id="int_fil_demo",
        form_type="10-K",
        files=[sample_file],
        overwrite=False,
        meta={"ingest_method": "upload"},
    )

    assert create_result.status == "uploaded"
    assert update_result.status == "uploaded"
    meta = context.source_repository.get_source_meta("AAPL", "fil_demo", SourceKind.FILING)
    assert str(meta.get("document_version", "")).startswith("v")


def test_execute_upload_update_missing_target_raises(tmp_path: Path) -> None:
    """验证 update 目标不存在时抛异常。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    _, service = _build_service_context(tmp_path)
    sample_file = tmp_path / "deck.pdf"
    sample_file.write_text("hello", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Document not found for update"):
        service.execute_upload(
            ticker="AAPL",
            source_kind=SourceKind.MATERIAL,
            action="update",
            document_id="mat_missing",
            internal_document_id="mat_missing",
            form_type="MATERIAL_OTHER",
            files=[sample_file],
            overwrite=False,
            meta={"material_name": "Deck", "ingest_method": "upload"},
        )


def test_resolve_document_id_by_internal_returns_match(tmp_path: Path) -> None:
    """验证 internal_document_id 可反查 document_id。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    _, service = _build_service_context(tmp_path)
    sample_file = tmp_path / "deck.pdf"
    sample_file.write_text("hello", encoding="utf-8")
    service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        action="create",
        document_id="mat_abc",
        internal_document_id="mat_abc",
        form_type="MATERIAL_OTHER",
        files=[sample_file],
        overwrite=False,
        meta={"material_name": "Deck", "ingest_method": "upload"},
    )

    matched = service.resolve_document_id_by_internal(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        internal_document_id="mat_abc",
    )
    missing = service.resolve_document_id_by_internal(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        internal_document_id="mat_not_exists",
    )

    assert matched == "mat_abc"
    assert missing is None


def test_pick_primary_docling_file_returns_none_when_missing() -> None:
    """验证缺少 docling 文件时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    value = _pick_primary_docling_file([{"name": "a.pdf"}])
    assert value is None


def test_validate_source_files_raises_on_invalid_input(tmp_path: Path) -> None:
    """验证文件校验在空列表/不存在/不支持后缀时抛异常。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    missing = tmp_path / "missing.pdf"
    bad_suffix = tmp_path / "data.zip"
    bad_suffix.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="上传文件不能为空"):
        _validate_source_files([])
    with pytest.raises(FileNotFoundError, match="上传文件不存在"):
        _validate_source_files([missing])
    with pytest.raises(ValueError, match="不支持的文件类型"):
        _validate_source_files([bad_suffix])


def test_resolve_helpers_cover_edge_branches() -> None:
    """验证版本/模式/ticker/财期等辅助函数分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _increment_document_version("v1") == "v2"
    assert _increment_document_version("abc") == "v2"
    assert _increment_document_version("vX") == "v2"
    assert _resolve_document_version(None, "fp") == "v1"
    assert _resolve_document_version({"document_version": "v3", "source_fingerprint": "x"}, "y") == "v4"
    assert _resolve_document_version({"document_version": "v3", "source_fingerprint": "x"}, "x") == "v3"
    assert _resolve_upsert_mode("create", None, False) == "create"
    assert _resolve_upsert_mode("create", {"x": 1}, True) == "update"
    assert _resolve_upsert_mode("update", {"x": 1}, False) == "update"
    assert _resolve_upsert_mode("update", None, True) == "create"
    with pytest.raises(FileNotFoundError, match="更新目标不存在"):
        _resolve_upsert_mode("update", None, False)
    with pytest.raises(FileExistsError, match="创建目标已存在"):
        _resolve_upsert_mode("create", {"x": 1}, False)
    assert _can_skip_upload({"ingest_complete": True, "source_fingerprint": "fp"}, "fp", False) is True
    assert _can_skip_upload(None, "fp", False) is False
    assert _can_skip_upload({"ingest_complete": False, "source_fingerprint": "fp"}, "fp", False) is False
    assert _can_skip_upload({"ingest_complete": True, "source_fingerprint": "fp"}, "fp", True) is False
    with pytest.raises(ValueError, match="ticker 不能为空"):
        _normalize_ticker("   ")
    assert normalize_cn_fiscal_period("q1") == "Q1"
    with pytest.raises(ValueError, match="不支持的 fiscal_period"):
        normalize_cn_fiscal_period("M1")
    assert derive_report_kind("FY") == "annual"
    assert derive_report_kind("H1") == "semi_annual"
    assert derive_report_kind("Q2") == "quarterly"


def test_build_material_and_filing_ids() -> None:
    """验证材料 ID、CN filing ID 与 SEC filing ID 的生成逻辑。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    create_doc, create_internal = build_material_ids(
        form_type="MATERIAL_OTHER",
        material_name="Deck",
        fiscal_year=None,
        fiscal_period=None,
    )
    assert create_doc.startswith("mat_")
    assert create_doc == create_internal
    with pytest.raises(ValueError, match="form_type 不能为空"):
        build_material_ids(
            form_type=" ",
            material_name="Deck",
            fiscal_year=None,
            fiscal_period=None,
        )
    with pytest.raises(ValueError, match="material_name 不能为空"):
        build_material_ids(
            form_type="MATERIAL_OTHER",
            material_name=" ",
            fiscal_year=None,
            fiscal_period=None,
        )
    upd_doc, upd_internal = build_material_ids(
        form_type="MATERIAL_OTHER",
        material_name="Deck",
        fiscal_year=2025,
        fiscal_period="q1",
    )
    assert upd_doc.startswith("mat_")
    assert upd_doc == upd_internal
    assert upd_doc != create_doc

    doc_id, internal_id = build_cn_filing_ids(
        ticker="AAPL",
        form_type="q1",
        fiscal_year=2025,
        fiscal_period="q1",
        amended=False,
    )
    assert internal_id.startswith("cn_")
    assert doc_id == f"fil_{internal_id}"
    sec_doc_id, sec_internal_id = build_sec_filing_ids(
        ticker="AAPL",
        fiscal_year=2025,
        fiscal_period="q1",
        amended=False,
    )
    assert sec_internal_id.startswith("sec_")
    assert sec_doc_id == f"fil_{sec_internal_id}"
    assert sec_doc_id != doc_id
    with pytest.raises(ValueError, match="ticker 不能为空"):
        build_cn_filing_ids(
            ticker="  ",
            form_type="Q1",
            fiscal_year=2025,
            fiscal_period="Q1",
            amended=False,
        )
    with pytest.raises(ValueError, match="ticker 不能为空"):
        build_sec_filing_ids(
            ticker="",
            fiscal_year=2025,
            fiscal_period="Q1",
            amended=False,
        )
    with pytest.raises(ValueError, match="form_type 不能为空"):
        build_cn_filing_ids(
            ticker="AAPL",
            form_type=" ",
            fiscal_year=2025,
            fiscal_period="Q1",
            amended=False,
        )
    with pytest.raises(ValueError, match="fiscal_period 不能为空"):
        build_sec_filing_ids(
            ticker="AAPL",
            fiscal_year=2025,
            fiscal_period=" ",
            amended=False,
        )


def test_resolve_upload_action_and_validate_material_ids() -> None:
    """验证自动动作解析与 material 显式 ID 一致性校验。"""

    assert resolve_upload_action(None, None) == "create"
    assert resolve_upload_action(None, {"document_id": "mat_x"}) == "update"
    assert resolve_upload_action("delete", {"document_id": "mat_x"}) == "delete"

    with pytest.raises(ValueError, match="显式 --document-id"):
        validate_material_upload_ids(
            stable_document_id="mat_stable",
            stable_internal_document_id="mat_stable",
            document_id="mat_other",
            internal_document_id=None,
        )
    with pytest.raises(ValueError, match="显式 --internal-document-id"):
        validate_material_upload_ids(
            stable_document_id="mat_stable",
            stable_internal_document_id="mat_stable",
            document_id=None,
            internal_document_id="mat_other",
        )
    resolved_document_id, resolved_internal_document_id = validate_material_upload_ids(
        stable_document_id="mat_stable",
        stable_internal_document_id="mat_stable",
        document_id="mat_stable",
        internal_document_id="mat_stable",
    )
    assert resolved_document_id == "mat_stable"
    assert resolved_internal_document_id == "mat_stable"


def test_build_upload_source_fingerprint_is_stable_for_order() -> None:
    """验证指纹对文件顺序不敏感。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    asset_a = _PendingFileAsset(
        name="a.txt",
        data=b"a",
        content_type="text/plain",
        sha256="sha_a",
        size=1,
        source="original",
    )
    asset_b = _PendingFileAsset(
        name="b_docling.json",
        data=b"b",
        content_type="application/json",
        sha256="sha_b",
        size=1,
        source="docling",
    )
    fp1 = _build_upload_source_fingerprint([asset_a, asset_b])
    fp2 = _build_upload_source_fingerprint([asset_b, asset_a])
    assert fp1 == fp2


def test_convert_bytes_with_docling_import_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Docling 未安装时抛出明确异常。

    Args:
        tmp_path: 临时目录（占位，本用例不使用）。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    _ = tmp_path
    original_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """对 docling import 注入 ImportError。

        Args:
            name: 模块名。
            *args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            原始 import 结果。

        Raises:
            ImportError: 模拟 docling 缺失。
        """

        if name.startswith("docling"):
            raise ImportError("docling missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    from dayu.docling_runtime import DoclingRuntimeInitializationError

    with pytest.raises(DoclingRuntimeInitializationError, match="Docling 未安装"):
        _convert_bytes_with_docling(b"x", "sample.pdf")


def test_convert_bytes_with_docling_conversion_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 Docling convert 抛错会被包装为 RuntimeError。

    Args:
        tmp_path: 临时目录（占位，本用例不使用）。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    _ = tmp_path

    def _raise_convert_failure(*args: Any, **kwargs: Any) -> Any:
        """模拟统一 Docling 运行时在转换阶段抛错。

        Args:
            *args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            无。

        Raises:
            RuntimeError: 固定抛出。
        """

        _ = (args, kwargs)
        raise RuntimeError("convert boom")

    monkeypatch.setattr(
        "dayu.fins.docling_export.convert_pdf_bytes_with_docling",
        _raise_convert_failure,
    )

    with pytest.raises(RuntimeError, match="Docling 转换失败"):
        _convert_bytes_with_docling(b"x", "sample.pdf")
