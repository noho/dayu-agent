"""DoclingProcessor 真实 PDF 集成测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dayu.engine.processors.docling_processor import DoclingProcessor
from dayu.fins.pipelines.docling_upload_service import _convert_file_with_docling
from dayu.fins.storage.local_file_source import LocalFileSource

pytestmark = pytest.mark.integration


def _fixture_pdf_path() -> Path:
    """返回真实 Docling PDF fixture 路径。

    Args:
        无。

    Returns:
        真实 PDF fixture 文件路径。

    Raises:
        FileNotFoundError: fixture 缺失时抛出。
    """

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "docling"
        / "dayu_docling_integration_fixture.pdf"
    )
    if not fixture_path.exists():
        raise FileNotFoundError(f"Docling PDF fixture 不存在: {fixture_path}")
    return fixture_path


@pytest.fixture(scope="module")
def real_docling_json_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """基于真实 PDF fixture 生成 Docling JSON。

    Args:
        tmp_path_factory: pytest 模块级临时目录工厂。

    Returns:
        生成后的 `*_docling.json` 路径。

    Raises:
        OSError: 写文件失败时抛出。
        RuntimeError: Docling 转换失败时抛出。
    """

    output_dir = tmp_path_factory.mktemp("docling_processor_integration")
    docling_json_path = output_dir / "dayu_docling_integration_fixture_docling.json"
    docling_payload = _convert_file_with_docling(_fixture_pdf_path())
    docling_json_path.write_text(
        json.dumps(docling_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return docling_json_path


@pytest.fixture(scope="module")
def processor_source(real_docling_json_path: Path) -> LocalFileSource:
    """构建真实 Docling JSON 对应的本地 Source。

    Args:
        real_docling_json_path: 真实 `*_docling.json` 路径。

    Returns:
        指向真实 fixture 的 `LocalFileSource`。

    Raises:
        无。
    """

    return LocalFileSource(
        path=real_docling_json_path,
        uri=real_docling_json_path.name,
        media_type="application/json",
    )


@pytest.fixture(scope="module")
def processor(processor_source: LocalFileSource) -> DoclingProcessor:
    """构建真实 Docling JSON 对应的处理器实例。

    Args:
        processor_source: 指向真实 fixture 的 `LocalFileSource`。

    Returns:
        `DoclingProcessor` 实例。

    Raises:
        ValueError: 文件不合法时抛出。
        RuntimeError: Docling 处理器加载失败时抛出。
    """

    return DoclingProcessor(processor_source)


def test_docling_processor_reads_real_pdf_table(
    processor: DoclingProcessor,
    processor_source: LocalFileSource,
) -> None:
    """验证真实 PDF 经 Docling 后可读取稳定表格结构。

    Args:
        processor: 真实 `DoclingProcessor` 实例。
        processor_source: 指向真实 fixture 的 `LocalFileSource`。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert DoclingProcessor.supports(processor_source) is True

    tables = processor.list_tables()
    assert len(tables) == 1

    table_summary = tables[0]
    assert table_summary["section_ref"] is not None
    assert (
        table_summary["context_before"]
        == "The table below is the canonical assertion target for integration tests."
    )

    table_content = processor.read_table(table_summary["table_ref"])
    assert table_content["data_format"] == "records"
    assert table_content["columns"] == ["Metric", "FY2024", "FY2025"]

    rows = table_content["data"]
    assert isinstance(rows, list)
    assert len(rows) == 3

    first_row = rows[0]
    assert isinstance(first_row, dict)
    assert first_row["Metric"] == "Revenue"
    assert first_row["FY2024"] == "120"
    assert first_row["FY2025"] == "135"

    second_row = rows[1]
    assert isinstance(second_row, dict)
    assert second_row["Metric"] == "Operating Margin"
    assert second_row["FY2024"] == "18%"
    assert second_row["FY2025"] == "21%"


def test_docling_processor_sections_keep_table_reference(processor: DoclingProcessor) -> None:
    """验证真实章节内容仍能关联表格引用，不接受静默退化。

    Args:
        processor: 真实 `DoclingProcessor` 实例。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    sections = processor.list_sections()
    titles = [section.get("title") for section in sections]
    assert "Dayu Docling Integration Fixture" in titles
    assert "Financial Summary" in titles

    summary_section = next(
        section for section in sections if section.get("title") == "Financial Summary"
    )
    section_content = processor.read_section(summary_section["ref"])

    assert section_content["tables"]
    table_ref = section_content["tables"][0]
    assert f"[[{table_ref}]]" in section_content["content"]
    assert "canonical assertion target" in section_content["content"]
