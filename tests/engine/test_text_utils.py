"""text_utils 中 ref / placeholder contract 测试。"""

from __future__ import annotations

import pytest

from dayu.engine.processors import text_utils


@pytest.mark.unit
def test_format_table_placeholder_and_extract_refs() -> None:
    """验证表格占位符格式化与提取逻辑。"""

    placeholder = text_utils.format_table_placeholder("t_0001")

    assert placeholder == "[[t_0001]]"
    assert text_utils.extract_table_refs_from_text(
        "before [[t_0001]] middle [[t_0002]] end [[t_0001]]"
    ) == ["t_0001", "t_0002"]


@pytest.mark.unit
def test_append_missing_table_placeholders() -> None:
    """验证缺失占位符补齐逻辑。"""

    assert text_utils.append_missing_table_placeholders("", ["t_0001"]) == "[[t_0001]]"
    assert (
        text_utils.append_missing_table_placeholders("body", ["t_0001", "t_0002"])
        == "body\n[[t_0001]]\n[[t_0002]]"
    )
    assert text_utils.append_missing_table_placeholders("[[t_0001]]", ["t_0001"]) == "[[t_0001]]"


@pytest.mark.unit
def test_ref_patterns_and_validation() -> None:
    """验证 section/table ref 正则与空输入校验。"""

    assert text_utils.SECTION_REF_PATTERN.fullmatch("s_0001") is not None
    assert text_utils.SECTION_REF_PATTERN.fullmatch("sec_0001") is None
    assert text_utils.TABLE_REF_PATTERN.fullmatch("t_0001") is not None
    assert text_utils.TABLE_REF_PATTERN.fullmatch("tbl_0001") is None
    assert text_utils.TABLE_PLACEHOLDER_PATTERN.findall("[[t_0001]] xx [[t_0002]]") == ["t_0001", "t_0002"]

    with pytest.raises(ValueError, match="table_ref"):
        text_utils.format_table_placeholder("  ")
