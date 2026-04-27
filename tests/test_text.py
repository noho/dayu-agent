"""dayu.text 跨层文本工具测试。"""

from __future__ import annotations

import pytest

from dayu.text import strip_markdown_fence


@pytest.mark.unit
class TestStripMarkdownFence:
    """strip_markdown_fence 测试集。"""

    def test_plain_text_passthrough(self) -> None:
        """无围栏文本原样返回。"""
        assert strip_markdown_fence("hello world") == "hello world"

    def test_empty_input(self) -> None:
        """空输入返回空字符串。"""
        assert strip_markdown_fence("") == ""
        assert strip_markdown_fence("   ") == ""

    def test_none_input(self) -> None:
        """None 输入返回空字符串。"""
        assert strip_markdown_fence(None) == ""

    def test_basic_fence(self) -> None:
        """裸 ``` 围栏正确剥离。"""
        fenced = "```\ncontent here\n```"
        assert strip_markdown_fence(fenced) == "content here"

    def test_markdown_language_tag(self) -> None:
        """```markdown 语言标签围栏正确剥离。"""
        fenced = "```markdown\n# Title\n\nBody text\n```"
        assert strip_markdown_fence(fenced) == "# Title\n\nBody text"

    def test_json_language_tag(self) -> None:
        """```json 语言标签围栏正确剥离。"""
        fenced = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fence(fenced) == '{"key": "value"}'

    def test_incomplete_fence_two_lines(self) -> None:
        """不足 3 行的围栏不剥离。"""
        fenced = "```\ncontent only"
        assert strip_markdown_fence(fenced) == "```\ncontent only"

    def test_multiline_content(self) -> None:
        """多行内容围栏正确剥离。"""
        fenced = "```markdown\nline1\nline2\nline3\n```"
        assert strip_markdown_fence(fenced) == "line1\nline2\nline3"

    def test_fence_with_surrounding_whitespace(self) -> None:
        """首尾空白不影响围栏检测。"""
        fenced = "  ```\ncontent\n```  "
        assert strip_markdown_fence(fenced) == "content"
