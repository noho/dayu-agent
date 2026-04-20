"""BsTenQFormProcessor 构造与 marker 代理测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from dayu.fins.processors import bs_ten_q_processor as module


class _SourceStub:
    """最小 Source 桩。"""

    uri = "local://test.html"
    media_type = "text/html"
    content_length = None
    etag = None

    def open(self) -> Any:
        """打开资源。

        Args:
            无。

        Returns:
            无。

        Raises:
            OSError: 固定抛错。
        """

        raise OSError("not used")

    def materialize(self, suffix: Optional[str] = None) -> Path:
        """返回物化路径。

        Args:
            suffix: 可选后缀。

        Returns:
            固定路径。

        Raises:
            无。
        """

        del suffix
        return Path("/tmp/placeholder.html")


@pytest.mark.unit
def test_bs_ten_q_init_and_build_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖 BsTenQFormProcessor 构造与 marker 构建代理。"""

    captured: dict[str, Any] = {}

    def _fake_init(self: Any, *, source: Any, form_type: Optional[str] = None, media_type: Optional[str] = None) -> None:
        """替代父类构造函数。

        Args:
            self: 实例对象。
            source: 来源对象。
            form_type: 表单类型。
            media_type: 媒体类型。

        Returns:
            无。

        Raises:
            无。
        """

        captured["source"] = source
        captured["form_type"] = form_type
        captured["media_type"] = media_type
        self._virtual_sections = []

    monkeypatch.setattr(module._BaseBsReportFormProcessor, "__init__", _fake_init)
    monkeypatch.setattr(module, "_build_ten_q_markers", lambda text: [(456, "Item 1")])
    monkeypatch.setattr(
        module.BsTenQFormProcessor,
        "_collect_document_text",
        lambda self: "document text",
    )
    monkeypatch.setattr(
        module.BsTenQFormProcessor,
        "_postprocess_virtual_sections",
        lambda self, full_text: captured.setdefault("postprocess_text", full_text),
    )

    processor = module.BsTenQFormProcessor(_SourceStub(), form_type="10-Q", media_type="text/html")

    assert captured["form_type"] == "10-Q"
    assert captured["media_type"] == "text/html"
    assert captured["postprocess_text"] == "document text"
    assert processor._build_markers("dummy") == [(456, "Item 1")]
