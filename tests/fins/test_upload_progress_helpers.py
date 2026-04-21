"""upload_progress_helpers 模块单元测试。"""

from __future__ import annotations

import pytest
from typing import cast

from dayu.fins.pipelines.docling_upload_service import UploadFileEventPayload
from dayu.fins.pipelines.docling_upload_service import UploadFileEventType
from dayu.fins.pipelines.upload_filing_events import UploadFilingEventType
from dayu.fins.pipelines.upload_material_events import UploadMaterialEventType
from dayu.fins.pipelines.upload_progress_helpers import (
    map_upload_file_event_to_filing_event_type,
    map_upload_file_event_to_material_event_type,
)


def test_map_upload_file_event_to_filing_event_type_covers_supported_events() -> None:
    """验证财报上传文件事件映射覆盖当前受支持的全部事件类型。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert (
        map_upload_file_event_to_filing_event_type(
            UploadFileEventPayload(
                event_type="conversion_started",
                name="report.pdf",
                payload={"message": "正在 convert"},
            )
        )
        == UploadFilingEventType.CONVERSION_STARTED
    )
    assert (
        map_upload_file_event_to_filing_event_type(
            UploadFileEventPayload(
                event_type="file_uploaded",
                name="report.pdf",
                payload={"source": "original"},
            )
        )
        == UploadFilingEventType.FILE_UPLOADED
    )
    assert (
        map_upload_file_event_to_filing_event_type(
            UploadFileEventPayload(
                event_type="file_skipped",
                name="report.pdf",
                payload={"reason": "already_uploaded"},
            )
        )
        == UploadFilingEventType.FILE_SKIPPED
    )
    assert (
        map_upload_file_event_to_filing_event_type(
            UploadFileEventPayload(
                event_type="file_failed",
                name="report.pdf",
                payload={"error": "boom"},
            )
        )
        == UploadFilingEventType.FILE_FAILED
    )


def test_map_upload_file_event_to_material_event_type_covers_supported_events() -> None:
    """验证材料上传文件事件映射覆盖当前受支持的全部事件类型。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert (
        map_upload_file_event_to_material_event_type(
            UploadFileEventPayload(
                event_type="conversion_started",
                name="deck.pdf",
                payload={"message": "正在 convert"},
            )
        )
        == UploadMaterialEventType.CONVERSION_STARTED
    )
    assert (
        map_upload_file_event_to_material_event_type(
            UploadFileEventPayload(
                event_type="file_uploaded",
                name="deck.pdf",
                payload={"source": "original"},
            )
        )
        == UploadMaterialEventType.FILE_UPLOADED
    )
    assert (
        map_upload_file_event_to_material_event_type(
            UploadFileEventPayload(
                event_type="file_skipped",
                name="deck.pdf",
                payload={"reason": "already_uploaded"},
            )
        )
        == UploadMaterialEventType.FILE_SKIPPED
    )
    assert (
        map_upload_file_event_to_material_event_type(
            UploadFileEventPayload(
                event_type="file_failed",
                name="deck.pdf",
                payload={"error": "boom"},
            )
        )
        == UploadMaterialEventType.FILE_FAILED
    )


def test_map_upload_file_event_to_filing_event_type_rejects_unknown_event() -> None:
    """验证财报上传文件事件映射遇到未知类型会显式失败。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    with pytest.raises(ValueError, match="未知上传文件事件类型"):
        map_upload_file_event_to_filing_event_type(
            cast(
                UploadFileEventPayload,
                UploadFileEventPayload(
                    event_type="conversion_started",
                    name="report.pdf",
                    payload={},
                ),
            ).__class__(
                event_type=cast(UploadFileEventType, "unknown"),
                name="report.pdf",
                payload={},
            ),
        )


def test_map_upload_file_event_to_material_event_type_rejects_unknown_event() -> None:
    """验证材料上传文件事件映射遇到未知类型会显式失败。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    with pytest.raises(ValueError, match="未知上传文件事件类型"):
        map_upload_file_event_to_material_event_type(
            cast(
                UploadFileEventPayload,
                UploadFileEventPayload(
                    event_type="conversion_started",
                    name="deck.pdf",
                    payload={},
                ),
            ).__class__(
                event_type=cast(UploadFileEventType, "unknown"),
                name="deck.pdf",
                payload={},
            ),
        )
