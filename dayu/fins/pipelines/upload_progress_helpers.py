"""上传进度事件辅助函数。

该模块负责 upload stream 的文件级事件类型映射。
"""

from __future__ import annotations

from .docling_upload_service import UploadFileEventPayload
from .upload_filing_events import UploadFilingEventType
from .upload_material_events import UploadMaterialEventType

_UPLOAD_FILE_TO_FILING_EVENT_TYPE: dict[str, UploadFilingEventType] = {
    "conversion_started": UploadFilingEventType.CONVERSION_STARTED,
    "file_uploaded": UploadFilingEventType.FILE_UPLOADED,
    "file_skipped": UploadFilingEventType.FILE_SKIPPED,
    "file_failed": UploadFilingEventType.FILE_FAILED,
}

_UPLOAD_FILE_TO_MATERIAL_EVENT_TYPE: dict[str, UploadMaterialEventType] = {
    "conversion_started": UploadMaterialEventType.CONVERSION_STARTED,
    "file_uploaded": UploadMaterialEventType.FILE_UPLOADED,
    "file_skipped": UploadMaterialEventType.FILE_SKIPPED,
    "file_failed": UploadMaterialEventType.FILE_FAILED,
}


def map_upload_file_event_to_filing_event_type(
    file_event: UploadFileEventPayload,
) -> UploadFilingEventType:
    """将上传文件事件映射为财报上传事件类型。

    Args:
        file_event: 上传服务返回的文件级事件。

    Returns:
        `upload_filing_stream` 对外事件类型。

    Raises:
        ValueError: 事件类型未知时抛出。
    """

    event_type = _UPLOAD_FILE_TO_FILING_EVENT_TYPE.get(file_event.event_type)
    if event_type is None:
        raise ValueError(f"未知上传文件事件类型: {file_event.event_type}")
    return event_type


def map_upload_file_event_to_material_event_type(
    file_event: UploadFileEventPayload,
) -> UploadMaterialEventType:
    """将上传文件事件映射为材料上传事件类型。

    Args:
        file_event: 上传服务返回的文件级事件。

    Returns:
        `upload_material_stream` 对外事件类型。

    Raises:
        ValueError: 事件类型未知时抛出。
    """

    event_type = _UPLOAD_FILE_TO_MATERIAL_EVENT_TYPE.get(file_event.event_type)
    if event_type is None:
        raise ValueError(f"未知上传文件事件类型: {file_event.event_type}")
    return event_type
