"""腾讯云 COS 文件托管辅助模块。

提供将本地文件上传到腾讯云对象存储（COS）并获取公开 URL 的能力，
主要用于 MinerU 云 API 需要通过 URL 访问 PDF 文件的场景。

设计目标：

- 临时文件托管：MinerU API 需要公开可访问的 URL，本地文件需先上传到 COS。
- 自动清理：提供删除接口，解析完成后清理临时文件。
- 配置集中：所有 COS 配置通过环境变量注入。
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from dayu.log import Log

if TYPE_CHECKING:
    pass

_MODULE = __name__

# ---------------------------------------------------------------------------
# 配置（环境变量）
# ---------------------------------------------------------------------------

_COS_SECRET_ID: str = os.environ.get("DAYU_COS_SECRET_ID", "")
_COS_SECRET_KEY: str = os.environ.get("DAYU_COS_SECRET_KEY", "")
_COS_BUCKET: str = os.environ.get("DAYU_COS_BUCKET", "")
_COS_REGION: str = os.environ.get("DAYU_COS_REGION", "ap-guangzhou")


class COSUploadError(RuntimeError):
    """COS 上传失败异常。"""


def _check_cos_config() -> None:
    """检查 COS 配置是否完整。

    Raises:
        COSUploadError: 配置缺失时抛出。
    """
    missing: list[str] = []
    if not _COS_SECRET_ID:
        missing.append("DAYU_COS_SECRET_ID")
    if not _COS_SECRET_KEY:
        missing.append("DAYU_COS_SECRET_KEY")
    if not _COS_BUCKET:
        missing.append("DAYU_COS_BUCKET")
    if missing:
        raise COSUploadError(
            f"COS 配置缺失: {', '.join(missing)}。"
            f"请设置对应环境变量。"
        )


def upload_pdf_to_cos(
    pdf_bytes: bytes,
    *,
    filename: str = "filing.pdf",
) -> str:
    """将 PDF 字节流上传到腾讯云 COS 并返回公开 URL。

    每次调用都创建新的 CosS3Client，因为 CosS3Client 不是线程安全的
    （内部有连接池和签名逻辑，多线程共享会导致签名错误）。

    Args:
        pdf_bytes: PDF 原始字节。
        filename: COS 对象键中的文件名。

    Returns:
        上传后的公开可访问 URL。

    Raises:
        COSUploadError: 配置缺失或上传失败时抛出。
    """
    _check_cos_config()

    try:
        from qcloud_cos import CosConfig, CosS3Client
    except ImportError as exc:
        raise COSUploadError(
            "cos-python-sdk-v5 未安装，无法上传到 COS。"
            "请执行 pip install cos-python-sdk-v5"
        ) from exc

    config = CosConfig(
        Region=_COS_REGION,
        SecretId=_COS_SECRET_ID,
        SecretKey=_COS_SECRET_KEY,
    )
    client = CosS3Client(config)

    # 使用时间戳避免文件名冲突
    timestamp = int(time.time())
    cos_key = f"dayu-mineru-temp/{timestamp}_{filename}"

    Log.info(
        f"上传 PDF 到 COS: bucket={_COS_BUCKET}, key={cos_key}, "
        f"size={len(pdf_bytes)} bytes",
        module=_MODULE,
    )

    try:
        client.put_object(
            Bucket=_COS_BUCKET,
            Body=pdf_bytes,
            Key=cos_key,
            ContentType="application/pdf",
        )
    except Exception as exc:
        raise COSUploadError(f"COS 上传失败: {exc}") from exc

    # 构造公开 URL
    url = f"https://{_COS_BUCKET}.cos.{_COS_REGION}.myqcloud.com/{cos_key}"
    Log.info(f"COS 上传成功: {url}", module=_MODULE)
    return url


def delete_from_cos(cos_key: str) -> None:
    """从 COS 删除指定对象。

    Args:
        cos_key: COS 对象键。
    """
    if not _COS_BUCKET:
        return
    try:
        from qcloud_cos import CosConfig, CosS3Client

        config = CosConfig(
            Region=_COS_REGION,
            SecretId=_COS_SECRET_ID,
            SecretKey=_COS_SECRET_KEY,
        )
        client = CosS3Client(config)
        client.delete_object(Bucket=_COS_BUCKET, Key=cos_key)
        Log.debug(f"COS 对象已删除: {cos_key}", module=_MODULE)
    except Exception as exc:
        Log.warn(f"COS 对象删除失败: {cos_key}, error={exc}", module=_MODULE)


def extract_cos_key_from_url(url: str) -> str:
    """从 COS 公开 URL 中提取对象键。

    Args:
        url: COS 公开 URL。

    Returns:
        COS 对象键。
    """
    # URL 格式: https://bucket.cos.region.myqcloud.com/key
    parts = url.split("/", 3)
    if len(parts) >= 4:
        return parts[3]
    return url


__all__ = [
    "COSUploadError",
    "upload_pdf_to_cos",
    "delete_from_cos",
    "extract_cos_key_from_url",
]
