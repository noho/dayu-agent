"""MinerU 云 API v4 运行时。

本模块提供将 PDF 字节流通过 MinerU 云 API v4 转换为统一中间格式
``ConvertedDocument`` 的完整流程，包含：

1. PDF 上传到腾讯云 COS，获取公开 URL；
2. 按 page_ranges 分批（服务端分页，≤200 页/批）；
3. 异步并发提交 + 并发轮询（指数退避 + jitter）；
4. 结果合并（简化版：直接 append）；
5. 配额检查（本地计数器 + 文件持久化）；
6. 任一环节失败自动回退 Docling。

调用链：``parse_pdf_bytes_with_mineru`` → ``_poll_all_tasks`` → ``_poll_one``。

回退链（五层）：

1. MinerU 云 API v4 单次（≤200 页且配额充足）
2. MinerU 云 API v4 分批（>200 页，page_ranges 分段）
3. MinerU 本地 CLI（``mineru`` 命令可用）
4. MinerU 本地 Python API（``magic_pdf`` 已安装）
5. Docling（终极兜底）

MinerU 云 API v4 规格（2026-05-06 确认）：

- Base URL: ``https://mineru.net``
- 提交: ``POST /api/v4/extract/task`` （JSON body，需 url 字段）
- 查询: ``GET /api/v4/extract/task/{task_id}``
- 认证: ``Authorization: Bearer <token>``
- 支持 ``page_ranges`` 参数实现服务端分页
- 不支持直接文件上传，需提供公开可访问的 URL
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import httpx

from dayu.document_protocol import (
    ConvertedDocument,
    DocumentBackend,
    DocumentSection,
    DocumentTable,
    DocumentImage,
)
from dayu.log import Log
from dayu.quota_tracker import QuotaTracker, QuotaExhaustedError

if TYPE_CHECKING:
    pass

_MODULE = __name__

# ---------------------------------------------------------------------------
# 配置常量（可通过环境变量覆盖）
# ---------------------------------------------------------------------------

#: MinerU 云 API v4 基础地址。
_MINERU_API_BASE: str = "https://mineru.net"

#: MinerU 云 API Token。
_MINERU_API_TOKEN: str = os.environ.get("DAYU_MINERU_TOKEN", "")

#: 每批最多页数（MinerU 云 API 限制 200 页）。
_MAX_PAGES_PER_BATCH: int = int(os.environ.get("DAYU_MINERU_CHUNK_SIZE", "200"))

#: 单批轮询超时（秒）。
_TIMEOUT: float = float(os.environ.get("DAYU_MINERU_TIMEOUT", "1800"))

#: 轮询初始间隔（秒）。
_POLL_INITIAL: float = 2.0

#: 轮询最大间隔（秒）。
_POLL_MAX: float = 30.0

#: 每日配额上限（页）。
_QUOTA_DAILY_LIMIT: int = int(os.environ.get("DAYU_QUOTA_DAILY_LIMIT", "5000"))

#: 模型版本。
_MODEL_VERSION: str = os.environ.get("DAYU_MINERU_MODEL_VERSION", "vlm")


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class MinerUAPIError(RuntimeError):
    """MinerU 云 API 调用异常。"""


class MinerUTimeoutError(MinerUAPIError):
    """MinerU 云 API 轮询超时。"""


class MinerUTaskFailedError(MinerUAPIError):
    """MinerU 云 API 任务失败。"""


# ---------------------------------------------------------------------------
# 任务状态机
# ---------------------------------------------------------------------------


class TaskState(str, Enum):
    """MinerU 任务状态。"""

    SUBMITTED = "submitted"
    POLLING = "polling"
    DONE = "done"
    FAILED = "failed"


@dataclass
class _TaskDescriptor:
    """单个 API 任务的描述。"""

    task_id: str
    page_range: str
    state: TaskState = TaskState.SUBMITTED
    result: dict[str, object] = field(default_factory=dict)
    error: str = ""


# ---------------------------------------------------------------------------
# 模块级配额跟踪器（单例）
# ---------------------------------------------------------------------------

_quota_tracker: QuotaTracker | None = None


def _get_quota_tracker() -> QuotaTracker:
    """获取或创建模块级配额跟踪器单例。

    Returns:
        ``QuotaTracker`` 实例。
    """
    global _quota_tracker  # noqa: PLW0603
    if _quota_tracker is None:
        _quota_tracker = QuotaTracker(daily_limit=_QUOTA_DAILY_LIMIT)
    return _quota_tracker


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------


def _build_http_client() -> httpx.AsyncClient:
    """构造 MinerU 云 API v4 专用的异步 HTTP 客户端。

    Returns:
        配置好认证头和超时的 ``httpx.AsyncClient``。
    """
    return httpx.AsyncClient(
        base_url=_MINERU_API_BASE,
        headers={
            "Authorization": f"Bearer {_MINERU_API_TOKEN}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(60.0),
    )


# ---------------------------------------------------------------------------
# Phase 1: 提交任务
# ---------------------------------------------------------------------------


async def _submit_task(
    client: httpx.AsyncClient,
    pdf_url: str,
    page_range: str,
) -> str:
    """向 MinerU 云 API v4 提交一个提取任务。

    Args:
        client: 异步 HTTP 客户端。
        pdf_url: PDF 文件的公开可访问 URL。
        page_range: 页码范围字符串，如 ``"1-200"``。

    Returns:
        MinerU 返回的 task_id。

    Raises:
        MinerUAPIError: API 调用失败或响应格式异常。
    """
    payload: dict[str, object] = {
        "url": pdf_url,
        "is_ocr": True,
        "enable_formula": True,
        "enable_table": True,
        "language": "ch",
        "model_version": _MODEL_VERSION,
        "page_ranges": page_range,
    }

    try:
        response = await client.post("/api/v4/extract/task", json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise MinerUAPIError("MinerU API 限流 (429)") from exc
        raise MinerUAPIError(
            f"MinerU 提交失败: HTTP {exc.response.status_code}, "
            f"body={exc.response.text[:200]}"
        ) from exc
    except httpx.RequestError as exc:
        raise MinerUAPIError(f"MinerU 提交请求失败: {exc}") from exc

    data = response.json()

    # v4 响应格式: {"code": 0, "msg": "ok", "data": {"task_id": "..."}}
    if not isinstance(data, dict):
        raise MinerUAPIError(f"MinerU 响应格式异常: {type(data).__name__}")

    code = data.get("code", -1)
    if code != 0:
        raise MinerUAPIError(f"MinerU 业务错误: code={code}, msg={data.get('msg')}")

    task_data = data.get("data")
    if not isinstance(task_data, dict):
        raise MinerUAPIError(f"MinerU 响应 data 格式异常: {type(task_data).__name__}")

    task_id = task_data.get("task_id", "")
    if not task_id:
        raise MinerUAPIError(f"MinerU 响应缺少 task_id: {data}")

    Log.debug(
        f"MinerU 任务已提交: page_range={page_range}, task_id={task_id}",
        module=_MODULE,
    )
    return str(task_id)


# ---------------------------------------------------------------------------
# Phase 2: 轮询（指数退避 + jitter）
# ---------------------------------------------------------------------------


async def _poll_one(
    client: httpx.AsyncClient,
    td: _TaskDescriptor,
    poll_initial: float = _POLL_INITIAL,
    poll_max: float = _POLL_MAX,
    timeout: float = _TIMEOUT,
) -> dict[str, object]:
    """单个任务的轮询循环（指数退避 + jitter）。

    Args:
        client: 异步 HTTP 客户端。
        td: 任务描述。
        poll_initial: 初始轮询间隔（秒）。
        poll_max: 最大轮询间隔（秒）。
        timeout: 总超时（秒）。

    Returns:
        MinerU 返回的结果字典。

    Raises:
        MinerUTaskFailedError: 任务失败。
        MinerUTimeoutError: 轮询超时。
        asyncio.CancelledError: 被 gather 取消（其他任务失败）。
    """
    delay = poll_initial
    elapsed = 0.0
    try:
        while elapsed < timeout:
            result = await _query_task(client, td.task_id)

            # v4 响应: {"code": 0, "data": {"state": "done"/"pending"/"failed", ...}}
            task_data = result.get("data", {})
            if isinstance(task_data, dict):
                state = str(task_data.get("state", "pending"))
            else:
                state = "pending"

            td.state = TaskState.POLLING

            if state == "done":
                td.state = TaskState.DONE
                td.result = result
                Log.debug(
                    f"MinerU 任务完成: page_range={td.page_range}, "
                    f"task_id={td.task_id}",
                    module=_MODULE,
                )
                return result

            if state == "failed":
                td.state = TaskState.FAILED
                td.error = f"MinerU 任务失败: {td.task_id}"
                raise MinerUTaskFailedError(td.error)

            # 指数退避 + jitter
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(delay * jitter)
            elapsed += delay * jitter
            delay = min(delay * 2, poll_max)

        # 超时
        td.state = TaskState.FAILED
        td.error = f"MinerU 轮询超时: {timeout}s"
        raise MinerUTimeoutError(td.error)

    except asyncio.CancelledError:
        td.state = TaskState.FAILED
        td.error = "因其他任务失败被取消"
        raise


async def _query_task(
    client: httpx.AsyncClient,
    task_id: str,
) -> dict[str, object]:
    """查询 MinerU 任务状态。

    Args:
        client: 异步 HTTP 客户端。
        task_id: 任务 ID。

    Returns:
        完整响应字典。

    Raises:
        MinerUAPIError: API 调用失败。
    """
    try:
        response = await client.get(f"/api/v4/extract/task/{task_id}")
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise MinerUAPIError(
            f"MinerU 查询失败: HTTP {exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        raise MinerUAPIError(f"MinerU 查询请求失败: {exc}") from exc

    data = response.json()
    if not isinstance(data, dict):
        raise MinerUAPIError(f"MinerU 查询响应格式异常: {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Phase 2: 并发轮询编排
# ---------------------------------------------------------------------------


async def _poll_all_tasks(
    tasks: list[_TaskDescriptor],
    poll_initial: float = _POLL_INITIAL,
    poll_max: float = _POLL_MAX,
    timeout: float = _TIMEOUT,
) -> list[dict[str, object]]:
    """并发轮询所有任务。任一失败则取消全部。

    Args:
        tasks: 已提交的任务描述列表。
        poll_initial: 初始轮询间隔（秒）。
        poll_max: 最大轮询间隔（秒）。
        timeout: 单任务总超时（秒）。

    Returns:
        各任务的结果字典列表，顺序与输入一致。

    Raises:
        MinerUAPIError: 任一任务查询失败。
    """
    async with _build_http_client() as client:
        results = await asyncio.gather(
            *(_poll_one(client, td, poll_initial, poll_max, timeout) for td in tasks),
            return_exceptions=False,
        )
        return list(results)


# ---------------------------------------------------------------------------
# 页码范围计算
# ---------------------------------------------------------------------------


def _build_page_ranges(
    total_pages: int,
    max_per_batch: int = _MAX_PAGES_PER_BATCH,
) -> list[str]:
    """根据总页数和每批上限，生成 page_ranges 列表。

    Args:
        total_pages: PDF 总页数。
        max_per_batch: 每批最多页数。

    Returns:
        page_ranges 字符串列表，如 ``["1-200", "201-374"]``。
    """
    if total_pages <= max_per_batch:
        return [f"1-{total_pages}"]

    ranges: list[str] = []
    start = 1
    while start <= total_pages:
        end = min(start + max_per_batch - 1, total_pages)
        ranges.append(f"{start}-{end}")
        start = end + 1
    return ranges


def _estimate_pages(pdf_bytes: bytes) -> int:
    """从 PDF 字节大小粗略估算页数。

    Args:
        pdf_bytes: PDF 原始字节。

    Returns:
        估算的页数（至少为 1）。
    """
    return max(1, len(pdf_bytes) // 50_000)


# ---------------------------------------------------------------------------
# 结果转换
# ---------------------------------------------------------------------------


def _download_and_parse_zip(
    zip_url: str,
) -> tuple[str, list[dict[str, object]]]:
    """下载 MinerU 结果 zip 包并解析 markdown 和 content_list。

    v4 API 不在 JSON 响应中直接返回内容，而是通过 ``full_zip_url``
    指向一个 zip 包，内含 ``full.md`` 和 ``content_list_v2.json``。

    Args:
        zip_url: zip 包的下载 URL。

    Returns:
        (raw_markdown, content_list_blocks) 元组。

    Raises:
        MinerUAPIError: 下载或解析失败时抛出。
    """
    import io as _io
    import json as _json
    import zipfile

    Log.debug(f"下载 MinerU 结果 zip: {zip_url}", module=_MODULE)
    try:
        response = httpx.get(zip_url, follow_redirects=True, timeout=120)
        response.raise_for_status()
    except Exception as exc:
        raise MinerUAPIError(f"MinerU 结果 zip 下载失败: {exc}") from exc

    raw_markdown = ""
    flat_blocks: list[dict[str, object]] = []

    try:
        with zipfile.ZipFile(_io.BytesIO(response.content)) as zf:
            names = zf.namelist()

            # 读取 markdown
            for name in names:
                if name.endswith("full.md") or name.endswith(".md"):
                    raw_markdown = zf.read(name).decode("utf-8")
                    break

            # 读取 content_list_v2.json（格式：list[list[dict]]，外层按页分组）
            cl_name = None
            for name in names:
                if "content_list" in name and name.endswith(".json"):
                    cl_name = name
                    break
            if cl_name is not None:
                raw_cl = _json.loads(zf.read(cl_name))
                if isinstance(raw_cl, list):
                    for page_idx, page_blocks in enumerate(raw_cl):
                        if isinstance(page_blocks, list):
                            for block in page_blocks:
                                if isinstance(block, dict):
                                    flat_blocks.append(
                                        {**block, "page_idx": page_idx}
                                    )
                        elif isinstance(page_blocks, dict):
                            flat_blocks.append(
                                {**page_blocks, "page_idx": page_idx}
                            )
    except zipfile.BadZipFile as exc:
        raise MinerUAPIError(f"MinerU 结果 zip 损坏: {exc}") from exc
    except Exception as exc:
        raise MinerUAPIError(f"MinerU 结果 zip 解析失败: {exc}") from exc

    return raw_markdown, flat_blocks


def _convert_mineru_result(
    result: dict[str, object],
    backend: DocumentBackend,
) -> ConvertedDocument:
    """将 MinerU v4 结果字典转换为统一中间格式。

    v4 API 的结果通过 ``full_zip_url`` 返回 zip 包，内含
    ``full.md``（Markdown）和 ``content_list_v2.json``（结构化块列表）。
    本函数下载 zip 并解析为 ``ConvertedDocument``。

    Args:
        result: MinerU 返回的完整结果字典（含 data.full_zip_url）。
        backend: 后端标识。

    Returns:
        转换完成的 ``ConvertedDocument``。
    """
    data = result.get("data", {})
    if not isinstance(data, dict):
        data = {}

    # 尝试从 zip 下载内容
    zip_url = str(data.get("full_zip_url", ""))
    raw_markdown = ""
    content_blocks: list[dict[str, object]] = []

    if zip_url:
        try:
            raw_markdown, content_blocks = _download_and_parse_zip(zip_url)
        except MinerUAPIError as exc:
            Log.warn(f"zip 下载/解析失败，回退到空内容: {exc}", module=_MODULE)

    sections: list[DocumentSection] = []
    tables: list[DocumentTable] = []
    images: list[DocumentImage] = []

    for block in content_blocks:
        block_type = str(block.get("type", "unknown"))
        page_idx_raw = block.get("page_idx")
        page_idx = (
            int(page_idx_raw)
            if isinstance(page_idx_raw, (int, float))
            else None
        )

        if block_type in ("text", "paragraph"):
            sections.append(
                DocumentSection(
                    title="",
                    level=1,
                    content=str(block.get("content", block.get("text", ""))),
                    page_idx=page_idx,
                )
            )
        elif block_type == "title":
            sections.append(
                DocumentSection(
                    title=str(block.get("content", block.get("title", ""))),
                    level=int(str(block.get("level", "1"))),
                    content="",
                    page_idx=page_idx,
                )
            )
        elif block_type == "table":
            tables.append(
                DocumentTable(
                    caption=str(block.get("caption", "")),
                    html=str(block.get("html", block.get("content", ""))),
                    page_idx=page_idx,
                )
            )
        elif block_type in ("figure", "image"):
            images.append(
                DocumentImage(
                    path=str(block.get("image_path", block.get("content", ""))),
                    caption=str(block.get("caption", "")),
                    page_idx=page_idx,
                )
            )
        else:
            Log.debug(
                f"跳过未知 MinerU block type: {block_type}",
                module=_MODULE,
            )

    return ConvertedDocument(
        backend=backend,
        sections=tuple(sections),
        tables=tuple(tables),
        images=tuple(images),
        raw_markdown=raw_markdown,
    )


def _merge_chunk_results(
    results: list[dict[str, object]],
    backend: DocumentBackend,
) -> ConvertedDocument:
    """合并多个批次的 MinerU 结果。

    简化版：直接 append 各列表，raw_markdown 用换行连接。

    Args:
        results: 各批次的结果字典列表。
        backend: 后端标识。

    Returns:
        合并后的 ``ConvertedDocument``。
    """
    all_sections: list[DocumentSection] = []
    all_tables: list[DocumentTable] = []
    all_images: list[DocumentImage] = []
    all_markdown_parts: list[str] = []

    for result in results:
        doc = _convert_mineru_result(result, backend)
        all_sections.extend(doc.sections)
        all_tables.extend(doc.tables)
        all_images.extend(doc.images)
        if doc.raw_markdown.strip():
            all_markdown_parts.append(doc.raw_markdown)

    return ConvertedDocument(
        backend=backend,
        sections=tuple(all_sections),
        tables=tuple(all_tables),
        images=tuple(all_images),
        raw_markdown="\n\n".join(all_markdown_parts),
        metadata={"chunk_count": str(len(results))},
    )


# ---------------------------------------------------------------------------
# 本地回退（层3 CLI + 层4 Python API）
# ---------------------------------------------------------------------------


def _try_parse_with_mineru_cli(pdf_bytes: bytes) -> ConvertedDocument | None:
    """尝试使用 MinerU 本地 CLI 解析 PDF（层3）。

    Args:
        pdf_bytes: PDF 原始字节。

    Returns:
        解析成功返回 ``ConvertedDocument``，CLI 不可用返回 ``None``。
    """
    if not shutil.which("mineru"):
        Log.debug("MinerU CLI 不可用，跳过层3", module=_MODULE)
        return None

    Log.info("MinerU 本地 CLI 可用，尝试解析", module=_MODULE)
    # TODO: 实现 CLI 调用逻辑
    return None


def _try_parse_with_mineru_python_api(
    pdf_bytes: bytes,
) -> ConvertedDocument | None:
    """尝试使用 MinerU 本地 Python API 解析 PDF（层4）。

    Args:
        pdf_bytes: PDF 原始字节。

    Returns:
        解析成功返回 ``ConvertedDocument``，API 不可用返回 ``None``。
    """
    try:
        import magic_pdf  # type: ignore[import-unfound]  # noqa: F401
    except ImportError:
        Log.debug("magic_pdf 未安装，跳过层4", module=_MODULE)
        return None

    Log.info("MinerU 本地 Python API 可用，尝试解析", module=_MODULE)
    # TODO: 实现 Python API 调用逻辑
    return None


# ---------------------------------------------------------------------------
# Docling 回退（层5）
# ---------------------------------------------------------------------------


def _parse_with_docling(pdf_bytes: bytes) -> ConvertedDocument:
    """使用 Docling 解析 PDF（层5 终极兜底）。

    Args:
        pdf_bytes: PDF 原始字节。

    Returns:
        Docling 解析结果的 ``ConvertedDocument``。

    Raises:
        RuntimeError: Docling 解析失败。
    """
    from dayu.fins.docling_export import convert_pdf_bytes_to_docling_payload

    Log.info("回退到 Docling 解析", module=_MODULE)
    try:
        payload = convert_pdf_bytes_to_docling_payload(
            pdf_bytes, stream_name="fallback.pdf"
        )
    except Exception as exc:
        raise RuntimeError(f"Docling 解析失败: {exc}") from exc

    raw_markdown = ""
    if isinstance(payload, dict):
        raw_markdown = str(payload.get("main-text", ""))

    return ConvertedDocument(
        backend=DocumentBackend.DOCLING,
        raw_markdown=raw_markdown,
    )


# ---------------------------------------------------------------------------
# 云 API 编排（层1 + 层2）
# ---------------------------------------------------------------------------


async def _submit_and_poll(
    pdf_url: str,
    page_ranges: list[str],
) -> list[dict[str, object]]:
    """并发提交 + 并发轮询所有批次。

    Args:
        pdf_url: PDF 文件的公开 URL。
        page_ranges: 页码范围列表。

    Returns:
        各批次结果字典列表。
    """
    async with _build_http_client() as client:
        # Phase 1: 并发提交
        submit_coroutines = [
            _submit_task(client, pdf_url, pr) for pr in page_ranges
        ]
        task_ids = await asyncio.gather(*submit_coroutines)
        tasks = [
            _TaskDescriptor(task_id=tid, page_range=pr)
            for tid, pr in zip(task_ids, page_ranges)
        ]
        Log.info(
            f"MinerU 批次已提交: {len(tasks)} 个任务",
            module=_MODULE,
        )

        # Phase 2: 并发轮询
        results = await asyncio.gather(
            *(_poll_one(client, td) for td in tasks),
            return_exceptions=False,
        )
        return list(results)


def _parse_with_cloud_api(
    pdf_bytes: bytes,
    total_pages: int,
    *,
    filename: str = "filing.pdf",
) -> ConvertedDocument:
    """通过 MinerU 云 API v4 解析 PDF（层1 + 层2）。

    流程：上传 COS → 配额检查 → 计算 page_ranges → 并发提交 → 并发轮询 → 合并结果。
    配额检查放在 COS 上传成功之后，避免 COS 上传失败时配额被白扣。

    Args:
        pdf_bytes: PDF 原始字节。
        total_pages: PDF 总页数。
        filename: COS 上传时的文件名。

    Returns:
        解析完成的 ``ConvertedDocument``。

    Raises:
        MinerUAPIError: 云 API 调用失败。
    """
    from dayu.cos_helper import upload_pdf_to_cos, delete_from_cos, extract_cos_key_from_url

    # Step 1: 上传到 COS
    Log.info(f"上传 PDF 到 COS: {len(pdf_bytes)} bytes", module=_MODULE)
    pdf_url = upload_pdf_to_cos(pdf_bytes, filename=filename)
    cos_key = extract_cos_key_from_url(pdf_url)

    try:
        # Step 2: 配额检查（COS 上传成功后才扣，避免白扣）
        tracker = _get_quota_tracker()
        if not tracker.check_and_consume(total_pages):
            raise MinerUAPIError(
                f"MinerU 配额不足: 请求 {total_pages} 页，"
                f"剩余 {tracker.get_remaining()} 页"
            )

        # Step 3: 计算 page_ranges
        page_ranges = _build_page_ranges(total_pages)
        Log.info(
            f"MinerU 云 API: {total_pages} 页, "
            f"{len(page_ranges)} 批, ranges={page_ranges}",
            module=_MODULE,
        )

        # Step 4: 并发提交 + 轮询
        results = asyncio.run(_submit_and_poll(pdf_url, page_ranges))
        Log.info(f"MinerU 批次全部完成: {len(results)} 个结果", module=_MODULE)

        # Step 5: 合并
        return _merge_chunk_results(results, DocumentBackend.MINERU_CLOUD)

    finally:
        # 清理 COS 临时文件
        delete_from_cos(cos_key)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def parse_pdf_bytes_with_mineru(
    pdf_bytes: bytes,
    *,
    total_pages: int | None = None,
    filename: str = "filing.pdf",
) -> ConvertedDocument:
    """将 PDF 字节流通过 MinerU 解析为统一中间格式。

    五层回退链：
    1. MinerU 云 API v4 单次（≤200 页且配额充足）
    2. MinerU 云 API v4 分批（>200 页，page_ranges）
    3. MinerU 本地 CLI
    4. MinerU 本地 Python API
    5. Docling

    ⚠️ 同步边界：此函数是同步的，内部调用 asyncio.run()。
    如果调用链变成 async（如 FastAPI 集成），需要重构为纯 async 函数。

    Args:
        pdf_bytes: PDF 原始字节。
        total_pages: PDF 总页数。为 None 时通过 pikepdf 精确获取。
        filename: COS 上传时的文件名（用于生成 COS 对象键）。

    Returns:
        解析完成的 ``ConvertedDocument``。
    """
    Log.info(
        f"MinerU 解析启动: pdf_size={len(pdf_bytes)} bytes",
        module=_MODULE,
    )

    # 获取精确页数
    if total_pages is None:
        try:
            import pikepdf
            import io

            with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
                total_pages = len(pdf.pages)
        except ImportError:
            total_pages = _estimate_pages(pdf_bytes)
            Log.warn(
                f"pikepdf 未安装，使用估算页数: {total_pages}",
                module=_MODULE,
            )

    Log.info(f"PDF 总页数: {total_pages}", module=_MODULE)

    # --- 层 1/2: 云 API ---
    # 配额检查已移入 _parse_with_cloud_api（COS 上传成功之后才扣配额，
    # 避免 COS 上传失败时配额被白扣）。
    if _MINERU_API_TOKEN:
        try:
            return _parse_with_cloud_api(pdf_bytes, total_pages, filename=filename)
        except Exception as exc:
            Log.warn(
                f"MinerU 云 API 失败，尝试本地回退: {exc}",
                module=_MODULE,
            )
    else:
        Log.debug("MinerU API Token 未配置，跳过云 API", module=_MODULE)

    # --- 层 3: 本地 CLI ---
    cli_result = _try_parse_with_mineru_cli(pdf_bytes)
    if cli_result is not None:
        return cli_result

    # --- 层 4: 本地 Python API ---
    api_result = _try_parse_with_mineru_python_api(pdf_bytes)
    if api_result is not None:
        return api_result

    # --- 层 5: Docling ---
    return _parse_with_docling(pdf_bytes)


__all__ = [
    "MinerUAPIError",
    "MinerUTaskFailedError",
    "MinerUTimeoutError",
    "parse_pdf_bytes_with_mineru",
]
