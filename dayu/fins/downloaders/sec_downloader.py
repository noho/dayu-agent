"""SEC EDGAR 下载器（低层网络与文件下载）。

职责范围：
- SEC 接口访问（ticker -> CIK、submissions、index.json）
- 远端文件列表获取（含 XBRL/exhibits）
- 文件下载与重试/间隔/UA/304 跳过

不包含业务流程逻辑（例如 form 选择、时间窗口、meta/manifest 写入）。

维护说明(不拆分本模块):
    本模块约 2050 行, 由 SecDownloader 类(1134 行, 7 个方法)和 33 个
    模块级工具函数组成, 全部围绕 SEC EDGAR HTTP 访问与文件解析这一个
    I/O 关注点. 类方法与工具函数间存在密集调用, 拆分只会增加 import
    复杂度. 外部仅消费 SecDownloader / RemoteFileDescriptor /
    DownloaderEvent 等少数公开符号.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import html
import inspect
import json
import os
import posixpath
import re
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, BinaryIO, Callable, Literal, Optional, TypeVar, cast, overload
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx
from dayu.workspace_paths import build_sec_throttle_dir
from dayu.contracts.env_keys import SEC_USER_AGENT_ENV

if sys.platform != "win32":
    import fcntl

from dayu.log import Log
from dayu.fins._converters import normalize_optional_text, optional_int
from dayu.fins.domain.document_models import FileObjectMeta

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dash}/"
ARCHIVES_INDEX_JSON = ARCHIVES_BASE + "index.json"
ARCHIVES_INDEX_HEADERS_HTML = ARCHIVES_BASE + "{accession}-index-headers.html"
BROWSE_EDGAR_ATOM_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?"
    "action=getcompany&filenum={filenum}&owner=include&count={count}&output=atom"
)
BROWSE_EDGAR_TICKER_ATOM_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?"
    "action=getcompany&CIK={ticker}&owner=exclude&count={count}&output=atom"
)

DEFAULT_SLEEP_SECONDS = 0.2
_UNCONFIGURED_USER_AGENT = "DayuAgent/1.0 unconfigured@example.com"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 0.8
_GLOBAL_SEC_THROTTLE_STATE_FILENAME = "state.json"
_GLOBAL_SEC_THROTTLE_LOCK_FILENAME = "state.lock"

# SEC 速率限制：最小请求间隔（秒），SEC 要求 ≤10 请求/秒，留安全余量
_SEC_MIN_REQUEST_INTERVAL_SECONDS = 0.12
# 触发限流退避的 HTTP 状态码
_THROTTLE_STATUS_CODES: frozenset[int] = frozenset({429, 503})
# 限流默认退避秒数（无 Retry-After 头时使用）
_SEC_THROTTLE_BACKOFF_SECONDS = 5.0
# SEC 公布口径：超过阈值后，即便恢复到阈值以下，也可能需要等待 10 分钟才解除限制。
_SEC_THROTTLE_RECOVERY_SECONDS = 600.0
# 限流额外重试次数（不消耗正常重试预算）
_SEC_THROTTLE_MAX_RETRIES = 3
_ETAG_WEAK_PREFIX = "W/"
_ETAG_GZIP_SUFFIX = "-gzip"
_AwaitedValueT = TypeVar("_AwaitedValueT")
_HttpResultT = TypeVar("_HttpResultT")


@dataclass(frozen=True)
class RemoteFileDescriptor:
    """远端文件描述。"""

    name: str
    source_url: str
    http_etag: Optional[str]
    http_last_modified: Optional[str]
    remote_size: Optional[int]
    http_status: Optional[int] = None
    sec_document_type: Optional[str] = None
    sec_description: Optional[str] = None


@dataclass(frozen=True)
class BrowseEdgarFiling:
    """browse-edgar 记录。"""

    form_type: str
    filing_date: str
    accession_number: str
    cik: str
    index_url: str


@dataclass(frozen=True)
class Sc13PartyRoles:
    """SC 13 申报方向角色。"""

    filed_by_cik: str
    subject_cik: str


DownloaderEventType = Literal["file_downloaded", "file_skipped", "file_failed"]


@dataclass(frozen=True)
class DownloaderEvent:
    """下载器文件级事件。"""

    event_type: DownloaderEventType
    name: str
    source_url: str
    http_etag: Optional[str]
    http_last_modified: Optional[str]
    http_status: Optional[int]
    file_meta: Optional[FileObjectMeta] = None
    reason_code: Optional[str] = None
    reason_message: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class _SecThrottleState:
    """SEC 限流共享状态。"""

    next_request_at: float = 0.0
    cooldown_until: float = 0.0


def _build_empty_content_failure_event(
    descriptor: RemoteFileDescriptor,
    http_status: Optional[int],
) -> DownloaderEvent:
    """构造 0 字节下载失败事件。

    Args:
        descriptor: 当前远端文件描述。
        http_status: 本次下载对应的 HTTP 状态码。

    Returns:
        `empty_content` 类型的失败事件。

    Raises:
        无。
    """

    return DownloaderEvent(
        event_type="file_failed",
        name=descriptor.name,
        source_url=descriptor.source_url,
        http_etag=descriptor.http_etag,
        http_last_modified=descriptor.http_last_modified,
        http_status=http_status,
        reason_code="empty_content",
        reason_message="下载内容为 0 字节，视为下载失败",
        error="下载内容为 0 字节，视为下载失败",
    )


def _should_abort_after_empty_primary(
    descriptor_name: str,
    primary_document: Optional[str],
) -> bool:
    """判断 0 字节文件是否应中止整个 filing 下载。

    Args:
        descriptor_name: 当前文件名。
        primary_document: 调用方指定的主文档文件名。

    Returns:
        若当前文件就是主文档且返回 0 字节，则返回 `True`。

    Raises:
        无。
    """

    return bool(primary_document) and descriptor_name == primary_document


def _handle_conditional_download_response(response: httpx.Response) -> tuple[int, Optional[bytes]]:
    """处理条件下载响应。

    Args:
        response: HTTP 响应对象。

    Returns:
        `(status_code, payload)`；若命中 `304`，则 payload 为 `None`。

    Raises:
        httpx.HTTPError: 非成功状态且非 `304` 时抛出。
    """

    if response.status_code == 304:
        return 304, None
    response.raise_for_status()
    return response.status_code, response.content


def _parse_http_json_response(response: httpx.Response) -> dict[str, Any]:
    """解析 JSON 响应体。

    Args:
        response: HTTP 响应对象。

    Returns:
        JSON 字典。

    Raises:
        httpx.HTTPError: HTTP 状态异常时抛出。
        ValueError: JSON 解析失败时抛出。
    """

    response.raise_for_status()
    return cast(dict[str, Any], response.json())


def _read_http_binary_response(response: httpx.Response) -> bytes:
    """读取字节响应体。

    Args:
        response: HTTP 响应对象。

    Returns:
        响应体字节。

    Raises:
        httpx.HTTPError: HTTP 状态异常时抛出。
    """

    response.raise_for_status()
    return response.content


def _return_http_response(response: httpx.Response) -> httpx.Response:
    """原样返回 HTTP 响应。

    Args:
        response: HTTP 响应对象。

    Returns:
        原始响应对象。

    Raises:
        无。
    """

    return response


class _RelativeHtmlLinkExtractor(HTMLParser):
    """提取 HTML 中的超链接地址。"""

    def __init__(self) -> None:
        """初始化链接提取器。"""

        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
    ) -> None:
        """处理起始标签并提取 `href`。

        Args:
            tag: 标签名。
            attrs: 标签属性列表。

        Returns:
            无。

        Raises:
            无。
        """

        if tag.lower() != "a":
            return
        for attribute_name, attribute_value in attrs:
            if attribute_name.lower() != "href":
                continue
            if attribute_value is None:
                continue
            normalized_href = attribute_value.strip()
            if normalized_href:
                self.hrefs.append(normalized_href)


def accession_to_no_dash(accession_number: str) -> str:
    """移除 accession 中的连字符。

    Args:
        accession_number: 原始 accession。

    Returns:
        无连字符 accession。

    Raises:
        ValueError: accession 为空时抛出。
    """

    normalized = accession_number.strip()
    if not normalized:
        raise ValueError("accession_number 不能为空")
    return normalized.replace("-", "")


def pick_extracted_instance_xml(items: list[dict[str, Any]]) -> Optional[str]:
    """从 index 项中选择 XBRL instance xml。

    Args:
        items: index.json 中的 `directory.item` 列表。

    Returns:
        匹配到的文件名；无匹配时返回 `None`。

    Raises:
        无。
    """

    names = [str(item.get("name", "")) for item in items if isinstance(item, dict)]
    for name in names:
        if name.endswith("_htm.xml"):
            return name
    for name in names:
        if name.endswith(".xml") and "htm" in name:
            return name
    non_linkbase = [
        name
        for name in names
        if name.endswith(".xml")
        and not name.lower().endswith(("_pre.xml", "_lab.xml", "_cal.xml", "_def.xml"))
        and name.lower() not in {"filingsummary.xml"}
    ]
    if not non_linkbase:
        return None
    non_linkbase.sort(key=lambda item: (0 if re.search(r"[-_]\d{8}\.xml$", item.lower()) else 1, len(item)))
    return non_linkbase[0]


def pick_taxonomy_files(items: list[dict[str, Any]]) -> list[str]:
    """选择 taxonomy/linkbase 文件名。

    Args:
        items: index.json 条目列表。

    Returns:
        taxonomy 文件名列表（去重后）。

    Raises:
        无。
    """

    names = [str(item.get("name", "")) for item in items if isinstance(item, dict)]
    selected: list[str] = []
    for suffix in [".xsd", "_pre.xml", "_cal.xml", "_def.xml", "_lab.xml"]:
        for name in names:
            if name.endswith(suffix):
                selected.append(name)
                break
    return sorted(set(selected))


def pick_exhibit_files(items: list[dict[str, Any]]) -> list[str]:
    """选择 6-K exhibit 文件。

    识别策略：
    1. 优先使用 SEC 文档类型（`EX-99.x`）；
    2. 若无类型字段，回退到文件名模式（`dex99*` / `ex99*`）。

    Args:
        items: 文件条目列表（可来自 `index.json` 或 `index-headers` 解析结果）。

    Returns:
        exhibit 文件名列表。

    Raises:
        无。
    """

    exhibits: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        lowered = name.lower()
        if not lowered.endswith((".htm", ".html")):
            continue
        document_type = _normalize_sec_document_type(item.get("type"))
        if document_type and document_type.startswith("EX-99"):
            exhibits.append(name)
            continue
        # Donnelley 格式: dex991.htm, dex992.htm, ...
        # Edgar Filing Services 格式: xxx_ex99-1.htm, xxx_ex99_1.htm, ...
        if "dex99" in lowered or "ex99" in lowered:
            exhibits.append(name)
    return sorted(set(exhibits))


def pick_form_document_files(items: list[dict[str, Any]], form_type: str) -> list[str]:
    """选择与 filing form 同型的 HTML 文件。

    当前主要用于 6-K：有些 foreign issuer 会把真正的季度结果正文挂在
    `TYPE=6-K` 的 cover html 上，而不是 `EX-99.x` exhibit。若只保留
    primary_document 与 exhibit，会漏掉这类候选，导致预筛选无法收敛到
    正确主文件。

    Args:
        items: 文件条目列表（可来自 `index.json` 或 `index-headers` 解析结果）。
        form_type: filing form 类型。

    Returns:
        与 form_type 同型的 HTML 文件名列表。

    Raises:
        无。
    """

    normalized_form_type = _normalize_sec_document_type(form_type)
    if normalized_form_type is None:
        return []
    matched: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        lowered = name.lower()
        if not lowered.endswith((".htm", ".html")):
            continue
        document_type = _normalize_sec_document_type(item.get("type"))
        if document_type == normalized_form_type:
            matched.append(name)
    return sorted(set(matched))


def extract_same_filing_linked_html_files(
    payload: bytes,
    primary_document: str,
) -> list[str]:
    """从主文档中提取同 filing 的相对 HTML 链接。

    只保留高把握的补链目标：
    - 必须是相对路径；
    - 归一化后不能越出当前归档目录；
    - 仅接受同目录下的 `.htm/.html` 文件；
    - 跳过锚点、脚本、邮件链接及主文档自身。

    Args:
        payload: 主文档 HTML 字节。
        primary_document: 主文档文件名。

    Returns:
        归一化后的相对 HTML 文件名列表。

    Raises:
        无。
    """

    if not payload:
        return []
    extractor = _RelativeHtmlLinkExtractor()
    extractor.feed(payload.decode("utf-8", errors="ignore"))
    candidates: set[str] = set()
    for raw_href in extractor.hrefs:
        normalized_href = _normalize_same_filing_relative_html_href(
            href=raw_href,
            primary_document=primary_document,
        )
        if normalized_href is not None:
            candidates.add(normalized_href)
    return sorted(candidates)


def _normalize_same_filing_relative_html_href(
    href: str,
    primary_document: str,
) -> Optional[str]:
    """规范化主文档中的同 filing 相对 HTML 链接。

    Args:
        href: 原始链接地址。
        primary_document: 主文档文件名。

    Returns:
        归一化后的相对文件名；若不是高把握的同 filing HTML 链接则返回 `None`。

    Raises:
        无。
    """

    normalized_href = html.unescape(href).strip().replace("\\", "/")
    if not normalized_href or normalized_href.startswith("#"):
        return None
    lowered_href = normalized_href.lower()
    if lowered_href.startswith(("javascript:", "mailto:", "tel:", "data:")):
        return None
    parsed = urlparse(normalized_href)
    if parsed.scheme or parsed.netloc:
        return None
    normalized_path = posixpath.normpath(parsed.path.strip())
    if normalized_path in {"", "."}:
        return None
    if normalized_path.startswith("/") or normalized_path.startswith("../"):
        return None
    # 只接受同目录文件名，避免把上层/子目录中的非 filing 正文资源一起拉进来。
    if "/" in normalized_path:
        return None
    if not normalized_path.lower().endswith((".htm", ".html")):
        return None
    if normalized_path == primary_document:
        return None
    return normalized_path


def build_source_fingerprint(descriptors: list[RemoteFileDescriptor]) -> str:
    """构建 source_fingerprint。

    Args:
        descriptors: 远端文件描述列表。

    Returns:
        指纹字符串（sha256）。

    Raises:
        无。
    """

    payload = [
        {
            "name": descriptor.name,
            # 说明：
            # - SEC/CloudFront 在相同内容下可能返回 transport 变体 ETag（如追加 -gzip）；
            # - Content-Length 也可能受压缩传输影响抖动；
            # 指纹需聚焦“内容身份”而非传输细节，避免误判远端变更。
            "etag": _normalize_fingerprint_etag(descriptor.http_etag),
            "last_modified": _normalize_fingerprint_last_modified(descriptor.http_last_modified),
        }
        for descriptor in sorted(descriptors, key=lambda item: item.name)
    ]
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_fingerprint_etag(raw_etag: Optional[str]) -> Optional[str]:
    """标准化用于指纹的 ETag。

    Args:
        raw_etag: 原始 HTTP ETag。

    Returns:
        标准化 ETag；无有效值时返回 `None`。

    Raises:
        无。
    """

    if raw_etag is None:
        return None
    normalized = str(raw_etag).strip()
    if not normalized:
        return None
    if normalized.upper().startswith(_ETAG_WEAK_PREFIX):
        normalized = normalized[2:].strip()
    if normalized.startswith('"') and normalized.endswith('"') and len(normalized) >= 2:
        normalized = normalized[1:-1]
    if normalized.lower().endswith(_ETAG_GZIP_SUFFIX):
        normalized = normalized[: -len(_ETAG_GZIP_SUFFIX)]
    normalized = normalized.strip().lower()
    return normalized or None


def _normalize_fingerprint_last_modified(raw_last_modified: Optional[str]) -> Optional[str]:
    """标准化用于指纹的 Last-Modified。

    Args:
        raw_last_modified: 原始 Last-Modified。

    Returns:
        标准化时间字符串；无有效值时返回 `None`。

    Raises:
        无。
    """

    if raw_last_modified is None:
        return None
    normalized = str(raw_last_modified).strip()
    return normalized or None


def hash_file_sha256(file_path: Path) -> str:
    """计算文件 sha256。

    Args:
        file_path: 文件路径。

    Returns:
        十六进制摘要。

    Raises:
        OSError: 文件读取失败时抛出。
    """

    sha256 = hashlib.sha256()
    with file_path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 64)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()



def _load_sec_throttle_state(state_path: Path) -> _SecThrottleState:
    """读取 SEC 共享限流状态。

    Args:
        state_path: 状态文件路径。

    Returns:
        共享限流状态；文件不存在或损坏时返回默认状态。

    Raises:
        无。
    """

    if not state_path.exists():
        return _SecThrottleState()
    try:
        with state_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return _SecThrottleState()
    if not isinstance(payload, dict):
        return _SecThrottleState()
    next_request_at = payload.get("next_request_at", 0.0)
    cooldown_until = payload.get("cooldown_until", 0.0)
    try:
        return _SecThrottleState(
            next_request_at=max(float(next_request_at), 0.0),
            cooldown_until=max(float(cooldown_until), 0.0),
        )
    except (TypeError, ValueError):
        return _SecThrottleState()


def _save_sec_throttle_state(state_path: Path, state: _SecThrottleState) -> None:
    """写入 SEC 共享限流状态。

    Args:
        state_path: 状态文件路径。
        state: 待写入的共享限流状态。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "next_request_at": state.next_request_at,
        "cooldown_until": state.cooldown_until,
    }
    with state_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


@contextlib.contextmanager
def _sec_throttle_lock(lock_path: Path) -> Any:
    """获取 SEC 共享限流文件锁。

    Args:
        lock_path: 锁文件路径。

    Yields:
        已打开并持锁的文件对象。

    Raises:
        OSError: 文件打开失败时抛出。
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        if sys.platform != "win32":
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield stream
        finally:
            if sys.platform != "win32":
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _resolve_sec_throttle_delay(response: httpx.Response) -> float:
    """解析 SEC 限流后的恢复等待时间。

    Args:
        response: HTTP 响应对象。

    Returns:
        应等待的秒数。若响应未提供足够信息，则至少等待 10 分钟。

    Raises:
        无。
    """

    return max(_parse_retry_after(response), _SEC_THROTTLE_RECOVERY_SECONDS)


class SecDownloader:
    """SEC 下载器。"""

    MODULE = "FINS.SEC_DOWNLOADER"

    def __init__(
        self,
        workspace_root: Path,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """初始化下载器。

        Args:
            workspace_root: 工作区根目录。
            client: 可选 `httpx.AsyncClient`（便于测试注入）。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.workspace_root = workspace_root.resolve()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()
        self._sleep_seconds = DEFAULT_SLEEP_SECONDS
        self._request_timeout_seconds = DEFAULT_REQUEST_TIMEOUT_SECONDS
        self._max_retries = DEFAULT_MAX_RETRIES
        self._user_agent = self._resolve_user_agent(None)
        self._last_request_time: float = 0.0  # 上次请求时间（monotonic clock）
        self._throttle_state_dir = build_sec_throttle_dir(self.workspace_root)
        self._throttle_state_path = self._throttle_state_dir / _GLOBAL_SEC_THROTTLE_STATE_FILENAME
        self._throttle_lock_path = self._throttle_state_dir / _GLOBAL_SEC_THROTTLE_LOCK_FILENAME
        Log.debug(
            f"初始化 SecDownloader: workspace_root={self.workspace_root}",
            module=self.MODULE,
        )

    async def close(self) -> None:
        """关闭底层 HTTP 客户端。

        Args:
            无。

        Returns:
            无。

        Raises:
            RuntimeError: 关闭失败时抛出。
        """

        if self._owns_client:
            await self._client.aclose()

    def normalize_ticker(self, ticker: str) -> str:
        """标准化 ticker。

        Args:
            ticker: 原始 ticker。

        Returns:
            大写 ticker。

        Raises:
            ValueError: ticker 为空时抛出。
        """

        normalized = ticker.strip().upper()
        if not normalized:
            raise ValueError("ticker 不能为空")
        return normalized

    def configure(
        self,
        user_agent: Optional[str],
        sleep_seconds: float,
        max_retries: int,
    ) -> None:
        """配置下载参数。

        Args:
            user_agent: User-Agent 字符串。
            sleep_seconds: 请求间隔秒数。
            max_retries: 最大重试次数。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        if max_retries <= 0:
            raise ValueError("max_retries 必须大于 0")
        if sleep_seconds < 0:
            raise ValueError("sleep_seconds 不能为负数")
        self._max_retries = max_retries
        self._sleep_seconds = sleep_seconds
        self._user_agent = self._resolve_user_agent(user_agent)
        Log.debug(
            f"更新下载配置: ua={self._user_agent} sleep_seconds={self._sleep_seconds} max_retries={self._max_retries}",
            module=self.MODULE,
        )

    async def resolve_company(self, ticker: str) -> tuple[str, str, str]:
        """解析 ticker 对应的 CIK 与公司名。

        Args:
            ticker: 股票代码。

        Returns:
            `(cik, company_name, cik10)`。

        Raises:
            RuntimeError: ticker 不存在时抛出。
        """

        normalized = self.normalize_ticker(ticker)
        mapping = await _await_if_needed(self._http_get_json(SEC_TICKER_MAP_URL))
        for row in mapping.values():
            if str(row.get("ticker", "")).upper() != normalized:
                continue
            cik = str(row.get("cik_str", "")).strip()
            company_name = str(row.get("title", "")).strip()
            if cik.isdigit():
                cik10 = str(int(cik)).zfill(10)
                return cik, company_name, cik10
        fallback_result = await _await_if_needed(self._resolve_company_via_browse_edgar_ticker(normalized))
        if fallback_result is not None:
            return fallback_result
        raise RuntimeError(f"无法在 SEC ticker map 中找到 ticker={normalized}")

    async def _resolve_company_via_browse_edgar_ticker(
        self,
        ticker: str,
        count: int = 40,
    ) -> Optional[tuple[str, str, str]]:
        """通过 browse-edgar 反查 ticker 对应公司信息。

        Args:
            ticker: 已标准化的大写 ticker。
            count: browse-edgar 拉取条数上限。

        Returns:
            命中时返回 `(cik, company_name, cik10)`，否则返回 `None`。

        Raises:
            无。
        """

        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            return None
        url = BROWSE_EDGAR_TICKER_ATOM_URL.format(ticker=normalized_ticker, count=count)
        try:
            payload = await _await_if_needed(self._http_get_bytes(url))
        except RuntimeError as exc:
            Log.warn(
                f"browse-edgar ticker 反查失败: ticker={normalized_ticker} error={exc}",
                module=self.MODULE,
            )
            return None
        try:
            entries = _parse_browse_edgar_atom(payload)
        except RuntimeError as exc:
            Log.warn(
                f"browse-edgar XML 解析失败，跳过 ticker={normalized_ticker}: {exc}",
                module=self.MODULE,
            )
            return None
        if not entries:
            return None
        for entry in entries:
            raw_cik = str(entry.cik).strip()
            if not raw_cik.isdigit():
                continue
            # SEC 下游接口使用无前导零/10位两种 CIK 形式，这里统一构造。
            cik = str(int(raw_cik))
            cik10 = cik.zfill(10)
            try:
                submissions = await _await_if_needed(self.fetch_submissions(cik10))
            except RuntimeError as exc:
                Log.warn(
                    (
                        "browse-edgar ticker 命中后拉取 submissions 失败: "
                        f"ticker={normalized_ticker} cik10={cik10} error={exc}"
                    ),
                    module=self.MODULE,
                )
                continue
            company_name = str(submissions.get("name", "")).strip()
            return cik, (company_name or normalized_ticker), cik10
        return None

    async def fetch_submissions(self, cik10: str) -> dict[str, Any]:
        """拉取 submissions JSON。

        Args:
            cik10: 10 位 CIK 字符串。

        Returns:
            submissions JSON 字典。

        Raises:
            RuntimeError: 请求失败时抛出。
        """

        url = SEC_SUBMISSIONS_URL.format(cik10=cik10)
        return await _await_if_needed(self._http_get_json(url))

    async def fetch_json(self, url: str) -> dict[str, Any]:
        """拉取任意 JSON URL。

        Args:
            url: 目标 JSON URL。

        Returns:
            JSON 字典。

        Raises:
            RuntimeError: 请求失败时抛出。
        """

        return await _await_if_needed(self._http_get_json(url))

    async def fetch_browse_edgar_filenum(
        self,
        filenum: str,
        count: int = 100,
    ) -> list[BrowseEdgarFiling]:
        """通过 browse-edgar 拉取 filenum 对应的 filings。

        Args:
            filenum: SEC 文件编号（如 005-XXXX）。
            count: 拉取条数上限。

        Returns:
            filings 列表。

        Raises:
            RuntimeError: 请求失败时抛出。
        """

        normalized = filenum.strip()
        if not normalized:
            return []
        url = BROWSE_EDGAR_ATOM_URL.format(filenum=normalized, count=count)
        payload = await _await_if_needed(self._http_get_bytes(url))
        return _parse_browse_edgar_atom(payload)

    async def resolve_primary_document(
        self,
        cik: str,
        accession_no_dash: str,
        form_type: str,
    ) -> str:
        """根据 index.json 推断主文件名。

        Args:
            cik: CIK（无前导零）。
            accession_no_dash: 无连字符 accession。
            form_type: 表单类型。

        Returns:
            主文件名。

        Raises:
            RuntimeError: 无法定位主文件时抛出。
        """

        index_url = ARCHIVES_INDEX_JSON.format(cik=str(int(cik)), accession_no_dash=accession_no_dash)
        index_json = await _await_if_needed(self._http_get_json(index_url))
        items = list(index_json.get("directory", {}).get("item", []) or [])
        primary = _select_primary_from_index_items(items, form_type)
        if primary:
            return primary
        raise RuntimeError("无法解析 primary_document")

    async def fetch_sc13_party_roles(
        self,
        archive_cik: str,
        accession_number: str,
    ) -> Optional[Sc13PartyRoles]:
        """解析 SC 13 filing 的申报方与被申报方 CIK。

        数据来源：`-index-headers.html` 页面中的 `FILED BY` 与
        `SUBJECT COMPANY` 段落内的 `CENTRAL INDEX KEY`。

        Args:
            archive_cik: archive 路径中的 CIK（无前导零或有前导零均可）。
            accession_number: accession（含连字符）。

        Returns:
            解析成功时返回 `Sc13PartyRoles`；网络失败或字段缺失返回 `None`。

        Raises:
            无。
        """

        normalized_archive_cik = _normalize_cik_value(archive_cik)
        normalized_accession = accession_number.strip()
        if normalized_archive_cik is None or not normalized_accession:
            return None
        accession_no_dash = accession_to_no_dash(normalized_accession)
        url = ARCHIVES_INDEX_HEADERS_HTML.format(
            cik=normalized_archive_cik,
            accession_no_dash=accession_no_dash,
            accession=normalized_accession,
        )
        try:
            payload = await _await_if_needed(self._http_get_bytes(url))
        except RuntimeError as exc:
            Log.warn(
                (
                    "SC13 index-headers 抓取失败: "
                    f"archive_cik={normalized_archive_cik} accession={normalized_accession} error={exc}"
                ),
                module=self.MODULE,
            )
            return None
        return _parse_sc13_party_roles_from_index_headers(payload)

    async def fetch_file_bytes(self, url: str) -> bytes:
        """下载文件并返回字节内容。

        Args:
            url: 文件 URL。

        Returns:
            文件内容字节。

        Raises:
            RuntimeError: 下载失败时抛出。
        """

        return await _await_if_needed(self._http_download(url))

    async def list_filing_files(
        self,
        cik: str,
        accession_no_dash: str,
        primary_document: str,
        form_type: str,
        include_xbrl: bool = True,
        include_exhibits: bool = True,
        include_http_metadata: bool = True,
    ) -> list[RemoteFileDescriptor]:
        """列出 filing 相关的远端文件。

        Args:
            cik: CIK（无前导零）。
            accession_no_dash: 无连字符 accession。
            primary_document: primaryDocument 文件名。
            form_type: filing form 类型。
            include_xbrl: 是否包含 XBRL 文件。
            include_exhibits: 是否包含 exhibit 文件（6-K）。
            include_http_metadata: 是否额外拉取文件级 HTTP 元数据。

        Returns:
            远端文件描述列表。

        Raises:
            RuntimeError: 网络请求连续失败时抛出。
        """

        archive_base = ARCHIVES_BASE.format(cik=str(int(cik)), accession_no_dash=accession_no_dash)
        filenames: list[str] = [primary_document]
        index_items: list[dict[str, Any]] = []
        index_header_documents: list[dict[str, Any]] = []
        if include_xbrl or include_exhibits:
            index_items = await _await_if_needed(
                self._try_fetch_index_items(cik=cik, accession_no_dash=accession_no_dash)
            )
        if include_exhibits and form_type == "6-K":
            index_header_documents = await _await_if_needed(
                self._try_fetch_index_header_documents(
                    cik=cik,
                    accession_no_dash=accession_no_dash,
                )
            )
        if include_xbrl:
            extracted_xml = pick_extracted_instance_xml(index_items)
            if extracted_xml:
                filenames.append(extracted_xml)
            filenames.extend(pick_taxonomy_files(index_items))
        if include_exhibits and form_type == "6-K":
            filenames.extend(pick_form_document_files(index_items, form_type))
            filenames.extend(pick_form_document_files(index_header_documents, form_type))
            filenames.extend(pick_exhibit_files(index_items))
            filenames.extend(pick_exhibit_files(index_header_documents))
            filenames.extend(
                await _await_if_needed(
                    self._try_fetch_primary_linked_html_files(
                        archive_base=archive_base,
                        primary_document=primary_document,
                    )
                )
            )
        file_meta_map = _build_file_metadata_map(index_items, index_header_documents)
        unique_filenames = sorted(set(filenames))
        Log.verbose(
            f"远端文件列表: {unique_filenames}",
            module=self.MODULE,
        )
        descriptors: list[RemoteFileDescriptor] = []
        for filename in unique_filenames:
            source_url = archive_base + filename
            metadata = file_meta_map.get(filename, {})
            head_response: Optional[httpx.Response] = None
            if include_http_metadata:
                head_response = await _await_if_needed(self._http_head(source_url, allow_redirects=True))
            descriptors.append(
                RemoteFileDescriptor(
                    name=filename,
                    source_url=source_url,
                    http_etag=_safe_header(head_response, "ETag"),
                    http_last_modified=_safe_header(head_response, "Last-Modified"),
                    remote_size=optional_int(_safe_header(head_response, "Content-Length")),
                    http_status=head_response.status_code if head_response else None,
                    sec_document_type=_normalize_sec_document_type(metadata.get("type")),
                    sec_description=normalize_optional_text(metadata.get("description")),
                )
            )
        return descriptors

    async def download_files_stream(
        self,
        remote_files: list[RemoteFileDescriptor],
        overwrite: bool,
        store_file: Callable[[str, BinaryIO], FileObjectMeta],
        existing_files: Optional[dict[str, dict[str, Any]]] = None,
        primary_document: Optional[str] = None,
    ) -> AsyncIterator[DownloaderEvent]:
        """下载远端文件列表并流式返回文件级事件。

        Args:
            remote_files: 远端文件描述列表。
            overwrite: 是否覆盖下载。
            store_file: 文件存储回调（入参：文件名、二进制流）。
            existing_files: 既有文件元数据映射（按文件名）。
            primary_document: 主文档文件名（如 *.htm）；若指定且该文件下载为 0 字节，
                立即停止生成器，后续文件不再下载，确保整个 filing 不落盘。

        Yields:
            文件级下载事件。

        Raises:
            OSError: 本地落盘失败时抛出。
        """

        previous_map = existing_files or {}
        for descriptor in remote_files:
            previous = previous_map.get(descriptor.name, {})
            previous_etag = str(previous.get("http_etag") or previous.get("etag") or "").strip() or None
            previous_last_modified = (
                str(previous.get("http_last_modified") or previous.get("last_modified") or "").strip() or None
            )
            if not overwrite:
                try:
                    status_code, payload = await _await_if_needed(
                        self._http_download_if_modified(
                            url=descriptor.source_url,
                            etag=previous_etag,
                            last_modified=previous_last_modified,
                        )
                    )
                except RuntimeError as exc:
                    # 捕获下载异常（如503等HTTP错误），转换为file_failed事件
                    yield DownloaderEvent(
                        event_type="file_failed",
                        name=descriptor.name,
                        source_url=descriptor.source_url,
                        http_etag=descriptor.http_etag,
                        http_last_modified=descriptor.http_last_modified,
                        http_status=None,
                        reason_code="download_error",
                        reason_message=str(exc),
                        error=str(exc),
                    )
                    continue
                if status_code == 304:
                    yield DownloaderEvent(
                        event_type="file_skipped",
                        name=descriptor.name,
                        source_url=descriptor.source_url,
                        http_etag=descriptor.http_etag,
                        http_last_modified=descriptor.http_last_modified,
                        http_status=304,
                        reason_code="not_modified",
                        reason_message="远端文件未修改，跳过重新下载",
                    )
                    continue
                if payload is None:
                    yield DownloaderEvent(
                        event_type="file_failed",
                        name=descriptor.name,
                        source_url=descriptor.source_url,
                        http_etag=descriptor.http_etag,
                        http_last_modified=descriptor.http_last_modified,
                        http_status=status_code,
                        reason_code="empty_response",
                        reason_message="下载失败，未返回内容",
                        error="下载失败，未返回内容",
                    )
                    continue
                if len(payload) == 0:
                    yield _build_empty_content_failure_event(descriptor, status_code)
                    if _should_abort_after_empty_primary(descriptor.name, primary_document):
                        # 主文档 0 字节：整个 filing 无效，中止下载，确保后续文件不落盘。
                        return
                    continue
                file_meta = store_file(descriptor.name, _to_binary_stream(payload))
                yield DownloaderEvent(
                    event_type="file_downloaded",
                    name=descriptor.name,
                    source_url=descriptor.source_url,
                    http_etag=descriptor.http_etag,
                    http_last_modified=descriptor.http_last_modified,
                    http_status=status_code,
                    file_meta=file_meta,
                )
                continue
            try:
                payload = await _await_if_needed(self._http_download(descriptor.source_url))
                if len(payload) == 0:
                    yield _build_empty_content_failure_event(descriptor, descriptor.http_status)
                    if _should_abort_after_empty_primary(descriptor.name, primary_document):
                        # 主文档 0 字节：整个 filing 无效，中止下载，确保后续文件不落盘。
                        return
                    continue
                file_meta = store_file(descriptor.name, _to_binary_stream(payload))
                yield DownloaderEvent(
                    event_type="file_downloaded",
                    name=descriptor.name,
                    source_url=descriptor.source_url,
                    http_etag=descriptor.http_etag,
                    http_last_modified=descriptor.http_last_modified,
                    http_status=descriptor.http_status,
                    file_meta=file_meta,
                )
            except RuntimeError as exc:
                yield DownloaderEvent(
                    event_type="file_failed",
                    name=descriptor.name,
                    source_url=descriptor.source_url,
                    http_etag=descriptor.http_etag,
                    http_last_modified=descriptor.http_last_modified,
                    http_status=descriptor.http_status,
                    reason_code="download_error",
                    reason_message=str(exc),
                    error=str(exc),
                )

    async def download_files(
        self,
        remote_files: list[RemoteFileDescriptor],
        overwrite: bool,
        store_file: Callable[[str, BinaryIO], FileObjectMeta],
        existing_files: Optional[dict[str, dict[str, Any]]] = None,
        primary_document: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """下载远端文件列表并聚合返回结果。

        Args:
            remote_files: 远端文件描述列表。
            overwrite: 是否覆盖下载。
            store_file: 文件存储回调（入参：文件名、二进制流）。
            existing_files: 既有文件元数据映射（按文件名）。
            primary_document: 主文档文件名，转发给 download_files_stream。

        Returns:
            单文件下载结果列表。

        Raises:
            OSError: 本地落盘失败时抛出。
        """

        results: list[dict[str, Any]] = []
        async for event in self.download_files_stream(
            remote_files=remote_files,
            overwrite=overwrite,
            store_file=store_file,
            existing_files=existing_files,
            primary_document=primary_document,
        ):
            if event.event_type == "file_downloaded":
                results.append(
                    {
                        "name": event.name,
                        "status": "downloaded",
                        "file_meta": event.file_meta,
                        "source_url": event.source_url,
                        "http_etag": event.http_etag,
                        "http_last_modified": event.http_last_modified,
                        "http_status": event.http_status,
                    }
                )
            elif event.event_type == "file_skipped":
                results.append(
                    {
                        "name": event.name,
                        "status": "skipped",
                        "source_url": event.source_url,
                        "http_etag": event.http_etag,
                        "http_last_modified": event.http_last_modified,
                        "http_status": event.http_status,
                        "reason_code": event.reason_code,
                        "reason_message": event.reason_message,
                    }
                )
            else:
                results.append(
                    {
                        "name": event.name,
                        "status": "failed",
                        "source_url": event.source_url,
                        "http_etag": event.http_etag,
                        "http_last_modified": event.http_last_modified,
                        "http_status": event.http_status,
                        "reason_code": event.reason_code,
                        "reason_message": event.reason_message,
                        "error": event.error,
                    }
                )
        return results

    async def _execute_sec_request(
        self,
        *,
        url: str,
        method: Literal["GET", "HEAD"],
        response_handler: Callable[[httpx.Response], _HttpResultT],
        handled_exceptions: tuple[type[Exception], ...],
        attempt_log_prefix: str,
        failure_prefix: str,
        extra_headers: Optional[dict[str, str]] = None,
        allow_redirects: bool = False,
    ) -> _HttpResultT:
        """执行带 SEC 限流与重试策略的 HTTP 请求。

        Args:
            url: 请求地址。
            method: HTTP 方法。
            response_handler: 成功响应处理器，可在其中执行 `raise_for_status()` 或解析响应体。
            handled_exceptions: 需要触发重试的异常类型集合。
            attempt_log_prefix: 单次失败时的日志前缀。
            failure_prefix: 重试耗尽后的异常前缀。
            extra_headers: 额外请求头。
            allow_redirects: `HEAD` 请求是否跟随重定向。

        Returns:
            `response_handler` 产出的结果。

        Raises:
            RuntimeError: 重试耗尽后抛出。
        """

        headers = self._build_headers()
        if extra_headers:
            headers.update(extra_headers)
        last_exception: Optional[Exception] = None
        throttle_retries_remaining = _SEC_THROTTLE_MAX_RETRIES
        attempt_index = 0
        while attempt_index < self._max_retries:
            await self._rate_limit()
            try:
                if method == "GET":
                    response = await self._client.get(
                        url=url,
                        headers=headers,
                        timeout=self._request_timeout_seconds,
                    )
                else:
                    response = await self._client.head(
                        url=url,
                        headers=headers,
                        timeout=self._request_timeout_seconds,
                        follow_redirects=allow_redirects,
                    )
                if response.status_code in _THROTTLE_STATUS_CODES and throttle_retries_remaining > 0:
                    throttle_retries_remaining -= 1
                    delay = _resolve_sec_throttle_delay(response)
                    self._register_global_throttle_cooldown(delay)
                    Log.warn(
                        f"SEC 限流 {response.status_code}: url={url} 等待 {delay:.1f}s",
                        module=self.MODULE,
                    )
                    await asyncio.sleep(delay)
                    continue
                return response_handler(response)
            except handled_exceptions as exc:
                last_exception = exc
                Log.debug(
                    f"{attempt_log_prefix}: url={url} attempt={attempt_index + 1} error={exc}",
                    module=self.MODULE,
                )
                await self._retry_backoff(attempt_index)
                attempt_index += 1
        raise RuntimeError(f"{failure_prefix}: url={url} error={last_exception}")

    async def _http_download_if_modified(
        self,
        url: str,
        etag: Optional[str],
        last_modified: Optional[str],
    ) -> tuple[int, Optional[bytes]]:
        """按条件请求下载文件，未修改时返回 304。

        Args:
            url: 文件 URL。
            etag: 远端 ETag（可选）。
            last_modified: 远端 Last-Modified（可选）。

        Returns:
            (HTTP 状态码, 内容字节)；`304` 表示未修改且内容为空。

        Raises:
            RuntimeError: 下载失败时抛出。
        """

        conditional_headers: dict[str, str] = {}
        if etag:
            conditional_headers["If-None-Match"] = etag
        if last_modified:
            conditional_headers["If-Modified-Since"] = last_modified
        if not conditional_headers:
            return 200, await self._http_download(url=url)

        return await self._execute_sec_request(
            url=url,
            method="GET",
            response_handler=_handle_conditional_download_response,
            handled_exceptions=(httpx.HTTPError,),
            attempt_log_prefix="条件下载失败",
            failure_prefix="条件下载失败",
            extra_headers=conditional_headers,
        )

    async def _http_get_json(self, url: str) -> dict[str, Any]:
        """执行 GET JSON 请求。

        Args:
            url: 请求地址。

        Returns:
            JSON 字典。

        Raises:
            RuntimeError: 请求失败时抛出。
        """

        return await self._execute_sec_request(
            url=url,
            method="GET",
            response_handler=_parse_http_json_response,
            handled_exceptions=(httpx.HTTPError, ValueError),
            attempt_log_prefix="GET JSON 失败",
            failure_prefix="GET JSON 失败",
        )

    async def _http_head(self, url: str, allow_redirects: bool) -> Optional[httpx.Response]:
        """执行 HEAD 请求。

        Args:
            url: 请求地址。
            allow_redirects: 是否跟随重定向。

        Returns:
            Response 对象；失败时返回 `None`。

        Raises:
            无。
        """

        try:
            return await self._execute_sec_request(
                url=url,
                method="HEAD",
                response_handler=_return_http_response,
                handled_exceptions=(httpx.HTTPError,),
                attempt_log_prefix="HEAD 失败",
                failure_prefix="HEAD 失败",
                allow_redirects=allow_redirects,
            )
        except RuntimeError:
            return None

    async def _http_download(self, url: str) -> bytes:
        """下载文件并返回内容。

        Args:
            url: 文件 URL。

        Returns:
            文件内容字节。

        Raises:
            RuntimeError: 下载失败时抛出。
        """

        return await self._execute_sec_request(
            url=url,
            method="GET",
            response_handler=_read_http_binary_response,
            handled_exceptions=(httpx.HTTPError,),
            attempt_log_prefix="下载失败",
            failure_prefix="下载失败",
        )

    async def _http_get_bytes(self, url: str) -> bytes:
        """执行 GET 请求并返回字节内容。

        Args:
            url: 请求地址。

        Returns:
            响应体字节。

        Raises:
            RuntimeError: 请求失败时抛出。
        """

        return await self._execute_sec_request(
            url=url,
            method="GET",
            response_handler=_read_http_binary_response,
            handled_exceptions=(httpx.HTTPError,),
            attempt_log_prefix="GET bytes 失败",
            failure_prefix="GET bytes 失败",
        )

    async def _try_fetch_index_items(self, cik: str, accession_no_dash: str) -> list[dict[str, Any]]:
        """尝试拉取 index.json 中的 item 列表。

        Args:
            cik: CIK（无前导零）。
            accession_no_dash: 无连字符 accession。

        Returns:
            index.json 的 item 列表；失败时返回空列表。

        Raises:
            无。
        """

        index_url = ARCHIVES_INDEX_JSON.format(
            cik=str(int(cik)),
            accession_no_dash=accession_no_dash,
        )
        try:
            index_json = await _await_if_needed(self._http_get_json(index_url))
        except RuntimeError as exc:
            Log.warn(f"读取 index.json 失败: {index_url} error={exc}", module=self.MODULE)
            return []
        return list(index_json.get("directory", {}).get("item", []) or [])

    async def _try_fetch_index_header_documents(
        self,
        cik: str,
        accession_no_dash: str,
    ) -> list[dict[str, Any]]:
        """尝试从 index-headers 页面解析文档条目。

        Args:
            cik: CIK（无前导零）。
            accession_no_dash: 无连字符 accession。

        Returns:
            文档条目列表；请求失败或解析失败时返回空列表。

        Raises:
            无。
        """

        accession = _format_accession_with_dash(accession_no_dash)
        index_headers_url = ARCHIVES_INDEX_HEADERS_HTML.format(
            cik=str(int(cik)),
            accession_no_dash=accession_no_dash,
            accession=accession,
        )
        try:
            payload = await _await_if_needed(self._http_get_bytes(index_headers_url))
        except RuntimeError as exc:
            Log.warn(
                f"读取 index-headers 失败: {index_headers_url} error={exc}",
                module=self.MODULE,
            )
            return []
        return _parse_index_header_document_entries(payload)

    async def _try_fetch_primary_linked_html_files(
        self,
        archive_base: str,
        primary_document: str,
    ) -> list[str]:
        """尝试从主文档补链同 filing 的相对 HTML 文件。

        Args:
            archive_base: filing archive 基础 URL。
            primary_document: 主文档文件名。

        Returns:
            归一化后的相对 HTML 文件名列表；请求失败时返回空列表。

        Raises:
            无。
        """

        primary_document_url = archive_base + primary_document
        try:
            payload = await _await_if_needed(self._http_get_bytes(primary_document_url))
        except RuntimeError as exc:
            Log.warn(
                f"读取主文档补链失败: {primary_document_url} error={exc}",
                module=self.MODULE,
            )
            return []
        return extract_same_filing_linked_html_files(
            payload=payload,
            primary_document=primary_document,
        )

    def _resolve_user_agent(self, configured_user_agent: Optional[str]) -> str:
        """解析 User-Agent。

        优先级：显式传入 > 环境变量 SEC_USER_AGENT > 未配置 fallback（附警告）。

        Args:
            configured_user_agent: 显式传入的 User-Agent。

        Returns:
            最终 User-Agent。

        Raises:
            无。
        """

        value = (configured_user_agent or os.environ.get(SEC_USER_AGENT_ENV) or "").strip()
        if value:
            return value
        Log.warning(
            f"SEC User-Agent 未配置。SEC 要求提供真实联系信息，否则可能限流或封禁。"
            f"请通过环境变量 {SEC_USER_AGENT_ENV} 或 dayu-cli init 配置。",
            module=self.MODULE,
        )
        return _UNCONFIGURED_USER_AGENT

    def _build_headers(self) -> dict[str, str]:
        """构建请求头。

        Args:
            无。

        Returns:
            请求头字典。

        Raises:
            无。
        """

        return {
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _reserve_global_request_slot(self, min_interval: float) -> float:
        """在共享状态中预留下一个请求时间片。

        设计目标：
        - 同一工作区下多个下载进程共享一个速率上限；
        - 命中 SEC 429/503 后，所有进程共享同一冷却窗口；
        - 保持决策与数据同源，避免每个进程各自“自认为没超速”。

        Args:
            min_interval: 请求最小间隔秒数。

        Returns:
            当前请求应等待的秒数。

        Raises:
            OSError: 状态文件读写失败时抛出。
        """

        now = time.time()
        with _sec_throttle_lock(self._throttle_lock_path):
            state = _load_sec_throttle_state(self._throttle_state_path)
            allowed_at = max(now, state.cooldown_until, state.next_request_at)
            next_state = _SecThrottleState(
                next_request_at=allowed_at + min_interval,
                cooldown_until=state.cooldown_until,
            )
            _save_sec_throttle_state(self._throttle_state_path, next_state)
        return max(allowed_at - now, 0.0)

    def _register_global_throttle_cooldown(self, delay_seconds: float) -> None:
        """登记 SEC 限流后的共享冷却窗口。

        Args:
            delay_seconds: 冷却秒数。

        Returns:
            无。

        Raises:
            OSError: 状态文件写入失败时抛出。
        """

        cooldown_until = time.time() + max(delay_seconds, 0.0)
        with _sec_throttle_lock(self._throttle_lock_path):
            state = _load_sec_throttle_state(self._throttle_state_path)
            next_state = _SecThrottleState(
                next_request_at=max(state.next_request_at, cooldown_until),
                cooldown_until=max(state.cooldown_until, cooldown_until),
            )
            _save_sec_throttle_state(self._throttle_state_path, next_state)

    async def _rate_limit(self) -> None:
        """基于单调时钟的请求速率限制器。

        确保相邻请求间隔 ≥ max(_SEC_MIN_REQUEST_INTERVAL_SECONDS, sleep_seconds)。
        在每次 HTTP 请求 **之前** 调用。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        min_interval = max(_SEC_MIN_REQUEST_INTERVAL_SECONDS, self._sleep_seconds)
        if min_interval <= 0:
            return
        # 先执行实例内限流，避免同一 event loop 内短时间打爆共享锁。
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        # 再执行跨进程共享限流，确保同一工作区下所有下载进程共用节流状态。
        shared_wait_seconds = self._reserve_global_request_slot(min_interval)
        if shared_wait_seconds > 0:
            await asyncio.sleep(shared_wait_seconds)
        self._last_request_time = time.monotonic()

    async def _retry_backoff(self, attempt_index: int) -> None:
        """执行指数退避。

        Args:
            attempt_index: 重试序号（从 0 开始）。

        Returns:
            无。

        Raises:
            无。
        """

        if attempt_index >= self._max_retries - 1:
            return
        delay = RETRY_BACKOFF_BASE_SECONDS * (2**attempt_index)
        await asyncio.sleep(delay)


def _parse_retry_after(response: httpx.Response) -> float:
    """解析 Retry-After 响应头，返回等待秒数。

    Args:
        response: HTTP 响应对象。

    Returns:
        等待秒数。如无 Retry-After 头则使用默认退避时间。

    Raises:
        无。
    """

    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    return _SEC_THROTTLE_BACKOFF_SECONDS


def _safe_header(response: Optional[httpx.Response], key: str) -> Optional[str]:
    """安全读取响应头。

    Args:
        response: Response 对象。
        key: 头字段名。

    Returns:
        响应头内容或 `None`。

    Raises:
        无。
    """

    if response is None:
        return None
    return response.headers.get(key)


def _to_binary_stream(payload: bytes) -> BinaryIO:
    """将 bytes 包装为可读二进制流。

    Args:
        payload: 内容字节。

    Returns:
        二进制流对象。

    Raises:
        无。
    """

    return BytesIO(payload)


@overload
async def _await_if_needed(value: Awaitable[_AwaitedValueT]) -> _AwaitedValueT:
    """按需等待可等待对象（awaitable 重载）。"""


@overload
async def _await_if_needed(value: _AwaitedValueT) -> _AwaitedValueT:
    """按需等待可等待对象（普通值重载）。"""


async def _await_if_needed(value: Awaitable[_AwaitedValueT] | _AwaitedValueT) -> _AwaitedValueT:
    """按需等待可等待对象。

    Args:
        value: 普通值或可等待对象。

    Returns:
        已解析后的值。

    Raises:
        RuntimeError: 可等待对象执行失败时抛出。
    """

    if inspect.isawaitable(value):
        return await cast(Awaitable[_AwaitedValueT], value)
    return cast(_AwaitedValueT, value)


def _parse_sc13_party_roles_from_index_headers(payload: bytes) -> Optional[Sc13PartyRoles]:
    """从 index-headers 页面内容解析 SC 13 双方 CIK。

    Args:
        payload: `-index-headers.html` 响应字节。

    Returns:
        解析成功时返回 `Sc13PartyRoles`，否则返回 `None`。

    Raises:
        无。
    """

    if not payload:
        return None
    text = payload.decode("utf-8", errors="ignore")
    filed_by_section = _extract_sc13_section_text(text=text, section_name="FILED BY")
    subject_section = _extract_sc13_section_text(text=text, section_name="SUBJECT COMPANY")
    if filed_by_section is None or subject_section is None:
        return None
    filed_by_cik = _extract_first_cik_from_text(filed_by_section)
    subject_cik = _extract_first_cik_from_text(subject_section)
    if filed_by_cik is None or subject_cik is None:
        return None
    return Sc13PartyRoles(filed_by_cik=filed_by_cik, subject_cik=subject_cik)


def _format_accession_with_dash(accession_no_dash: str) -> str:
    """将无连字符 accession 转换为 SEC 标准格式。

    Args:
        accession_no_dash: 无连字符 accession。

    Returns:
        标准 accession（`xxxxxxxxxx-xx-xxxxxx`）；无法匹配时返回原值。

    Raises:
        无。
    """

    normalized = str(accession_no_dash or "").strip()
    matched = re.fullmatch(r"(\d{10})(\d{2})(\d{6})", normalized)
    if matched is None:
        return normalized
    return f"{matched.group(1)}-{matched.group(2)}-{matched.group(3)}"


def _parse_index_header_document_entries(payload: bytes) -> list[dict[str, str]]:
    """解析 `-index-headers.html` 中的 `<DOCUMENT>` 条目。

    SEC 的 index-headers 页面通常把 SGML 文本转义后放入 `<pre>`，本函数先做
    HTML 反转义，再按 `<DOCUMENT>...</DOCUMENT>` 块提取 `TYPE/FILENAME/DESCRIPTION`。

    Args:
        payload: index-headers 响应字节。

    Returns:
        解析出的文档条目列表（每项包含 `name/type/description`）。

    Raises:
        无。
    """

    if not payload:
        return []
    raw_text = payload.decode("utf-8", errors="ignore")
    normalized_text = html.unescape(raw_text)
    document_blocks = re.findall(
        r"(?is)<DOCUMENT>\s*(.*?)\s*</DOCUMENT>",
        normalized_text,
    )
    documents: list[dict[str, str]] = []
    for block in document_blocks:
        filename_match = re.search(r"(?im)^\s*<FILENAME>\s*([^\n<]+)\s*$", block)
        if filename_match is None:
            continue
        name = filename_match.group(1).strip()
        if not name:
            continue
        type_match = re.search(r"(?im)^\s*<TYPE>\s*([^\n<]+)\s*$", block)
        description_match = re.search(r"(?im)^\s*<DESCRIPTION>\s*([^\n<]+)\s*$", block)
        documents.append(
            {
                "name": name,
                "type": (type_match.group(1).strip() if type_match else ""),
                "description": (description_match.group(1).strip() if description_match else ""),
            }
        )
    return documents


def _extract_sc13_section_text(
    text: str,
    section_name: str,
) -> Optional[str]:
    """抽取 index-headers 中指定主体段落文本。

    Args:
        text: 页面纯文本。
        section_name: 段落名（如 `FILED BY`、`SUBJECT COMPANY`）。

    Returns:
        段落文本；未找到时返回 `None`。

    Raises:
        无。
    """

    headers = list(re.finditer(r"(?im)^\s*(FILED BY|SUBJECT COMPANY)\s*:\s*$", text))
    if not headers:
        return None
    normalized_target = section_name.strip().upper()
    for index, matched in enumerate(headers):
        current_name = matched.group(1).strip().upper()
        if current_name != normalized_target:
            continue
        section_start = matched.end()
        section_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        return text[section_start:section_end]
    return None


def _normalize_sec_document_type(value: Any) -> Optional[str]:
    """规范化 SEC 文档类型。

    `index.json` 中常见 `type=text.gif`，并非真实文档类型，这里会过滤掉。

    Args:
        value: 原始类型值。

    Returns:
        规范化后的类型字符串；无效值返回 `None`。

    Raises:
        无。
    """

    normalized = normalize_optional_text(value)
    if normalized is None:
        return None
    upper = normalized.upper()
    if upper == "TEXT.GIF":
        return None
    return upper


def _build_file_metadata_map(
    index_items: list[dict[str, Any]],
    index_header_documents: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """构建文件名到元数据的映射。

    合并优先级：
    - 先写入 `index.json` 条目；
    - 再用 `index-headers` 条目覆盖（其 `TYPE/DESCRIPTION` 更可靠）。

    Args:
        index_items: `index.json` 文档条目。
        index_header_documents: `index-headers` 解析文档条目。

    Returns:
        文件名到元数据的映射。

    Raises:
        无。
    """

    mapping: dict[str, dict[str, str]] = {}
    for item in [*index_items, *index_header_documents]:
        if not isinstance(item, dict):
            continue
        name = normalize_optional_text(item.get("name"))
        if name is None:
            continue
        target = mapping.setdefault(name, {})
        document_type = _normalize_sec_document_type(item.get("type"))
        if document_type is not None:
            target["type"] = document_type
        description = normalize_optional_text(item.get("description"))
        if description is not None:
            target["description"] = description
    return mapping


def _extract_first_cik_from_text(text: str) -> Optional[str]:
    """从段落文本中提取首个 `CENTRAL INDEX KEY`。

    Args:
        text: 段落文本。

    Returns:
        规范化后的 CIK 字符串；未命中返回 `None`。

    Raises:
        无。
    """

    matched = re.search(r"(?im)^\s*CENTRAL INDEX KEY:\s*([0-9]+)\s*$", text)
    if matched is None:
        return None
    return _normalize_cik_value(matched.group(1))


def _normalize_cik_value(raw_cik: Any) -> Optional[str]:
    """将 CIK 规范化为不带前导零的数字字符串。

    Args:
        raw_cik: 原始 CIK。

    Returns:
        规范化 CIK；非法输入返回 `None`。

    Raises:
        无。
    """

    normalized = str(raw_cik or "").strip()
    if not normalized.isdigit():
        return None
    return str(int(normalized))


def _parse_browse_edgar_atom(payload: bytes) -> list[BrowseEdgarFiling]:
    """解析 browse-edgar Atom 输出。

    Args:
        payload: XML 字节内容。

    Returns:
        filings 列表。

    Raises:
        RuntimeError: XML 解析失败时抛出。
    """

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:  # noqa: BLE001
        raise RuntimeError("browse-edgar XML 解析失败") from exc
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results: list[BrowseEdgarFiling] = []
    for entry in root.findall("a:entry", ns):
        title = entry.findtext("a:title", default="", namespaces=ns).strip()
        updated = entry.findtext("a:updated", default="", namespaces=ns).strip()
        link = entry.find("a:link", ns)
        href = link.get("href", "") if link is not None else ""
        if not title or not href:
            continue
        form_type = _extract_form_from_title(title)
        filing_date = updated.split("T")[0] if updated else ""
        accession_number, cik = _parse_browse_edgar_href(href)
        if not accession_number or not cik or not filing_date:
            continue
        results.append(
            BrowseEdgarFiling(
                form_type=form_type,
                filing_date=filing_date,
                accession_number=accession_number,
                cik=cik,
                index_url=href,
            )
        )
    return results


def _extract_form_from_title(title: str) -> str:
    """从 browse-edgar title 中提取 form 类型。

    Args:
        title: 原始 title 文本。

    Returns:
        form 类型。

    Raises:
        无。
    """

    prefix = title.split(" - ", 1)[0].strip()
    return re.sub(r"\s*\[.*\]\s*$", "", prefix).strip()


def _parse_browse_edgar_href(href: str) -> tuple[str, str]:
    """从 browse-edgar 链接提取 accession 与 CIK。

    Args:
        href: 链接地址。

    Returns:
        (accession_number, cik)；解析失败返回空字符串。

    Raises:
        无。
    """

    accession_match = re.search(r"/(\d{10}-\d{2}-\d{6})-index\.htm", href)
    if accession_match is None:
        accession_match = re.search(r"/(\d{10}-\d{2}-\d{6})-index\.html", href)
    accession_number = accession_match.group(1) if accession_match else ""
    cik_match = re.search(r"/data/(\d+)/", href)
    cik = cik_match.group(1) if cik_match else ""
    return accession_number, cik


def _select_primary_from_index_items(items: list[dict[str, Any]], form_type: str) -> str:
    """从 index.json 条目中选择主文件名。

    Args:
        items: index.json 的 item 列表。
        form_type: 表单类型。

    Returns:
        主文件名；无法判断时返回空字符串。

    Raises:
        无。
    """

    normalized_form = str(form_type).strip().upper()
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip().upper()
        name = str(item.get("name", "")).strip()
        if item_type == normalized_form and name:
            return name
    candidates: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        lower = name.lower()
        if lower.endswith((".htm", ".html", ".txt")):
            candidates.append(name)
    if not candidates:
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name:
                candidates.append(name)
    return candidates[0] if candidates else ""
