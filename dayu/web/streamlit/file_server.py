"""Streamlit 应用配套的本地文件服务。

设计目标：
- 在 Streamlit 进程内随启动一次后台 HTTP 服务，仅监听 127.0.0.1，
  暴露工作区下的财报文件供浏览器在新标签内查看。
- 严格限制可访问范围在 `workspace_root/portfolio/<ticker>/filings/<document_id>/`
  之内，禁止任何路径穿越或目录列表。
- 与 Streamlit / FastAPI 主进程解耦：使用标准库 `http.server`，零额外依赖。

启动后通过 `start_file_server()` 返回的 `FileServerHandle` 可获取实际监听端口，
用于前端拼接 `http://127.0.0.1:<port>/files/<ticker>/filing/<document_id>/<filename>`
形式的下载/预览链接。
"""

from __future__ import annotations

import logging
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlparse

_LOGGER = logging.getLogger(__name__)

_FILING_URL_PREFIX = "/files/"
_SUPPORTED_SOURCE_KIND = "filing"
_HOST_BIND = "127.0.0.1"
_FILING_DIRECTORY_NAME = "filings"
_PORTFOLIO_DIRECTORY_NAME = "portfolio"
_HTML_DEFAULT_CONTENT_TYPE = "text/html; charset=utf-8"
_BINARY_DEFAULT_CONTENT_TYPE = "application/octet-stream"


class FileServerHandle(NamedTuple):
    """文件服务运行句柄。

    Attributes:
        host: 实际监听的主机地址，固定 127.0.0.1。
        port: 实际监听的端口（随机分配后稳定）。
        workspace_root: 服务允许访问的工作区根目录。
    """

    host: str
    port: int
    workspace_root: Path

    @property
    def base_url(self) -> str:
        """返回服务的基础 URL（不带末尾斜杠）。"""

        return f"http://{self.host}:{self.port}"

    def build_filing_url(self, ticker: str, document_id: str, filename: str) -> str:
        """构造财报文件访问 URL。

        Args:
            ticker: 股票代码。
            document_id: 财报文档 ID。
            filename: 文件名（不含目录分隔符）。

        Returns:
            可在浏览器新标签中打开的 HTTP URL。

        Raises:
            ValueError: 当任一参数为空或包含路径分隔符时抛出。
        """

        from urllib.parse import quote

        for component_name, component_value in (
            ("ticker", ticker),
            ("document_id", document_id),
            ("filename", filename),
        ):
            if not component_value or "/" in component_value or "\\" in component_value:
                raise ValueError(f"非法的 URL 组件 {component_name}: {component_value!r}")

        encoded_ticker = quote(ticker, safe="")
        encoded_document_id = quote(document_id, safe="")
        encoded_filename = quote(filename, safe="")
        return (
            f"{self.base_url}{_FILING_URL_PREFIX}"
            f"{encoded_ticker}/{_SUPPORTED_SOURCE_KIND}/"
            f"{encoded_document_id}/{encoded_filename}"
        )


def start_file_server(workspace_root: Path) -> FileServerHandle:
    """启动 Streamlit 配套的本地文件服务。

    Args:
        workspace_root: 工作区根目录，服务仅允许访问该目录下的财报文件。

    Returns:
        文件服务运行句柄，包含监听端口与允许访问的根目录。

    Raises:
        FileNotFoundError: 当 `workspace_root` 不存在或不是目录时抛出。
        OSError: 启动 HTTP 监听失败时抛出。
    """

    resolved_root = workspace_root.resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"工作区目录不存在或不是目录: {resolved_root}")

    handler_class = _build_handler_class(resolved_root)
    server = ThreadingHTTPServer((_HOST_BIND, 0), handler_class)
    actual_host, actual_port = server.server_address[:2]
    if not isinstance(actual_port, int):
        server.server_close()
        raise OSError(f"无法解析文件服务监听端口: {server.server_address!r}")

    thread = threading.Thread(
        target=server.serve_forever,
        name=f"dayu-file-server-{actual_port}",
        daemon=True,
    )
    thread.start()
    _LOGGER.info(
        "本地文件服务已启动: host=%s port=%s workspace=%s",
        actual_host,
        actual_port,
        resolved_root,
    )
    return FileServerHandle(
        host=str(actual_host),
        port=int(actual_port),
        workspace_root=resolved_root,
    )


def resolve_filing_file(
    workspace_root: Path,
    ticker: str,
    document_id: str,
    filename: str,
) -> Path:
    """解析受信的财报文件绝对路径，禁止越权访问。

    Args:
        workspace_root: 工作区根目录的绝对路径。
        ticker: 股票代码。
        document_id: 财报文档 ID。
        filename: 文件名。

    Returns:
        受信的绝对文件路径。

    Raises:
        PermissionError: 当解析后路径越出 `filings/<document_id>/` 范围时抛出。
        FileNotFoundError: 当目标文件不存在或不是常规文件时抛出。
    """

    if not ticker or not document_id or not filename:
        raise PermissionError("ticker / document_id / filename 不能为空")

    if any(_is_unsafe_segment(segment) for segment in (ticker, document_id, filename)):
        raise PermissionError("路径段包含非法字符")

    document_root = (
        workspace_root
        / _PORTFOLIO_DIRECTORY_NAME
        / ticker
        / _FILING_DIRECTORY_NAME
        / document_id
    ).resolve()
    candidate = (document_root / filename).resolve()

    if not _is_within_directory(candidate, document_root):
        raise PermissionError("目标路径越出允许范围")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"文件不存在: {candidate}")
    return candidate


def _build_handler_class(workspace_root: Path) -> type[BaseHTTPRequestHandler]:
    """构造与给定 workspace 绑定的 HTTP 处理器类。

    Args:
        workspace_root: 工作区根目录的绝对路径。

    Returns:
        `BaseHTTPRequestHandler` 子类，用于 `ThreadingHTTPServer`。

    Raises:
        无。
    """

    class _FilingHttpRequestHandler(BaseHTTPRequestHandler):
        """财报文件只读 HTTP 处理器。"""

        server_version = "DayuFileServer/1.0"

        def do_GET(self) -> None:  # noqa: N802 - http.server 约定方法名
            """处理 GET 请求，仅允许 `/files/<ticker>/filing/<doc_id>/<filename>`。"""

            self._handle_request(write_body=True)

        def do_HEAD(self) -> None:  # noqa: N802 - http.server 约定方法名
            """处理 HEAD 请求，返回与 GET 相同的响应头但不写入响应体。"""

            self._handle_request(write_body=False)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - 与父类签名一致
            """将访问日志重定向到 `logging`，避免污染 stderr。"""

            _LOGGER.debug("file-server: " + format, *args)

        def _handle_request(self, *, write_body: bool) -> None:
            """统一处理 GET/HEAD，按需要决定是否写入响应体。

            Args:
                write_body: True 表示写入文件内容，False 表示仅返回响应头。
            """

            try:
                target_path, content_type = self._resolve_target()
            except PermissionError as exc:
                self._send_text_response(HTTPStatus.FORBIDDEN, str(exc))
                return
            except FileNotFoundError as exc:
                self._send_text_response(HTTPStatus.NOT_FOUND, str(exc))
                return
            except ValueError as exc:
                self._send_text_response(HTTPStatus.BAD_REQUEST, str(exc))
                return

            try:
                content_length = target_path.stat().st_size
            except OSError as exc:
                self._send_text_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

            if not write_body:
                return
            try:
                with target_path.open("rb") as stream:
                    while True:
                        chunk = stream.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (OSError, ConnectionError) as exc:
                _LOGGER.warning("写入文件响应失败: %s", exc)

        def _resolve_target(self) -> tuple[Path, str]:
            """根据请求路径解析目标文件与响应 Content-Type。

            Returns:
                二元组 `(目标文件绝对路径, Content-Type)`。

            Raises:
                ValueError: URL 结构不合法时抛出。
                PermissionError: 命中安全校验失败时抛出。
                FileNotFoundError: 目标文件不存在时抛出。
            """

            parsed_url = urlparse(self.path)
            url_path = parsed_url.path
            if not url_path.startswith(_FILING_URL_PREFIX):
                raise ValueError("非法路径前缀")

            relative_path = url_path[len(_FILING_URL_PREFIX):]
            segments = relative_path.split("/")
            if len(segments) != 4:
                raise ValueError("URL 必须形如 /files/<ticker>/filing/<doc_id>/<filename>")

            ticker_segment, source_kind_segment, document_id_segment, filename_segment = segments
            if source_kind_segment != _SUPPORTED_SOURCE_KIND:
                raise PermissionError(f"不支持的 source_kind: {source_kind_segment}")

            ticker = unquote(ticker_segment)
            document_id = unquote(document_id_segment)
            filename = unquote(filename_segment)

            target_path = resolve_filing_file(
                workspace_root=workspace_root,
                ticker=ticker,
                document_id=document_id,
                filename=filename,
            )
            return target_path, _guess_content_type(target_path)

        def _send_text_response(self, status: HTTPStatus, message: str) -> None:
            """统一发送纯文本错误响应。

            Args:
                status: HTTP 状态码。
                message: 错误描述文本。
            """

            payload = message.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (OSError, ConnectionError) as exc:
                _LOGGER.debug("写入错误响应失败: %s", exc)

    return _FilingHttpRequestHandler


def _is_unsafe_segment(segment: str) -> bool:
    """判断单个路径段是否包含非法字符。

    Args:
        segment: URL 路径单段。

    Returns:
        True 表示包含 `/`、`\\`、`..` 或为空。
    """

    if not segment:
        return True
    if segment in (".", ".."):
        return True
    if "/" in segment or "\\" in segment:
        return True
    return False


def _is_within_directory(candidate: Path, directory: Path) -> bool:
    """判断 `candidate` 是否处于 `directory` 内（含相等）。

    Args:
        candidate: 待检查的绝对路径。
        directory: 受信目录的绝对路径。

    Returns:
        True 表示候选路径处于目录范围内。
    """

    try:
        candidate.relative_to(directory)
        return True
    except ValueError:
        return False


def _guess_content_type(path: Path) -> str:
    """推断文件的 HTTP Content-Type。

    Args:
        path: 文件绝对路径。

    Returns:
        Content-Type 字符串；HTML 文件默认带 UTF-8 charset，
        无法识别时退回 `application/octet-stream`。
    """

    guessed_type, _ = mimetypes.guess_type(str(path))
    if guessed_type is None:
        return _BINARY_DEFAULT_CONTENT_TYPE
    if guessed_type.startswith("text/html"):
        return _HTML_DEFAULT_CONTENT_TYPE
    if guessed_type.startswith("text/") and "charset" not in guessed_type:
        return f"{guessed_type}; charset=utf-8"
    return guessed_type
