"""网页抓取的 Playwright backend 基础设施。

本模块只承载浏览器单例、同步 worker、storage state 解析与
Playwright 回退执行逻辑，不包含 requests 主路径编排或工具注册。
"""

from __future__ import annotations

import atexit
import math
import multiprocessing
import os
import pickle
import time
from multiprocessing.process import BaseProcess
from queue import Empty
from threading import Lock
from typing import Any, Callable
from urllib.parse import urlparse

import requests
from dayu.contracts.cancellation import CancellationToken

from dayu.log import Log
from dayu.contracts.cancellation import CancelledError

MODULE = "ENGINE.WEB_PLAYWRIGHT"

_DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
_DEFAULT_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
_DEFAULT_ACCEPT_LANGUAGE = "zh-CN,zh;q=0.9,en;q=0.8"
_DEFAULT_SEC_CH_UA = '"Chromium";v="131", "Google Chrome";v="131", "Not_A Brand";v="24"'
_DEFAULT_SEC_CH_UA_MOBILE = "?0"
_DEFAULT_SEC_CH_UA_PLATFORM = '"macOS"'

_PW_INSTANCE: Any | None = None
_PW_BROWSER: Any | None = None
_PW_BROWSER_KEY: tuple[str | None, bool] | None = None
_PW_LOCK = Lock()
_PW_RESULT_EXTRA_TIMEOUT_SECONDS = 10
_PW_NAVIGATION_WAIT_UNTIL = "domcontentloaded"
_PW_POST_NAVIGATION_SETTLE_MS = 1000
_PW_HOME_WARMUP_TIMEOUT_MS = 2500
_PW_LOAD_STATE_TIMEOUT_MS = 2500
_PW_NETWORK_IDLE_TIMEOUT_MS = 1500
_PW_RESULT_POLL_INTERVAL_SECONDS = 0.05
_PW_RESULT_DRAIN_GRACE_SECONDS = 0.5
_PW_PROCESS_TERMINATE_GRACE_SECONDS = 1.0


def _playwright_process_entry(
    result_queue: Any,
    worker_callable: Callable[..., dict[str, Any]],
    worker_kwargs: dict[str, Any],
) -> None:
    """子进程入口：执行同步 Playwright worker 并回传结果。

    Args:
        result_queue: 结果队列。
        worker_callable: 同步 worker 函数。
        worker_kwargs: worker 关键字参数。

    Returns:
        无。

    Raises:
        无。
    """

    try:
        result_queue.put({
            "kind": "result",
            "payload": worker_callable(**worker_kwargs),
        })
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )


def _is_picklable_worker(worker_callable: Callable[..., dict[str, Any]]) -> bool:
    """判断给定 worker 是否可安全发送到子进程。"""

    try:
        pickle.dumps(worker_callable)
    except Exception:
        return False
    return True


def _terminate_playwright_process(process: BaseProcess) -> None:
    """尽力终止 Playwright worker 进程。"""

    if not process.is_alive():
        process.join(timeout=0)
        return
    process.terminate()
    process.join(timeout=_PW_PROCESS_TERMINATE_GRACE_SECONDS)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=_PW_PROCESS_TERMINATE_GRACE_SECONDS)


def _poll_playwright_result_queue(
    *,
    result_queue: Any,
    timeout: float,
) -> dict[str, Any] | None:
    """在限定时间内轮询 Playwright worker 结果队列。

    Args:
        result_queue: 子进程结果队列。
        timeout: 本次轮询允许等待的秒数。

    Returns:
        读到结果时返回结果字典；本轮无结果时返回 ``None``。

    Raises:
        无。
    """

    try:
        if timeout <= 0:
            return result_queue.get_nowait()
        return result_queue.get(timeout=timeout)
    except Empty:
        return None


def _close_playwright_result_queue(result_queue: Any) -> None:
    """关闭父进程侧结果队列句柄。

    Args:
        result_queue: 子进程结果队列。

    Returns:
        无。

    Raises:
        无。
    """

    try:
        result_queue.close()
    except Exception:
        pass
    try:
        result_queue.join_thread()
    except Exception:
        pass


def _run_playwright_worker_process(
    *,
    playwright_sync_worker: Callable[..., dict[str, Any]],
    worker_kwargs: dict[str, Any],
    total_timeout: float,
    cancellation_token: CancellationToken | None,
) -> dict[str, Any]:
    """在子进程边界执行 Playwright worker，并在超时或取消时硬终止。

    Args:
        playwright_sync_worker: 同步 worker 函数。
        worker_kwargs: worker 关键字参数。
        total_timeout: 父进程等待总时长。
        cancellation_token: 当前工具调用的取消令牌。

    Returns:
        worker 返回的抓取结果。

    Raises:
        TimeoutError: worker 超时未返回时抛出。
        CancelledError: 当前调用已被取消时抛出。
        RuntimeError: worker 异常退出或未回传结果时抛出。
    """

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_playwright_process_entry,
        args=(result_queue, playwright_sync_worker, worker_kwargs),
    )
    process.daemon = True
    process.start()
    deadline = time.monotonic() + max(total_timeout, 0.0)
    result_drain_deadline: float | None = None
    try:
        while True:
            if cancellation_token is not None and cancellation_token.is_cancelled():
                _terminate_playwright_process(process)
                cancellation_token.raise_if_cancelled()
            current_time = time.monotonic()
            active_deadline = result_drain_deadline if result_drain_deadline is not None else deadline
            if current_time >= active_deadline:
                if result_drain_deadline is not None:
                    raise RuntimeError("playwright worker exited without result")
                _terminate_playwright_process(process)
                raise TimeoutError("playwright worker timeout")

            payload = _poll_playwright_result_queue(
                result_queue=result_queue,
                timeout=min(_PW_RESULT_POLL_INTERVAL_SECONDS, max(0.0, active_deadline - current_time)),
            )
            if payload is not None:
                process.join(timeout=0)
                break

            if not process.is_alive() and result_drain_deadline is None:
                process.join(timeout=0)
                result_drain_deadline = min(
                    deadline,
                    time.monotonic() + _PW_RESULT_DRAIN_GRACE_SECONDS,
                )
                continue

        if payload.get("kind") == "error":
            raise RuntimeError(
                f"{payload.get('error_type')}: {payload.get('message')}"
            )
        return payload["payload"]
    finally:
        if process.is_alive():
            _terminate_playwright_process(process)
        _close_playwright_result_queue(result_queue)


def _close_playwright_browser() -> None:
    """关闭 Playwright Browser 和运行时单例。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """

    global _PW_BROWSER, _PW_INSTANCE, _PW_BROWSER_KEY
    try:
        if _PW_BROWSER is not None:
            _PW_BROWSER.close()
    except Exception:
        pass
    try:
        if _PW_INSTANCE is not None:
            _PW_INSTANCE.stop()
    except Exception:
        pass
    _PW_BROWSER = None
    _PW_INSTANCE = None
    _PW_BROWSER_KEY = None


atexit.register(_close_playwright_browser)


def _normalize_playwright_channel(playwright_channel: str | None) -> str | None:
    """标准化 Playwright channel 配置。

    Args:
        playwright_channel: 原始 channel 配置。

    Returns:
        规整后的 channel；空字符串时返回 `None`。

    Raises:
        无。
    """

    if playwright_channel is None:
        return None
    normalized = str(playwright_channel).strip()
    return normalized or None


def _normalize_playwright_storage_state_dir(path_value: str | None) -> str | None:
    """标准化 Playwright storage state 目录路径。

    Args:
        path_value: 原始路径配置。

    Returns:
        目录路径字符串；未配置时返回 `None`。

    Raises:
        无。
    """

    if path_value is None:
        return None
    normalized = str(path_value).strip()
    if not normalized:
        return None
    return os.path.expanduser(normalized)


def _resolve_playwright_storage_state_path(
    *,
    url: str,
    playwright_storage_state_dir: str | None,
) -> str:
    """按 host 解析 Playwright storage state 文件路径。

    Args:
        url: 当前抓取 URL。
        playwright_storage_state_dir: storage state 目录配置。

    Returns:
        命中的 storage state 文件绝对路径；未命中时返回空字符串。

    Raises:
        无。
    """

    normalized_dir = _normalize_playwright_storage_state_dir(playwright_storage_state_dir)
    if normalized_dir is None:
        return ""
    host = (urlparse(url).hostname or "").strip().lower()
    if not host:
        return ""
    host_candidates = [host]
    if host.startswith("www."):
        stripped_host = host[4:]
        if stripped_host:
            host_candidates.append(stripped_host)
    else:
        host_candidates.append(f"www.{host}")

    for candidate_host in dict.fromkeys(host_candidates):
        candidate = os.path.join(normalized_dir, f"{candidate_host}.json")
        if os.path.isfile(candidate):
            return candidate
    return ""


def _get_playwright_browser(
    *,
    playwright_channel: str | None = None,
    headless: bool = True,
) -> Any | None:
    """获取或懒初始化全局 Playwright Browser 单例。

    Args:
        playwright_channel: 浏览器 channel 配置。
        headless: 是否以 headless 方式启动。

    Returns:
        Playwright Browser 单例；不可用时返回 `None`。

    Raises:
        无。
    """

    global _PW_INSTANCE, _PW_BROWSER, _PW_BROWSER_KEY
    browser_key = (_normalize_playwright_channel(playwright_channel), bool(headless))
    if _PW_BROWSER is not None and _PW_BROWSER_KEY == browser_key:
        return _PW_BROWSER
    with _PW_LOCK:
        if _PW_BROWSER is not None and _PW_BROWSER_KEY == browser_key:
            return _PW_BROWSER
        if _PW_BROWSER is not None or _PW_INSTANCE is not None:
            _close_playwright_browser()
        try:
            from playwright.sync_api import sync_playwright

            pw = sync_playwright().start()
            launch_kwargs: dict[str, Any] = {"headless": bool(headless)}
            if browser_key[0] is not None:
                launch_kwargs["channel"] = browser_key[0]
            launch_kwargs["args"] = ["--disable-blink-features=AutomationControlled"]
            browser = pw.chromium.launch(**launch_kwargs)
            _PW_INSTANCE = pw
            _PW_BROWSER = browser
            _PW_BROWSER_KEY = browser_key
        except Exception as exc:
            Log.warning(f"Playwright 浏览器初始化失败，回退不可用: {exc}", module=MODULE)
            return None
    return _PW_BROWSER


def _route_handler_abort_resources(route: Any) -> None:
    """中止图片、字体、媒体请求，放行其余资源。

    Args:
        route: Playwright Route 对象。

    Returns:
        无。

    Raises:
        无。
    """

    abort_resource_types = {"image", "font", "media"}
    if route.request.resource_type in abort_resource_types:
        route.abort()
    else:
        route.continue_()


def _get_remaining_playwright_timeout_ms(
    deadline_monotonic: float,
    *,
    time_monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """计算 Playwright 当前阶段剩余超时。

    Args:
        deadline_monotonic: 本次浏览器抓取总预算 deadline。
        time_monotonic: 可注入的单调时钟函数。

    Returns:
        剩余可用毫秒数；预算已耗尽时返回 0。

    Raises:
        无。
    """

    remaining_seconds = max(0.0, deadline_monotonic - time_monotonic())
    return max(0, math.ceil(remaining_seconds * 1000))


def _require_playwright_timeout_ms(
    deadline_monotonic: float,
    *,
    time_monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """为必需的 Playwright 阶段解析剩余超时。

    Args:
        deadline_monotonic: 本次浏览器抓取总预算 deadline。
        time_monotonic: 可注入的单调时钟函数。

    Returns:
        当前阶段可用的毫秒超时。

    Raises:
        RuntimeError: 当浏览器总预算已耗尽时抛出。
    """

    remaining_timeout_ms = _get_remaining_playwright_timeout_ms(
        deadline_monotonic,
        time_monotonic=time_monotonic,
    )
    if remaining_timeout_ms <= 0:
        raise RuntimeError("Playwright 页面加载超时: browser deadline exceeded")
    return remaining_timeout_ms


def _maybe_warmup_playwright_page(
    *,
    page: Any,
    url: str,
    deadline_monotonic: float,
    build_domain_home_url: Callable[[str], str],
    normalize_url_for_http: Callable[[str], str],
    time_monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """在浏览器回退前先做一次同域首页预热。

    Args:
        page: Playwright Page。
        url: 目标 URL。
        deadline_monotonic: 本次浏览器抓取总预算 deadline。
        build_domain_home_url: 同域首页构造函数。
        normalize_url_for_http: URL 规范化函数。
        time_monotonic: 可注入的单调时钟函数。

    Returns:
        无。

    Raises:
        无。
    """

    try:
        home_url = build_domain_home_url(url)
        normalized_url = normalize_url_for_http(url)
    except ValueError:
        return

    if home_url == normalized_url:
        return

    remaining_timeout_ms = _get_remaining_playwright_timeout_ms(
        deadline_monotonic,
        time_monotonic=time_monotonic,
    )
    warmup_timeout_ms = min(remaining_timeout_ms, _PW_HOME_WARMUP_TIMEOUT_MS)
    if warmup_timeout_ms <= 0:
        return

    try:
        page.goto(home_url, wait_until="domcontentloaded", timeout=warmup_timeout_ms)
    except Exception:
        return


def _settle_playwright_page(
    *,
    page: Any,
    deadline_monotonic: float,
    time_monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """在浏览器导航后做有上限的页面稳定化等待。

    Args:
        page: Playwright Page。
        deadline_monotonic: 本次浏览器抓取总预算 deadline。
        time_monotonic: 可注入的单调时钟函数。

    Returns:
        无。

    Raises:
        无。
    """

    wait_for_load_state = getattr(page, "wait_for_load_state", None)
    if callable(wait_for_load_state):
        for state, budget_ms in (
            ("load", _PW_LOAD_STATE_TIMEOUT_MS),
            ("networkidle", _PW_NETWORK_IDLE_TIMEOUT_MS),
        ):
            remaining_timeout_ms = _get_remaining_playwright_timeout_ms(
                deadline_monotonic,
                time_monotonic=time_monotonic,
            )
            step_timeout_ms = min(remaining_timeout_ms, budget_ms)
            if step_timeout_ms <= 0:
                return
            try:
                wait_for_load_state(state, timeout=step_timeout_ms)
            except Exception:
                continue

    wait_for_timeout = getattr(page, "wait_for_timeout", None)
    if callable(wait_for_timeout):
        remaining_timeout_ms = _get_remaining_playwright_timeout_ms(
            deadline_monotonic,
            time_monotonic=time_monotonic,
        )
        step_timeout_ms = min(remaining_timeout_ms, _PW_POST_NAVIGATION_SETTLE_MS)
        if step_timeout_ms <= 0:
            return
        wait_for_timeout(step_timeout_ms)


def _playwright_sync_worker(
    *,
    url: str,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    playwright_channel: str | None = None,
    playwright_storage_state_path: str = "",
    get_playwright_browser: Callable[..., Any | None],
    build_domain_home_url: Callable[[str], str],
    normalize_url_for_http: Callable[[str], str],
    sanitize_response_headers: Callable[[Any], dict[str, str]],
    build_text_excerpt: Callable[[str], str],
    convert_html_to_markdown: Callable[..., Any],
    time_monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """在独立线程中执行完整的 Playwright 同步抓取流程。

    Args:
        url: 已通过安全校验的网页链接。
        timeout_seconds: 本次浏览器回退总预算秒数。
        headers: 可选额外请求头；当前保留作未来扩展。
        playwright_channel: 浏览器回退使用的 Chromium channel。
        playwright_storage_state_path: 浏览器回退可选 storage state 文件路径。
        get_playwright_browser: Browser 单例获取函数。
        build_domain_home_url: 同域首页构造函数。
        normalize_url_for_http: URL 规范化函数。
        sanitize_response_headers: 响应头裁剪函数。
        build_text_excerpt: 文本摘录构造函数。
        convert_html_to_markdown: HTML 四段式转换函数。
        time_monotonic: 可注入的单调时钟函数。

    Returns:
        成功时返回含 `ok=True` 的结果字典；失败时抛出异常由调用方处理。

    Raises:
        RuntimeError: Playwright 不可用、页面加载失败、内容转换失败等。
    """

    _ = headers
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    except ImportError as exc:
        raise RuntimeError(f"Playwright 未安装，无法执行浏览器回退抓取: {exc}") from exc

    stealth_class: type | None = None
    has_stealth = False
    try:
        from playwright_stealth import Stealth

        stealth_class = Stealth
        has_stealth = True
    except ImportError:
        Log.warning("未安装 playwright-stealth，将跳过指纹隐蔽步骤，反爬绕过率可能降低", module=MODULE)

    browser = get_playwright_browser(playwright_channel=playwright_channel, headless=True)
    if browser is None:
        raise RuntimeError("Playwright Browser 单例不可用，无法执行浏览器回退抓取。")

    context_kwargs: dict[str, Any] = {
        "viewport": {"width": 1280, "height": 800},
        "user_agent": _DEFAULT_BROWSER_USER_AGENT,
        "locale": "zh-CN",
        "accept_downloads": False,
        "ignore_https_errors": True,
        "extra_http_headers": {
            "Accept": _DEFAULT_ACCEPT,
            "Accept-Language": _DEFAULT_ACCEPT_LANGUAGE,
            "Sec-Ch-Ua": _DEFAULT_SEC_CH_UA,
            "Sec-Ch-Ua-Mobile": _DEFAULT_SEC_CH_UA_MOBILE,
            "Sec-Ch-Ua-Platform": _DEFAULT_SEC_CH_UA_PLATFORM,
            "Upgrade-Insecure-Requests": "1",
        },
    }
    storage_state_path = str(playwright_storage_state_path).strip()
    if storage_state_path:
        context_kwargs["storage_state"] = storage_state_path
    context = browser.new_context(**context_kwargs)
    try:
        page = context.new_page()
        if has_stealth and stealth_class is not None:
            stealth_class().apply_stealth_sync(page)
        page.route("**/*", _route_handler_abort_resources)

        deadline_monotonic = time_monotonic() + max(float(timeout_seconds), 0.0)
        _maybe_warmup_playwright_page(
            page=page,
            url=url,
            deadline_monotonic=deadline_monotonic,
            build_domain_home_url=build_domain_home_url,
            normalize_url_for_http=normalize_url_for_http,
            time_monotonic=time_monotonic,
        )
        try:
            response = page.goto(
                url,
                wait_until=_PW_NAVIGATION_WAIT_UNTIL,
                timeout=_require_playwright_timeout_ms(
                    deadline_monotonic,
                    time_monotonic=time_monotonic,
                ),
            )
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(f"Playwright 页面加载超时: {exc}") from exc

        if response is None:
            raise RuntimeError("Playwright page.goto 未返回 response 对象。")

        content_type_value = (response.headers.get("content-type") or "").lower()
        if "text/html" not in content_type_value and content_type_value:
            context.close()
            return {
                "ok": False,
                "availability": "unprocessable",
                "reason": "non_html_content_type",
                "http_status": getattr(response, "status", None),
                "response_headers": sanitize_response_headers(getattr(response, "headers", {})),
                "content_type": content_type_value,
            }

        _settle_playwright_page(
            page=page,
            deadline_monotonic=deadline_monotonic,
            time_monotonic=time_monotonic,
        )
        html = page.content()
        final_url = page.url
        try:
            page_text = page.evaluate("() => document.body ? document.body.innerText : ''")
        except Exception:
            page_text = html
    except Exception:
        context.close()
        raise
    else:
        context.close()

    pipeline_result = convert_html_to_markdown(html, url=final_url)
    return {
        "ok": True,
        "title": pipeline_result.title,
        "content": pipeline_result.markdown,
        "final_url": final_url,
        "extraction_source": pipeline_result.extractor_source,
        "renderer_source": pipeline_result.renderer_source,
        "normalization_applied": pipeline_result.normalization_applied,
        "quality_flags": list(pipeline_result.quality_flags),
        "content_stats": dict(pipeline_result.content_stats),
        "http_status": getattr(response, "status", None),
        "response_headers": sanitize_response_headers(getattr(response, "headers", {})),
        "response_excerpt": build_text_excerpt(page_text),
    }


def _fetch_and_convert_with_playwright(
    *,
    url: str,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    playwright_channel: str | None = None,
    playwright_storage_state_path: str = "",
    cancellation_token: CancellationToken | None = None,
    resolve_timeout_budget: Callable[..., float],
    playwright_sync_worker: Callable[..., dict[str, Any]],
    detect_bot_challenge: Callable[..., Any],
) -> dict[str, Any]:
    """使用 Playwright 执行浏览器抓取并转换为 Markdown。

    Args:
        url: 已通过安全校验的网页链接。
        timeout_seconds: 浏览器回退总预算秒数。
        headers: 可选请求头。
        timeout_budget: Runner 注入的单次 tool call 总预算。
        deadline_monotonic: 当前工具调用的单调时钟 deadline。
        playwright_channel: 浏览器回退使用的 Chromium channel。
        playwright_storage_state_path: 浏览器回退可选 storage state 文件路径。
        resolve_timeout_budget: timeout 预算解析函数。
        playwright_sync_worker: 同步 worker 函数。
        detect_bot_challenge: challenge 检测函数。

    Returns:
        成功时返回 `ok=True` 结果；失败时返回标准化失败字典。

    Raises:
        无。
    """

    try:
        import playwright  # noqa: F401
    except ImportError:
        Log.warning("playwright 未安装，浏览器回退不可用。", module=MODULE)
        return {
            "ok": False,
            "availability": "unprocessable",
            "reason": "playwright_not_installed",
        }

    try:
        effective_timeout = resolve_timeout_budget(
            timeout_seconds,
            timeout_budget=timeout_budget,
            deadline_monotonic=deadline_monotonic,
            reserve_seconds=_PW_RESULT_EXTRA_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return {
            "ok": False,
            "availability": "timeout",
            "reason": "playwright_timeout",
        }

    total_timeout = effective_timeout + _PW_RESULT_EXTRA_TIMEOUT_SECONDS
    try:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        if _is_picklable_worker(playwright_sync_worker):
            result = _run_playwright_worker_process(
                playwright_sync_worker=playwright_sync_worker,
                worker_kwargs={
                    "url": url,
                    "timeout_seconds": effective_timeout,
                    "headers": headers,
                    "playwright_channel": playwright_channel,
                    "playwright_storage_state_path": playwright_storage_state_path,
                },
                total_timeout=total_timeout,
                cancellation_token=cancellation_token,
            )
        else:
            result = playwright_sync_worker(
                url=url,
                timeout_seconds=effective_timeout,
                headers=headers,
                playwright_channel=playwright_channel,
                playwright_storage_state_path=playwright_storage_state_path,
            )
    except TimeoutError:
        Log.debug(f"Playwright 浏览器回退在 {total_timeout}s 内未返回结果: {url}", module=MODULE)
        return {
            "ok": False,
            "availability": "timeout",
            "reason": "playwright_timeout",
        }
    except CancelledError:
        raise
    except Exception as exc:
        Log.debug(f"Playwright 浏览器回退失败: {exc}", module=MODULE)
        return {
            "ok": False,
            "availability": "unprocessable",
            "reason": "playwright_error",
        }

    if result.get("ok"):
        challenge = detect_bot_challenge(
            response=None,
            response_headers=result.get("response_headers"),
            http_status=result.get("http_status"),
            content_text=result.get("content") or result.get("response_excerpt", ""),
        )
        if challenge.challenge_detected:
            return {
                "ok": False,
                "availability": "blocked",
                "reason": "bot_challenge",
                "http_status": result.get("http_status"),
                "response_headers": result.get("response_headers", {}),
                "response_excerpt": result.get("response_excerpt", ""),
                "challenge_signals": list(challenge.challenge_signals),
            }
    return result