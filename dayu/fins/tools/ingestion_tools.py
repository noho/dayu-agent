"""财报长事务工具注册模块。

该模块负责将 `download/process` 两类长事务暴露为适合 LLM 调用的
`start/status/cancel` job 工具，并严格控制 schema 与返回数据的认知负担。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from dayu.engine.exceptions import ToolArgumentError
from dayu.engine.tool_contracts import DupCallSpec
from dayu.engine.tool_registry import ToolRegistry
from dayu.engine.tools.base import tool
from dayu.fins.ingestion.job_manager import (
    IngestionJobManager,
    get_or_create_ingestion_job_manager,
)
from dayu.fins.resolver.market_resolver import MarketResolver
from dayu.fins.ingestion.service import FinsIngestionService

MODULE = "FINS.INGESTION_TOOLS"
INGESTION_TOOL_TAGS = frozenset({"ingestion"})

_SUPPORTED_DOWNLOAD_FORM_TYPES = [
    "10-K",
    "10-Q",
    "20-F",
    "6-K",
    "8-K",
    "8-K/A",
    "DEF 14A",
    "SC 13D",
    "SC 13D/A",
    "SC 13G",
    "SC 13G/A",
]

_SUPPORTED_DOWNLOAD_MARKETS = frozenset({"US"})
_JOB_TERMINAL_STATUSES = ["succeeded", "failed", "cancelled"]

def register_ingestion_tools(
    registry: ToolRegistry,
    *,
    service_factory: Callable[[str], FinsIngestionService],
    manager_key: str,
    timeout_budget: float | None = None,
) -> int:
    """注册财报长事务 job 工具。

    Args:
        registry: 工具注册表。
        service_factory: `ticker -> FinsIngestionService` 工厂。
        manager_key: 长事务 job 管理器稳定标识。
        timeout_budget: Runner 为单次 tool call 提供的预算秒数；当前 ingestion 工具预留该参数，
            暂未消费。

    Returns:
        新注册的工具数量。

    Raises:
        ValueError: 参数非法时抛出。
    """
    del timeout_budget
    if registry is None:
        raise ValueError("registry 不能为空")
    if service_factory is None:
        raise ValueError("service_factory 不能为空")
    if not str(manager_key).strip():
        raise ValueError("manager_key 不能为空")

    manager = get_or_create_ingestion_job_manager(
        manager_key=manager_key,
        service_factory=service_factory,
    )
    tool_factories = [
        _create_start_download_job_tool,
        _create_get_download_job_status_tool,
        _create_cancel_download_job_tool,
    ]
    for factory in tool_factories:
        name, func, schema = factory(registry=registry, manager=manager)
        registry.register(name, func, schema)
    return len(tool_factories)


def _create_start_download_job_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建下载 job 启动工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "公司代码。第一次直接传最自然的写法即可。",
            },
            "form_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": _SUPPORTED_DOWNLOAD_FORM_TYPES,
                },
                "uniqueItems": True,
                "description": "可选表单过滤。只在你明确要缩小下载范围时填写；留空表示下载该公司当前支持的全部表单。",
            },
            "filed_date_from": {
                "type": "string",
                "description": "可选 filed date 下界。只在你明确要限制时间范围时填写；格式 YYYY、YYYY-MM 或 YYYY-MM-DD。",
            },
            "filed_date_to": {
                "type": "string",
                "description": "可选 filed date 上界。只在你明确要限制时间范围时填写；格式 YYYY、YYYY-MM 或 YYYY-MM-DD。",
            },
            "overwrite": {
                "type": "boolean",
                "description": "是否覆盖已有本地下载结果。仅在你明确要重下时设为 true。",
                "default": False,
            },
        },
        "required": ["ticker"],
    }

    @tool(
        registry,
        name="start_financial_filing_download_job",
        description=(
            "启动单个 ticker 的下载任务。拿到返回里的 job.job_id 后，下一步只用状态工具轮询，直到 job.status 进入 succeeded / failed / cancelled。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
    )
    def start_financial_filing_download_job(
        ticker: str,
        form_types: Optional[list[str]] = None,
        filed_date_from: Optional[str] = None,
        filed_date_to: Optional[str] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """启动财报下载 job。

        Args:
            ticker: 股票代码。
            form_types: 可选表单过滤。
            filed_date_from: 可选 filing date 下界。
            filed_date_to: 可选 filing date 上界。
            overwrite: 是否覆盖已有下载结果。

        Returns:
            启动结果，包含 job 摘要与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_ticker = _normalize_required_text(
            tool_name="start_financial_filing_download_job",
            arg_name="ticker",
            value=ticker,
        )
        market_profile = MarketResolver.resolve(normalized_ticker)
        if market_profile.market not in _SUPPORTED_DOWNLOAD_MARKETS:
            return _build_not_implemented_start_response(
                ticker=normalized_ticker,
                market=market_profile.market,
            )
        normalized_form_types = _normalize_form_types(form_types)
        request_outcome, snapshot = manager.start_download_job(
            ticker=normalized_ticker,
            form_types=normalized_form_types,
            filed_date_from=_normalize_optional_text(filed_date_from),
            filed_date_to=_normalize_optional_text(filed_date_to),
            overwrite=bool(overwrite),
        )
        return _build_start_response(
            request_outcome=request_outcome,
            snapshot=snapshot,
            start_tool_name="start_financial_filing_download_job",
            status_tool_name="get_financial_filing_download_job_status",
        )

    return (
        start_financial_filing_download_job.__tool_name__,
        start_financial_filing_download_job,
        start_financial_filing_download_job.__tool_schema__,
    )


def _create_get_download_job_status_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建下载 job 状态查询工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "任务 ID。直接使用启动工具返回的 job.job_id。",
            }
        },
        "required": ["job_id"],
    }

    @tool(
        registry,
        name="get_financial_filing_download_job_status",
        description=(
            "查询下载任务状态。启动后反复调用本工具，直到 job.status 进入 succeeded / failed / cancelled。优先按 next_step.action 决定是继续轮询、停止还是重新启动。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
        dup_call=DupCallSpec(
            mode="poll_until_terminal",
            status_path="job.status",
            terminal_values=_JOB_TERMINAL_STATUSES,
        ),
    )
    def get_financial_filing_download_job_status(job_id: str) -> dict[str, Any]:
        """查询下载 job 状态。

        Args:
            job_id: job 标识。

        Returns:
            状态摘要、失败信息与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_job_id = _normalize_required_text(
            tool_name="get_financial_filing_download_job_status",
            arg_name="job_id",
            value=job_id,
        )
        snapshot = manager.get_job_snapshot(normalized_job_id)
        return _build_status_response(
            snapshot=snapshot,
            requested_job_id=normalized_job_id,
            status_tool_name="get_financial_filing_download_job_status",
            start_tool_name="start_financial_filing_download_job",
        )

    return (
        get_financial_filing_download_job_status.__tool_name__,
        get_financial_filing_download_job_status,
        get_financial_filing_download_job_status.__tool_schema__,
    )


def _create_cancel_download_job_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建下载 job 取消工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "任务 ID。直接使用启动工具或状态工具返回的 job_id。",
            }
        },
        "required": ["job_id"],
    }

    @tool(
        registry,
        name="cancel_financial_filing_download_job",
        description=(
            "请求取消下载任务。取消不是立即完成的；调用后继续用状态工具确认是否进入 cancelled。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
    )
    def cancel_financial_filing_download_job(job_id: str) -> dict[str, Any]:
        """请求取消下载 job。

        Args:
            job_id: job 标识。

        Returns:
            取消请求结果、当前 job 摘要与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_job_id = _normalize_required_text(
            tool_name="cancel_financial_filing_download_job",
            arg_name="job_id",
            value=job_id,
        )
        cancellation_outcome, snapshot = manager.cancel_job(normalized_job_id)
        return _build_cancel_response(
            cancellation_outcome=cancellation_outcome,
            snapshot=snapshot,
            requested_job_id=normalized_job_id,
            status_tool_name="get_financial_filing_download_job_status",
            start_tool_name="start_financial_filing_download_job",
        )

    return (
        cancel_financial_filing_download_job.__tool_name__,
        cancel_financial_filing_download_job,
        cancel_financial_filing_download_job.__tool_schema__,
    )


def _create_start_process_job_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建预处理 job 启动工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "公司代码。第一次直接传最自然的写法即可。",
            },
            "overwrite": {
                "type": "boolean",
                "description": "是否覆盖已有快照。仅在你明确要重做处理时设为 true。",
                "default": False,
            },
        },
        "required": ["ticker"],
    }

    @tool(
        registry,
        name="start_financial_document_preprocess_job",
        description=(
            "启动单个 ticker 的预处理任务。拿到返回里的 job.job_id 后，下一步只用状态工具轮询，直到 job.status 进入 succeeded / failed / cancelled。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
    )
    def start_financial_document_preprocess_job(
        ticker: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """启动预处理 job。

        Args:
            ticker: 股票代码。
            overwrite: 是否覆盖已有快照。

        Returns:
            启动结果，包含 job 摘要与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_ticker = _normalize_required_text(
            tool_name="start_financial_document_preprocess_job",
            arg_name="ticker",
            value=ticker,
        )
        request_outcome, snapshot = manager.start_process_job(
            ticker=normalized_ticker,
            overwrite=bool(overwrite),
        )
        return _build_start_response(
            request_outcome=request_outcome,
            snapshot=snapshot,
            start_tool_name="start_financial_document_preprocess_job",
            status_tool_name="get_financial_document_preprocess_job_status",
        )

    return (
        start_financial_document_preprocess_job.__tool_name__,
        start_financial_document_preprocess_job,
        start_financial_document_preprocess_job.__tool_schema__,
    )


def _create_get_process_job_status_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建预处理 job 状态查询工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "任务 ID。直接使用启动工具返回的 job.job_id。",
            }
        },
        "required": ["job_id"],
    }

    @tool(
        registry,
        name="get_financial_document_preprocess_job_status",
        description=(
            "查询预处理任务状态。启动后反复调用本工具，直到 job.status 进入 succeeded / failed / cancelled。优先按 next_step.action 决定是继续轮询、停止还是重新启动。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
        dup_call=DupCallSpec(
            mode="poll_until_terminal",
            status_path="job.status",
            terminal_values=_JOB_TERMINAL_STATUSES,
        ),
    )
    def get_financial_document_preprocess_job_status(job_id: str) -> dict[str, Any]:
        """查询预处理 job 状态。

        Args:
            job_id: job 标识。

        Returns:
            状态摘要、失败信息与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_job_id = _normalize_required_text(
            tool_name="get_financial_document_preprocess_job_status",
            arg_name="job_id",
            value=job_id,
        )
        snapshot = manager.get_job_snapshot(normalized_job_id)
        return _build_status_response(
            snapshot=snapshot,
            requested_job_id=normalized_job_id,
            status_tool_name="get_financial_document_preprocess_job_status",
            start_tool_name="start_financial_document_preprocess_job",
        )

    return (
        get_financial_document_preprocess_job_status.__tool_name__,
        get_financial_document_preprocess_job_status,
        get_financial_document_preprocess_job_status.__tool_schema__,
    )


def _create_cancel_process_job_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建预处理 job 取消工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "任务 ID。直接使用启动工具或状态工具返回的 job_id。",
            }
        },
        "required": ["job_id"],
    }

    @tool(
        registry,
        name="cancel_financial_document_preprocess_job",
        description=(
            "请求取消预处理任务。取消不是立即完成的；调用后继续用状态工具确认是否进入 cancelled。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
    )
    def cancel_financial_document_preprocess_job(job_id: str) -> dict[str, Any]:
        """请求取消预处理 job。

        Args:
            job_id: job 标识。

        Returns:
            取消请求结果、当前 job 摘要与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_job_id = _normalize_required_text(
            tool_name="cancel_financial_document_preprocess_job",
            arg_name="job_id",
            value=job_id,
        )
        cancellation_outcome, snapshot = manager.cancel_job(normalized_job_id)
        return _build_cancel_response(
            cancellation_outcome=cancellation_outcome,
            snapshot=snapshot,
            requested_job_id=normalized_job_id,
            status_tool_name="get_financial_document_preprocess_job_status",
            start_tool_name="start_financial_document_preprocess_job",
        )

    return (
        cancel_financial_document_preprocess_job.__tool_name__,
        cancel_financial_document_preprocess_job,
        cancel_financial_document_preprocess_job.__tool_schema__,
    )


def _normalize_required_text(
    *,
    tool_name: str,
    arg_name: str,
    value: Any,
) -> str:
    """标准化必填文本参数。

    Args:
        tool_name: 工具名。
        arg_name: 参数名。
        value: 输入值。

    Returns:
        去空白后的非空字符串。

    Raises:
        ToolArgumentError: 参数为空时抛出。
    """

    normalized = str(value or "").strip()
    if not normalized:
        raise ToolArgumentError(tool_name, arg_name, value, "不能为空")
    return normalized


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    """标准化可选文本参数。

    Args:
        value: 原始值。

    Returns:
        去空白后的文本；为空时返回 `None`。

    Raises:
        无。
    """

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_form_types(form_types: Optional[list[str]]) -> Optional[list[str]]:
    """标准化表单数组。

    Args:
        form_types: 原始表单数组。

    Returns:
        去重、排序后的表单数组；为空时返回 `None`。

    Raises:
        ToolArgumentError: 表单值为空白时抛出。
    """

    if form_types is None:
        return None
    normalized_items: list[str] = []
    for item in form_types:
        normalized = str(item or "").strip()
        if not normalized:
            raise ToolArgumentError(
                "start_financial_filing_download_job",
                "form_types",
                form_types,
                "不能包含空白表单类型",
            )
        normalized_items.append(normalized)
    if not normalized_items:
        return None
    return sorted(set(normalized_items))


def _build_status_response(
    *,
    snapshot: Optional[dict[str, Any]],
    requested_job_id: str,
    status_tool_name: str,
    start_tool_name: str,
) -> dict[str, Any]:
    """构建状态查询返回。

    Args:
        snapshot: job 快照。
        requested_job_id: 请求的 job_id。
        status_tool_name: 对应 status 工具名。
        start_tool_name: 对应 start 工具名（供重试建议）。

    Returns:
        面向 LLM 的极简状态结构。

    Raises:
        无。
    """

    if snapshot is None:
        return {
            "job": None,
            "progress": None,
            "result_summary": None,
            "recent_issues": None,
            "failure": _build_failure(
                code="job_not_found",
                message="找不到这个 job_id，或该任务已过期",
                retryable=False,
            ),
            "next_step": _build_stop_next_step(job_id=requested_job_id),
        }
    return {
        "job": _build_public_job(snapshot),
        "progress": _build_public_progress(snapshot),
        "result_summary": snapshot.get("result_summary"),
        "recent_issues": snapshot.get("recent_issues"),
        "failure": snapshot.get("failure"),
        "next_step": _build_next_step(
            snapshot=snapshot,
            status_tool_name=status_tool_name,
            start_tool_name=start_tool_name,
        ),
    }


def _build_start_response(
    *,
    request_outcome: str,
    snapshot: Optional[dict[str, Any]],
    start_tool_name: str,
    status_tool_name: str,
    failure: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构建启动工具返回。

    Args:
        request_outcome: 启动结果枚举。
        snapshot: 可选 job 快照。
        start_tool_name: 启动工具名。
        status_tool_name: 状态查询工具名。
        failure: 可选失败信息。

    Returns:
        面向 LLM 的极简启动结构。

    Raises:
        无。
    """

    if snapshot is None:
        return {
            "request_outcome": request_outcome,
            "job": None,
            "failure": failure,
            "next_step": _build_stop_next_step(job_id=None),
        }
    return {
        "request_outcome": request_outcome,
        "job": _build_public_job(snapshot),
        "failure": failure,
        "next_step": _build_next_step(
            snapshot=snapshot,
            status_tool_name=status_tool_name,
            start_tool_name=start_tool_name,
        ),
    }


def _build_cancel_response(
    *,
    cancellation_outcome: str,
    snapshot: Optional[dict[str, Any]],
    requested_job_id: str,
    status_tool_name: str,
    start_tool_name: str,
) -> dict[str, Any]:
    """构建取消工具返回。

    Args:
        cancellation_outcome: 取消结果枚举。
        snapshot: job 快照。
        requested_job_id: 请求的 job_id。
        status_tool_name: 对应 status 工具名。
        start_tool_name: 对应 start 工具名（供重试建议）。

    Returns:
        面向 LLM 的取消结果结构。

    Raises:
        无。
    """

    # 显式守卫：无论 manager 层返回何种 outcome，snapshot 为 None 时
    # 统一映射为 "job_not_found"，避免内部枚举值泄漏到 LLM。
    if snapshot is None or cancellation_outcome == "not_found":
        return {
            "cancellation_outcome": "job_not_found",
            "job": None,
            "progress": None,
            "result_summary": None,
            "recent_issues": None,
            "failure": _build_failure(
                code="job_not_found",
                message="找不到这个 job_id，或该任务已过期",
                retryable=False,
            ),
            "next_step": _build_stop_next_step(job_id=requested_job_id),
        }
    return {
        "cancellation_outcome": cancellation_outcome,
        "job": _build_public_job(snapshot),
        "progress": _build_public_progress(snapshot),
        "result_summary": snapshot.get("result_summary"),
        "recent_issues": snapshot.get("recent_issues"),
        "failure": snapshot.get("failure"),
        "next_step": _build_next_step(
            snapshot=snapshot,
            status_tool_name=status_tool_name,
            start_tool_name=start_tool_name,
        ),
    }


def _build_public_job(snapshot: dict[str, Any]) -> dict[str, Any]:
    """将内部 job 快照转换为对外 job 摘要。

    Args:
        snapshot: 内部快照。

    Returns:
        对外 job 摘要。

    Raises:
        ValueError: 快照缺失 `job` 时抛出。
    """

    job = snapshot.get("job")
    if not isinstance(job, dict):
        raise ValueError("snapshot.job 缺失")
    return {
        "job_id": job.get("job_id"),
        "job_type": _map_public_job_type(job.get("job_type")),
        "ticker": job.get("ticker"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }


def _build_public_progress(snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    """将内部快照转换为对外进度摘要。

    Args:
        snapshot: 内部快照。

    Returns:
        进度结构；缺失时返回 `None`。

    Raises:
        无。
    """

    progress = snapshot.get("progress")
    if not isinstance(progress, dict):
        return None
    return {
        "unit": progress.get("unit"),
        "completed": progress.get("completed"),
        "total": progress.get("total"),
        "percent": progress.get("percent"),
    }


def _build_next_step(
    *,
    snapshot: dict[str, Any],
    status_tool_name: str,
    start_tool_name: str,
) -> dict[str, Any]:
    """根据 job 状态生成下一步建议。

    Args:
        snapshot: job 快照。
        status_tool_name: 对应状态查询工具名。
        start_tool_name: 对应启动工具名（供重试建议）。

    Returns:
        机器可判别的下一步建议。

    Raises:
        ValueError: 快照缺失 `job` 时抛出。
    """

    job = _build_public_job(snapshot)
    result_summary = snapshot.get("result_summary")
    if job["status"] in {"queued", "running", "cancelling"}:
        return {
            "action": "poll_status",
            "tool_name": status_tool_name,
            "job_id": job["job_id"],
            "suggested_wait_seconds": 5,
        }
    if job["status"] == "failed":
        return {
            "action": "stop_or_retry",
            "tool_name": start_tool_name,
            "job_id": job["job_id"],
            "suggested_wait_seconds": None,
        }
    if _has_failed_units(result_summary):
        return {
            "action": "stop_or_retry",
            "tool_name": start_tool_name,
            "job_id": job["job_id"],
            "suggested_wait_seconds": None,
        }
    return _build_stop_next_step(job_id=str(job["job_id"]))


def _build_stop_next_step(*, job_id: Optional[str]) -> dict[str, Any]:
    """构建停止建议。

    Args:
        job_id: 可选关联 job_id。

    Returns:
        停止建议结构。

    Raises:
        无。
    """

    return {
        "action": "stop",
        "tool_name": None,
        "job_id": job_id,
        "suggested_wait_seconds": None,
    }


def _build_failure(*, code: str, message: str, retryable: bool) -> dict[str, Any]:
    """构建统一失败结构。

    Args:
        code: 错误码。
        message: 错误消息。
        retryable: 是否可重试。

    Returns:
        失败结构。

    Raises:
        无。
    """

    return {
        "code": code,
        "message": message,
        "retryable": retryable,
    }


def _build_not_implemented_start_response(*, ticker: str, market: str) -> dict[str, Any]:
    """构建不支持市场的启动返回。

    Args:
        ticker: 股票代码。
        market: 市场类型。

    Returns:
        `request_outcome=not_implemented` 的极简返回结构。

    Raises:
        无。
    """

    failure = _build_failure(
        code="not_implemented",
        message=f"当前市场暂不支持下载任务：market={market}，ticker={ticker}",
        retryable=False,
    )
    return _build_start_response(
        request_outcome="not_implemented",
        snapshot=None,
        start_tool_name="start_financial_filing_download_job",
        status_tool_name="get_financial_filing_download_job_status",
        failure=failure,
    )


def _map_public_job_type(value: Any) -> str:
    """映射对外 job 类型。

    Args:
        value: 内部 job 类型。

    Returns:
        对外稳定枚举值。

    Raises:
        无。
    """

    normalized = str(value or "").strip().lower()
    if normalized == "download":
        return "filing_download"
    if normalized == "process":
        return "document_preprocess"
    return "unknown"


def _has_failed_units(result_summary: Any) -> bool:
    """判断结果摘要中是否包含失败单元。

    Args:
        result_summary: 结果摘要。

    Returns:
        只要任一失败计数大于 0 即返回 `True`。

    Raises:
        无。
    """

    if not isinstance(result_summary, dict):
        return False
    for key, value in result_summary.items():
        if not str(key).endswith("_failed"):
            continue
        try:
            if int(value or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False
