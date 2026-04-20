"""CLI / WeChat 共用的执行选项参数与构建逻辑。

该模块负责两类稳定职责：
1. 为参数解析器注册请求级 `ExecutionOptions` 覆盖参数；
2. 将 `argparse.Namespace` 收敛为 `ExecutionOptions`。

该逻辑被 CLI 与 WeChat 入口共同消费，避免多入口字段清单漂移。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dayu.contracts.toolset_config import ToolsetConfigSnapshot, build_toolset_config_snapshot
from dayu.execution.options import ExecutionOptions


def add_execution_option_arguments(parser: argparse.ArgumentParser) -> None:
    """为解析器注册请求级执行选项参数。

    Args:
        parser: 待扩展的参数解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--web-provider",
        type=str,
        default=None,
        choices=["auto", "tavily", "serper", "duckduckgo"],
        help="联网检索 provider（默认读取 run.json.web_tools_config.provider，未配置时为 auto）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="模型 temperature（覆盖 llm_models.runtime_hints.temperature_profiles[scene.temperature_profile].temperature）",
    )
    parser.add_argument("--debug-sse", action="store_true", help="开启 SSE 高频调试日志（覆盖 run.json）")
    parser.add_argument("--debug-tool-delta", action="store_true", help="开启工具调用参数增量日志（覆盖 run.json）")
    parser.add_argument("--debug-sse-sample-rate", type=float, help="调试日志采样率（0-1，覆盖 run.json）")
    parser.add_argument("--debug-sse-throttle-sec", type=float, help="调试日志节流窗口秒数（覆盖 run.json）")
    parser.add_argument("--tool-timeout-seconds", type=float, help="工具执行超时时间（秒，覆盖 run.json）")
    parser.add_argument(
        "--enable-tool-trace",
        action="store_true",
        help="启用工具调用请求/返回 JSONL 追踪（覆盖 run.json）",
    )
    parser.add_argument(
        "--tool-trace-dir",
        type=str,
        help="工具调用追踪输出目录（可相对 workspace，覆盖 run.json）",
    )
    parser.add_argument("--max-iterations", type=int, help="Agent 最大迭代次数（覆盖 run.json）")
    parser.add_argument(
        "--fallback-mode",
        type=str,
        choices=["force_answer", "raise_error"],
        help="超限处理模式（覆盖 run.json）",
    )
    parser.add_argument("--fallback-prompt", type=str, help="超限时补充提示（覆盖 run.json）")
    parser.add_argument(
        "--max-consecutive-failed-tool-batches",
        type=int,
        help="连续失败工具批次上限（覆盖 run.json）",
    )
    parser.add_argument(
        "--max-duplicate-tool-calls",
        type=int,
        help="同一工具无信息增量重复调用的连续上限（覆盖 run.json）",
    )
    parser.add_argument(
        "--duplicate-tool-hint-prompt",
        type=str,
        help="检测到重复工具调用时注入给模型的提示词（覆盖 run.json）",
    )
    parser.add_argument(
        "--doc-limits-json",
        type=str,
        default=None,
        help="文档工具 limits 覆盖 JSON（覆盖 run.json.doc_tool_limits）",
    )
    parser.add_argument(
        "--fins-limits-json",
        type=str,
        default=None,
        help="财报工具 limits 覆盖 JSON（覆盖 run.json.fins_tool_limits）",
    )


def build_execution_options_from_args(args: argparse.Namespace) -> ExecutionOptions:
    """从参数对象构建请求级执行选项。

    Args:
        args: 命令行参数对象。

    Returns:
        执行选项对象。

    Raises:
        SystemExit: limits JSON 或 temperature 参数非法时抛出。
    """

    from dayu.cli.arg_parsing import parse_limits_override, parse_temperature_argument

    doc_limits = parse_limits_override(
        getattr(args, "doc_limits_json", None),
        field_name="--doc-limits-json",
    )
    fins_limits = parse_limits_override(
        getattr(args, "fins_limits_json", None),
        field_name="--fins-limits-json",
    )

    return ExecutionOptions(
        model_name=(raw_model_name if (raw_model_name := str(getattr(args, "model_name", "") or "").strip()) else None),
        temperature=parse_temperature_argument(getattr(args, "temperature", None), field_name="--temperature"),
        debug_sse=bool(getattr(args, "debug_sse", False)),
        debug_tool_delta=bool(getattr(args, "debug_tool_delta", False)),
        debug_sse_sample_rate=getattr(args, "debug_sse_sample_rate", None),
        debug_sse_throttle_sec=getattr(args, "debug_sse_throttle_sec", None),
        tool_timeout_seconds=getattr(args, "tool_timeout_seconds", None),
        max_iterations=getattr(args, "max_iterations", None),
        fallback_mode=getattr(args, "fallback_mode", None),
        fallback_prompt=getattr(args, "fallback_prompt", None),
        max_consecutive_failed_tool_batches=getattr(args, "max_consecutive_failed_tool_batches", None),
        max_duplicate_tool_calls=getattr(args, "max_duplicate_tool_calls", None),
        duplicate_tool_hint_prompt=getattr(args, "duplicate_tool_hint_prompt", None),
        web_provider=getattr(args, "web_provider", None),
        trace_enabled=(True if bool(getattr(args, "enable_tool_trace", False)) else None),
        trace_output_dir=Path(getattr(args, "tool_trace_dir")).expanduser().resolve()
        if getattr(args, "tool_trace_dir", None)
        else None,
        toolset_config_overrides=_build_toolset_override_snapshots(
            doc_limits=doc_limits,
            fins_limits=fins_limits,
        ),
    )


def _build_toolset_override_snapshots(
    *,
    doc_limits: object | None,
    fins_limits: object | None,
) -> tuple[ToolsetConfigSnapshot, ...]:
    """把 override 载荷收敛为通用 toolset 快照。

    Args:
        doc_limits: 文档工具限制 override。
        fins_limits: 财报工具限制 override。

    Returns:
        通用 toolset override 快照序列。

    Raises:
        TypeError: override 无法构造成通用快照时抛出。
        ValueError: toolset 名称非法时抛出。
    """

    snapshots: list[ToolsetConfigSnapshot] = []
    for snapshot in (
        build_toolset_config_snapshot("doc", doc_limits),
        build_toolset_config_snapshot("fins", fins_limits),
    ):
        if snapshot is not None:
            snapshots.append(snapshot)
    return tuple(snapshots)


__all__ = [
    "add_execution_option_arguments",
    "build_execution_options_from_args",
]
