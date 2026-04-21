"""CLI 参数定义模块。

模块职责：
- 定义命令行解析器（``DayuCliArgumentParser``）。
- 注册各子命令及其参数（interactive / prompt / write / download / upload_* / process* / host）。
- 提供 ``parse_arguments()`` 入口供 ``main()`` 调用。
"""
from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, NoReturn

from dayu.execution.cli_execution_options import add_execution_option_arguments

if TYPE_CHECKING:
    from dayu.execution.options import ExecutionOptionsOverridePayload



class DayuCliArgumentParser(argparse.ArgumentParser):
    """`dayu.cli` 顶层参数解析器。

    设计意图：
    - 统一固定 `python -m dayu.cli` 作为程序名，避免暴露 `__main__.py`。
    - 在缺少顶层子命令时输出完整帮助，而不是仅输出一行难读的 usage。
    """

    def error(self, message: str) -> NoReturn:
        """输出更适合人读的参数错误信息。

        Args:
            message: argparse 生成的错误文案。

        Returns:
            无。

        Raises:
            SystemExit: 参数解析失败时退出。
        """

        if "required: command" in message:
            self.print_help(sys.stderr)
            self.exit(2, "\n错误: 缺少子命令。请先选择一个子命令，再用 `--help` 查看该命令的具体参数。\n")
        super().error(message)

def _add_global_args(parser: argparse.ArgumentParser) -> None:
    """追加各子命令共享的全局参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    _add_workspace_args(parser)
    _add_logging_args(parser)


def _add_workspace_args(parser: argparse.ArgumentParser) -> None:
    """追加工作区与配置目录参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--base", "-b", "--workspace",
        type=str,
        default="./workspace",
        help="工作区根目录（默认 ./workspace）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置目录（默认 workspace/config）",
    )


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    """追加日志等级参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    log_level_group = parser.add_mutually_exclusive_group()
    log_level_group.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "verbose", "info", "warn", "error"],
        help="设置日志级别",
    )
    log_level_group.add_argument("--debug", action="store_true", help="日志级别设为 DEBUG")
    log_level_group.add_argument("--verbose", action="store_true", help="日志级别设为 VERBOSE")
    log_level_group.add_argument("--info", action="store_true", help="日志级别设为 INFO")
    log_level_group.add_argument("--quiet", action="store_true", help="日志级别设为 ERROR")


def _add_ticker_arg(
    parser: argparse.ArgumentParser,
    *,
    required: bool,
    help_text: str,
) -> None:
    """追加股票代码参数。

    Args:
        parser: 子命令解析器。
        required: 是否必填。
        help_text: 帮助文案。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--ticker",
        type=str,
        required=required,
        default=None,
        help=help_text,
    )


def _add_fins_common_args(parser: argparse.ArgumentParser) -> None:
    """追加财报命令通用参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    _add_global_args(parser)


def _add_model_name_arg(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    """追加模型名称参数。

    Args:
        parser: 子命令解析器。
        help_text: 帮助文案。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--model-name", "-m",
        type=str,
        default=None,
        help=help_text,
    )


def _add_date_args(
    parser: argparse.ArgumentParser,
    *,
    filing_date_help: str,
    report_date_help: str,
) -> None:
    """追加披露日期与报告日期参数。

    Args:
        parser: 子命令解析器。
        filing_date_help: `--filing-date` 帮助文案。
        report_date_help: `--report-date` 帮助文案。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--filing-date", dest="filing_date", default=None, help=filing_date_help)
    parser.add_argument("--report-date", dest="report_date", default=None, help=report_date_help)


def _add_company_meta_args(
    parser: argparse.ArgumentParser,
    *,
    company_id_help: str,
    company_name_help: str,
    infer_help: str,
) -> None:
    """追加公司元信息与别名推断参数。

    Args:
        parser: 子命令解析器。
        company_id_help: `--company-id` 帮助文案。
        company_name_help: `--company-name` 帮助文案。
        infer_help: `--infer` 帮助文案。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--company-id", dest="company_id", default=None, help=company_id_help)
    parser.add_argument(
        "--company-name",
        dest="company_name",
        default=None,
        help=company_name_help,
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help=infer_help,
    )


def _add_overwrite_arg(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    """追加覆盖开关参数。

    Args:
        parser: 子命令解析器。
        help_text: 帮助文案。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--overwrite", action="store_true", help=help_text)


def _add_ci_arg(parser: argparse.ArgumentParser) -> None:
    """追加 CI 快照导出开关。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--ci", action="store_true", help="是否追加导出 search_document 与 query_xbrl_facts 快照")


def _add_fins_download_args(parser: argparse.ArgumentParser) -> None:
    """追加 `download` 子命令参数。"""

    _add_ticker_arg(
        parser,
        required=True,
        help_text="股票代码；支持 CSV，如 BABA,9988,9988.HK，其中第一个值为 canonical ticker",
    )
    parser.add_argument(
        "--forms",
        dest="form_type",
        nargs="+",
        default=None,
        help="可选 form 列表（支持简写，如 10Q 10K DEF14A）",
    )
    parser.add_argument("--start", dest="start_date", default=None, help="可选开始日期（YYYY/ YYYY-MM/ YYYY-MM-DD）")
    parser.add_argument("--end", dest="end_date", default=None, help="可选结束日期（YYYY/ YYYY-MM/ YYYY-MM-DD）")
    _add_overwrite_arg(parser, help_text="是否覆盖已存在结果")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="是否基于本地已下载 filings 重建 meta/manifest（不重新下载）",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="使用 FMP 推断 ticker_aliases；infer 成功时与显式 CSV alias 合并，下载阶段还会继续与 SEC alias 合并，失败时回退到显式 CSV alias",
    )
    _add_fins_common_args(parser)


def _add_fins_upload_filing_args(parser: argparse.ArgumentParser) -> None:
    """追加 `upload_filing` 子命令参数。"""

    _add_ticker_arg(
        parser,
        required=True,
        help_text="股票代码；支持 CSV，如 BABA,9988,9988.HK，其中第一个值为 canonical ticker",
    )
    parser.add_argument(
        "--action",
        dest="action",
        default=None,
        choices=["create", "update", "delete"],
        help="财报动作类型（默认仅自动判定 create/update；delete 必须显式传入）",
    )
    parser.add_argument("--files", nargs="+", default=None, help="上传文件列表")
    parser.add_argument("--fiscal-year", dest="fiscal_year", type=int, required=True, help="财年")
    parser.add_argument("--fiscal-period", dest="fiscal_period", required=True, help="财季或年度标识（Q1/Q2/Q3/Q4/FY/H1）")
    parser.add_argument("--amended", action="store_true", help="财报是否修订版")
    _add_date_args(
        parser,
        filing_date_help="可选披露日期",
        report_date_help="可选报告日期",
    )
    _add_company_meta_args(
        parser,
        company_id_help="公司 ID（仅在 meta.json 不存在时 create/update 必填）",
        company_name_help="公司名称（仅在 meta.json 不存在时 create/update 必填；若显式传入，则优先于 --infer 返回值）",
        infer_help="使用 FMP 推断 ticker_aliases；成功时与显式 CSV alias 合并，且仅在未传 --company-name 时回退使用 FMP 公司名",
    )
    _add_overwrite_arg(parser, help_text="是否覆盖已存在结果")
    _add_fins_common_args(parser)


def _add_fins_upload_material_args(parser: argparse.ArgumentParser) -> None:
    """追加 `upload_material` 子命令参数。"""

    _add_ticker_arg(
        parser,
        required=True,
        help_text="股票代码；支持 CSV，如 BABA,9988,9988.HK，其中第一个值为 canonical ticker",
    )
    parser.add_argument(
        "--action",
        dest="action",
        default=None,
        choices=["create", "update", "delete"],
        help="材料动作类型（默认仅自动判定 create/update；delete 必须显式传入）",
    )
    parser.add_argument("--forms", dest="form_type", required=True, help="材料 form_type")
    parser.add_argument("--material-name", dest="material_name", required=True, help="材料名称")
    parser.add_argument("--files", nargs="+", default=None, help="上传文件列表")
    parser.add_argument(
        "--document-id",
        dest="document_id",
        default=None,
        help="文档 ID；若传入则必须与按 form_type/material_name/fiscal 生成的稳定 ID 一致",
    )
    parser.add_argument(
        "--internal-document-id",
        dest="internal_document_id",
        default=None,
        help="内部文档 ID；material 场景下与 document_id 恒等，若传入则必须与稳定 ID 一致",
    )
    parser.add_argument("--fiscal-year", dest="fiscal_year", type=int, default=None, help="可选财年")
    parser.add_argument("--fiscal-period", dest="fiscal_period", default=None, help="可选财期")
    _add_date_args(
        parser,
        filing_date_help="可选披露日期",
        report_date_help="可选报告日期",
    )
    _add_company_meta_args(
        parser,
        company_id_help="公司 ID（仅在 meta.json 不存在时 create/update 必填）",
        company_name_help="公司名称（仅在 meta.json 不存在时 create/update 必填；若显式传入，则优先于 --infer 返回值）",
        infer_help="使用 FMP 推断 ticker_aliases；成功时与显式 CSV alias 合并，且仅在未传 --company-name 时回退使用 FMP 公司名",
    )
    _add_overwrite_arg(parser, help_text="是否覆盖已存在结果")
    _add_fins_common_args(parser)


def _add_fins_upload_filings_from_args(parser: argparse.ArgumentParser) -> None:
    """追加 `upload_filings_from` 子命令参数。"""

    _add_ticker_arg(
        parser,
        required=True,
        help_text="股票代码；支持 CSV，如 BABA,9988,9988.HK，其中第一个值为 canonical ticker",
    )
    parser.add_argument("--from", dest="source_dir", required=True, help="待扫描文件目录")
    parser.add_argument(
        "--action",
        dest="action",
        default=None,
        choices=["create", "update"],
        help="可选生成脚本中的固定上传动作（默认留空，执行时自动判定）",
    )
    parser.add_argument("--output", dest="output_script", default=None, help="输出脚本路径，默认写到 --base 指向的 workspace 根目录下")
    parser.add_argument("--recursive", action="store_true", help="是否递归扫描子目录")
    parser.add_argument("--amended", action="store_true", help="生成命令时附加 --amended")
    _add_date_args(
        parser,
        filing_date_help="批量附加披露日期",
        report_date_help="批量附加报告日期",
    )
    _add_company_meta_args(
        parser,
        company_id_help="公司 ID（仅在工作区缺少 meta.json 时用于首条生成命令）",
        company_name_help="公司名称（仅在工作区缺少 meta.json 时用于首条生成命令；若显式传入，则优先于 --infer 返回值）",
        infer_help="使用 FMP 推断 ticker_aliases；成功时与显式 CSV alias 合并，且仅在未传 --company-name 时回退使用 FMP 公司名",
    )
    _add_overwrite_arg(parser, help_text="生成命令时附加 --overwrite")
    parser.add_argument(
        "--material-forms",
        dest="material_forms",
        default=None,
        help="强制覆盖 material 的 form_type；留空则按路由表自动识别",
    )
    _add_fins_common_args(parser)


def _add_fins_process_args(parser: argparse.ArgumentParser) -> None:
    """追加 `process` 子命令参数。"""

    _add_ticker_arg(parser, required=True, help_text="股票代码")
    parser.add_argument(
        "--document-id",
        dest="document_ids",
        action="append",
        default=None,
        help="仅处理指定文档 ID；可重复传入，也支持单个参数中用逗号分隔多个 ID",
    )
    _add_overwrite_arg(parser, help_text="是否覆盖已存在结果")
    _add_ci_arg(parser)
    _add_fins_common_args(parser)


def _add_fins_process_single_args(parser: argparse.ArgumentParser) -> None:
    """追加 `process_filing/process_material` 子命令参数。"""

    _add_ticker_arg(parser, required=True, help_text="股票代码")
    parser.add_argument("--document-id", dest="document_id", required=True, help="文档 ID")
    _add_overwrite_arg(parser, help_text="是否覆盖已存在结果")
    _add_ci_arg(parser)
    _add_fins_common_args(parser)


def _add_agent_args(parser: argparse.ArgumentParser) -> None:
    """追加 Agent 运行时参数（interactive / write 子命令共用，不含 --model-name）。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    add_execution_option_arguments(parser)


def _add_thinking_args(parser: argparse.ArgumentParser) -> None:
    """追加 thinking 回显开关。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否回显 thinking 增量（默认: --no-thinking）",
    )


def _add_write_args(parser: argparse.ArgumentParser) -> None:
    """追加 write 子命令专用参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--audit-model-name",
        type=str,
        default=None,
        help="审计模型配置名称（未传时使用 audit/confirm scene manifest 的 model.default_name）",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="写作模板文件路径（默认: workspace/assets/定性分析模板.md，回退 dayu/assets/定性分析模板.md）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="写作输出目录（默认: workspace/draft/{ticker}）",
    )
    parser.add_argument(
        "--write-max-retries",
        type=int,
        default=2,
        help="章节审计失败后的最大重写次数（默认: 2）",
    )
    parser.add_argument(
        "--chapter",
        type=str,
        default=None,
        help="仅写指定章节（如 '业务分析'），省略时执行全部章节",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用断点恢复（默认: --resume）",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="仅执行写作，不运行 audit/confirm/repair；全文和 --chapter 模式均生效",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="放宽第0章和第10章的 audit 前置门禁；全文和单章模式均生效",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="仅执行公司级 facet 归因并写回 manifest，不进入写作阶段",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="仅读取写作输出目录并打印上次写作流水线运行报告，不进入写作阶段",
    )


def _create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器。

    Args:
        无。

    Returns:
        已配置参数的解析器。

    Raises:
        无。
    """

    parser = DayuCliArgumentParser(
        prog="python -m dayu.cli",
        description="公司财报分析工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    interactive_parser = subparsers.add_parser("interactive", help="多轮交互终端对话")
    _add_global_args(interactive_parser)
    _add_agent_args(interactive_parser)
    _add_model_name_arg(
        interactive_parser,
        help_text="LLM 配置名称（未传时使用 interactive scene manifest 的 model.default_name）",
    )
    _add_thinking_args(interactive_parser)
    interactive_parser.add_argument(
        "--new-session",
        action="store_true",
        help="删除当前 interactive 会话绑定，并从新会话开始。",
    )

    prompt_parser = subparsers.add_parser("prompt", help="执行单次 prompt 并输出结果")
    _add_global_args(prompt_parser)
    _add_ticker_arg(
        prompt_parser,
        required=False,
        help_text="公司股票代码（可选，指定时启用财报工具）",
    )
    _add_agent_args(prompt_parser)
    prompt_parser.add_argument(
        "prompt",
        type=str,
        help="单次执行的 prompt 文本",
    )
    _add_model_name_arg(
        prompt_parser,
        help_text="LLM 配置名称（未传时使用 interactive scene manifest 的 model.default_name）",
    )
    _add_thinking_args(prompt_parser)

    write_parser = subparsers.add_parser("write", help="逐章写作或打印上次写作报告")
    _add_global_args(write_parser)
    _add_ticker_arg(
        write_parser,
        required=False,
        help_text="公司股票代码（可选，指定时启用财报工具）",
    )
    _add_agent_args(write_parser)
    _add_model_name_arg(
        write_parser,
        help_text="主写作场景模型名（未传时使用各 scene manifest 的 model.default_name）",
    )
    _add_write_args(write_parser)

    download_parser = subparsers.add_parser("download", help="下载 filings")
    _add_fins_download_args(download_parser)

    upload_filing_parser = subparsers.add_parser("upload_filing", help="上传财报")
    _add_fins_upload_filing_args(upload_filing_parser)

    upload_filings_from_parser = subparsers.add_parser(
        "upload_filings_from",
        help="从目录批量识别财报并生成上传脚本",
    )
    _add_fins_upload_filings_from_args(upload_filings_from_parser)

    upload_material_parser = subparsers.add_parser("upload_material", help="上传材料")
    _add_fins_upload_material_args(upload_material_parser)

    process_parser = subparsers.add_parser("process", help="全量预处理")
    _add_fins_process_args(process_parser)

    process_filing_parser = subparsers.add_parser("process_filing", help="处理单个 filing")
    _add_fins_process_single_args(process_filing_parser)

    process_material_parser = subparsers.add_parser("process_material", help="处理单个 material")
    _add_fins_process_single_args(process_material_parser)

    # 初始化子命令
    init_parser = subparsers.add_parser("init", help="初始化工作区并配置模型供应商")
    init_parser.add_argument(
        "--base", "-b",
        type=str,
        default="./workspace",
        help="工作区根目录（默认 ./workspace）",
    )
    _add_overwrite_arg(init_parser, help_text="覆盖已有配置文件")

    # 宿主管理子命令
    _register_host_subcommands(subparsers)

    return parser


def _register_host_subcommands(subparsers: argparse._SubParsersAction[DayuCliArgumentParser]) -> None:
    """注册宿主管理子命令的参数定义。

    这里仅保留 argparse 结构，避免 ``--help``/``parse_args`` 阶段
    提前导入宿主运行时实现模块。

    Args:
        subparsers: 顶层子命令注册器。

    Returns:
        无。

    Raises:
        无。
    """

    sessions_parser = subparsers.add_parser("sessions", help="管理会话")
    _add_global_args(sessions_parser)
    sessions_parser.add_argument("--all", action="store_true", dest="show_all", help="列出全部会话（含已关闭）")
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_action")
    close_parser = sessions_subparsers.add_parser("close", help="关闭会话")
    close_parser.add_argument("session_id", help="要关闭的 session ID")

    runs_parser = subparsers.add_parser("runs", help="管理运行记录")
    _add_global_args(runs_parser)
    runs_parser.add_argument("--all", action="store_true", dest="show_all", help="列出全部 run（含已完成）")
    runs_parser.add_argument("--session", dest="session_id", help="按 session 过滤")

    cancel_parser = subparsers.add_parser("cancel", help="取消运行")
    _add_global_args(cancel_parser)
    cancel_parser.add_argument("run_id", nargs="?", help="要取消的 run ID")
    cancel_parser.add_argument("--session", dest="session_id", help="取消 session 下所有活跃 run")

    host_parser = subparsers.add_parser("host", help="宿主维护")
    _add_global_args(host_parser)
    host_subparsers = host_parser.add_subparsers(dest="host_action")
    host_subparsers.add_parser("cleanup", help="清理孤儿 run 和过期 permit")
    host_subparsers.add_parser("status", help="显示宿主状态")


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。

    Args:
        无。

    Returns:
        解析后的命令行参数。

    Raises:
        无。
    """

    return _create_parser().parse_args()


# ---------------------------------------------------------------------------
# CLI 参数解析共享工具函数
# ---------------------------------------------------------------------------

def parse_limits_override(
    raw_json: str | None,
    *,
    field_name: str,
) -> "ExecutionOptionsOverridePayload | None":
    """解析工具 limits JSON 覆盖字符串。

    将 CLI / WeChat 等入口传入的 JSON 字符串解析为
    ``ExecutionOptionsOverridePayload``，校验值类型（仅允许标量）。

    Args:
        raw_json: 原始 JSON 字符串；``None`` 表示未提供。
        field_name: 当前参数名，用于错误提示。

    Returns:
        归一化后的覆盖字典；未提供时返回 ``None``。

    Raises:
        SystemExit: 当 JSON 非法、不是对象或包含非标量值时退出。
    """

    if raw_json is None:
        return None
    import json

    from dayu.execution.options import ExecutionOptionsOverridePayload
    from dayu.log import Log

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        Log.error(f"{field_name} 不是合法 JSON: {exc}", module=_PARSE_MODULE)
        raise SystemExit(2) from exc
    if not isinstance(parsed, dict):
        Log.error(f"{field_name} 必须是 JSON 对象", module=_PARSE_MODULE)
        raise SystemExit(2)
    normalized: ExecutionOptionsOverridePayload = {}
    for key, value in parsed.items():
        if value is None or isinstance(value, str | int | float | bool):
            normalized[str(key)] = value
            continue
        Log.error(f"{field_name} 只允许 JSON 标量值，字段 {key!r} 非法", module=_PARSE_MODULE)
        raise SystemExit(2)
    return normalized


def parse_temperature_argument(
    raw_value: str | int | float | None,
    *,
    field_name: str,
) -> float | None:
    """解析 temperature 参数。

    Args:
        raw_value: 原始参数值。
        field_name: 参数名，仅用于错误提示。

    Returns:
        标准化后的 temperature；未传时返回 ``None``。

    Raises:
        SystemExit: 当 temperature 非法时退出。
    """

    from dayu.execution.options import normalize_temperature
    from dayu.log import Log

    try:
        return normalize_temperature(raw_value, field_name=field_name)
    except ValueError as exc:
        Log.error(str(exc), module=_PARSE_MODULE)
        raise SystemExit(2) from exc


_PARSE_MODULE = "CLI.ARGS"
