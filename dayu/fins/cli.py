"""财报域 CLI 薄层。

模块职责：
- 提供子命令风格的 CLI 入口，调用对应 Pipeline 方法。
- 解析 `ticker` 后通过 `MarketResolver` 自动路由到 SEC/CN Pipeline。
- 负责参数校验与基础格式转换（如 `--forms` 组装为 form 字符串）。

命令风格：
- `fins download ...`
- `fins upload_filing ...`
- `fins upload_filings_from ...`
- `fins upload_material ...`
- `fins process ...`
- `fins process_filing ...`
- `fins process_material ...`

使用说明：
- 各子命令均支持：
  - `--base`：工作区根目录（默认 `./workspace`）。
  - `--log-level/--verbose/--debug/--info/--quiet`：控制日志级别（默认 `INFO`，互斥）。
- `download` 支持：
  - `--ticker` 支持 CSV：`canonical,alias1,alias2`。
  - `--forms`：多个 form（可简写，如 `10Q 10K DEF14A`）。
  - `--start/--end`：日期格式支持 `YYYY`、`YYYY-MM`、`YYYY-MM-DD`。
  - `--rebuild`：基于本地已下载 filings 重建 `meta/manifest`（不重新下载）。
  - `--infer`：用 FMP 推断 alias；成功时与显式 CSV alias 合并，随后再与 SEC alias 合并，失败时回退到显式 CSV alias。
- `upload_filing` 支持 `--action`（`create|update|delete`，默认 `create`）：
  - `create/update` 必须提供 `--files`。
    - 若 `meta.json` 不存在，则 `create/update` 必须提供 `--company-id`；`--company-name` 可由 `--infer` 成功后补齐。
    - 若 `meta.json` 已存在，则重复传入 `--company-id/--company-name` 会被忽略并告警。
- `upload_material` 需要 `--action` 与 `--forms`：
  - `create/update` 必须提供 `--files`。
    - 若 `meta.json` 不存在，则 `create/update` 必须提供 `--company-id`；`--company-name` 可由 `--infer` 成功后补齐。
    - 若 `meta.json` 已存在，则重复传入 `--company-id/--company-name` 会被忽略并告警。
  - `update/delete` 必须提供 `--document-id` 或 `--internal-document-id`。
- `upload_filings_from`：
  - 扫描目录并从文件名中识别 `fiscal_year/fiscal_period`。
    - 若 `meta.json` 不存在，则仅首条生成命令附带 `--company-id` 与 `--company-name` 以初始化公司元数据；`--infer` 成功时会把“显式 CSV alias + FMP alias”的合并结果，以及最终公司名 bake 到脚本正文。
    - 若 `meta.json` 已存在，则生成脚本不再附带 `--company-id/--company-name`，重复传入会被忽略并告警。
  - 仅生成批量上传脚本，不直接执行上传；脚本格式跟随当前运行平台，且头部附带可复制的重生成命令注释。

示例：
- `python -m dayu.cli download --ticker AAPL --forms 10Q 10K DEF14A --start 2024 --end 2025-02`
- `python -m dayu.cli upload_filing --ticker 0300 --action create --files ./tmp/a.pdf --fiscal-year 2025 --fiscal-period FY --company-id 000333 --company-name 美的集团`
- `python -m dayu.cli upload_filings_from --ticker 0300 --from ./workspace/source --output ./workspace/upload_0300.sh`
- `python -m dayu.cli upload_material --ticker AAPL --action update --forms MATERIAL_OTHER --material-name deck --document-id mat_xxx --files ./tmp/deck.pdf --company-id 320193 --company-name "Apple Inc."`
- `python -m dayu.cli process --ticker AAPL --overwrite`
"""

from __future__ import annotations

import asyncio
import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from dayu.log import Log, LogLevel

from .ingestion.process_events import ProcessEvent
from .pipelines import PipelineProtocol, get_pipeline_from_market_profile
from .pipelines.download_events import DownloadEvent
from .pipelines.upload_filing_events import UploadFilingEvent
from .pipelines.upload_material_events import UploadMaterialEvent
from .resolver.market_resolver import MarketResolver
from .resolver.fmp_company_alias_resolver import (
    FmpAliasInferenceError,
    infer_company_aliases_from_fmp,
)
from .cli_formatters import (
    coerce_cli_result as _coerce_cli_result,
    format_cli_result as _format_cli_result,
    format_download_stream_event_line as _format_download_stream_event_line,
    format_process_stream_event_line as _format_process_stream_event_line,
    format_upload_stream_event_line as _format_upload_stream_event_line,
)
from dayu.contracts.fins import FinsResultData
from .upload_recognition import (
    SUPPORTED_UPLOAD_FROM_SUFFIXES,
    FISCAL_YEAR_PATTERN,
    Q1_PATTERN,
    Q2_PATTERN,
    Q3_PATTERN,
    Q4_PATTERN,
    H1_PATTERN,
    FY_PATTERN,
    _Q4_QUARTERLY_MARKER_PATTERN,
    _MATERIAL_ROUTING_TABLE,
    _PRIORITY_LONG_SCOPE_REPORT,
    _PRIORITY_QUARTERLY_REPORT,
    _PRIORITY_GENERIC_REPORT,
    _PRIORITY_ANNOUNCEMENT,
    _PRIORITY_SUPPLEMENTARY,
    _UPLOAD_ANNUAL_PERIODS,
    _UPLOAD_PERIODIC_PERIODS,
    _UPLOAD_MAX_ANNUAL,
    _UPLOAD_MAX_PERIODIC,
    _UPLOAD_MAX_PRESENTATION,
    _DEFAULT_MATERIAL_FORMS,
    _YEAR_SUBDIR_PATTERN,
    _detect_year_subdir_layout,
    _collect_upload_from_files,
    _infer_fiscal_from_filename,
    _infer_fiscal_from_path,
    _infer_fiscal_period_from_filename,
    _match_material_form_type,
    VALID_MATERIAL_FORM_TYPES,
    _compute_main_report_priority,
    _pick_best_per_period,
    _filter_upload_entries,
    _filter_material_entries,
    _derive_material_name,
)
from .storage.fs_company_meta_repository import FsCompanyMetaRepository

MODULE = "FINS.CLI"
_TICKER_CSV_COMMANDS = frozenset({"download", "upload_filing", "upload_material", "upload_filings_from"})


@dataclass(frozen=True)
class _ParsedTickerArgument:
    """CLI ticker CSV 解析结果。

    Attributes:
        raw_value: 原始输入值。
        canonical_ticker: 规范 ticker。
        explicit_aliases: CSV 中显式传入的 alias（不含 canonical）。
        normalized_csv: 规范化后的 CSV 字符串。
    """

    raw_value: str
    canonical_ticker: str
    explicit_aliases: list[str]
    normalized_csv: str


def _create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器。

    Args:
        无。

    Returns:
        已配置参数的解析器。

    Raises:
        无。
    """

    parser = argparse.ArgumentParser(description="Fins Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="下载 filings")
    _add_download_args(download_parser)

    upload_filing_parser = subparsers.add_parser("upload_filing", help="上传财报")
    _add_upload_filing_args(upload_filing_parser)

    upload_filings_from_parser = subparsers.add_parser(
        "upload_filings_from",
        help="从目录批量识别财报并生成上传脚本",
    )
    _add_upload_filings_from_args(upload_filings_from_parser)

    upload_material_parser = subparsers.add_parser("upload_material", help="上传材料")
    _add_upload_material_args(upload_material_parser)

    process_parser = subparsers.add_parser("process", help="全量预处理")
    _add_process_args(process_parser)

    process_filing_parser = subparsers.add_parser("process_filing", help="处理单个 filing")
    _add_process_single_args(process_filing_parser)

    process_material_parser = subparsers.add_parser("process_material", help="处理单个 material")
    _add_process_single_args(process_material_parser)

    return parser


def _add_global_args(parser: argparse.ArgumentParser) -> None:
    """追加各子命令共享的全局参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--base",
        dest="base",
        default="./workspace",
        help="工作区根目录（默认 ./workspace）",
    )
    log_level_group = parser.add_mutually_exclusive_group()
    log_level_group.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "verbose", "info", "warn", "error"],
        default=None,
        help="显式设置日志级别（debug/verbose/info/warn/error）",
    )
    log_level_group.add_argument(
        "--verbose",
        action="store_true",
        help="输出 verbose 级别日志",
    )
    log_level_group.add_argument(
        "--debug",
        action="store_true",
        help="输出 debug 级别日志",
    )
    log_level_group.add_argument(
        "--info",
        action="store_true",
        help="输出 info 级别日志",
    )
    log_level_group.add_argument(
        "--quiet",
        action="store_true",
        help="仅输出 error 级别日志",
    )


def _add_download_args(parser: argparse.ArgumentParser) -> None:
    """追加 download 子命令参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--ticker",
        required=True,
        help="股票代码；支持 CSV，如 BABA,9988,9988.HK，其中第一个值为 canonical ticker",
    )
    parser.add_argument(
        "--forms",
        dest="form_type",
        nargs="+",
        default=None,
        help="可选 form 列表（支持简写，如 10Q 10K DEF14A）",
    )
    parser.add_argument(
        "--start",
        dest="start_date",
        default=None,
        help="可选开始日期（YYYY/ YYYY-MM/ YYYY-MM-DD）",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        default=None,
        help="可选结束日期（YYYY/ YYYY-MM/ YYYY-MM-DD）",
    )
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在结果")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="是否基于本地已下载 filings 重建 meta/manifest（不重新下载）",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="使用 FMP 推断 ticker_aliases；infer 成功时与 --ticker CSV 中显式 alias 合并，失败时回退到显式 alias",
    )
    _add_global_args(parser)


def _add_upload_filing_args(parser: argparse.ArgumentParser) -> None:
    """追加 upload_filing 子命令参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--ticker",
        required=True,
        help="股票代码；支持 CSV，如 BABA,9988,9988.HK，其中第一个值为 canonical ticker",
    )
    parser.add_argument(
        "--action",
        dest="action",
        default="create",
        choices=["create", "update", "delete"],
        help="财报动作类型（默认 create）",
    )
    parser.add_argument("--files", nargs="+", default=None, help="上传文件列表")
    parser.add_argument("--fiscal-year", dest="fiscal_year", type=int, required=True, help="财年")
    parser.add_argument(
        "--fiscal-period",
        dest="fiscal_period",
        required=True,
        help="财季或年度标识（Q1/Q2/Q3/Q4/FY/H1）",
    )
    parser.add_argument("--amended", action="store_true", help="财报是否修订版")
    parser.add_argument("--filing-date", dest="filing_date", default=None, help="可选披露日期")
    parser.add_argument("--report-date", dest="report_date", default=None, help="可选报告日期")
    parser.add_argument(
        "--company-id",
        dest="company_id",
        default=None,
        help="公司 ID（仅在 meta.json 不存在时 create/update 必填）",
    )
    parser.add_argument(
        "--company-name",
        dest="company_name",
        default=None,
        help="公司名称（仅在 meta.json 不存在时 create/update 必填；--infer 成功时会被 FMP 覆盖）",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="使用 FMP 推断 ticker_aliases，并在成功时以 FMP 公司名覆盖 --company-name",
    )
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在结果")
    _add_global_args(parser)


def _add_upload_material_args(parser: argparse.ArgumentParser) -> None:
    """追加 upload_material 子命令参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--ticker",
        required=True,
        help="股票代码；支持 CSV，如 BABA,9988,9988.HK，其中第一个值为 canonical ticker",
    )
    parser.add_argument(
        "--action",
        dest="action",
        required=True,
        choices=["create", "update", "delete"],
        help="材料动作类型",
    )
    parser.add_argument("--forms", dest="form_type", required=True, help="材料 form_type")
    parser.add_argument("--material-name", dest="material_name", required=True, help="材料名称")
    parser.add_argument("--files", nargs="+", default=None, help="上传文件列表")
    parser.add_argument("--document-id", dest="document_id", default=None, help="文档 ID")
    parser.add_argument(
        "--internal-document-id",
        dest="internal_document_id",
        default=None,
        help="内部文档 ID",
    )
    parser.add_argument("--filing-date", dest="filing_date", default=None, help="可选披露日期")
    parser.add_argument("--report-date", dest="report_date", default=None, help="可选报告日期")
    parser.add_argument(
        "--company-id",
        dest="company_id",
        default=None,
        help="公司 ID（仅在 meta.json 不存在时 create/update 必填）",
    )
    parser.add_argument(
        "--company-name",
        dest="company_name",
        default=None,
        help="公司名称（仅在 meta.json 不存在时 create/update 必填；--infer 成功时会被 FMP 覆盖）",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="使用 FMP 推断 ticker_aliases，并在成功时以 FMP 公司名覆盖 --company-name",
    )
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在结果")
    _add_global_args(parser)


def _add_upload_filings_from_args(parser: argparse.ArgumentParser) -> None:
    """追加 upload_filings_from 子命令参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument(
        "--ticker",
        required=True,
        help="股票代码；支持 CSV，如 BABA,9988,9988.HK，其中第一个值为 canonical ticker",
    )
    parser.add_argument(
        "--from",
        dest="source_dir",
        required=True,
        help="待扫描文件目录",
    )
    parser.add_argument(
        "--action",
        dest="action",
        default="create",
        choices=["create", "update"],
        help="生成脚本中的上传动作（默认 create）",
    )
    parser.add_argument(
        "--output",
        dest="output_script",
        default=None,
        help="输出脚本路径，默认写到 --base 指向的 workspace 根目录下",
    )
    parser.add_argument("--recursive", action="store_true", help="是否递归扫描子目录")
    parser.add_argument("--amended", action="store_true", help="生成命令时附加 --amended")
    parser.add_argument("--filing-date", dest="filing_date", default=None, help="批量附加披露日期")
    parser.add_argument("--report-date", dest="report_date", default=None, help="批量附加报告日期")
    parser.add_argument(
        "--company-id",
        dest="company_id",
        default=None,
        help="公司 ID（仅在工作区缺少 meta.json 时用于首条生成命令）",
    )
    parser.add_argument(
        "--company-name",
        dest="company_name",
        default=None,
        help="公司名称（仅在工作区缺少 meta.json 时用于首条生成命令；--infer 成功时会被 FMP 覆盖）",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="使用 FMP 推断 ticker_aliases，并在成功时以 FMP 公司名覆盖 --company-name",
    )
    parser.add_argument("--overwrite", action="store_true", help="生成命令时附加 --overwrite")
    parser.add_argument(
        "--material-forms",
        dest="material_forms",
        default=None,
        help="强制覆盖 material 的 form_type；留空则按路由表自动识别（财务报表→FINANCIAL_STATEMENTS，业绩演示→EARNINGS_PRESENTATION 等）",
    )
    _add_global_args(parser)


def _add_process_args(parser: argparse.ArgumentParser) -> None:
    """追加 process 子命令参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--ticker", required=True, help="股票代码")
    parser.add_argument(
        "--document-id",
        dest="document_ids",
        action="append",
        default=None,
        help="仅处理指定文档 ID；可重复传入，也支持单个参数中用逗号分隔多个 ID",
    )
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在结果")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="是否追加导出 search_document 与 query_xbrl_facts 快照",
    )
    _add_global_args(parser)


def _add_process_single_args(parser: argparse.ArgumentParser) -> None:
    """追加单文档 process 子命令参数。

    Args:
        parser: 子命令解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--ticker", required=True, help="股票代码")
    parser.add_argument("--document-id", dest="document_id", required=True, help="文档 ID")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在结果")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="是否追加导出 search_document 与 query_xbrl_facts 快照",
    )
    _add_global_args(parser)


def _normalize_cli_ticker_token(raw_token: str) -> str:
    """规范化 CLI ticker token。

    Args:
        raw_token: 原始 ticker token。

    Returns:
        去空白并大写后的 ticker token。

    Raises:
        无。
    """

    compact_token = "".join(str(raw_token).strip().split())
    return compact_token.upper()


def _merge_ticker_alias_groups(
    *,
    canonical_ticker: str,
    alias_groups: Sequence[Optional[Sequence[str]]],
) -> list[str]:
    """按顺序合并多组 ticker alias。

    Args:
        canonical_ticker: 规范 ticker。
        alias_groups: 待合并的 alias 组，前组优先级更高。

    Returns:
        去重后的 alias 列表，首项恒为 canonical ticker。

    Raises:
        ValueError: 规范 ticker 为空时抛出。
    """

    normalized_canonical = _normalize_cli_ticker_token(canonical_ticker)
    if not normalized_canonical:
        raise ValueError("ticker 不能为空")
    merged_aliases: list[str] = [normalized_canonical]
    for alias_group in alias_groups:
        for raw_alias in alias_group or []:
            normalized_alias = _normalize_cli_ticker_token(str(raw_alias))
            if not normalized_alias or normalized_alias in merged_aliases:
                continue
            merged_aliases.append(normalized_alias)
    return merged_aliases


def _encode_ticker_csv(*, canonical_ticker: str, ticker_aliases: Sequence[str]) -> str:
    """把 canonical ticker 与 alias 编码为 CLI CSV。

    Args:
        canonical_ticker: 规范 ticker。
        ticker_aliases: alias 列表。

    Returns:
        规范化后的 CSV 字符串。

    Raises:
        ValueError: 规范 ticker 为空时抛出。
    """

    merged_aliases = _merge_ticker_alias_groups(
        canonical_ticker=canonical_ticker,
        alias_groups=[ticker_aliases],
    )
    return ",".join(merged_aliases)


def _parse_ticker_argument(raw_ticker: str) -> _ParsedTickerArgument:
    """解析 CLI `--ticker` 参数。

    支持：
    - 单值：`BABA`
    - CSV：`BABA,9988,9988.HK`

    其中首个 token 永远视为 canonical ticker。

    Args:
        raw_ticker: 原始 `--ticker` 输入。

    Returns:
        解析后的 ticker 参数对象。

    Raises:
        ValueError: 输入为空或 canonical ticker 缺失时抛出。
    """

    raw_value = str(raw_ticker or "").strip()
    if not raw_value:
        raise ValueError("--ticker 不能为空")
    raw_tokens = [token for token in raw_value.split(",")]
    normalized_tokens = [
        _normalize_cli_ticker_token(token)
        for token in raw_tokens
        if _normalize_cli_ticker_token(token)
    ]
    if not normalized_tokens:
        raise ValueError("--ticker 不能为空")
    canonical_ticker = normalized_tokens[0]
    explicit_aliases = _merge_ticker_alias_groups(
        canonical_ticker=canonical_ticker,
        alias_groups=[normalized_tokens[1:]],
    )[1:]
    return _ParsedTickerArgument(
        raw_value=raw_value,
        canonical_ticker=canonical_ticker,
        explicit_aliases=explicit_aliases,
        normalized_csv=_encode_ticker_csv(
            canonical_ticker=canonical_ticker,
            ticker_aliases=explicit_aliases,
        ),
    )


def _resolve_prepared_explicit_ticker_aliases(
    *,
    canonical_ticker: str,
    ticker_aliases: object,
) -> list[str]:
    """从预规范化参数中恢复显式 ticker alias。

    Args:
        canonical_ticker: 当前规范 ticker。
        ticker_aliases: 参数对象中已有的 ticker alias 列表。

    Returns:
        去重后的显式 alias 列表，不包含 canonical ticker。

    Raises:
        无。
    """

    if not isinstance(ticker_aliases, Sequence) or isinstance(ticker_aliases, str | bytes):
        return []
    merged_aliases = _merge_ticker_alias_groups(
        canonical_ticker=canonical_ticker,
        alias_groups=[[str(item) for item in ticker_aliases]],
    )
    return merged_aliases[1:]


def _ensure_supported_ticker_argument(args: argparse.Namespace) -> None:
    """校验并规范化 CLI ticker 参数。

    Args:
        args: 解析后的 CLI 参数。

    Returns:
        无。

    Raises:
        ValueError: 当非 CSV 命令传入 alias，或 ticker 非法时抛出。
    """

    parsed_ticker = _parse_ticker_argument(args.ticker)
    explicit_aliases = parsed_ticker.explicit_aliases
    if args.command in _TICKER_CSV_COMMANDS and not explicit_aliases:
        explicit_aliases = _resolve_prepared_explicit_ticker_aliases(
            canonical_ticker=parsed_ticker.canonical_ticker,
            ticker_aliases=getattr(args, "ticker_aliases", None),
        )
    if explicit_aliases:
        args.original_ticker = _encode_ticker_csv(
            canonical_ticker=parsed_ticker.canonical_ticker,
            ticker_aliases=explicit_aliases,
        )
    else:
        args.original_ticker = parsed_ticker.raw_value
    args.ticker = parsed_ticker.canonical_ticker
    args.explicit_ticker_aliases = explicit_aliases
    args.ticker_aliases = _merge_ticker_alias_groups(
        canonical_ticker=parsed_ticker.canonical_ticker,
        alias_groups=[explicit_aliases],
    )
    args.generated_ticker_csv = _encode_ticker_csv(
        canonical_ticker=parsed_ticker.canonical_ticker,
        ticker_aliases=args.ticker_aliases,
    )
    if args.command in _TICKER_CSV_COMMANDS:
        return
    if explicit_aliases:
        raise ValueError(f"{args.command} 仅支持单 ticker，不支持 CSV alias")


def _resolve_inferred_aliases(
    *,
    ticker: str,
    command_name: str,
) -> tuple[Optional[str], Optional[list[str]], Optional[Exception]]:
    """调用 FMP 推断公司名与 ticker alias。

    Args:
        ticker: 规范 ticker。
        command_name: 当前命令名，用于日志。

    Returns:
        `(company_name, ticker_aliases, error)` 三元组；成功时 `error=None`。

    Raises:
        无。失败会通过返回值暴露，便于调用方决定 fallback 还是 fail-fast。
    """

    try:
        inference_result = infer_company_aliases_from_fmp(ticker)
    except Exception as exc:
        Log.warn(
            f"{command_name}: FMP infer 失败，ticker={ticker} error={exc}",
            module=MODULE,
        )
        return None, None, exc
    return (
        inference_result.company_name,
        inference_result.ticker_aliases,
        None,
    )


def _prepare_download_args(args: argparse.Namespace) -> None:
    """准备 download 命令的最终 alias 参数。

    Args:
        args: CLI 参数。

    Returns:
        无。

    Raises:
        无。
    """

    if not bool(getattr(args, "infer", False)):
        args.generated_ticker_csv = _encode_ticker_csv(
            canonical_ticker=args.ticker,
            ticker_aliases=args.ticker_aliases,
        )
        return
    _, inferred_aliases, infer_error = _resolve_inferred_aliases(
        ticker=args.ticker,
        command_name="download",
    )
    if infer_error is None and inferred_aliases is not None:
        args.ticker_aliases = _merge_ticker_alias_groups(
            canonical_ticker=args.ticker,
            alias_groups=[args.explicit_ticker_aliases, inferred_aliases],
        )
    else:
        args.ticker_aliases = _merge_ticker_alias_groups(
            canonical_ticker=args.ticker,
            alias_groups=[args.explicit_ticker_aliases],
        )
    args.generated_ticker_csv = _encode_ticker_csv(
        canonical_ticker=args.ticker,
        ticker_aliases=args.ticker_aliases,
    )


def _prepare_upload_like_args(args: argparse.Namespace) -> None:
    """准备 upload / upload_filings_from 命令的最终 alias 与公司名参数。

    Args:
        args: CLI 参数。

    Returns:
        无。

    Raises:
        ValueError: infer 失败且缺少必需公司信息时抛出。
    """

    args.original_company_name = args.company_name
    base_dir = Path(args.base).expanduser().resolve()
    args.existing_company_meta = _load_existing_company_meta(base_dir=base_dir, ticker=args.ticker)
    args.ticker_aliases = _merge_ticker_alias_groups(
        canonical_ticker=args.ticker,
        alias_groups=[args.explicit_ticker_aliases],
    )
    if bool(getattr(args, "infer", False)) and args.existing_company_meta is None:
        inferred_company_name, inferred_aliases, infer_error = _resolve_inferred_aliases(
            ticker=args.ticker,
            command_name=args.command,
        )
        if infer_error is None and inferred_aliases is not None and inferred_company_name is not None:
            if not str(args.company_name or "").strip():
                args.company_name = inferred_company_name
            args.ticker_aliases = _merge_ticker_alias_groups(
                canonical_ticker=args.ticker,
                alias_groups=[args.explicit_ticker_aliases, inferred_aliases],
            )
    args.generated_ticker_csv = _encode_ticker_csv(
        canonical_ticker=args.ticker,
        ticker_aliases=args.ticker_aliases,
    )
    if args.existing_company_meta is not None:
        return
    _validate_company_meta_args(
        action_name=getattr(args, "action", "create"),
        company_id=getattr(args, "company_id", None),
        company_name=getattr(args, "company_name", None),
        command_name=args.command,
    )


def _prepare_cli_args(args: argparse.Namespace) -> None:
    """在真正分发前统一准备 CLI 参数。

    Args:
        args: 原始 CLI 参数。

    Returns:
        无。

    Raises:
        ValueError: 参数非法时抛出。
    """

    if bool(getattr(args, "_cli_args_prepared", False)):
        return
    if not hasattr(args, "base"):
        # 测试或内部辅助调用可能只构造最小 Namespace，这里补齐 CLI 默认值。
        args.base = "./workspace"
    _ensure_supported_ticker_argument(args)
    if args.command == "download":
        _prepare_download_args(args)
    elif args.command in {"upload_filing", "upload_material", "upload_filings_from"}:
        _prepare_upload_like_args(args)
    elif args.command == "process":
        args.document_ids = _coerce_document_ids_input(getattr(args, "document_ids", None))
    args._cli_args_prepared = True


def _build_pipeline_for_ticker(
    ticker: str,
    workspace_root: Path,
) -> PipelineProtocol:
    """按 ticker 构建 pipeline。

    Args:
        ticker: 股票代码。
        workspace_root: 工作区根目录。
    Returns:
        Pipeline 实例。

    Raises:
        ValueError: ticker 不合法时抛出。
    """

    market_profile = MarketResolver.resolve(ticker)
    Log.debug(
        f"ticker 解析完成: ticker={market_profile.ticker} "
        f"market={market_profile.market}",
        module=MODULE,
    )
    return get_pipeline_from_market_profile(
        market_profile=market_profile,
        workspace_root=workspace_root,
    )


def _dispatch_action(
    pipeline: PipelineProtocol,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """分发执行动作。

    Args:
        pipeline: 管线实例。
        args: 解析后的命令行参数。

    Returns:
        动作执行结果。

    Raises:
        ValueError: 当动作不支持时抛出。
    """

    supported_commands = {
        "upload_filings_from",
        "download",
        "upload_filing",
        "upload_material",
        "process",
        "process_filing",
        "process_material",
    }
    if args.command not in supported_commands:
        raise ValueError(f"不支持的 command: {args.command}")
    _prepare_cli_args(args)
    if args.command == "upload_filings_from":
        return _generate_upload_filings_script(args)
    if args.command == "download":
        return pipeline.download(
            ticker=args.ticker,
            form_type=_coerce_forms_input(args.form_type),
            start_date=args.start_date,
            end_date=args.end_date,
            overwrite=args.overwrite,
            rebuild=args.rebuild,
            ticker_aliases=args.ticker_aliases,
        )
    if args.command == "upload_filing":
        _validate_upload_filing_args(args)
        return pipeline.upload_filing(
            ticker=args.ticker,
            action=args.action,
            files=_to_paths(args.files),
            fiscal_year=args.fiscal_year,
            fiscal_period=args.fiscal_period,
            amended=args.amended,
            filing_date=args.filing_date,
            report_date=args.report_date,
            company_id=args.company_id,
            company_name=args.company_name,
            ticker_aliases=args.ticker_aliases,
            overwrite=args.overwrite,
        )
    if args.command == "upload_material":
        _validate_upload_material_args(args)
        return pipeline.upload_material(
            ticker=args.ticker,
            action=args.action,
            form_type=args.form_type,
            material_name=args.material_name,
            files=_to_paths(args.files),
            document_id=args.document_id,
            internal_document_id=args.internal_document_id,
            filing_date=args.filing_date,
            report_date=args.report_date,
            company_id=args.company_id,
            company_name=args.company_name,
            ticker_aliases=args.ticker_aliases,
            overwrite=args.overwrite,
        )
    if args.command == "process":
        return pipeline.process(
            ticker=args.ticker,
            overwrite=args.overwrite,
            ci=args.ci,
            document_ids=args.document_ids,
        )
    if args.command == "process_filing":
        return pipeline.process_filing(
            ticker=args.ticker,
            document_id=args.document_id,
            overwrite=args.overwrite,
            ci=args.ci,
        )
    if args.command == "process_material":
        return pipeline.process_material(
            ticker=args.ticker,
            document_id=args.document_id,
            overwrite=args.overwrite,
            ci=args.ci,
        )
    raise ValueError(f"不支持的 command: {args.command}")


def _dispatch_upload_action_with_stream_feedback(
    pipeline: PipelineProtocol,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """分发上传动作并输出流式逐文件回显。

    Args:
        pipeline: 管线实例。
        args: 解析后的命令行参数。

    Returns:
        上传动作最终结果。

    Raises:
        ValueError: 当命令不是上传动作时抛出。
        RuntimeError: 流式事件未返回最终结果时抛出。
    """

    _prepare_cli_args(args)
    if args.command == "upload_filing":
        _validate_upload_filing_args(args)
        stream = pipeline.upload_filing_stream(
            ticker=args.ticker,
            action=args.action,
            files=_to_paths(args.files),
            fiscal_year=args.fiscal_year,
            fiscal_period=args.fiscal_period,
            amended=args.amended,
            filing_date=args.filing_date,
            report_date=args.report_date,
            company_id=args.company_id,
            company_name=args.company_name,
            ticker_aliases=args.ticker_aliases,
            overwrite=args.overwrite,
        )
        return _run_async_upload_stream_sync(
            _collect_upload_result_with_feedback(
                stream=stream,
                stream_name="upload_filing_stream",
            )
        )
    if args.command == "upload_material":
        _validate_upload_material_args(args)
        stream = pipeline.upload_material_stream(
            ticker=args.ticker,
            action=args.action,
            form_type=args.form_type,
            material_name=args.material_name,
            files=_to_paths(args.files),
            document_id=args.document_id,
            internal_document_id=args.internal_document_id,
            filing_date=args.filing_date,
            report_date=args.report_date,
            company_id=args.company_id,
            company_name=args.company_name,
            ticker_aliases=args.ticker_aliases,
            overwrite=args.overwrite,
        )
        return _run_async_upload_stream_sync(
            _collect_upload_result_with_feedback(
                stream=stream,
                stream_name="upload_material_stream",
            )
        )
    raise ValueError(f"不支持的上传命令: {args.command}")


def _dispatch_ingestion_action_with_stream_feedback(
    pipeline: PipelineProtocol,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """分发长事务动作并输出流式进度回显。

    Args:
        pipeline: 管线实例。
        args: 解析后的命令行参数。

    Returns:
        长事务最终结果。

    Raises:
        ValueError: 当命令不是下载或预处理时抛出。
        RuntimeError: 流式事件未返回最终结果时抛出。
    """

    _prepare_cli_args(args)
    if args.command == "download":
        stream = pipeline.download_stream(
            ticker=args.ticker,
            form_type=_coerce_forms_input(args.form_type),
            start_date=args.start_date,
            end_date=args.end_date,
            overwrite=args.overwrite,
            rebuild=args.rebuild,
            ticker_aliases=args.ticker_aliases,
        )
        return _run_async_upload_stream_sync(
            _collect_pipeline_result_with_feedback(
                stream=stream,
                stream_name="download_stream",
                formatter=_format_download_stream_event_line,
            )
        )
    if args.command == "process":
        stream = pipeline.process_stream(
            ticker=args.ticker,
            overwrite=args.overwrite,
            ci=args.ci,
            document_ids=args.document_ids,
        )
        return _run_async_upload_stream_sync(
            _collect_pipeline_result_with_feedback(
                stream=stream,
                stream_name="process_stream",
                formatter=_format_process_stream_event_line,
            )
        )
    raise ValueError(f"不支持的长事务命令: {args.command}")


async def _collect_upload_result_with_feedback(
    stream: AsyncIterator[UploadFilingEvent | UploadMaterialEvent],
    *,
    stream_name: str,
) -> dict[str, Any]:
    """消费上传事件流并打印逐文件回显。

    Args:
        stream: 上传事件流。
        stream_name: 事件流名称（用于错误提示）。

    Returns:
        上传最终结果字典。

    Raises:
        RuntimeError: 事件流未返回最终结果时抛出。
    """

    result: Optional[dict[str, Any]] = None
    async for event in stream:
        line = _format_upload_stream_event_line(event)
        if line is not None:
            Log.info(line, module=MODULE)
        if event.event_type in {"upload_completed", "upload_failed"}:
            payload_result = event.payload.get("result")
            if isinstance(payload_result, dict):
                result = payload_result
    if result is None:
        raise RuntimeError(f"{stream_name} 未返回最终结果")
    return result


async def _collect_pipeline_result_with_feedback(
    stream: AsyncIterator[DownloadEvent | ProcessEvent],
    *,
    stream_name: str,
    formatter: Any,
) -> dict[str, Any]:
    """消费 download/process 事件流并打印进度回显。

    Args:
        stream: 长事务事件流。
        stream_name: 事件流名称（用于错误提示）。
        formatter: 单行格式化函数。

    Returns:
        最终结果字典。

    Raises:
        RuntimeError: 事件流未返回最终结果时抛出。
    """

    result: Optional[dict[str, Any]] = None
    async for event in stream:
        line = formatter(event)
        if line is not None:
            Log.verbose(line, module=MODULE)
        if event.event_type != "pipeline_completed":
            continue
        payload_result = event.payload.get("result")
        if isinstance(payload_result, dict):
            result = payload_result
    if result is None:
        raise RuntimeError(f"{stream_name} 未返回最终结果")
    return result


def _run_async_upload_stream_sync(coro: Any) -> dict[str, Any]:
    """在同步上下文运行上传事件消费协程。

    Args:
        coro: 协程对象。

    Returns:
        协程结果字典。

    Raises:
        RuntimeError: 当前线程已有运行中的事件循环时抛出。
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("检测到正在运行的事件循环，请改用 stream 异步接口")
def _to_paths(files: Optional[list[str]]) -> list[Path]:
    """将字符串路径列表转换为 Path 列表。

    Args:
        files: 可选字符串路径列表。

    Returns:
        Path 列表；当输入为空时返回空列表。

    Raises:
        无。
    """

    if files is None:
        return []
    return [Path(item) for item in files]


def _coerce_forms_input(raw_forms: Optional[Sequence[str] | str]) -> Optional[str]:
    """将 CLI forms 参数规范化为 form_type 字符串。

    Args:
        raw_forms: CLI 解析后的 forms 输入。

    Returns:
        规范化后的字符串；当输入为空时返回 `None`。

    Raises:
        ValueError: 当输入为空列表或仅包含空白时抛出。
    """

    if raw_forms is None:
        return None
    if isinstance(raw_forms, str):
        cleaned = raw_forms.strip()
        if not cleaned:
            raise ValueError("forms 不能为空")
        return cleaned
    cleaned_items = [str(item).strip() for item in raw_forms if str(item).strip()]
    if not cleaned_items:
        raise ValueError("forms 不能为空")
    return " ".join(cleaned_items)


def _coerce_document_ids_input(
    raw_document_ids: Optional[Sequence[str] | str],
) -> Optional[list[str]]:
    """将 CLI document_ids 参数规范化为文档 ID 列表。

    Args:
        raw_document_ids: CLI 解析后的 document_ids 输入。

    Returns:
        规范化后的文档 ID 列表；当输入为空时返回 `None`。

    Raises:
        ValueError: 当输入为空列表或仅包含空白时抛出。
    """

    if raw_document_ids is None:
        return None
    if isinstance(raw_document_ids, str):
        candidates = [raw_document_ids]
    else:
        candidates = list(raw_document_ids)
    normalized_ids: list[str] = []
    for item in candidates:
        for part in str(item).split(","):
            cleaned = part.strip()
            if cleaned:
                normalized_ids.append(cleaned)
    if not normalized_ids:
        raise ValueError("document_ids 不能为空")
    return list(dict.fromkeys(normalized_ids))


def _validate_upload_material_args(args: argparse.Namespace) -> None:
    """校验 `upload_material` 的动作级约束。

    校验顺序：
    1. 规范化 form_type 为大写，确保与 _MATERIAL_FORM_TYPE_TO_DOCUMENT_TYPE 等大写键一致。
    2. 校验 form_type 是否为合法枚举值（VALID_MATERIAL_FORM_TYPES），拒绝 typo。
    3. 校验动作与文件参数的组合约束。

    Args:
        args: 解析后的命令行参数；会原地规范化 args.form_type 为大写。

    Returns:
        无。

    Raises:
        ValueError: 当 form_type 不合法或动作与参数组合不合法时抛出。
    """

    # 规范化 form_type 为大写，确保与内部映射表（大写键）对齐，同时兼容用户小写输入
    args.form_type = args.form_type.strip().upper()
    if args.form_type not in VALID_MATERIAL_FORM_TYPES:
        raise ValueError(
            f"--forms '{args.form_type}' 不是合法的 material form_type，"
            f"支持的值：{sorted(VALID_MATERIAL_FORM_TYPES)}"
        )
    action_name = args.action
    if action_name in {"create", "update"} and not args.files:
        raise ValueError("upload_material 在 create/update 时必须提供 --files")
    if action_name in {"update", "delete"} and not (args.document_id or args.internal_document_id):
        raise ValueError(
            "upload_material 在 update/delete 时必须提供 --document-id 或 --internal-document-id"
        )


def _validate_upload_filing_args(args: argparse.Namespace) -> None:
    """校验 `upload_filing` 的动作级约束。

    Args:
        args: 解析后的命令行参数。

    Returns:
        无。

    Raises:
        ValueError: 当动作与参数组合不合法时抛出。
    """

    action_name = str(args.action).strip().lower()
    if action_name in {"create", "update"} and not args.files:
        raise ValueError("upload_filing 在 create/update 时必须提供 --files")


def _validate_company_meta_args(
    *,
    action_name: str,
    company_id: Optional[str],
    company_name: Optional[str],
    command_name: str,
) -> None:
    """校验 create/update 场景下的 company meta 参数。

    Args:
        action_name: 动作名称。
        company_id: 公司 ID 参数。
        company_name: 公司名称参数。
        command_name: 子命令名称（用于错误信息）。

    Returns:
        无。

    Raises:
        ValueError: create/update 缺失 company meta 时抛出。
    """

    normalized_action = str(action_name).strip().lower()
    if normalized_action not in {"create", "update"}:
        return
    if not str(company_id or "").strip():
        raise ValueError(f"{command_name} 在 create/update 时必须提供 --company-id")
    if not str(company_name or "").strip():
        raise ValueError(f"{command_name} 在 create/update 时必须提供 --company-name")


def _generate_upload_filings_script(args: argparse.Namespace) -> dict[str, Any]:
    """扫描目录并生成批量上传脚本。

    处理逻辑：
    1. 按文件名识别 fiscal_year/fiscal_period；``财务报表`` 文件单独归入 material 列表。
    2. 每个（fiscal_year, fiscal_period）组只保留主报告（按优先级去重）。
    3. 按收集上限过滤（年报≤5份，季报/半年报仅最新一年≤3份）。
    4. 脚本头部为 upload_filing 命令，尾部为 upload_material（财务报表）命令。

    Args:
        args: `upload_filings_from` 子命令参数。

    Returns:
        生成结果字典（包含脚本路径、识别结果、材料结果与跳过原因）。

    Raises:
        ValueError: 参数非法时抛出。
        FileNotFoundError: 源目录不存在时抛出。
        OSError: 脚本写入失败时抛出。
    """

    _prepare_cli_args(args)
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir 不存在或不是目录: {source_dir}")
    base_dir = Path(args.base).expanduser().resolve()
    script_platform = _get_current_upload_script_platform()
    output_script = _resolve_upload_script_path(
        base_dir=base_dir,
        ticker=args.ticker,
        output_script=args.output_script,
        script_platform=script_platform,
    )
    existing_company_meta = getattr(args, "existing_company_meta", None)
    if existing_company_meta is None:
        existing_company_meta = _load_existing_company_meta(base_dir=base_dir, ticker=args.ticker)
    include_company_meta_once = existing_company_meta is None
    if include_company_meta_once:
        _validate_company_meta_args(
            action_name=args.action,
            company_id=args.company_id,
            company_name=args.company_name,
            command_name="upload_filings_from",
        )
    else:
        _warn_ignored_company_meta_args(
            ticker=args.ticker,
            company_id=args.company_id,
            company_name=args.company_name,
            command_name="upload_filings_from",
        )
    candidate_files = _collect_upload_from_files(source_dir=source_dir, recursive=args.recursive)

    filing_candidates: list[dict[str, Any]] = []
    material_entries: list[dict[str, Any]] = []
    skipped_entries: list[dict[str, str]] = []

    for file_path in candidate_files:
        # material 路由：按 _MATERIAL_ROUTING_TABLE 匹配，命中则生成 upload_material 命令
        matched_form_type = _match_material_form_type(file_path.name)
        if args.material_forms is not None:
            # CLI 显式传入 --material-forms 时强制覆盖自动识别结果
            matched_form_type = matched_form_type and args.material_forms
        if matched_form_type is not None:
            mat_name = _derive_material_name(file_path.name, parent_dir_name=file_path.parent.name)
            material_entries.append(
                {
                    "file": str(file_path),
                    "material_name": mat_name,
                    "material_forms": matched_form_type,
                }
            )
            continue

        inferred = _infer_fiscal_from_path(file_path)
        if inferred is None:
            skipped_entries.append(
                {"file": str(file_path), "reason": "无法从文件名或目录名识别 fiscal_year/fiscal_period"}
            )
            continue

        fiscal_year, fiscal_period = inferred
        filing_candidates.append(
            {"file": str(file_path), "fiscal_year": fiscal_year, "fiscal_period": fiscal_period}
        )

    # 同期去重：每（year, period）只保留优先级最高的主报告
    filing_candidates, duped_entries = _pick_best_per_period(filing_candidates)
    for entry in duped_entries:
        skipped_entries.append({"file": entry["file"], "reason": "同期存在更高优先级报告，已去重"})

    # 按收集上限过滤
    recognized_entries, dropped_entries = _filter_upload_entries(filing_candidates)
    for entry in dropped_entries:
        skipped_entries.append(
            {
                "file": entry["file"],
                "reason": (
                    f"超出收集上限（annual≤{_UPLOAD_MAX_ANNUAL}，"
                    f"periodic≤{_UPLOAD_MAX_PERIODIC}/最新年）"
                ),
            }
        )

    # 按 form_type 上限过滤 material：
    # EARNINGS_PRESENTATION 最多 _UPLOAD_MAX_PRESENTATION 份（业绩演示同季报上限）;
    # EARNINGS_CALL 数量同年报+季报总数（每份报告至多对应一条电话会议记录）。
    material_caps: dict[str, int] = {
        "EARNINGS_PRESENTATION": _UPLOAD_MAX_PRESENTATION,
        "EARNINGS_CALL": len(recognized_entries),
    }
    material_entries, dropped_materials = _filter_material_entries(material_entries, material_caps)
    for entry in dropped_materials:
        skipped_entries.append({"file": entry["file"], "reason": "超出 material 收集上限"})

    _attach_upload_commands(
        recognized_entries=recognized_entries,
        material_entries=material_entries,
        base_dir=base_dir,
        ticker=args.generated_ticker_csv,
        action=args.action,
        amended=args.amended,
        filing_date=args.filing_date,
        report_date=args.report_date,
        company_id=args.company_id,
        company_name=args.company_name,
        overwrite=args.overwrite,
        include_company_meta_once=include_company_meta_once,
    )

    all_commands = (
        [e["command"] for e in recognized_entries]
        + [e["command"] for e in material_entries]
    )
    regenerate_command = _build_upload_filings_from_regenerate_command(
        args=args,
        source_dir=source_dir,
        base_dir=base_dir,
        output_script=output_script,
        script_platform=script_platform,
        include_company_meta_args=include_company_meta_once,
    )
    _write_upload_script(
        output_script=output_script,
        commands=all_commands,
        regenerate_command=regenerate_command,
        script_platform=script_platform,
    )
    return {
        "action": "upload_filings_from",
        "status": "ok",
        "ticker": args.ticker,
        "generated_ticker_csv": args.generated_ticker_csv,
        "source_dir": str(source_dir),
        "script_path": str(output_script),
        "script_platform": script_platform,
        "total_files": len(candidate_files),
        "recognized_count": len(recognized_entries),
        "material_count": len(material_entries),
        "skipped_count": len(skipped_entries),
        "recognized": recognized_entries,
        "material": material_entries,
        "skipped": skipped_entries,
    }


def _resolve_upload_script_path(
    *,
    base_dir: Optional[Path] = None,
    ticker: str,
    output_script: Optional[str],
    script_platform: str,
    source_dir: Optional[Path] = None,
) -> Path:
    """解析上传脚本输出路径。

    Args:
        base_dir: 工作区根目录。
        ticker: 股票代码。
        output_script: 用户传入脚本路径。
        script_platform: 脚本平台标识。
        source_dir: 旧接口遗留参数，当前仅用于兼容调用方；不参与路径计算。

    Returns:
        脚本绝对路径。

    Raises:
        ValueError: 路径参数非法时抛出。
    """

    default_script_name = _build_default_upload_script_name(
        ticker=ticker,
        script_platform=script_platform,
    )
    del source_dir
    if output_script:
        resolved_output = Path(output_script).expanduser().resolve()
        # 兼容目录输入：当 --output 指向目录时，自动使用默认脚本文件名。
        if resolved_output.exists() and resolved_output.is_dir():
            return resolved_output / default_script_name
        return resolved_output
    if base_dir is None:
        raise ValueError("base_dir 不能为空")
    return base_dir / default_script_name


def _get_current_upload_script_platform() -> str:
    """识别当前运行平台对应的脚本类型。

    Args:
        无。

    Returns:
        平台标识：`mac`、`linux` 或 `windows`。

    Raises:
        无。
    """

    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "mac"
    return "linux"


def _build_default_upload_script_name(*, ticker: str, script_platform: str) -> str:
    """按运行平台生成默认脚本文件名。

    Args:
        ticker: 股票代码。
        script_platform: 脚本平台标识。

    Returns:
        默认脚本文件名。

    Raises:
        无。
    """

    safe_ticker = ticker.strip().upper()
    if script_platform == "windows":
        return f"upload_filings_{safe_ticker}.cmd"
    return f"upload_filings_{safe_ticker}.sh"


def _join_upload_script_command(parts: Sequence[str], *, script_platform: str) -> str:
    """按平台拼接命令行字符串。

    Args:
        parts: 命令参数列表。
        script_platform: 脚本平台标识。

    Returns:
        单行命令字符串。

    Raises:
        无。
    """

    if script_platform == "windows":
        return subprocess.list2cmdline([str(part) for part in parts])
    return " ".join(shlex.quote(str(part)) for part in parts)


def _build_upload_filings_from_regenerate_command(
    *,
    args: argparse.Namespace,
    source_dir: Path,
    base_dir: Path,
    output_script: Path,
    script_platform: str,
    include_company_meta_args: bool,
) -> str:
    """构建脚本头部的重生成命令注释。

    Args:
        args: `upload_filings_from` 子命令参数。
        source_dir: 已解析的源目录。
        base_dir: 已解析的工作区根目录。
        output_script: 已解析的输出脚本路径。
        script_platform: 脚本平台标识。
        include_company_meta_args: 是否在重生成命令中保留 company meta 参数。

    Returns:
        可直接复制执行的 `upload_filings_from` 命令字符串。

    Raises:
        无。
    """

    parts = [
        "dayu-cli",
        "upload_filings_from",
        "--ticker",
        str(getattr(args, "original_ticker", args.ticker)),
        "--from",
        str(source_dir),
        "--base",
        str(base_dir),
        "--output",
        str(output_script),
    ]
    if getattr(args, "action", "create") != "create":
        parts.extend(["--action", str(args.action)])
    if bool(getattr(args, "recursive", False)):
        parts.append("--recursive")
    if bool(getattr(args, "amended", False)):
        parts.append("--amended")
    if bool(getattr(args, "infer", False)):
        parts.append("--infer")
    filing_date = getattr(args, "filing_date", None)
    if filing_date:
        parts.extend(["--filing-date", str(filing_date)])
    report_date = getattr(args, "report_date", None)
    if report_date:
        parts.extend(["--report-date", str(report_date)])
    company_id = getattr(args, "company_id", None)
    company_name = getattr(args, "original_company_name", getattr(args, "company_name", None))
    if include_company_meta_args and company_id:
        parts.extend(["--company-id", str(company_id)])
    if include_company_meta_args and company_name:
        parts.extend(["--company-name", str(company_name)])
    if bool(getattr(args, "overwrite", False)):
        parts.append("--overwrite")
    return _join_upload_script_command(parts, script_platform=script_platform)


def _build_upload_material_command(
    *,
    base_dir: Path,
    ticker: str,
    action: str,
    file_path: Path,
    material_forms: str,
    material_name: str,
    filing_date: Optional[str],
    report_date: Optional[str],
    company_id: Optional[str],
    company_name: Optional[str],
    overwrite: bool,
) -> str:
    """构建单条 upload_material 命令。

    Args:
        base_dir: CLI 工作区目录。
        ticker: 股票代码或已 bake 的 ticker CSV。
        action: 上传动作。
        file_path: 文件路径。
        material_forms: material form_type（如 ``FINANCIAL_STATEMENTS``）。
        material_name: 材料名称。
        filing_date: 披露日期。
        report_date: 报告日期。
        company_id: 公司 ID。
        company_name: 公司名称。
        overwrite: 是否覆盖。

    Returns:
        可直接在 shell 执行的命令行字符串。

    Raises:
        无。
    """

    parts = [
        "dayu-cli",
        "upload_material",
        "--base",
        str(base_dir),
        "--ticker",
        ticker,
        "--action",
        action,
        "--forms",
        material_forms,
        "--material-name",
        material_name,
        "--files",
        str(file_path),
    ]
    if company_id:
        parts.extend(["--company-id", company_id])
    if company_name:
        parts.extend(["--company-name", company_name])
    if filing_date:
        parts.extend(["--filing-date", filing_date])
    if report_date:
        parts.extend(["--report-date", report_date])
    if overwrite:
        parts.append("--overwrite")
    return _join_upload_script_command(parts, script_platform=_get_current_upload_script_platform())


def _build_upload_filing_command(
    *,
    base_dir: Path,
    ticker: str,
    action: str,
    file_path: Path,
    fiscal_year: int,
    fiscal_period: str,
    amended: bool,
    filing_date: Optional[str],
    report_date: Optional[str],
    company_id: Optional[str],
    company_name: Optional[str],
    overwrite: bool,
) -> str:
    """构建单条 upload_filing 命令。

    Args:
        base_dir: CLI 工作区目录。
        ticker: 股票代码或已 bake 的 ticker CSV。
        action: 上传动作。
        file_path: 文件路径。
        fiscal_year: 财年。
        fiscal_period: 财期。
        amended: 是否修订版。
        filing_date: 披露日期。
        report_date: 报告日期。
        company_id: 公司 ID。
        company_name: 公司名称。
        overwrite: 是否覆盖。

    Returns:
        可直接在 shell 执行的命令行字符串。

    Raises:
        无。
    """

    parts = [
        "dayu-cli",
        "upload_filing",
        "--base",
        str(base_dir),
        "--ticker",
        ticker,
        "--action",
        action,
        "--files",
        str(file_path),
        "--fiscal-year",
        str(fiscal_year),
        "--fiscal-period",
        fiscal_period,
    ]
    if company_id:
        parts.extend(["--company-id", company_id])
    if company_name:
        parts.extend(["--company-name", company_name])
    if amended:
        parts.append("--amended")
    if filing_date:
        parts.extend(["--filing-date", filing_date])
    if report_date:
        parts.extend(["--report-date", report_date])
    if overwrite:
        parts.append("--overwrite")
    return _join_upload_script_command(parts, script_platform=_get_current_upload_script_platform())


def _attach_upload_commands(
    *,
    recognized_entries: list[dict[str, Any]],
    material_entries: list[dict[str, Any]],
    base_dir: Path,
    ticker: str,
    action: str,
    amended: bool,
    filing_date: Optional[str],
    report_date: Optional[str],
    company_id: Optional[str],
    company_name: Optional[str],
    overwrite: bool,
    include_company_meta_once: bool,
) -> None:
    """为已筛选的 upload 条目补齐最终命令字符串。

    Args:
        recognized_entries: 已保留的 filing 条目。
        material_entries: 已保留的 material 条目。
        base_dir: CLI 工作区目录。
        ticker: 股票代码。
        action: 上传动作。
        amended: 是否修订版。
        filing_date: 披露日期。
        report_date: 报告日期。
        company_id: 公司 ID。
        company_name: 公司名称。
        overwrite: 是否覆盖。
        include_company_meta_once: 是否仅在首条命令里注入 company meta。

    Returns:
        无。

    Raises:
        无。
    """

    should_include_company_meta = include_company_meta_once
    for entry in recognized_entries:
        include_company_meta = should_include_company_meta
        entry["command"] = _build_upload_filing_command(
            base_dir=base_dir,
            ticker=ticker,
            action=action,
            file_path=Path(entry["file"]),
            fiscal_year=int(entry["fiscal_year"]),
            fiscal_period=str(entry["fiscal_period"]),
            amended=amended,
            filing_date=filing_date,
            report_date=report_date,
            company_id=company_id if include_company_meta else None,
            company_name=company_name if include_company_meta else None,
            overwrite=overwrite,
        )
        should_include_company_meta = False

    for entry in material_entries:
        include_company_meta = should_include_company_meta
        entry["command"] = _build_upload_material_command(
            base_dir=base_dir,
            ticker=ticker,
            action=action,
            file_path=Path(entry["file"]),
            material_forms=str(entry["material_forms"]),
            material_name=str(entry["material_name"]),
            filing_date=filing_date,
            report_date=report_date,
            company_id=company_id if include_company_meta else None,
            company_name=company_name if include_company_meta else None,
            overwrite=overwrite,
        )
        should_include_company_meta = False


def _load_existing_company_meta(*, base_dir: Path, ticker: str) -> Optional[Any]:
    """读取工作区中现有的公司元数据。

    Args:
        base_dir: 工作区根目录。
        ticker: 股票代码。

    Returns:
        已存在的公司元数据对象；若不存在则返回 `None`。

    Raises:
        ValueError: 元数据文件格式非法时抛出。
        OSError: 仓储读取失败时抛出。
    """

    repository = FsCompanyMetaRepository(base_dir)
    try:
        return repository.get_company_meta(ticker)
    except FileNotFoundError:
        return None


def _warn_ignored_company_meta_args(
    *,
    ticker: str,
    company_id: Optional[str],
    company_name: Optional[str],
    command_name: str,
) -> None:
    """记录 CLI 层忽略 company meta 参数的告警。

    Args:
        ticker: 股票代码。
        company_id: 公司 ID 参数。
        company_name: 公司名称参数。
        command_name: 子命令名称。

    Returns:
        无。

    Raises:
        无。
    """

    normalized_company_id = str(company_id or "").strip()
    normalized_company_name = str(company_name or "").strip()
    if not normalized_company_id and not normalized_company_name:
        return
    Log.warn(
        (
            f"{command_name}: ticker={ticker} 已存在公司元数据，"
            "将忽略本次传入的 --company-id/--company-name，并在生成脚本时复用现有 meta.json"
        ),
        module=MODULE,
    )


def _build_upload_script_header(
    *,
    script_platform: str,
    regenerate_command: str,
) -> list[str]:
    """构建批量上传脚本头部。

    Args:
        script_platform: 脚本平台标识。
        regenerate_command: 重生成命令。

    Returns:
        脚本头部行列表。

    Raises:
        无。
    """

    if script_platform == "windows":
        return [
            "@echo off",
            "setlocal",
            "",
            "REM 重新生成脚本：",
            f"REM {regenerate_command}",
            "",
        ]
    shebang = "#!/bin/zsh" if script_platform == "mac" else "#!/usr/bin/env bash"
    return [
        shebang,
        "set -euo pipefail",
        "",
        "# 重新生成脚本：",
        f"# {regenerate_command}",
        "",
    ]


def _write_upload_script(
    *,
    output_script: Path,
    commands: list[str],
    regenerate_command: str,
    script_platform: str,
) -> None:
    """将批量命令写入平台对应脚本。

    Args:
        output_script: 输出脚本路径。
        commands: 命令列表。
        regenerate_command: 重生成命令注释。
        script_platform: 脚本平台标识。

    Returns:
        无。

    Raises:
        OSError: 文件写入失败时抛出。
    """

    lines = _build_upload_script_header(
        script_platform=script_platform,
        regenerate_command=regenerate_command,
    )
    if commands:
        lines.extend(commands)
    else:
        if script_platform == "windows":
            lines.append("echo 没有识别到可上传的文件")
        else:
            lines.append("echo '没有识别到可上传的文件'")
    output_script.parent.mkdir(parents=True, exist_ok=True)
    output_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if script_platform != "windows":
        output_script.chmod(0o755)


def _configure_logging(
    verbose: bool,
    debug: bool,
    info: bool,
    quiet: bool = False,
    log_level: Optional[str] = None,
) -> None:
    """配置 CLI 日志级别。

    Args:
        verbose: 是否启用 VERBOSE。
        debug: 是否启用 debug（DEBUG）。
        info: 是否启用 info（INFO）。
        quiet: 是否仅输出 ERROR 日志。
        log_level: 显式日志级别字符串（debug/verbose/info/warn/error）。

    Returns:
        无。

    Raises:
        无。
    """

    if log_level:
        Log.set_level(LogLevel[log_level.upper()])
    elif debug:
        Log.set_level(LogLevel.DEBUG)
    elif verbose:
        Log.set_level(LogLevel.VERBOSE)
    elif info:
        Log.set_level(LogLevel.INFO)
    elif quiet:
        Log.set_level(LogLevel.ERROR)
    else:
        # 默认采用 INFO，仍需显式调用 set_level 以触发第三方库日志抑制逻辑。
        Log.set_level(LogLevel.INFO)
    Log.debug("日志初始化完成", module=MODULE)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口函数。

    Args:
        argv: 可选命令行参数列表；为空时读取系统参数。

    Returns:
        退出码，0 表示成功。

    Raises:
        Exception: 参数错误或执行错误时抛出。
    """

    parser = _create_parser()
    args = parser.parse_args(argv)
    _configure_logging(
        verbose=args.verbose,
        debug=args.debug,
        info=args.info,
        quiet=args.quiet,
        log_level=args.log_level,
    )
    try:
        _prepare_cli_args(args)
        if args.command == "upload_filings_from":
            result = _generate_upload_filings_script(args)
        else:
            workspace_root = Path(args.base).expanduser().resolve()
            pipeline = _build_pipeline_for_ticker(
                ticker=args.ticker,
                workspace_root=workspace_root,
            )
            if args.command in {"upload_filing", "upload_material"}:
                result = _dispatch_upload_action_with_stream_feedback(
                    pipeline=pipeline,
                    args=args,
                )
            elif args.command in {"download", "process"}:
                result = _dispatch_ingestion_action_with_stream_feedback(
                    pipeline=pipeline,
                    args=args,
                )
            else:
                result = _dispatch_action(pipeline=pipeline, args=args)
    except (RuntimeError, ValueError) as exc:
        Log.error(f"命令执行失败: {exc}", module=MODULE)
        return 1
    typed_result: FinsResultData = _coerce_cli_result(args.command, result)
    print(_format_cli_result(args.command, typed_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "入口已迁移，请使用 `python -m dayu.cli <subcommand>`。"
    )
