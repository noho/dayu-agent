#!/usr/bin/env python3
"""CN/HK Docling CI 批量 score runner。

该脚本固定调用 ``python -m dayu.fins.score_docling_ci``，把 CN/HK Docling
CI 的评分输出收敛到 ``workspace/tmp/docling_ci_score/{tag}``。它不处理
SEC form，不调用 ``score_sec_ci``，也不修改 scorer/profile。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence, TypeAlias, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dayu.fins.domain.enums import SourceKind
from dayu.fins.storage import FsCompanyMetaRepository, FsSourceDocumentRepository
from dayu.fins.ticker_normalization import try_normalize_ticker

DEFAULT_BASE = "workspace"
DEFAULT_TAG = "manual"
DEFAULT_SCORE_DIRNAME = "docling_ci_score"
REPORT_PROFILE_ID = "cn_hk_docling_v1"
SOURCE_KIND_ALL = "all"
SOURCE_KIND_FILING = "filing"
SOURCE_KIND_MATERIAL = "material"
REPORT_KIND_ALL = "all"
REPORT_KIND_ANNUAL = "annual"
REPORT_KIND_SEMIANNUAL = "semiannual"
REPORT_KIND_QUARTERLY = "quarterly"
REPORT_KIND_MATERIAL = "material"
SOURCE_KIND_CHOICES = (SOURCE_KIND_ALL, SOURCE_KIND_FILING, SOURCE_KIND_MATERIAL)
REPORT_KIND_CHOICES = (
    REPORT_KIND_ALL,
    REPORT_KIND_ANNUAL,
    REPORT_KIND_SEMIANNUAL,
    REPORT_KIND_QUARTERLY,
    REPORT_KIND_MATERIAL,
)
AVAILABLE_STATUS = "available"
CN_HK_MARKETS = ("CN", "HK")
NO_CN_HK_TICKERS_REASON = "没有发现 CN/HK active source ticker"
DEFAULT_SOURCE_KINDS = (SOURCE_KIND_FILING, SOURCE_KIND_MATERIAL)
DEFAULT_REPORT_KINDS = (
    REPORT_KIND_ANNUAL,
    REPORT_KIND_SEMIANNUAL,
    REPORT_KIND_QUARTERLY,
    REPORT_KIND_MATERIAL,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ScoreTarget:
    """单次 scorer 调用目标。

    Attributes:
        source_kind: scorer ``--source-kind`` 参数。
        report_kind: scorer ``--report-kind`` 参数。
        output_stem: 输出文件名主干。
        by_kind: 是否写入 ``by_kind`` 子目录。
    """

    source_kind: str
    report_kind: str
    output_stem: str
    by_kind: bool


@dataclass(frozen=True, slots=True)
class ScoreRunResult:
    """单次 scorer 执行结果。"""

    target: ScoreTarget
    return_code: int
    output_json: Path
    output_md: Path
    output_txt: Path
    summary: "ScoreSummaryRecord"
    markets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScoreSummaryRecord:
    """scorer JSON 中的 summary 摘要。"""

    avg: float
    p10: float
    hard_gate_failures: int
    completeness_failure_count: int
    document_count: int
    expected_document_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 命令行参数列表；为 ``None`` 时读取 ``sys.argv``。

    Returns:
        参数命名空间。

    Raises:
        SystemExit: 参数非法时由 ``argparse`` 抛出。
    """

    parser = argparse.ArgumentParser(description="CN/HK Docling CI 批量 score runner")
    parser.add_argument("--base", default=DEFAULT_BASE, help="workspace 根目录或 portfolio 目录")
    parser.add_argument("--tickers", default=None, help="逗号分隔 ticker；未传时通过 storage 扫描 CN/HK active source")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="本轮评分标签")
    parser.add_argument(
        "--source-kinds",
        default=",".join(DEFAULT_SOURCE_KINDS),
        help="逗号分隔 source kind；支持 all,filing,material",
    )
    parser.add_argument(
        "--report-kinds",
        default=",".join(DEFAULT_REPORT_KINDS),
        help="逗号分隔 report kind；支持 all,annual,semiannual,quarterly,material",
    )
    return parser.parse_args(argv)


def _resolve_workspace_root(base: str) -> Path:
    """把 CLI ``--base`` 解析为 workspace 根目录。

    Args:
        base: CLI 传入路径。

    Returns:
        workspace 根目录绝对路径。

    Raises:
        无。
    """

    normalized = Path(base).resolve()
    if normalized.name == "portfolio":
        return normalized.parent
    return normalized


def _resolve_project_root() -> Path:
    """解析仓库根目录。

    Args:
        无。

    Returns:
        仓库根目录绝对路径。

    Raises:
        无。
    """

    return Path(__file__).resolve().parents[1]


def _parse_csv_tokens(raw: str | None) -> list[str]:
    """解析逗号分隔字符串。

    Args:
        raw: 原始字符串。

    Returns:
        去空、去重后的 token 列表。

    Raises:
        无。
    """

    if raw is None:
        return []
    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    return list(dict.fromkeys(tokens))


def _resolve_choice_tokens(
    *,
    raw: str | None,
    allowed: tuple[str, ...],
    default: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    """解析并校验枚举型 CSV 参数。

    Args:
        raw: 原始 CSV 字符串。
        allowed: 允许值。
        default: 空输入时的默认值。
        label: 参数名称，用于错误信息。

    Returns:
        标准化 token 元组。

    Raises:
        ValueError: 存在不支持的 token 时抛出。
    """

    tokens = _parse_csv_tokens(raw)
    if not tokens:
        return default
    for token in tokens:
        if token not in allowed:
            raise ValueError(f"不支持的 {label}: {token}")
    return tuple(tokens)


def _is_active_source_meta(meta: JsonObject) -> bool:
    """判断 source meta 是否属于 active 文档。

    Args:
        meta: source meta。

    Returns:
        active source 返回 ``True``。

    Raises:
        无。
    """

    is_deleted = meta.get("is_deleted")
    if isinstance(is_deleted, bool) and is_deleted:
        return False
    ingest_complete = meta.get("ingest_complete")
    return not isinstance(ingest_complete, bool) or ingest_complete


def _ticker_has_active_cn_hk_source(workspace_root: Path, ticker: str) -> bool:
    """判断 ticker 是否存在 active CN/HK source 文档。

    Args:
        workspace_root: workspace 根目录。
        ticker: 股票代码。

    Returns:
        存在 active filing/material 时返回 ``True``。

    Raises:
        OSError: 仓储读取失败时抛出。
    """

    source_repository = FsSourceDocumentRepository(workspace_root)
    for source_kind in (SourceKind.FILING, SourceKind.MATERIAL):
        for document_id in source_repository.list_source_document_ids(ticker, source_kind):
            try:
                raw_meta = source_repository.get_source_meta(ticker, document_id, source_kind)
            except FileNotFoundError:
                continue
            if _is_active_source_meta(_as_json_object(cast(JsonValue, raw_meta))):
                return True
    return False


def _is_cn_hk_ticker_name(ticker: str) -> bool:
    """根据目录名 / ticker 名判断是否属于 CN/HK。

    Args:
        ticker: 公司目录名或 ticker。

    Returns:
        可归一为 CN/HK ticker 时返回 ``True``。

    Raises:
        无。
    """

    normalized = try_normalize_ticker(ticker)
    return normalized is not None and normalized.market in CN_HK_MARKETS


def _discover_cn_hk_tickers(workspace_root: Path) -> list[str]:
    """通过 storage 扫描 CN/HK active source ticker。

    CN/HK 与 SEC 的第一层区分使用目录名 / ticker 名归一化结果，而不是依赖
    scorer 的全量扫描或后续 snapshot meta。

    Args:
        workspace_root: workspace 根目录。

    Returns:
        已排序 ticker 列表。

    Raises:
        OSError: 仓储读取失败时抛出。
        ValueError: 公司元数据非法时抛出。
    """

    company_repository = FsCompanyMetaRepository(workspace_root)
    tickers: list[str] = []
    for entry in company_repository.scan_company_meta_inventory():
        if entry.status != AVAILABLE_STATUS or entry.company_meta is None:
            continue
        directory_ticker = entry.directory_name.strip().upper()
        meta_ticker = entry.company_meta.ticker.strip().upper()
        ticker = directory_ticker or meta_ticker
        if not ticker:
            continue
        if not _is_cn_hk_ticker_name(ticker) and not _is_cn_hk_ticker_name(meta_ticker):
            continue
        if _ticker_has_active_cn_hk_source(workspace_root, ticker):
            tickers.append(ticker)
    return sorted(set(tickers))


def _parse_tickers(raw: str | None, workspace_root: Path) -> list[str]:
    """解析 ticker CSV 参数。

    Args:
        raw: 原始 ticker 字符串。
        workspace_root: workspace 根目录。

    Returns:
        规范化 ticker 列表；空输入时扫描 CN/HK active source ticker。

    Raises:
        OSError: 仓储读取失败时抛出。
        ValueError: 公司元数据非法时抛出。
    """

    if raw is None:
        return _discover_cn_hk_tickers(workspace_root)
    parsed = list(dict.fromkeys(token.strip().upper() for token in raw.split(",") if token.strip()))
    return parsed if parsed else _discover_cn_hk_tickers(workspace_root)


def _is_valid_by_kind_pair(source_kind: str, report_kind: str) -> bool:
    """判断 source/report kind 组合是否属于 CN/HK Docling by-kind 观察口径。

    Args:
        source_kind: source kind。
        report_kind: report kind。

    Returns:
        是否应执行该组合。

    Raises:
        无。
    """

    if source_kind == SOURCE_KIND_ALL or report_kind == REPORT_KIND_ALL:
        return False
    if source_kind == SOURCE_KIND_FILING:
        return report_kind in {REPORT_KIND_ANNUAL, REPORT_KIND_SEMIANNUAL, REPORT_KIND_QUARTERLY}
    if source_kind == SOURCE_KIND_MATERIAL:
        return report_kind == REPORT_KIND_MATERIAL
    return False


def _build_score_targets(
    *,
    source_kinds: tuple[str, ...],
    report_kinds: tuple[str, ...],
) -> tuple[ScoreTarget, ...]:
    """构建本轮 scorer 调用目标。

    始终先执行一次 ``all/all`` 作为 overall 真源；by-kind 只执行 CN/HK 合法组合。

    Args:
        source_kinds: CLI 请求的 source kind 集合。
        report_kinds: CLI 请求的 report kind 集合。

    Returns:
        scorer 调用目标元组。

    Raises:
        无。
    """

    targets = [
        ScoreTarget(
            source_kind=SOURCE_KIND_ALL,
            report_kind=REPORT_KIND_ALL,
            output_stem="score",
            by_kind=False,
        )
    ]
    expanded_source_kinds = (
        DEFAULT_SOURCE_KINDS if SOURCE_KIND_ALL in source_kinds else source_kinds
    )
    expanded_report_kinds = (
        DEFAULT_REPORT_KINDS if REPORT_KIND_ALL in report_kinds else report_kinds
    )
    for source_kind in expanded_source_kinds:
        for report_kind in expanded_report_kinds:
            if not _is_valid_by_kind_pair(source_kind, report_kind):
                continue
            stem = f"score_{source_kind}_{report_kind}"
            targets.append(
                ScoreTarget(
                    source_kind=source_kind,
                    report_kind=report_kind,
                    output_stem=stem,
                    by_kind=True,
                )
            )
    return tuple(targets)


def _target_output_paths(tag_dir: Path, target: ScoreTarget) -> tuple[Path, Path, Path]:
    """构造单个 target 的输出路径。

    Args:
        tag_dir: ``workspace/tmp/docling_ci_score/{tag}`` 目录。
        target: scorer 调用目标。

    Returns:
        ``json, md, txt`` 路径元组。

    Raises:
        无。
    """

    output_dir = tag_dir / "by_kind" if target.by_kind else tag_dir
    return (
        output_dir / f"{target.output_stem}.json",
        output_dir / f"{target.output_stem}.md",
        output_dir / f"{target.output_stem}.txt",
    )


def _build_score_command(
    *,
    workspace_root: Path,
    tickers: list[str],
    target: ScoreTarget,
    output_json: Path,
    output_md: Path,
) -> tuple[str, ...]:
    """构造 ``score_docling_ci`` 子进程命令。

    Args:
        workspace_root: workspace 根目录。
        tickers: ticker 列表；为空时不传 ``--tickers``。
        target: scorer 调用目标。
        output_json: JSON 输出路径。
        output_md: Markdown 输出路径。

    Returns:
        子进程命令元组。

    Raises:
        无。
    """

    command = [
        sys.executable,
        "-m",
        "dayu.fins.score_docling_ci",
        "--base",
        str(workspace_root),
        "--source-kind",
        target.source_kind,
        "--report-kind",
        target.report_kind,
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    if tickers:
        command.extend(["--tickers", ",".join(tickers)])
    return tuple(command)


def _run_score_command(
    *,
    workspace_root: Path,
    project_root: Path,
    tickers: list[str],
    target: ScoreTarget,
    output_json: Path,
    output_md: Path,
    output_txt: Path,
) -> int:
    """执行单个 ``score_docling_ci`` 命令。

    Args:
        workspace_root: workspace 根目录。
        project_root: 仓库根目录。
        tickers: ticker 列表。
        target: scorer 调用目标。
        output_json: JSON 输出路径。
        output_md: Markdown 输出路径。
        output_txt: stdout/stderr 输出路径。

    Returns:
        子进程退出码。

    Raises:
        OSError: 子进程启动或输出写入失败时抛出。
    """

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    command = _build_score_command(
        workspace_root=workspace_root,
        tickers=tickers,
        target=target,
        output_json=output_json,
        output_md=output_md,
    )
    completed = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    output_txt.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
    return int(completed.returncode)


def _as_json_object(value: JsonValue) -> JsonObject:
    """把 JSON 值收窄为对象。

    Args:
        value: JSON 值。

    Returns:
        JSON 对象；非对象返回空字典。

    Raises:
        无。
    """

    if isinstance(value, dict):
        return cast(JsonObject, value)
    return {}


def _as_json_list(value: JsonValue) -> list[JsonValue]:
    """把 JSON 值收窄为列表。

    Args:
        value: JSON 值。

    Returns:
        JSON 列表；非列表返回空列表。

    Raises:
        无。
    """

    if isinstance(value, list):
        return cast(list[JsonValue], value)
    return []


def _number_value(data: JsonObject, key: str) -> float:
    """读取 JSON 数值。

    Args:
        data: JSON 对象。
        key: 字段名。

    Returns:
        浮点数；缺失或非法时返回 0。

    Raises:
        无。
    """

    value = data.get(key)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _int_value(data: JsonObject, key: str) -> int:
    """读取 JSON 整数。

    Args:
        data: JSON 对象。
        key: 字段名。

    Returns:
        整数；缺失或非法时返回 0。

    Raises:
        无。
    """

    value = data.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _string_value(data: JsonObject, key: str) -> str:
    """读取 JSON 字符串。

    Args:
        data: JSON 对象。
        key: 字段名。

    Returns:
        字符串；缺失或非标量时返回空字符串。

    Raises:
        无。
    """

    value = data.get(key)
    if value is None or isinstance(value, (list, dict)):
        return ""
    return str(value).strip()


def _load_score_json(path: Path) -> JsonObject:
    """读取 scorer JSON 输出。

    Args:
        path: JSON 文件路径。

    Returns:
        JSON 对象；文件不存在时返回空对象。

    Raises:
        OSError: 文件读取失败时抛出。
        ValueError: JSON 根节点不是对象时抛出。
    """

    if not path.exists():
        return {}
    parsed = cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(parsed, dict):
        raise ValueError(f"score JSON 根节点必须是对象: {path}")
    return cast(JsonObject, parsed)


def _extract_summary(payload: JsonObject) -> ScoreSummaryRecord:
    """从 scorer JSON 提取 summary。

    Args:
        payload: scorer JSON。

    Returns:
        摘要记录。

    Raises:
        无。
    """

    summary = _as_json_object(payload.get("summary", {}))
    return ScoreSummaryRecord(
        avg=round(_number_value(summary, "average_score"), 2),
        p10=round(_number_value(summary, "p10_score"), 2),
        hard_gate_failures=_int_value(summary, "hard_gate_failures"),
        completeness_failure_count=_int_value(summary, "completeness_failure_count"),
        document_count=_int_value(summary, "document_count"),
        expected_document_count=_int_value(summary, "expected_document_count"),
    )


def _extract_markets(payload: JsonObject) -> tuple[str, ...]:
    """从 scorer JSON 提取 market 集合。

    Args:
        payload: scorer JSON。

    Returns:
        已排序 market 元组。

    Raises:
        无。
    """

    markets: set[str] = set()
    for item in _as_json_list(payload.get("documents", [])):
        document = _as_json_object(item)
        market = _string_value(document, "market").upper()
        if market:
            markets.add(market)
    return tuple(sorted(markets))


def _document_scores(payload: JsonObject) -> list[float]:
    """从 scorer JSON 提取文档分数。

    Args:
        payload: scorer JSON。

    Returns:
        文档总分列表。

    Raises:
        无。
    """

    scores: list[float] = []
    for item in _as_json_list(payload.get("documents", [])):
        document = _as_json_object(item)
        score = _number_value(document, "total_score")
        scores.append(score)
    return scores


def _percentile_p10(values: list[float]) -> float:
    """计算 P10 分位。

    Args:
        values: 数值列表。

    Returns:
        P10；空列表返回 0。

    Raises:
        无。
    """

    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(float(sorted_values[0]), 2)
    position = 0.1 * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    ratio = position - lower
    return round(float(sorted_values[lower] * (1 - ratio) + sorted_values[upper] * ratio), 2)


def _run_target(
    *,
    workspace_root: Path,
    project_root: Path,
    tag_dir: Path,
    tickers: list[str],
    target: ScoreTarget,
) -> ScoreRunResult:
    """执行单个 score target 并读取输出。

    Args:
        workspace_root: workspace 根目录。
        project_root: 仓库根目录。
        tag_dir: 当前 tag 输出目录。
        tickers: ticker 列表。
        target: scorer 调用目标。

    Returns:
        单次执行结果。

    Raises:
        OSError: 子进程或文件读写失败时抛出。
        ValueError: score JSON 非法时抛出。
    """

    output_json, output_md, output_txt = _target_output_paths(tag_dir, target)
    return_code = _run_score_command(
        workspace_root=workspace_root,
        project_root=project_root,
        tickers=tickers,
        target=target,
        output_json=output_json,
        output_md=output_md,
        output_txt=output_txt,
    )
    payload = _load_score_json(output_json)
    return ScoreRunResult(
        target=target,
        return_code=return_code,
        output_json=output_json,
        output_md=output_md,
        output_txt=output_txt,
        summary=_extract_summary(payload),
        markets=_extract_markets(payload),
    )


def _empty_score_payload(target: ScoreTarget) -> JsonObject:
    """构造空 ticker fail-fast 的 scorer 兼容输出。

    Args:
        target: scorer 调用目标。

    Returns:
        空 score JSON 负载。

    Raises:
        无。
    """

    return {
        "profile_id": REPORT_PROFILE_ID,
        "report_kind": target.report_kind,
        "source_kind": target.source_kind,
        "summary": {
            "average_score": 0.0,
            "p10_score": 0.0,
            "hard_gate_failures": 0,
            "completeness_failure_count": 0,
            "expected_document_count": 0,
            "passed": False,
            "failed_reasons": [NO_CN_HK_TICKERS_REASON],
            "document_count": 0,
        },
        "documents": [],
        "completeness_failures": [],
    }


def _write_empty_target_outputs(tag_dir: Path, target: ScoreTarget) -> ScoreRunResult:
    """写出空 ticker fail-fast 的单 target 输出。

    Args:
        tag_dir: 当前 tag 输出目录。
        target: scorer 调用目标。

    Returns:
        单 target 执行结果，return_code 固定为 1。

    Raises:
        OSError: 文件写入失败时抛出。
    """

    output_json, output_md, output_txt = _target_output_paths(tag_dir, target)
    payload = _empty_score_payload(target)
    _write_json(output_json, payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        f"# CN/HK Docling CI 评分报告\n\n{NO_CN_HK_TICKERS_REASON}\n",
        encoding="utf-8",
    )
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(f"{NO_CN_HK_TICKERS_REASON}\n", encoding="utf-8")
    return ScoreRunResult(
        target=target,
        return_code=1,
        output_json=output_json,
        output_md=output_md,
        output_txt=output_txt,
        summary=_extract_summary(payload),
        markets=tuple(),
    )


def _summary_payload(results: tuple[ScoreRunResult, ...]) -> JsonObject:
    """构造 ``summary.json`` 负载。

    Args:
        results: 全部 score 执行结果。

    Returns:
        JSON 对象。

    Raises:
        无。
    """

    entries: dict[str, JsonValue] = {}
    for result in results:
        key = f"{result.target.source_kind}_{result.target.report_kind}"
        entries[key] = {
            "source_kind": result.target.source_kind,
            "report_kind": result.target.report_kind,
            "avg": result.summary.avg,
            "p10": result.summary.p10,
            "hard_gate_failures": result.summary.hard_gate_failures,
            "completeness_failure_count": result.summary.completeness_failure_count,
            "document_count": result.summary.document_count,
            "expected_document_count": result.summary.expected_document_count,
            "score_return_code": result.return_code,
            "output_json": str(result.output_json),
            "output_md": str(result.output_md),
            "output_txt": str(result.output_txt),
            "markets": list(result.markets),
        }
    return entries


def _overall_payload(
    *,
    overall_result: ScoreRunResult,
    overall_score_payload: JsonObject,
    results: tuple[ScoreRunResult, ...],
) -> JsonObject:
    """构造 ``overall_summary.json`` 负载。

    Args:
        overall_result: ``all/all`` 执行结果。
        overall_score_payload: ``all/all`` scorer JSON。
        results: 全部 score 执行结果。

    Returns:
        JSON 对象。

    Raises:
        无。
    """

    scores = _document_scores(overall_score_payload)
    avg = round(sum(scores) / len(scores), 2) if scores else overall_result.summary.avg
    source_kinds = sorted({item.target.source_kind for item in results})
    report_kinds = sorted({item.target.report_kind for item in results})
    markets = _extract_markets(overall_score_payload)
    return {
        "overall_avg": avg,
        "overall_p10": _percentile_p10(scores) if scores else overall_result.summary.p10,
        "overall_hard_gate_failures": overall_result.summary.hard_gate_failures,
        "overall_completeness_failure_count": overall_result.summary.completeness_failure_count,
        "overall_document_count": overall_result.summary.document_count,
        "overall_expected_document_count": overall_result.summary.expected_document_count,
        "source_kinds_included": source_kinds,
        "report_kinds_included": report_kinds,
        "markets_included": list(markets),
        "score_return_code": overall_result.return_code,
    }


def _write_json(path: Path, payload: JsonObject) -> None:
    """写出 JSON 文件。

    Args:
        path: 输出路径。
        payload: JSON 对象。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数列表；为 ``None`` 时读取 ``sys.argv``。

    Returns:
        任一 scorer 子进程失败时返回 1，否则返回 0。

    Raises:
        OSError: 仓储读取或文件写入失败时抛出。
        ValueError: 参数非法或 score JSON 非法时抛出。
    """

    args = parse_args(argv)
    workspace_root = _resolve_workspace_root(str(args.base))
    project_root = _resolve_project_root()
    tag_dir = workspace_root / "tmp" / DEFAULT_SCORE_DIRNAME / str(args.tag)
    tickers = _parse_tickers(
        str(args.tickers) if args.tickers is not None else None,
        workspace_root,
    )
    source_kinds = _resolve_choice_tokens(
        raw=str(args.source_kinds),
        allowed=SOURCE_KIND_CHOICES,
        default=DEFAULT_SOURCE_KINDS,
        label="source kind",
    )
    report_kinds = _resolve_choice_tokens(
        raw=str(args.report_kinds),
        allowed=REPORT_KIND_CHOICES,
        default=DEFAULT_REPORT_KINDS,
        label="report kind",
    )
    targets = _build_score_targets(source_kinds=source_kinds, report_kinds=report_kinds)
    if not targets:
        raise ValueError("没有可执行的 CN/HK Docling score target")
    if not tickers:
        results = tuple(_write_empty_target_outputs(tag_dir, target) for target in targets)
    else:
        results = tuple(
            _run_target(
                workspace_root=workspace_root,
                project_root=project_root,
                tag_dir=tag_dir,
                tickers=tickers,
                target=target,
            )
            for target in targets
        )
    overall_result = results[0]
    overall_score_payload = _load_score_json(overall_result.output_json)
    _write_json(tag_dir / "summary.json", _summary_payload(results))
    _write_json(
        tag_dir / "overall_summary.json",
        _overall_payload(
            overall_result=overall_result,
            overall_score_payload=overall_score_payload,
            results=results,
        ),
    )
    return 0 if all(item.return_code == 0 for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
