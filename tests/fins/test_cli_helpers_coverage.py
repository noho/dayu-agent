"""fins.cli_support 补充覆盖测试。"""

from __future__ import annotations

import asyncio
import shlex
from argparse import Namespace
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from dayu.fins import cli_support as module
from dayu.fins.ingestion.process_events import ProcessEvent, ProcessEventType
from dayu.fins.pipelines.base import PipelineProtocol
from dayu.fins.pipelines.download_events import DownloadEvent, DownloadEventType
from dayu.fins.pipelines.upload_filing_events import UploadFilingEvent, UploadFilingEventType
from dayu.fins.pipelines.upload_material_events import UploadMaterialEvent, UploadMaterialEventType
from dayu.fins.ticker_normalization import NormalizedTicker


class _PipelineStub(PipelineProtocol):
    """最小 pipeline 桩。"""

    def __init__(self) -> None:
        """初始化 pipeline 调用记录。"""

        self.called: list[tuple[str, dict[str, Any]]] = []

    def download_stream(
        self,
        ticker: str,
        form_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: list[str] | None = None,
    ) -> AsyncIterator[DownloadEvent]:
        """测试中不应调用 download_stream。"""

        del ticker, form_type, start_date, end_date, overwrite, rebuild, ticker_aliases
        raise AssertionError("download_stream 不应在该测试中被调用")

    def download(
        self,
        ticker: str,
        form_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: list[str] | None = None,
    ) -> dict[str, Any]:
        """测试中不应调用 download。"""

        del ticker, form_type, start_date, end_date, overwrite, rebuild, ticker_aliases
        raise AssertionError("download 不应在该测试中被调用")

    def upload_filing(
        self,
        ticker: str,
        action: str | None,
        files: list[Path],
        fiscal_year: int,
        fiscal_period: str,
        amended: bool = False,
        filing_date: str | None = None,
        report_date: str | None = None,
        company_id: str | None = None,
        company_name: str | None = None,
        ticker_aliases: list[str] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """测试中不应调用 upload_filing。"""

        del (
            ticker,
            action,
            files,
            fiscal_year,
            fiscal_period,
            amended,
            filing_date,
            report_date,
            company_id,
            company_name,
            ticker_aliases,
            overwrite,
        )
        raise AssertionError("upload_filing 不应在该测试中被调用")

    def upload_filing_stream(
        self,
        ticker: str,
        action: str | None,
        files: list[Path],
        fiscal_year: int,
        fiscal_period: str,
        amended: bool = False,
        filing_date: str | None = None,
        report_date: str | None = None,
        company_id: str | None = None,
        company_name: str | None = None,
        ticker_aliases: list[str] | None = None,
        overwrite: bool = False,
    ) -> AsyncIterator[UploadFilingEvent]:
        """测试中不应调用 upload_filing_stream。"""

        del (
            ticker,
            action,
            files,
            fiscal_year,
            fiscal_period,
            amended,
            filing_date,
            report_date,
            company_id,
            company_name,
            ticker_aliases,
            overwrite,
        )
        raise AssertionError("upload_filing_stream 不应在该测试中被调用")

    def upload_material(
        self,
        ticker: str,
        action: str | None,
        form_type: str,
        material_name: str,
        files: list[Path] | None = None,
        document_id: str | None = None,
        internal_document_id: str | None = None,
        fiscal_year: int | None = None,
        fiscal_period: str | None = None,
        filing_date: str | None = None,
        report_date: str | None = None,
        company_id: str | None = None,
        company_name: str | None = None,
        ticker_aliases: list[str] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """记录材料上传调用。"""

        self.called.append(
            (
                "upload_material",
                {
                    "ticker": ticker,
                    "action": action,
                    "form_type": form_type,
                    "material_name": material_name,
                    "files": files,
                    "document_id": document_id,
                    "internal_document_id": internal_document_id,
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "company_id": company_id,
                    "company_name": company_name,
                    "ticker_aliases": ticker_aliases,
                    "overwrite": overwrite,
                },
            )
        )
        return {"ok": True}

    def upload_material_stream(
        self,
        ticker: str,
        action: str | None,
        form_type: str,
        material_name: str,
        files: list[Path] | None = None,
        document_id: str | None = None,
        internal_document_id: str | None = None,
        fiscal_year: int | None = None,
        fiscal_period: str | None = None,
        filing_date: str | None = None,
        report_date: str | None = None,
        company_id: str | None = None,
        company_name: str | None = None,
        ticker_aliases: list[str] | None = None,
        overwrite: bool = False,
    ) -> AsyncIterator[UploadMaterialEvent]:
        """测试中不应调用 upload_material_stream。"""

        del (
            ticker,
            action,
            form_type,
            material_name,
            files,
            document_id,
            internal_document_id,
            fiscal_year,
            fiscal_period,
            filing_date,
            report_date,
            company_id,
            company_name,
            ticker_aliases,
            overwrite,
        )
        raise AssertionError("upload_material_stream 不应在该测试中被调用")

    def process(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """记录全量处理调用。"""

        self.called.append(
            (
                "process",
                {
                    "ticker": ticker,
                    "overwrite": overwrite,
                    "ci": ci,
                    "document_ids": document_ids,
                },
            )
        )
        return {"ok": True}

    def process_stream(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: list[str] | None = None,
    ) -> AsyncIterator[ProcessEvent]:
        """测试中不应调用 process_stream。"""

        del ticker, overwrite, ci, document_ids
        raise AssertionError("process_stream 不应在该测试中被调用")

    def process_filing(
        self,
        ticker: str,
        document_id: str,
        overwrite: bool = False,
        ci: bool = False,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """记录单个 filing 处理调用。"""

        self.called.append(
            (
                "process_filing",
                {
                    "ticker": ticker,
                    "document_id": document_id,
                    "overwrite": overwrite,
                    "ci": ci,
                    "cancel_checker": cancel_checker,
                },
            )
        )
        return {"ok": True}

    def process_material(
        self,
        ticker: str,
        document_id: str,
        overwrite: bool = False,
        ci: bool = False,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """记录单个 material 处理调用。"""

        self.called.append(
            (
                "process_material",
                {
                    "ticker": ticker,
                    "document_id": document_id,
                    "overwrite": overwrite,
                    "ci": ci,
                    "cancel_checker": cancel_checker,
                },
            )
        )
        return {"ok": True}


@pytest.mark.unit
def test_build_pipeline_for_ticker_delegates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """覆盖 ticker 解析日志与 pipeline 工厂调用路径。"""

    monkeypatch.setattr(module, "normalize_ticker", lambda ticker: NormalizedTicker(canonical=ticker, market="US", exchange=None, raw=ticker))
    monkeypatch.setattr(module.Log, "debug", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "get_pipeline_from_normalized_ticker", lambda **kwargs: {"pipeline": "ok", **kwargs})

    result = cast(Any, module._build_pipeline_for_ticker("AAPL", tmp_path))
    assert result["pipeline"] == "ok"
    assert result["normalized_ticker"].canonical == "AAPL"


@pytest.mark.unit
def test_dispatch_action_multiple_branches() -> None:
    """覆盖 dispatch_action 的 upload_material/process/unsupported 分支。"""

    pipeline = _PipelineStub()

    upload_args = Namespace(
        command="upload_material",
        ticker="AAPL",
        action="create",
        form_type="MATERIAL_OTHER",
        material_name="Deck",
        files=["/tmp/a.pdf"],
        document_id=None,
        internal_document_id=None,
        fiscal_year=None,
        fiscal_period=None,
        filing_date=None,
        report_date=None,
        company_id="1",
        company_name="Apple",
        overwrite=False,
    )
    assert module._dispatch_action(pipeline, upload_args) == {"ok": True}

    process_args = Namespace(
        command="process",
        ticker="AAPL",
        overwrite=False,
        ci=True,
        document_ids=["fil_001", "fil_002"],
    )
    assert module._dispatch_action(pipeline, process_args) == {"ok": True}
    assert pipeline.called[1] == (
        "process",
        {
            "ticker": "AAPL",
            "overwrite": False,
            "ci": True,
            "document_ids": ["fil_001", "fil_002"],
        },
    )

    process_material_args = Namespace(command="process_material", ticker="AAPL", document_id="m1", overwrite=False, ci=False)
    assert module._dispatch_action(pipeline, process_material_args) == {"ok": True}

    with pytest.raises(ValueError, match="不支持的 command"):
        module._dispatch_action(pipeline, Namespace(command="bad"))


@pytest.mark.unit
def test_upload_stream_feedback_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖上传流回显、未返回结果异常与同步运行器分支。"""

    emitted_lines: list[str] = []

    def _capture_verbose(message: str, *, module: str = "APP") -> None:
        """捕获 verbose 输出。"""

        del module
        emitted_lines.append(message)

    monkeypatch.setattr(module.Log, "info", _capture_verbose)

    async def _stream_with_result() -> AsyncIterator[UploadFilingEvent]:
        yield UploadFilingEvent(
            event_type=UploadFilingEventType.FILE_UPLOADED,
            ticker="AAPL",
            payload={"name": "a", "size": 1},
        )
        yield UploadFilingEvent(
            event_type=UploadFilingEventType.UPLOAD_COMPLETED,
            ticker="AAPL",
            payload={"result": {"status": "ok"}},
        )

    result = asyncio.run(module._collect_upload_result_with_feedback(_stream_with_result(), stream_name="x"))
    assert result == {"status": "ok"}
    assert any("file_uploaded" in line for line in emitted_lines)

    async def _stream_without_result() -> AsyncIterator[UploadMaterialEvent]:
        yield UploadMaterialEvent(event_type=UploadMaterialEventType.UPLOAD_STARTED, ticker="AAPL", payload={})

    with pytest.raises(RuntimeError, match="未返回最终结果"):
        asyncio.run(module._collect_upload_result_with_feedback(_stream_without_result(), stream_name="y"))

    assert module._run_async_upload_stream_sync(asyncio.sleep(0, result={"ok": True})) == {"ok": True}

    monkeypatch.setattr(module.asyncio, "get_running_loop", lambda: object())
    with pytest.raises(RuntimeError, match="正在运行的事件循环"):
        module._run_async_upload_stream_sync(None)


@pytest.mark.unit
def test_pipeline_stream_feedback_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖 download/process 事件流回显与最终结果提取。"""

    emitted_lines: list[str] = []

    def _capture_verbose(message: str, *, module: str = "APP") -> None:
        """捕获 verbose 输出。"""

        del module
        emitted_lines.append(message)

    monkeypatch.setattr(module.Log, "verbose", _capture_verbose)

    async def _download_stream() -> AsyncIterator[DownloadEvent]:
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_STARTED,
            ticker="AAPL",
            payload={"overwrite": False, "rebuild": False},
        )
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_COMPLETED,
            ticker="AAPL",
            payload={"result": {"status": "ok", "ticker": "AAPL"}},
        )

    async def _process_stream_without_result() -> AsyncIterator[ProcessEvent]:
        yield ProcessEvent(
            event_type=ProcessEventType.PIPELINE_STARTED,
            ticker="AAPL",
            payload={"total_documents": 1},
        )

    result = asyncio.run(
        module._collect_pipeline_result_with_feedback(
            _download_stream(),
            stream_name="download_stream",
            formatter=module._format_download_stream_event_line,
        )
    )
    assert result == {"status": "ok", "ticker": "AAPL"}
    assert any("[download] started ticker=AAPL" in line for line in emitted_lines)

    with pytest.raises(RuntimeError, match="未返回最终结果"):
        asyncio.run(
            module._collect_pipeline_result_with_feedback(
                _process_stream_without_result(),
                stream_name="process_stream",
                formatter=module._format_process_stream_event_line,
            )
        )


@pytest.mark.unit
def test_coerce_and_validate_helpers(tmp_path: Path) -> None:
    """覆盖 forms 规范化、参数校验与路径收集分支。"""

    assert module._coerce_forms_input(None) is None
    assert module._coerce_forms_input("10-K") == "10-K"
    assert module._coerce_document_ids_input(" fil_1 , fil_2 ") == ["fil_1", "fil_2"]
    assert module._coerce_document_ids_input(["fil_1", "fil_2", "fil_1"]) == ["fil_1", "fil_2"]
    with pytest.raises(ValueError, match="forms 不能为空"):
        module._coerce_forms_input("   ")
    with pytest.raises(ValueError, match="forms 不能为空"):
        module._coerce_forms_input([" ", ""])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="document_ids 不能为空"):
        module._coerce_document_ids_input([" ", ""])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="--files"):
        module._validate_upload_filing_args(Namespace(action="create", files=None, company_id="1", company_name="a"))
    with pytest.raises(ValueError, match="--files"):
        module._validate_upload_material_args(
            Namespace(action="update", files=None, form_type="EARNINGS_CALL", document_id="d1", internal_document_id=None, company_id="1", company_name="a")
        )

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "a.txt").write_text("x", encoding="utf-8")
    (source_dir / "b.pdf").write_text("x", encoding="utf-8")
    (source_dir / "sub").mkdir()
    assert [p.name for p in module._collect_upload_from_files(source_dir=source_dir, recursive=False)] == [
        "a.txt",
        "b.pdf",
    ]


@pytest.mark.unit
def test_filename_infer_and_script_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """覆盖文件名财期推断、脚本路径与脚本写入分支。"""

    assert module._infer_fiscal_from_filename("report_2024_unknown.pdf") is None
    assert module._infer_fiscal_period_from_filename("2024Q2_report.pdf") == "Q2"
    assert module._infer_fiscal_period_from_filename("2024Q3_report.pdf") == "Q3"
    # Q4 无「季报」关键词时升级为 FY（港股 Q4 通常指全年业绩）
    assert module._infer_fiscal_period_from_filename("2024Q4_report.pdf") == "FY"
    # 含「季报」时明确保留 Q4
    assert module._infer_fiscal_period_from_filename("2024Q4季报.pdf") == "Q4"

    monkeypatch.setattr(module, "_get_current_upload_script_platform", lambda: "linux")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    resolved = module._resolve_upload_script_path(
        source_dir=tmp_path,
        ticker="aapl",
        output_script=str(out_dir),
        script_platform="linux",
    )
    assert resolved.name == "upload_filings_AAPL.sh"
    assert module._build_default_upload_script_name(ticker="aapl", script_platform="windows") == "upload_filings_AAPL.cmd"

    command = module._build_upload_filing_command(
        base_dir=Path("/tmp"),
        ticker="AAPL",
        action="create",
        file_path=Path("/tmp/a.pdf"),
        fiscal_year=2024,
        fiscal_period="FY",
        amended=True,
        filing_date="2024-11-01",
        report_date="2024-09-28",
        company_id="1",
        company_name="Apple",
        overwrite=True,
    )
    assert "--amended" in command and "--overwrite" in command
    assert command.startswith("python -m dayu.cli upload_filing --base ")

    parser = module._create_parser()
    parsed = parser.parse_args(shlex.split(command)[3:])
    assert parsed.command == "upload_filing"

    material_command = module._build_upload_material_command(
        base_dir=Path("/tmp"),
        ticker="AAPL",
        action="create",
        file_path=Path("/tmp/m.pdf"),
        material_forms="EARNINGS_CALL",
        material_name="call",
        fiscal_year=None,
        fiscal_period=None,
        filing_date=None,
        report_date=None,
        company_id=None,
        company_name=None,
        overwrite=False,
    )
    assert material_command.startswith("python -m dayu.cli upload_material --base ")

    script_path = tmp_path / "script.sh"
    module._write_upload_script(
        output_script=script_path,
        commands=[],
        regenerate_command="python -m dayu.cli upload_filings_from --ticker AAPL",
        script_platform="linux",
    )
    assert "没有识别到可上传的文件" in script_path.read_text(encoding="utf-8")
    assert "# python -m dayu.cli upload_filings_from --ticker AAPL" in script_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_upload_command_builders_omit_action_when_none(tmp_path: Path) -> None:
    """验证批量脚本命令在自动模式下不会写入 `--action`。"""

    filing_command = module._build_upload_filing_command(
        base_dir=tmp_path,
        ticker="AAPL",
        action=None,
        file_path=tmp_path / "a.pdf",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
        filing_date=None,
        report_date=None,
        company_id=None,
        company_name=None,
        overwrite=False,
    )
    assert "--action" not in filing_command
    assert "None" not in filing_command

    material_command = module._build_upload_material_command(
        base_dir=tmp_path,
        ticker="AAPL",
        action=None,
        file_path=tmp_path / "m.pdf",
        material_forms="MATERIAL_OTHER",
        material_name="deck",
        fiscal_year=None,
        fiscal_period=None,
        filing_date=None,
        report_date=None,
        company_id=None,
        company_name=None,
        overwrite=False,
    )
    assert "--action" not in material_command
    assert "None" not in material_command


@pytest.mark.unit
def test_regenerate_command_omits_action_when_none(tmp_path: Path) -> None:
    """验证重生成命令在自动模式下不会写入 `--action`。"""

    args = Namespace(
        ticker="AAPL",
        original_ticker="AAPL",
        action=None,
        recursive=False,
        amended=False,
        infer=False,
        filing_date=None,
        report_date=None,
        company_id="1",
        company_name="Apple",
        original_company_name="Apple",
        overwrite=False,
    )
    command = module._build_upload_filings_from_regenerate_command(
        args=args,
        source_dir=tmp_path / "source",
        base_dir=tmp_path,
        output_script=tmp_path / "upload_filings_AAPL.sh",
        script_platform="linux",
        include_company_meta_args=True,
    )
    assert "--action" not in command
    assert "None" not in command


@pytest.mark.unit
def test_windows_script_helpers(tmp_path: Path) -> None:
    """覆盖 Windows 平台脚本输出分支。"""

    script_path = tmp_path / "script.cmd"
    module._write_upload_script(
        output_script=script_path,
        commands=["python -m dayu.cli upload_filing --ticker AAPL"],
        regenerate_command='python -m dayu.cli upload_filings_from --ticker AAPL --from "C:\\tmp"',
        script_platform="windows",
    )
    text = script_path.read_text(encoding="utf-8")

    assert text.startswith("@echo off")
    assert "REM python -m dayu.cli upload_filings_from --ticker AAPL" in text
    assert "python -m dayu.cli upload_filing --ticker AAPL %*" in text


@pytest.mark.unit
def test_logging_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖日志等级设置与默认级别。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    levels: list[Any] = []
    monkeypatch.setattr(module.Log, "set_level", lambda level: levels.append(level))
    monkeypatch.setattr(module.Log, "debug", lambda *args, **kwargs: None)
    module._configure_logging(verbose=False, debug=True, info=False, quiet=False, log_level=None)
    module._configure_logging(verbose=True, debug=False, info=False, quiet=False, log_level=None)
    module._configure_logging(verbose=False, debug=False, info=True, quiet=False, log_level=None)
    module._configure_logging(verbose=False, debug=False, info=False, quiet=True, log_level=None)
    module._configure_logging(verbose=False, debug=False, info=False, quiet=False, log_level=None)
    module._configure_logging(verbose=False, debug=True, info=False, quiet=False, log_level="warn")
    assert levels == [
        module.LogLevel.DEBUG,
        module.LogLevel.VERBOSE,
        module.LogLevel.INFO,
        module.LogLevel.ERROR,
        module.LogLevel.INFO,
        module.LogLevel.WARN,
    ]


@pytest.mark.unit
def test_logging_parser_mutual_exclusion() -> None:
    """覆盖日志参数互斥解析与 `--log-level` 单独生效分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
        SystemExit: argparse 参数校验失败时抛出。
    """

    parser = module._create_parser()
    args = parser.parse_args(["download", "--ticker", "AAPL", "--log-level", "warn"])
    assert args.log_level == "warn"
    assert args.debug is False
    assert args.verbose is False
    assert args.info is False
    assert args.quiet is False

    with pytest.raises(SystemExit):
        parser.parse_args(["download", "--ticker", "AAPL", "--debug", "--quiet"])
    with pytest.raises(SystemExit):
        parser.parse_args(["download", "--ticker", "AAPL", "--log-level", "info", "--debug"])
