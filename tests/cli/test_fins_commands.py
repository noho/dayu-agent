"""``dayu.cli.commands.fins`` 财报命令构建与执行测试。"""

from __future__ import annotations

import argparse
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from dayu.cli.commands.fins import (
    _build_fins_command,
    _consume_fins_stream,
    _format_fins_progress_line,
    run_fins_command,
    _should_log_fins_progress_as_info,
)
from dayu.contracts.fins import (
    DownloadCommandPayload,
    DownloadProgressPayload,
    DownloadResultData,
    FinsCommandName,
    FinsEvent,
    FinsEventType,
    FinsProgressEventName,
    FinsResult,
    ProcessCommandPayload,
    ProcessProgressPayload,
    ProcessResultData,
    ProcessSingleResultData,
    UploadFilingProgressPayload,
    UploadMaterialProgressPayload,
)


# --------------------------------------------------------------------------- #
#  _build_fins_command
# --------------------------------------------------------------------------- #


class TestBuildFinsCommand:
    """_build_fins_command 测试。"""

    def test_unknown_command_raises_value_error(self) -> None:
        """非财报命令名抛出 ValueError。"""
        args = argparse.Namespace(command="unknown_cmd")
        with pytest.raises(ValueError, match="不是财报命令"):
            _build_fins_command(args)

    def test_download_command(self) -> None:
        """download 命令构建 DownloadCommandPayload。"""
        args = argparse.Namespace(
            command="download",
            ticker="AAPL",
            form_type=["10-K", "10-Q"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            overwrite=True,
            rebuild=False,
            infer=False,
            ticker_aliases=(),
        )
        cmd = _build_fins_command(args)
        assert cmd.name == FinsCommandName.DOWNLOAD
        assert isinstance(cmd.payload, DownloadCommandPayload)
        assert cmd.payload.ticker == "AAPL"
        assert cmd.payload.form_type == ("10-K", "10-Q")
        assert cmd.stream is True

    def test_process_command(self) -> None:
        """process 命令构建 ProcessCommandPayload。"""
        args = argparse.Namespace(
            command="process",
            ticker="AAPL",
            document_ids=None,
            overwrite=False,
            ci=False,
        )
        cmd = _build_fins_command(args)
        assert cmd.name == FinsCommandName.PROCESS
        assert isinstance(cmd.payload, ProcessCommandPayload)
        assert cmd.stream is True

    def test_process_filing_command_not_streaming(self) -> None:
        """process_filing 命令不启用流式。"""
        args = argparse.Namespace(
            command="process_filing",
            ticker="AAPL",
            document_id="doc-123",
            overwrite=False,
            ci=False,
        )
        cmd = _build_fins_command(args)
        assert cmd.name == FinsCommandName.PROCESS_FILING
        assert cmd.stream is False

    def test_process_material_command_not_streaming(self) -> None:
        """process_material 命令不启用流式。"""
        args = argparse.Namespace(
            command="process_material",
            ticker="AAPL",
            document_id="doc-456",
            overwrite=False,
            ci=False,
        )
        cmd = _build_fins_command(args)
        assert cmd.name == FinsCommandName.PROCESS_MATERIAL
        assert cmd.stream is False


# --------------------------------------------------------------------------- #
#  _format_fins_progress_line
# --------------------------------------------------------------------------- #


class TestFormatFinsProgressLine:
    """_format_fins_progress_line 测试。"""

    def test_download_progress_with_form_type(self) -> None:
        """DownloadProgressPayload 携带 form_type 时输出 form_type 字段。"""
        payload = DownloadProgressPayload(
            event_type=FinsProgressEventName.FILE_DOWNLOADED,
            ticker="AAPL",
            document_id="doc-1",
            form_type="10-K",
        )
        line = _format_fins_progress_line(FinsCommandName.DOWNLOAD, payload)
        assert "form_type=10-K" in line
        assert "ticker=AAPL" in line
        assert "document_id=doc-1" in line

    def test_download_progress_without_form_type(self) -> None:
        """DownloadProgressPayload 不携带 form_type 时不输出 form_type 字段。"""
        payload = DownloadProgressPayload(
            event_type=FinsProgressEventName.FILING_STARTED,
            ticker="AAPL",
        )
        line = _format_fins_progress_line(FinsCommandName.DOWNLOAD, payload)
        assert "form_type" not in line

    def test_upload_filing_progress_with_action_and_name(self) -> None:
        """UploadFilingProgressPayload 携带 action 和 name 时输出。"""
        payload = UploadFilingProgressPayload(
            event_type=FinsProgressEventName.UPLOAD_STARTED,
            ticker="MSFT",
            action="create",
            name="filing-2023",
            file_count=3,
            size=1024,
        )
        line = _format_fins_progress_line(FinsCommandName.UPLOAD_FILING, payload)
        assert "action=create" in line
        assert "name=filing-2023" in line
        assert "file_count=3" in line
        assert "size=1024" in line

    def test_process_progress_with_message(self) -> None:
        """ProcessProgressPayload 携带 reason 时输出 message。"""
        payload = ProcessProgressPayload(
            event_type=FinsProgressEventName.DOCUMENT_STARTED,
            ticker="AAPL",
            document_id="doc-7",
            reason="processing",
        )
        line = _format_fins_progress_line(FinsCommandName.PROCESS, payload)
        assert "message=processing" in line

    def test_upload_material_progress_with_error(self) -> None:
        """UploadMaterialProgressPayload 携带 error 时输出 message。"""
        payload = UploadMaterialProgressPayload(
            event_type=FinsProgressEventName.UPLOAD_FAILED,
            ticker="AAPL",
            error="network timeout",
        )
        line = _format_fins_progress_line(FinsCommandName.UPLOAD_MATERIAL, payload)
        assert "message=network timeout" in line


# --------------------------------------------------------------------------- #
#  _should_log_fins_progress_as_info
# --------------------------------------------------------------------------- #


class TestShouldLogFinsProgressAsInfo:
    """_should_log_fins_progress_as_info 测试。"""

    def test_upload_filing_returns_true(self) -> None:
        """upload_filing 命令返回 True。"""
        assert _should_log_fins_progress_as_info("upload_filing") is True

    def test_upload_material_returns_true(self) -> None:
        """upload_material 命令返回 True。"""
        assert _should_log_fins_progress_as_info("upload_material") is True

    def test_download_returns_false(self) -> None:
        """download 命令返回 False。"""
        assert _should_log_fins_progress_as_info("download") is False

    def test_process_returns_false(self) -> None:
        """process 命令返回 False。"""
        assert _should_log_fins_progress_as_info("process") is False

    def test_process_filing_returns_false(self) -> None:
        """process_filing 命令返回 False。"""
        assert _should_log_fins_progress_as_info("process_filing") is False

    def test_unknown_command_returns_false(self) -> None:
        """未知命令返回 False。"""
        assert _should_log_fins_progress_as_info("unknown") is False


# --------------------------------------------------------------------------- #
#  _consume_fins_stream
# --------------------------------------------------------------------------- #


class TestConsumeFinsStream:
    """_consume_fins_stream 测试。"""

    @pytest.mark.asyncio
    async def test_yields_result_from_result_event(self) -> None:
        """收到 RESULT 事件时返回最终结果。"""
        result_data = ProcessResultData(
            pipeline="process",
            status="ok",
            ticker="AAPL",
        )

        async def _stream() -> AsyncIterator[FinsEvent]:
            yield FinsEvent(
                type=FinsEventType.RESULT,
                command=FinsCommandName.PROCESS,
                payload=result_data,
            )

        result = await _consume_fins_stream(_stream(), FinsCommandName.PROCESS)
        assert result is result_data

    @pytest.mark.asyncio
    async def test_progress_then_result(self) -> None:
        """先收到 PROGRESS 再收到 RESULT 时正确返回。"""
        result_data = ProcessResultData(
            pipeline="process",
            status="ok",
            ticker="AAPL",
        )

        async def _stream() -> AsyncIterator[FinsEvent]:
            yield FinsEvent(
                type=FinsEventType.PROGRESS,
                command=FinsCommandName.PROCESS,
                payload=ProcessProgressPayload(
                    event_type=FinsProgressEventName.DOCUMENT_STARTED,
                    ticker="AAPL",
                ),
            )
            yield FinsEvent(
                type=FinsEventType.RESULT,
                command=FinsCommandName.PROCESS,
                payload=result_data,
            )

        result = await _consume_fins_stream(_stream(), FinsCommandName.PROCESS)
        assert result is result_data

    @pytest.mark.asyncio
    async def test_no_result_raises_runtime_error(self) -> None:
        """未收到 RESULT 事件时抛出 RuntimeError。"""
        async def _stream() -> AsyncIterator[FinsEvent]:
            if False:
                yield  # pragma: no cover

        with pytest.raises(RuntimeError, match="流式执行未返回最终结果"):
            await _consume_fins_stream(_stream(), FinsCommandName.PROCESS)

    @pytest.mark.asyncio
    async def test_skips_non_fins_event(self) -> None:
        """非 FinsEvent 实例被跳过，最终仍返回结果。"""
        result_data = ProcessResultData(
            pipeline="process",
            status="ok",
            ticker="AAPL",
        )

        class NotAFinsEvent:
            """非 FinsEvent 类型。"""

        async def _stream() -> AsyncIterator[object]:  # type: ignore[misc]
            yield NotAFinsEvent()
            yield FinsEvent(
                type=FinsEventType.RESULT,
                command=FinsCommandName.PROCESS,
                payload=result_data,
            )

        result = await _consume_fins_stream(
            _stream(),  # type: ignore[arg-type]
            FinsCommandName.PROCESS,
        )
        assert result is result_data

    @pytest.mark.asyncio
    async def test_upload_filing_progress_logged_as_info(self) -> None:
        """upload_filing 进度按 INFO 输出。"""
        result_data = ProcessResultData(
            pipeline="process",
            status="ok",
            ticker="AAPL",
        )

        async def _stream() -> AsyncIterator[FinsEvent]:
            yield FinsEvent(
                type=FinsEventType.PROGRESS,
                command=FinsCommandName.UPLOAD_FILING,
                payload=UploadFilingProgressPayload(
                    event_type=FinsProgressEventName.UPLOAD_STARTED,
                    ticker="AAPL",
                ),
            )
            yield FinsEvent(
                type=FinsEventType.RESULT,
                command=FinsCommandName.PROCESS,
                payload=result_data,
            )

        # upload_filing 命令走 INFO 分支（_should_log_fins_progress_as_info 返回 True）
        result = await _consume_fins_stream(_stream(), FinsCommandName.UPLOAD_FILING)
        assert result is result_data

    @pytest.mark.asyncio
    async def test_multiple_result_events_keeps_last(self) -> None:
        """多个 RESULT 事件时保留最后一个。"""
        first_result = ProcessResultData(
            pipeline="process",
            status="partial",
            ticker="AAPL",
        )
        final_result = ProcessResultData(
            pipeline="process",
            status="ok",
            ticker="AAPL",
        )

        async def _stream() -> AsyncIterator[FinsEvent]:
            yield FinsEvent(
                type=FinsEventType.RESULT,
                command=FinsCommandName.PROCESS,
                payload=first_result,
            )
            yield FinsEvent(
                type=FinsEventType.RESULT,
                command=FinsCommandName.PROCESS,
                payload=final_result,
            )

        result = await _consume_fins_stream(_stream(), FinsCommandName.PROCESS)
        assert result is final_result


# --------------------------------------------------------------------------- #
#  run_fins_command
# --------------------------------------------------------------------------- #


class TestRunFinsCommand:
    """run_fins_command 测试。"""

    def test_run_command_configures_loglevel_before_execution(self) -> None:
        """命令入口应先应用日志级别参数。"""

        args = argparse.Namespace(command="not_a_fins_command")
        setup_calls: list[argparse.Namespace] = []

        with (
            patch("dayu.cli.commands.fins.setup_loglevel", side_effect=lambda namespace: setup_calls.append(namespace)),
            patch("dayu.cli.commands.fins._build_fins_ops_service", return_value=MagicMock()),
        ):
            exit_code = run_fins_command(args)

        assert exit_code == 1
        assert setup_calls == [args]

    def test_stream_command_receives_sync_result_raises_runtime_error(self) -> None:
        """流式命令收到同步结果时返回退出码 1（RuntimeError 被捕获）。"""
        args = argparse.Namespace(
            command="download",
            ticker="AAPL",
            form_type=(),
            start_date=None,
            end_date=None,
            overwrite=False,
            rebuild=False,
            infer=False,
            ticker_aliases=(),
            log_level=None,
            debug=False,
            verbose=False,
            info=False,
            quiet=False,
        )

        mock_service = MagicMock()
        mock_result = FinsResult(
            command=FinsCommandName.DOWNLOAD,
            data=DownloadResultData(
                pipeline="download",
                status="ok",
                ticker="AAPL",
            ),
        )
        mock_service.submit.return_value = MagicMock(
            execution=mock_result,  # stream=True 但返回 FinsResult，触发 RuntimeError
        )

        with patch("dayu.cli.commands.fins._build_fins_ops_service", return_value=mock_service):
            exit_code = run_fins_command(args)

        assert exit_code == 1

    def test_non_stream_command_returns_zero(self) -> None:
        """非流式命令成功时返回退出码 0。"""
        args = argparse.Namespace(
            command="process_filing",
            ticker="AAPL",
            document_id="doc-1",
            overwrite=False,
            ci=False,
            log_level=None,
            debug=False,
            verbose=False,
            info=False,
            quiet=False,
        )

        result_data = ProcessSingleResultData(
            pipeline="process_filing",
            action="process",
            status="ok",
            ticker="AAPL",
            document_id="doc-1",
        )
        mock_result = FinsResult(
            command=FinsCommandName.PROCESS_FILING,
            data=result_data,
        )
        mock_service = MagicMock()
        mock_service.submit.return_value = MagicMock(execution=mock_result)

        with patch("dayu.cli.commands.fins._build_fins_ops_service", return_value=mock_service):
            exit_code = run_fins_command(args)

        assert exit_code == 0

    def test_non_stream_command_returns_async_raises_runtime_error(self) -> None:
        """非流式命令返回异步流时返回退出码 1。"""
        args = argparse.Namespace(
            command="process_filing",
            ticker="AAPL",
            document_id="doc-1",
            overwrite=False,
            ci=False,
            log_level=None,
            debug=False,
            verbose=False,
            info=False,
            quiet=False,
        )

        async def _fake_stream() -> AsyncIterator[FinsEvent]:
            if False:
                yield  # pragma: no cover

        mock_service = MagicMock()
        mock_service.submit.return_value = MagicMock(execution=_fake_stream())

        with patch("dayu.cli.commands.fins._build_fins_ops_service", return_value=mock_service):
            exit_code = run_fins_command(args)

        assert exit_code == 1

    def test_build_command_error_returns_one(self) -> None:
        """构建命令失败时返回退出码 1。"""
        args = argparse.Namespace(
            command="not_a_fins_command",
            log_level=None,
            debug=False,
            verbose=False,
            info=False,
            quiet=False,
        )

        with patch("dayu.cli.commands.fins._build_fins_ops_service", return_value=MagicMock()):
            exit_code = run_fins_command(args)

        assert exit_code == 1
