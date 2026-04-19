"""`dayu.cli` 覆盖率补充测试。"""

from __future__ import annotations

import json
import sys
import builtins
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, cast

import pytest

from dayu import cli as app_cli
from dayu.cli.interactive_state import (
    FileInteractiveStateStore,
    InteractiveSessionState,
    build_interactive_session_id,
)
from dayu.cli import main as app_cli_main
from dayu.cli import interactive_ui as app_interactive
from dayu.cli.arg_parsing import _create_parser
from dayu.cli.dependency_setup import setup_loglevel, setup_paths
from dayu.contracts.events import AppEvent, AppEventType
from dayu.execution.options import ExecutionOptions
from dayu.engine.events import EventType, StreamEvent
from dayu.log import Log, LogLevel
from dayu.services.contracts import (
    ChatPendingTurnView,
    ChatResumeRequest,
    ChatTurnRequest,
    ChatTurnSubmission,
    PromptRequest,
    PromptSubmission,
    SessionResolutionPolicy,
)


def _build_workspace(tmp_path: Path, *, ticker: str = "AAPL") -> Path:
    """构造最小可运行 workspace 目录结构。

    Args:
        tmp_path: pytest 提供的临时目录。
        ticker: 股票代码目录名。

    Returns:
        workspace 根目录路径。

    Raises:
        OSError: 创建目录或文件失败时抛出。
    """

    workspace = tmp_path / "workspace"
    filings_dir = workspace / "portfolio" / ticker / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    (filings_dir / "sample.txt").write_text("hello", encoding="utf-8")
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text("{}", encoding="utf-8")
    return workspace


def _build_log_level_args(**overrides: Any) -> Namespace:
    """构建 `setup_loglevel` 需要的参数对象。

    Args:
        **overrides: 字段覆盖项。

    Returns:
        参数对象。

    Raises:
        ValueError: 参数非法时抛出。
    """

    payload = {
        "log_level": None,
        "debug": False,
        "verbose": False,
        "info": False,
        "quiet": False,
    }
    payload.update(overrides)
    return Namespace(**payload)


@pytest.mark.unit
def test_main_non_interactive_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `main()` 在非交互模式下可正常返回 0。

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace = _build_workspace(tmp_path)
    # 需要存在 llm_models.json 才能覆盖 ConfigLoader 的完整加载路径。
    llm_models_path = workspace / "config" / "llm_models.json"
    llm_models_path.write_text(
        json.dumps(
            {
                "mimo-v2-pro-thinking-plan": {
                    "runner_type": "openai_compatible",
                    "temperature": 0.8,
                    "runtime_hints": {
                        "temperature_profiles": {
                            "interactive": {
                                "temperature": 0.8,
                            }
                        }
                    },
                },
                "mimo-v2-pro-thinking": {
                    "runner_type": "openai_compatible",
                    "temperature": 0.8,
                    "runtime_hints": {
                        "temperature_profiles": {
                            "interactive": {
                                "temperature": 0.8,
                            }
                        }
                    },
                },
                "deepseek-thinking": {
                    "runner_type": "openai_compatible",
                    "temperature": 0.8,
                    "runtime_hints": {
                        "temperature_profiles": {
                            "interactive": {
                                "temperature": 0.8,
                            }
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["prog", "interactive", "--base", str(workspace)])
    # 降低日志噪声，避免测试输出污染。
    monkeypatch.setattr(Log, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(Log, "error", lambda *args, **kwargs: None)
    # 阻止进入真实的交互式循环。
    monkeypatch.setattr("dayu.cli.commands.interactive.interactive", lambda *_args, **_kwargs: None)

    result = app_cli.run_main()

    assert result == 0


@pytest.mark.unit
def test_interactive_rejects_removed_ticker_argument(tmp_path: Path) -> None:
    """验证 interactive 子命令不再接受 `--ticker`。"""

    workspace = _build_workspace(tmp_path)

    with pytest.raises(SystemExit):
        _create_parser().parse_args(["interactive", "--base", str(workspace), "--ticker", "AAPL"])


@pytest.mark.unit
def test_interactive_accepts_new_session_flag(tmp_path: Path) -> None:
    """验证 interactive 子命令支持 `--new-session`。"""

    workspace = _build_workspace(tmp_path)

    parsed = _create_parser().parse_args(["interactive", "--base", str(workspace), "--new-session"])

    assert parsed.command == "interactive"
    assert parsed.new_session is True


@pytest.mark.unit
def test_interactive_state_store_round_trip_and_clear(tmp_path: Path) -> None:
    """验证 interactive 状态仓储支持读写与清理。"""

    store = FileInteractiveStateStore(tmp_path / ".dayu" / "interactive")
    assert store.load() is None

    state = InteractiveSessionState(interactive_key="interactive_key_1")
    store.save(state)

    loaded = store.load()
    assert loaded == state
    assert build_interactive_session_id(state.interactive_key).startswith("interactive_")

    store.clear()
    assert store.load() is None


@pytest.mark.unit
def test_root_parser_uses_python_module_prog() -> None:
    """验证顶层 CLI usage 使用 `python -m dayu.cli`，不暴露 `__main__.py`。"""

    parser = _create_parser()

    assert parser.prog == "python -m dayu.cli"
    assert "__main__.py" not in parser.format_usage()


@pytest.mark.unit
def test_root_parser_missing_command_prints_help_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """验证缺少子命令时会输出完整帮助和子命令摘要。"""

    parser = _create_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "python -m dayu.cli" in captured.err
    assert "__main__.py" not in captured.err
    assert "interactive" in captured.err
    assert "prompt" in captured.err
    assert "download" in captured.err
    assert "错误: 缺少子命令" in captured.err


@pytest.mark.unit
def test_setup_paths_error_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `setup_paths()` 的异常分支（目录不存在/目录类型错误）。

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(Log, "error", lambda *args, **kwargs: None)

    missing_args = Namespace(base=str(tmp_path / "missing"), ticker="AAPL")
    with pytest.raises(SystemExit):
        setup_paths(missing_args)

    workspace = tmp_path / "workspace"
    filings_file = workspace / "portfolio" / "AAPL" / "filings"
    filings_file.parent.mkdir(parents=True, exist_ok=True)
    filings_file.write_text("not-a-directory", encoding="utf-8")
    bad_args = Namespace(base=str(workspace), ticker="AAPL")
    with pytest.raises(SystemExit):
        setup_paths(bad_args)


@pytest.mark.unit
def test_setup_loglevel_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 `setup_loglevel()` 的优先级分支。

    Args:
        monkeypatch: pytest monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    recorded: list[Any] = []
    monkeypatch.setattr(Log, "set_level", lambda level: recorded.append(level))

    setup_loglevel(_build_log_level_args(log_level="debug"))
    setup_loglevel(_build_log_level_args(debug=True))
    setup_loglevel(_build_log_level_args(verbose=True))
    setup_loglevel(_build_log_level_args(info=True))
    setup_loglevel(_build_log_level_args(quiet=True))
    setup_loglevel(_build_log_level_args())

    assert len(recorded) == 6
    assert recorded[0] == LogLevel.DEBUG
    assert recorded[1] == LogLevel.DEBUG
    assert recorded[2] == LogLevel.VERBOSE
    assert recorded[3] == LogLevel.INFO
    assert recorded[4] == LogLevel.ERROR
    assert recorded[5] == LogLevel.INFO


def _build_interactive_inputs(tmp_path: Path) -> tuple[Any, Any, Any]:
    """构建 `interactive()` 调用所需输入对象。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        `(workspace_config, run_config, model_config)` 三元组。

    Raises:
        OSError: 路径创建失败时抛出。
    """

    config_loader = SimpleNamespace(
        load_llm_model=lambda _: {"runner_type": "openai_compatible"},
    )
    prompt_asset_store = SimpleNamespace(
        load_scene_manifest=lambda scene_name: {"scene": scene_name, "fragments": []},
        load_fragment_template=lambda path, required=True: "template",
    )
    workspace_config = SimpleNamespace(
        workspace_dir=tmp_path,
        config_loader=config_loader,
        prompt_asset_store=prompt_asset_store,
        ticker=None,
    )
    run_config = SimpleNamespace(
        agent_running_config=object(),
        runner_running_config=object(),
        doc_tool_limits=object(),
        fins_tool_limits=object(),
        web_tools_config=SimpleNamespace(
            enabled=True,
            provider="auto",
            request_timeout_seconds=12.0,
            max_search_results=20,
            fetch_truncate_chars=80000,
        ),
        tool_trace_config=SimpleNamespace(enabled=False, output_dir=str(tmp_path)),
    )
    model_config = SimpleNamespace(model_name="deepseek-chat")
    return workspace_config, run_config, model_config


def _build_event_session(*event_sequences: list[StreamEvent], fail_with: Exception | None = None):
    """构建测试用事件会话类。

    Args:
        *event_sequences: 每次调用对应的事件序列。
        fail_with: 若提供，则每次调用都直接抛出该异常。

    Returns:
        供 monkeypatch 使用的会话类。

    Raises:
        无。
    """

    call_log: list[str] = []
    request_log: list[Any] = []

    class _FakeSession:
        """测试用会话。"""

        call_log: list[str]
        request_log: list[Any]

        @classmethod
        def create(cls, **_kwargs: Any) -> _FakeSession:
            """返回会话实例。"""

            return cls()

        async def _yield_events(self, prompt: str) -> AsyncIterator[StreamEvent]:
            """按调用顺序返回预置事件流。"""

            call_log.append(prompt)
            if fail_with is not None:
                raise fail_with
            index = len(call_log) - 1
            for event in event_sequences[index]:
                yield event

        async def stream(self, request: PromptRequest) -> AsyncIterator[StreamEvent]:
            """返回单轮 prompt 预置事件流。"""

            request_log.append(request)
            async for event in self._yield_events(request.user_text):
                yield event

        async def stream_turn(self, request: ChatTurnRequest) -> AsyncIterator[StreamEvent]:
            """返回聊天轮次预置事件流。"""

            request_log.append(request)
            async for event in self._yield_events(request.user_text):
                yield event

        async def submit(self, request: PromptRequest) -> PromptSubmission:
            """按 PromptServiceProtocol 返回提交结果。"""

            request_log.append(request)
            return PromptSubmission(
                session_id=request.session_id or "test-session",
                event_stream=cast(AsyncIterator[AppEvent], self._yield_events(request.user_text)),
            )

        async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
            """按 ChatServiceProtocol 返回聊天提交结果。"""

            request_log.append(request)
            return ChatTurnSubmission(
                session_id=request.session_id or "test-session",
                event_stream=cast(AsyncIterator[AppEvent], self._yield_events(request.user_text)),
            )

        async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
            """按 ChatServiceProtocol 返回恢复提交结果。"""

            return ChatTurnSubmission(
                session_id=request.session_id,
                event_stream=cast(AsyncIterator[AppEvent], self._yield_events(request.pending_turn_id)),
            )

        def list_resumable_pending_turns(
            self,
            *,
            session_id: str | None = None,
            scene_name: str | None = None,
        ) -> list[ChatPendingTurnView]:
            """返回空的 pending turn 列表。"""

            del session_id, scene_name
            return []

    _FakeSession.call_log = call_log
    _FakeSession.request_log = request_log
    return _FakeSession


@pytest.mark.unit
def test_run_chat_turn_stream_uses_submit_turn_protocol_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 interactive chat 只消费 `submit_turn()` 稳定协议。"""

    class _Session:
        """只暴露稳定聊天协议的测试桩。"""

        async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
            """返回稳定聊天提交结果。"""

            async def _stream() -> AsyncIterator[AppEvent]:
                yield AppEvent(
                    type=AppEventType.FINAL_ANSWER,
                    payload={"content": "ok", "degraded": False},
                    meta={},
                )

            return ChatTurnSubmission(
                session_id=request.session_id or "resolved-session",
                event_stream=_stream(),
            )

        def list_resumable_pending_turns(
            self,
            *,
            session_id: str | None = None,
            scene_name: str | None = None,
        ) -> list[ChatPendingTurnView]:
            """返回空 pending turn 列表。"""

            del session_id, scene_name
            return []

        async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
            """该测试不应走恢复路径。"""

            raise AssertionError(f"unexpected resume_pending_turn call: {request}")

        async def stream_turn(self, request: ChatTurnRequest) -> AsyncIterator[StreamEvent]:
            """旧兼容接口不应被 interactive 使用。"""

            raise AssertionError(f"unexpected stream_turn call: {request}")

    monkeypatch.setattr(
        app_interactive,
        "_SpinnerController",
        lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None),
    )

    final_content, resolved_session_id = app_interactive._run_chat_turn_stream(
        cast(Any, _Session()),
        "问题一",
        session_id="chat-session",
        show_thinking=False,
    )

    assert final_content == "ok"
    assert resolved_session_id == "chat-session"


@pytest.mark.unit
def test_run_prompt_stream_uses_submit_protocol_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 interactive prompt 只消费 `submit()` 稳定协议。"""

    class _Session:
        """只暴露稳定 Prompt 协议的测试桩。"""

        async def submit(self, request: PromptRequest) -> PromptSubmission:
            """返回稳定 Prompt 提交结果。"""

            async def _stream() -> AsyncIterator[AppEvent]:
                yield AppEvent(
                    type=AppEventType.FINAL_ANSWER,
                    payload={"content": "prompt-ok", "degraded": False},
                    meta={},
                )

            return PromptSubmission(
                session_id=request.session_id or "prompt-session",
                event_stream=_stream(),
            )

        async def stream(self, request: PromptRequest) -> AsyncIterator[StreamEvent]:
            """旧兼容接口不应被 interactive 使用。"""

            raise AssertionError(f"unexpected stream call: {request}")

    monkeypatch.setattr(
        app_interactive,
        "_SpinnerController",
        lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None),
    )

    final_content = app_interactive._run_prompt_stream(
        cast(Any, _Session()),
        "问题一",
        ticker="AAPL",
        show_thinking=False,
    )

    assert final_content == "prompt-ok"


@pytest.mark.unit
def test_interactive_returns_when_agent_creation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证交互模式在 Agent 创建失败时会提前返回。

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    agent_session = _build_event_session(fail_with=RuntimeError("无法创建 Agent")).create()
    errors: list[str] = []
    monkeypatch.setattr(app_interactive.Log, "error", lambda message, module=None: errors.append(message))
    monkeypatch.setattr(app_interactive.sys, "stdin", SimpleNamespace(isatty=lambda: True))

    class DummyKeyBindings:
        def add(self, *_args: Any, **_kwargs: Any):
            def _decorator(func):
                return func
            return _decorator

    class DummySession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def prompt(self, _text: str) -> str | None:
            return "问题一"

    prompts = iter(["问题一", None, None])
    DummySession.prompt = lambda self, _text: next(prompts)  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "prompt_toolkit", SimpleNamespace(PromptSession=DummySession))
    monkeypatch.setitem(sys.modules, "prompt_toolkit.key_binding", SimpleNamespace(KeyBindings=DummyKeyBindings))

    app_interactive.interactive(agent_session, session_id="test-session")

    assert any("无法创建 Agent" in message for message in errors)


@pytest.mark.unit
def test_interactive_returns_when_not_tty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证非 TTY 环境下交互模式会报错并返回。

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    agent_session = _build_event_session(
        [StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "ok", "degraded": False}, metadata={})]
    ).create()
    errors: list[str] = []
    monkeypatch.setattr(app_interactive.Log, "error", lambda message, module=None: errors.append(message))
    monkeypatch.setattr(app_interactive.sys, "stdin", SimpleNamespace(isatty=lambda: False))

    app_interactive.interactive(agent_session, session_id="test-session")

    assert any("TTY" in message for message in errors)


@pytest.mark.unit
def test_interactive_returns_when_prompt_toolkit_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证缺少 `prompt_toolkit` 时交互模式会报错并返回。

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    agent_session = _build_event_session(
        [StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "ok", "degraded": False}, metadata={})]
    ).create()
    errors: list[str] = []
    monkeypatch.setattr(app_interactive.Log, "error", lambda message, module=None: errors.append(message))
    monkeypatch.setattr(app_interactive.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setitem(sys.modules, "prompt_toolkit", None)
    monkeypatch.setitem(sys.modules, "prompt_toolkit.key_binding", None)

    app_interactive.interactive(agent_session, session_id="test-session")

    assert any("prompt_toolkit 未安装" in message for message in errors)


@pytest.mark.unit
def test_interactive_streams_content_delta_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证交互模式会消费事件流并输出 content_delta。

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    outputs: list[str] = []
    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.CONTENT_DELTA, data="hello", metadata={}),
            StreamEvent(type=EventType.CONTENT_DELTA, data=" world", metadata={}),
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "hello world", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()
    prompts = iter(["问题一", None, None])

    monkeypatch.setattr(app_interactive, "_SpinnerController", lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None))
    monkeypatch.setattr(app_interactive.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(app_interactive.Log, "error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builtins, "print", lambda message="", *args, **kwargs: outputs.append(str(message)))

    class DummyKeyBindings:
        def add(self, *_args: Any, **_kwargs: Any):
            def _decorator(func):
                return func
            return _decorator

    class DummySession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def prompt(self, _text: str) -> str | None:
            return next(prompts)

    monkeypatch.setitem(sys.modules, "prompt_toolkit", SimpleNamespace(PromptSession=DummySession))
    monkeypatch.setitem(sys.modules, "prompt_toolkit.key_binding", SimpleNamespace(KeyBindings=DummyKeyBindings))

    app_interactive.interactive(agent_session, session_id="test-session")

    assert "hello" in outputs
    assert " world" in outputs


@pytest.mark.unit
def test_interactive_shows_waiting_spinner_when_thinking_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 interactive 在非 thinking 模式下会显示等待 spinner。"""

    spinner_events: list[tuple[str, str] | str] = []
    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "done", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()
    prompts = iter(["问题一", None, None])

    class SpinnerRecorder:
        """记录 spinner 生命周期。"""

        def __init__(self, *, label: str = "Waiting") -> None:
            """初始化记录器。

            Args:
                label: spinner 标签。

            Returns:
                无。

            Raises:
                无。
            """

            spinner_events.append(("init", label))

        def start(self) -> None:
            """记录启动事件。"""

            spinner_events.append("start")

        def stop(self) -> None:
            """记录停止事件。"""

            spinner_events.append("stop")

    monkeypatch.setattr(app_interactive, "_SpinnerController", SpinnerRecorder)
    monkeypatch.setattr(app_interactive.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(app_interactive.Log, "error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    class DummyKeyBindings:
        """用于替代 prompt_toolkit 的 KeyBindings。"""

        def add(self, *_args: Any, **_kwargs: Any):
            """返回原样装饰器。"""

            def _decorator(func):
                return func

            return _decorator

    class DummySession:
        """用于替代 prompt_toolkit 的 PromptSession。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """初始化会话。"""

        def prompt(self, _text: str) -> str | None:
            """返回预置输入序列。"""

            return next(prompts)

    monkeypatch.setitem(sys.modules, "prompt_toolkit", SimpleNamespace(PromptSession=DummySession))
    monkeypatch.setitem(sys.modules, "prompt_toolkit.key_binding", SimpleNamespace(KeyBindings=DummyKeyBindings))

    app_interactive.interactive(agent_session, session_id="test-session", show_thinking=False)

    assert spinner_events == [("init", "Waiting"), "start", "stop"]


@pytest.mark.unit
def test_interactive_skips_waiting_spinner_when_thinking_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 interactive 在 thinking 模式下不会显示等待 spinner。"""

    spinner_events: list[tuple[str, str] | str] = []
    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "done", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()
    prompts = iter(["问题一", None, None])

    class SpinnerRecorder:
        """记录 spinner 生命周期。"""

        def __init__(self, *, label: str = "Waiting") -> None:
            """初始化记录器。

            Args:
                label: spinner 标签。

            Returns:
                无。

            Raises:
                无。
            """

            spinner_events.append(("init", label))

        def start(self) -> None:
            """记录启动事件。"""

            spinner_events.append("start")

        def stop(self) -> None:
            """记录停止事件。"""

            spinner_events.append("stop")

    monkeypatch.setattr(app_interactive, "_SpinnerController", SpinnerRecorder)
    monkeypatch.setattr(app_interactive.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(app_interactive.Log, "error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    class DummyKeyBindings:
        """用于替代 prompt_toolkit 的 KeyBindings。"""

        def add(self, *_args: Any, **_kwargs: Any):
            """返回原样装饰器。"""

            def _decorator(func):
                return func

            return _decorator

    class DummySession:
        """用于替代 prompt_toolkit 的 PromptSession。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """初始化会话。"""

        def prompt(self, _text: str) -> str | None:
            """返回预置输入序列。"""

            return next(prompts)

    monkeypatch.setitem(sys.modules, "prompt_toolkit", SimpleNamespace(PromptSession=DummySession))
    monkeypatch.setitem(sys.modules, "prompt_toolkit.key_binding", SimpleNamespace(KeyBindings=DummyKeyBindings))

    app_interactive.interactive(agent_session, session_id="test-session", show_thinking=True)

    assert spinner_events == []


@pytest.mark.unit
def test_prompt_hides_reasoning_delta_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证单次 prompt 模式默认不输出 reasoning 增量。"""

    stderr_outputs: list[str] = []
    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.REASONING_DELTA, data="step-1", metadata={}),
            StreamEvent(type=EventType.REASONING_DELTA, data=" step-2", metadata={}),
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "done", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()

    monkeypatch.setattr(app_interactive, "_SpinnerController", lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None))
    monkeypatch.setattr(builtins, "print", lambda message="", *args, **kwargs: stderr_outputs.append(str(message)) if kwargs.get("file") is sys.stderr else None)

    result = app_interactive.prompt(agent_session, "测试问题")

    assert result == 0
    assert stderr_outputs == []


@pytest.mark.unit
def test_prompt_shows_waiting_spinner_when_thinking_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 prompt 在非 thinking 模式下会显示等待 spinner。"""

    spinner_events: list[tuple[str, str] | str] = []
    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "done", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()

    class SpinnerRecorder:
        """记录 spinner 生命周期。"""

        def __init__(self, *, label: str = "Waiting") -> None:
            """初始化记录器。

            Args:
                label: spinner 标签。

            Returns:
                无。

            Raises:
                无。
            """

            spinner_events.append(("init", label))

        def start(self) -> None:
            """记录启动事件。

            Args:
                无。

            Returns:
                无。

            Raises:
                无。
            """

            spinner_events.append("start")

        def stop(self) -> None:
            """记录停止事件。

            Args:
                无。

            Returns:
                无。

            Raises:
                无。
            """

            spinner_events.append("stop")

    monkeypatch.setattr(app_interactive, "_SpinnerController", SpinnerRecorder)
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    result = app_interactive.prompt(agent_session, "测试问题", show_thinking=False)

    assert result == 0
    assert spinner_events == [("init", "Waiting"), "start", "stop"]


@pytest.mark.unit
def test_prompt_skips_waiting_spinner_when_thinking_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 prompt 在 thinking 模式下不会显示等待 spinner。"""

    spinner_events: list[tuple[str, str] | str] = []
    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "done", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()

    class SpinnerRecorder:
        """记录 spinner 生命周期。"""

        def __init__(self, *, label: str = "Waiting") -> None:
            """初始化记录器。

            Args:
                label: spinner 标签。

            Returns:
                无。

            Raises:
                无。
            """

            spinner_events.append(("init", label))

        def start(self) -> None:
            """记录启动事件。

            Args:
                无。

            Returns:
                无。

            Raises:
                无。
            """

            spinner_events.append("start")

        def stop(self) -> None:
            """记录停止事件。

            Args:
                无。

            Returns:
                无。

            Raises:
                无。
            """

            spinner_events.append("stop")

    monkeypatch.setattr(app_interactive, "_SpinnerController", SpinnerRecorder)
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    result = app_interactive.prompt(agent_session, "测试问题", show_thinking=True)

    assert result == 0
    assert spinner_events == []


@pytest.mark.unit
def test_prompt_passes_execution_options_into_prompt_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证单次 prompt 会把请求级执行选项透传到 `PromptRequest`。"""

    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "done", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()
    execution_options = ExecutionOptions(model_name="deepseek-thinking", max_iterations=9)

    monkeypatch.setattr(
        app_interactive,
        "_SpinnerController",
        lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    result = app_interactive.prompt(
        agent_session,
        "测试问题",
        execution_options=execution_options,
    )

    assert result == 0
    assert len(fake_session_cls.request_log) == 1
    assert fake_session_cls.request_log[0].execution_options == execution_options


@pytest.mark.unit
def test_interactive_passes_execution_options_into_chat_turn_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证交互模式会把请求级执行选项透传到 `ChatTurnRequest`。"""

    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "done", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()
    execution_options = ExecutionOptions(model_name="deepseek-thinking", max_iterations=9)
    prompts = iter(["问题一", None, None])

    monkeypatch.setattr(
        app_interactive,
        "_SpinnerController",
        lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(app_interactive.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(app_interactive.Log, "error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    class DummyKeyBindings:
        def add(self, *_args: Any, **_kwargs: Any):
            def _decorator(func):
                return func
            return _decorator

    class DummySession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def prompt(self, _text: str) -> str | None:
            return next(prompts)

    monkeypatch.setitem(sys.modules, "prompt_toolkit", SimpleNamespace(PromptSession=DummySession))
    monkeypatch.setitem(sys.modules, "prompt_toolkit.key_binding", SimpleNamespace(KeyBindings=DummyKeyBindings))

    app_interactive.interactive(
        agent_session,
        session_id="test-session",
        execution_options=execution_options,
    )

    assert len(fake_session_cls.request_log) == 1
    assert fake_session_cls.request_log[0].execution_options == execution_options
    assert fake_session_cls.request_log[0].scene_name == "interactive"
    assert fake_session_cls.request_log[0].session_resolution_policy == SessionResolutionPolicy.ENSURE_DETERMINISTIC


@pytest.mark.unit
def test_prompt_renders_reasoning_delta_to_stderr_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证显式开启 thinking 回显后会输出 reasoning 增量。"""

    stderr_outputs: list[str] = []
    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.REASONING_DELTA, data="step-1", metadata={}),
            StreamEvent(type=EventType.REASONING_DELTA, data=" step-2", metadata={}),
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "done", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()

    monkeypatch.setattr(app_interactive, "_SpinnerController", lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None))
    monkeypatch.setattr(
        builtins,
        "print",
        lambda message="", *args, **kwargs: stderr_outputs.append(str(message)) if kwargs.get("file") is sys.stderr else None,
    )

    result = app_interactive.prompt(agent_session, "测试问题", show_thinking=True)

    assert result == 0
    assert "step-1" in stderr_outputs
    assert " step-2" in stderr_outputs


@pytest.mark.unit
def test_prompt_inserts_newline_between_reasoning_and_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 reasoning 输出与正文输出之间会补换行。"""

    rendered: list[tuple[str, str]] = []
    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.REASONING_DELTA, data="step-1", metadata={}),
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "done", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()

    monkeypatch.setattr(
        app_interactive,
        "_SpinnerController",
        lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None),
    )

    def _capture_print(message="", *args: Any, **kwargs: Any) -> None:
        target = "stderr" if kwargs.get("file") is sys.stderr else "stdout"
        rendered.append((target, str(message)))

    monkeypatch.setattr(builtins, "print", _capture_print)

    result = app_interactive.prompt(agent_session, "测试问题", show_thinking=True)

    assert result == 0
    assert rendered[:5] == [
        ("stderr", "Thinking..."),
        ("stderr", "step-1"),
        ("stderr", ""),   # reasoning 结尾换行
        ("stdout", ""),   # reasoning→content 边界空行
        ("stdout", "done"),
    ]


@pytest.mark.unit
def test_prompt_no_extra_newlines_between_multiple_reasoning_deltas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证多个连续 reasoning_delta 之间不会插入额外换行，只在 reasoning→content 边界处出现一次换行。

    正确顺序：
        ("stderr", "chunk1") -> ("stderr", " chunk2") -> ("stderr", " chunk3")
        -> ("stderr", "")    # 边界换行
        -> ("stdout", "done")
    错误顺序（有 bug 时）：
        ("stderr", "chunk1") -> ("stderr", "") -> ("stderr", " chunk2") -> ...

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    rendered: list[tuple[str, str]] = []
    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.REASONING_DELTA, data="chunk1", metadata={}),
            StreamEvent(type=EventType.REASONING_DELTA, data=" chunk2", metadata={}),
            StreamEvent(type=EventType.REASONING_DELTA, data=" chunk3", metadata={}),
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "done", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()

    monkeypatch.setattr(
        app_interactive,
        "_SpinnerController",
        lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None),
    )

    def _capture_print(message="", *args: Any, **kwargs: Any) -> None:
        target = "stderr" if kwargs.get("file") is sys.stderr else "stdout"
        rendered.append((target, str(message)))

    monkeypatch.setattr(builtins, "print", _capture_print)

    result = app_interactive.prompt(agent_session, "测试问题", show_thinking=True)

    assert result == 0
    # 首条 reasoning_delta 前有 Thinking...，三个 reasoning_delta 连续输出，中间无空行
    assert rendered[:7] == [
        ("stderr", "Thinking..."),
        ("stderr", "chunk1"),
        ("stderr", " chunk2"),
        ("stderr", " chunk3"),
        ("stderr", ""),   # reasoning 结尾换行
        ("stdout", ""),   # reasoning→content 边界空行
        ("stdout", "done"),
    ]


@pytest.mark.unit
def test_interactive_rebuilds_agent_every_turn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证交互模式会按输入轮次依次消费同一会话服务。

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    responses = iter(["问题一", "问题二", None, None])
    fake_session_cls = _build_event_session(
        [StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "ok", "degraded": False}, metadata={})],
        [StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "ok", "degraded": False}, metadata={})],
    )
    agent_session = fake_session_cls.create()

    monkeypatch.setattr(app_interactive.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(app_interactive, "_SpinnerController", lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None))
    monkeypatch.setattr(app_interactive.Log, "error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    class DummyKeyBindings:
        """用于替代 prompt_toolkit 的 KeyBindings。"""

        def add(self, *_args: Any, **_kwargs: Any):
            """返回原样装饰器。

            Args:
                *_args: 键位参数。
                **_kwargs: 选项参数。

            Returns:
                原样装饰器。

            Raises:
                无。
            """

            def _decorator(func):
                return func

            return _decorator

    class DummySession:
        """用于替代 prompt_toolkit 的 PromptSession。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """初始化会话。

            Args:
                *args: 位置参数。
                **kwargs: 关键字参数。

            Returns:
                无。

            Raises:
                无。
            """

        def prompt(self, _text: str) -> str | None:
            """返回预置输入序列。

            Args:
                _text: 提示文本。

            Returns:
                下一条输入；读尽后返回 ``None``。

            Raises:
                StopIteration: 输入耗尽时抛出。
            """

            return next(responses)

    monkeypatch.setitem(sys.modules, "prompt_toolkit", SimpleNamespace(PromptSession=DummySession))
    monkeypatch.setitem(sys.modules, "prompt_toolkit.key_binding", SimpleNamespace(KeyBindings=DummyKeyBindings))

    app_interactive.interactive(agent_session, session_id="test-session")

    assert fake_session_cls.call_log == ["问题一", "问题二"]


@pytest.mark.unit
def test_interactive_reuses_provided_session_id_within_each_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 interactive 会在每次启动内复用调用方提供的 session_id。

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    prompts_first = iter(["问题一", "问题二", None, None])
    prompts_second = iter(["问题三", None, None])
    fake_session_cls = _build_event_session(
        [StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "ok-1", "degraded": False}, metadata={})],
        [StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "ok-2", "degraded": False}, metadata={})],
        [StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "ok-3", "degraded": False}, metadata={})],
    )
    first_agent_session = fake_session_cls.create()
    second_agent_session = fake_session_cls.create()

    monkeypatch.setattr(app_interactive.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        app_interactive,
        "_SpinnerController",
        lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(app_interactive.Log, "error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    class DummyKeyBindings:
        """用于替代 prompt_toolkit 的 KeyBindings。"""

        def add(self, *_args: Any, **_kwargs: Any):
            """返回原样装饰器。

            Args:
                *_args: 键位参数。
                **_kwargs: 选项参数。

            Returns:
                原样装饰器。

            Raises:
                无。
            """

            def _decorator(func):
                return func

            return _decorator

    class DummySession:
        """用于替代 prompt_toolkit 的 PromptSession。"""

        prompts: Any = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """初始化会话。

            Args:
                *args: 位置参数。
                **kwargs: 关键字参数。

            Returns:
                无。

            Raises:
                无。
            """

        def prompt(self, _text: str) -> str | None:
            """返回预置输入序列。

            Args:
                _text: 提示文本。

            Returns:
                下一条输入；读尽后返回 ``None``。

            Raises:
                StopIteration: 输入耗尽时抛出。
            """

            return next(self.prompts)

    monkeypatch.setitem(sys.modules, "prompt_toolkit", SimpleNamespace(PromptSession=DummySession))
    monkeypatch.setitem(sys.modules, "prompt_toolkit.key_binding", SimpleNamespace(KeyBindings=DummyKeyBindings))

    DummySession.prompts = prompts_first
    app_interactive.interactive(first_agent_session, session_id="interactive_sess_a")

    DummySession.prompts = prompts_second
    app_interactive.interactive(second_agent_session, session_id="interactive_sess_b")

    request_log = fake_session_cls.request_log
    assert [request.user_text for request in request_log] == ["问题一", "问题二", "问题三"]
    assert [request.session_id for request in request_log] == ["interactive_sess_a", "interactive_sess_a", "interactive_sess_b"]
    assert [request.session_resolution_policy for request in request_log] == [
        SessionResolutionPolicy.ENSURE_DETERMINISTIC,
        SessionResolutionPolicy.ENSURE_DETERMINISTIC,
        SessionResolutionPolicy.ENSURE_DETERMINISTIC,
    ]


@pytest.mark.unit
def test_prompt_returns_when_agent_creation_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证单次 prompt 模式在 Agent 创建失败时返回错误码。

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    agent_session = _build_event_session(fail_with=RuntimeError("无法创建 Agent")).create()
    errors: list[str] = []
    monkeypatch.setattr(app_interactive.Log, "error", lambda message, module=None: errors.append(message))

    result = app_interactive.prompt(agent_session, "测试问题")

    assert result == 2
    assert any("无法创建 Agent" in message for message in errors)


@pytest.mark.unit
def test_prompt_prints_single_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证单次 prompt 模式会执行一次 Agent 并打印结果。

    Args:
        monkeypatch: pytest monkeypatch 工具。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    outputs: list[str] = []
    fake_session_cls = _build_event_session(
        [
            StreamEvent(type=EventType.CONTENT_DELTA, data="single", metadata={}),
            StreamEvent(type=EventType.CONTENT_DELTA, data="-response", metadata={}),
            StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "single-response", "degraded": False}, metadata={}),
        ]
    )
    agent_session = fake_session_cls.create()

    monkeypatch.setattr(app_interactive, "_SpinnerController", lambda label="Waiting": SimpleNamespace(start=lambda: None, stop=lambda: None))
    monkeypatch.setattr(builtins, "print", lambda message="", *args, **kwargs: outputs.append(str(message)))

    result = app_interactive.prompt(agent_session, "测试问题")

    assert result == 0
    assert "single" in outputs
    assert "-response" in outputs
