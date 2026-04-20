"""perf_utils 模块单元测试。"""

from __future__ import annotations

import pytest

from dayu.contracts.env_keys import FINS_PROCESSOR_PROFILE_ENV
from dayu.engine.processors.perf_utils import (
    ProcessorStageProfiler,
    is_processor_profile_enabled,
)


# ---------------------------------------------------------------------------
# is_processor_profile_enabled
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsProcessorProfileEnabled:
    """is_processor_profile_enabled 函数测试组。"""

    def test_returns_false_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未设置环境变量时返回 False。"""
        monkeypatch.delenv(FINS_PROCESSOR_PROFILE_ENV, raising=False)
        assert is_processor_profile_enabled() is False

    def test_returns_false_for_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """空字符串时返回 False。"""
        monkeypatch.setenv(FINS_PROCESSOR_PROFILE_ENV, "")
        assert is_processor_profile_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_returns_true_for_enabled_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """标准启用值返回 True。"""
        monkeypatch.setenv(FINS_PROCESSOR_PROFILE_ENV, value)
        assert is_processor_profile_enabled() is True

    @pytest.mark.parametrize("value", ["TRUE", "YES", "ON", "True", "Yes"])
    def test_returns_true_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """大写或混合大小写也应返回 True。"""
        monkeypatch.setenv(FINS_PROCESSOR_PROFILE_ENV, value)
        assert is_processor_profile_enabled() is True

    def test_returns_true_with_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """前后空白字符应被剥离后识别。"""
        monkeypatch.setenv(FINS_PROCESSOR_PROFILE_ENV, "  1  ")
        assert is_processor_profile_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "off", "no", "disabled", "2"])
    def test_returns_false_for_non_enabled_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """非启用值均返回 False。"""
        monkeypatch.setenv(FINS_PROCESSOR_PROFILE_ENV, value)
        assert is_processor_profile_enabled() is False


# ---------------------------------------------------------------------------
# ProcessorStageProfiler.stage  (disabled)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessorStageProfilerDisabled:
    """enabled=False 时 stage 上下文管理器不计时。"""

    def test_stage_disabled_yields_without_recording(self) -> None:
        """关闭时 stage 正常 yield，records 保持为空。"""
        profiler = ProcessorStageProfiler(component="TestComp", enabled=False)
        with profiler.stage("fetch"):
            pass
        assert profiler.records == {}

    def test_stage_disabled_does_not_raise_on_exception(self) -> None:
        """关闭时 stage 内部抛出异常，异常正常传播（不被吃掉）。"""
        profiler = ProcessorStageProfiler(component="TestComp", enabled=False)
        with pytest.raises(ValueError):
            with profiler.stage("fetch"):
                raise ValueError("boom")
        assert profiler.records == {}


# ---------------------------------------------------------------------------
# ProcessorStageProfiler.stage  (enabled)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessorStageProfilerEnabled:
    """enabled=True 时 stage 上下文管理器正确计时并累加。"""

    def test_stage_records_elapsed_positive(self) -> None:
        """启用时 records 中记录正值耗时。"""
        profiler = ProcessorStageProfiler(component="TestComp", enabled=True)
        with profiler.stage("parse"):
            pass
        assert "parse" in profiler.records
        assert profiler.records["parse"] >= 0.0

    def test_stage_accumulates_multiple_calls(self) -> None:
        """同名 stage 多次调用时累加耗时。"""
        profiler = ProcessorStageProfiler(component="TestComp", enabled=True)
        with profiler.stage("parse"):
            pass
        first_value = profiler.records["parse"]
        with profiler.stage("parse"):
            pass
        assert profiler.records["parse"] >= first_value

    def test_stage_records_multiple_stages(self) -> None:
        """不同 stage 名称各自独立记录。"""
        profiler = ProcessorStageProfiler(component="TestComp", enabled=True)
        with profiler.stage("alpha"):
            pass
        with profiler.stage("beta"):
            pass
        assert "alpha" in profiler.records
        assert "beta" in profiler.records

    def test_stage_records_on_exception(self) -> None:
        """stage 内部抛出异常时，耗时仍被记录，异常正常传播。"""
        profiler = ProcessorStageProfiler(component="TestComp", enabled=True)
        with pytest.raises(RuntimeError):
            with profiler.stage("crash"):
                raise RuntimeError("fail")
        assert "crash" in profiler.records
        assert profiler.records["crash"] >= 0.0


# ---------------------------------------------------------------------------
# ProcessorStageProfiler.log_summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessorStageProfilerLogSummary:
    """log_summary 行为测试组。"""

    def test_log_summary_disabled_does_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """enabled=False 时 log_summary 不调用 Log.info。"""
        calls: list[str] = []
        monkeypatch.setattr(
            "dayu.engine.processors.perf_utils.Log",
            type("FakeLog", (), {"info": staticmethod(lambda msg, **kw: calls.append(msg))})(),
        )
        profiler = ProcessorStageProfiler(component="C", enabled=False, records={"a": 1.0})
        profiler.log_summary()
        assert calls == []

    def test_log_summary_enabled_no_records_does_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """enabled=True 但 records 为空时 log_summary 不调用 Log.info。"""
        calls: list[str] = []
        monkeypatch.setattr(
            "dayu.engine.processors.perf_utils.Log",
            type("FakeLog", (), {"info": staticmethod(lambda msg, **kw: calls.append(msg))})(),
        )
        profiler = ProcessorStageProfiler(component="C", enabled=True)
        profiler.log_summary()
        assert calls == []

    def test_log_summary_outputs_sorted_stages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """启用时按阶段名称字母序输出日志。"""
        calls: list[str] = []
        monkeypatch.setattr(
            "dayu.engine.processors.perf_utils.Log",
            type("FakeLog", (), {"info": staticmethod(lambda msg, **kw: calls.append(msg))})(),
        )
        profiler = ProcessorStageProfiler(
            component="MyComp",
            enabled=True,
            records={"beta": 20.0, "alpha": 10.0},
        )
        profiler.log_summary()
        assert len(calls) == 1
        msg = calls[0]
        assert "MyComp" in msg
        assert "alpha=" in msg
        assert "beta=" in msg
        # alpha 应排在 beta 之前
        assert msg.index("alpha") < msg.index("beta")

    def test_log_summary_includes_extra_suffix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """extra 参数不为空时日志末尾包含 extra 内容。"""
        calls: list[str] = []
        monkeypatch.setattr(
            "dayu.engine.processors.perf_utils.Log",
            type("FakeLog", (), {"info": staticmethod(lambda msg, **kw: calls.append(msg))})(),
        )
        profiler = ProcessorStageProfiler(
            component="C", enabled=True, records={"stage1": 5.0}
        )
        profiler.log_summary(extra="ticker=AAPL")
        assert "ticker=AAPL" in calls[0]

    def test_log_summary_no_extra_suffix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """extra 为空时日志不包含多余的 '|' 分隔符。"""
        calls: list[str] = []
        monkeypatch.setattr(
            "dayu.engine.processors.perf_utils.Log",
            type("FakeLog", (), {"info": staticmethod(lambda msg, **kw: calls.append(msg))})(),
        )
        profiler = ProcessorStageProfiler(
            component="C", enabled=True, records={"stage1": 5.0}
        )
        profiler.log_summary()
        assert " | " not in calls[0]

    def test_log_summary_format_ms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """日志中耗时格式应为 name=x.xxms。"""
        calls: list[str] = []
        monkeypatch.setattr(
            "dayu.engine.processors.perf_utils.Log",
            type("FakeLog", (), {"info": staticmethod(lambda msg, **kw: calls.append(msg))})(),
        )
        profiler = ProcessorStageProfiler(
            component="C", enabled=True, records={"fetch": 123.456}
        )
        profiler.log_summary()
        assert "fetch=123.46ms" in calls[0]
