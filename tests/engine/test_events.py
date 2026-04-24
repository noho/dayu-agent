"""
测试事件模型
"""
import pytest
from dayu.engine import (
    EventType,
    StreamEvent,
    content_delta,
    content_complete,
    tool_call_start,
    tool_call_delta,
    tool_call_dispatched,
    tool_call_result,
    tool_calls_batch_ready,
    tool_calls_batch_done,
    iteration_start_event,
    error_event,
    warning_event,
    done_event,
    metadata_event,
    final_answer_event,
)


class TestEventType:
    """测试 EventType 枚举"""
    
    def test_event_types_exist(self):
        """测试所有事件类型都存在"""
        assert EventType.CONTENT_DELTA
        assert EventType.CONTENT_COMPLETE
        assert EventType.TOOL_CALL_START
        assert EventType.TOOL_CALL_DELTA
        assert EventType.TOOL_CALL_DISPATCHED
        assert EventType.TOOL_CALL_RESULT
        assert EventType.TOOL_CALLS_BATCH_READY
        assert EventType.TOOL_CALLS_BATCH_DONE
        assert EventType.ERROR
        assert EventType.WARNING
        assert EventType.DONE
        assert EventType.METADATA
        assert EventType.FINAL_ANSWER
    
    def test_event_type_values(self):
        """测试事件类型的值"""
        assert EventType.CONTENT_DELTA.value == "content_delta"
        assert EventType.ERROR.value == "error"
        assert EventType.DONE.value == "done"


class TestStreamEvent:
    """测试 StreamEvent 数据类"""
    
    def test_create_event(self):
        """测试创建事件"""
        event = StreamEvent(
            type=EventType.CONTENT_DELTA,
            data="test content",
            metadata={"key": "value"}
        )
        
        assert event.type == EventType.CONTENT_DELTA
        assert event.data == "test content"
        assert event.metadata == {"key": "value"}
    
    def test_create_event_without_metadata(self):
        """测试创建没有元数据的事件"""
        event = StreamEvent(
            type=EventType.DONE,
            data={"summary": "complete"}
        )
        
        assert event.type == EventType.DONE
        assert event.data == {"summary": "complete"}
        assert event.metadata == {}
    
    def test_to_dict(self):
        """测试转换为字典"""
        event = StreamEvent(
            type=EventType.CONTENT_DELTA,
            data="test",
            metadata={"count": 1}
        )
        
        d = event.to_dict()
        
        assert d == {
            "type": "content_delta",
            "data": "test",
            "metadata": {"count": 1}
        }
    
    def test_from_dict(self):
        """测试从字典创建"""
        d = {
            "type": "content_delta",
            "data": "test",
            "metadata": {"count": 1}
        }
        
        event = StreamEvent.from_dict(d)
        
        assert event.type == EventType.CONTENT_DELTA
        assert event.data == "test"
        assert event.metadata == {"count": 1}
    
    def test_from_dict_without_metadata(self):
        """测试从没有元数据的字典创建"""
        d = {
            "type": "done",
            "data": {}
        }
        
        event = StreamEvent.from_dict(d)
        
        assert event.type == EventType.DONE
        assert event.metadata == {}


class TestConvenienceFunctions:
    """测试便捷构造函数"""
    
    def test_content_delta(self):
        """测试 content_delta 函数"""
        event = content_delta("hello", index=0)
        
        assert event.type == EventType.CONTENT_DELTA
        assert event.data == "hello"
        assert event.metadata == {"index": 0}
    
    def test_content_complete(self):
        """测试 content_complete 函数"""
        event = content_complete("full text", chars=9)
        
        assert event.type == EventType.CONTENT_COMPLETE
        assert event.data == "full text"
        assert event.metadata == {"chars": 9}
    
    def test_tool_call_start(self):
        """测试 tool_call_start 函数"""
        event = tool_call_start("read_file", "call_123")
        
        assert event.type == EventType.TOOL_CALL_START
        assert event.data == {"name": "read_file", "id": "call_123"}
    
    def test_tool_call_delta(self):
        """测试 tool_call_delta 函数"""
        event = tool_call_delta("call_123", "read_file", '{"path":')
        
        assert event.type == EventType.TOOL_CALL_DELTA
        assert event.data == {
            "id": "call_123",
            "name": "read_file",
            "arguments_delta": '{"path":',
        }
    
    def test_tool_call_dispatched(self):
        """测试 tool_call_dispatched 函数"""
        event = tool_call_dispatched(
            "call_123",
            "read_file",
            '{"path":"test.txt"}',
            index_in_iteration=0,
        )
        
        assert event.type == EventType.TOOL_CALL_DISPATCHED
        assert event.data == {
            "id": "call_123",
            "name": "read_file",
            "arguments": '{"path":"test.txt"}',
            "index_in_iteration": 0,
            "display_name": "read_file",
            "param_preview": "",
        }

    def test_tool_call_dispatched_with_object_arguments(self):
        """测试 tool_call_dispatched 支持对象参数。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        event = tool_call_dispatched(
            "call_123",
            "read_file",
            {"path": "test.txt"},
            index_in_iteration=0,
        )
        assert event.type == EventType.TOOL_CALL_DISPATCHED
        assert event.data == {
            "id": "call_123",
            "name": "read_file",
            "arguments": {"path": "test.txt"},
            "index_in_iteration": 0,
            "display_name": "read_file",
            "param_preview": "",
        }

    def test_tool_call_result(self):
        """测试 tool_call_result 函数"""
        event = tool_call_result(
            "call_123",
            {"ok": True, "value": "file content"},
            name="read_file",
            arguments={"path": "test.txt"},
            index_in_iteration=0,
        )
        
        assert event.type == EventType.TOOL_CALL_RESULT
        assert event.data == {
            "id": "call_123",
            "name": "read_file",
            "arguments": {"path": "test.txt"},
            "index_in_iteration": 0,
            "result": {"ok": True, "value": "file content"},
            "display_name": "read_file",
        }

    def test_tool_calls_batch_ready(self):
        """测试 tool_calls_batch_ready 函数"""
        event = tool_calls_batch_ready(["call_1", "call_2"])
        
        assert event.type == EventType.TOOL_CALLS_BATCH_READY
        assert event.data == {"call_ids": ["call_1", "call_2"], "count": 2}

    def test_tool_calls_batch_done(self):
        """测试 tool_calls_batch_done 函数"""
        event = tool_calls_batch_done(
            ["call_1", "call_2"],
            ok=1,
            error=1,
            timeout=0,
            cancelled=0,
        )
        
        assert event.type == EventType.TOOL_CALLS_BATCH_DONE
        assert event.data == {
            "call_ids": ["call_1", "call_2"],
            "ok": 1,
            "error": 1,
            "timeout": 0,
            "cancelled": 0,
        }
    
    def test_iteration_start_event(self):
        """测试 iteration_start_event 函数"""
        event = iteration_start_event(iteration=2, run_id="run_abc")

        assert event.type == EventType.ITERATION_START
        assert event.data == {"iteration": 2, "run_id": "run_abc"}

    def test_error_event_simple(self):
        """测试 error_event 函数（简单）"""
        event = error_event("something failed")
        
        assert event.type == EventType.ERROR
        assert event.data["message"] == "something failed"
        assert event.data["recoverable"] is False
    
    def test_error_event_with_exception(self):
        """测试 error_event 函数（带异常）"""
        exc = ValueError("invalid value")
        event = error_event("test error", exception=exc, recoverable=True)
        
        assert event.type == EventType.ERROR
        assert event.data["message"] == "test error"
        assert event.data["recoverable"] is True
        assert event.data["exception_type"] == "ValueError"
        assert event.data["exception_str"] == "invalid value"
    
    def test_warning_event(self):
        """测试 warning_event 函数"""
        event = warning_event("be careful", level="high")
        
        assert event.type == EventType.WARNING
        assert event.data == {"message": "be careful"}
        assert event.metadata == {"level": "high"}
    
    def test_done_event_empty(self):
        """测试 done_event 函数（空）"""
        event = done_event()
        
        assert event.type == EventType.DONE
        assert event.data == {}
    
    def test_done_event_with_summary(self):
        """测试 done_event 函数（带摘要）"""
        event = done_event(summary={"total": 100}, duration=1.5)
        
        assert event.type == EventType.DONE
        assert event.data == {"total": 100}
        assert event.metadata == {"duration": 1.5}
    
    def test_metadata_event(self):
        """测试 metadata_event 函数"""
        event = metadata_event("token_count", 150, model="gpt-4")
        
        assert event.type == EventType.METADATA
        assert event.data == {"token_count": 150}
        assert event.metadata == {"model": "gpt-4"}

    def test_final_answer_event(self):
        """测试 final_answer_event 函数"""
        event = final_answer_event("done", degraded=True, iteration_id="i1")
        
        assert event.type == EventType.FINAL_ANSWER
        assert event.data == {"content": "done", "degraded": True}
        assert event.metadata == {"iteration_id": "i1"}

    def test_final_answer_event_supports_filtered_payload(self):
        """测试 final_answer_event 可携带 filtered 与 finish_reason。"""
        event = final_answer_event(
            "partial",
            degraded=True,
            filtered=True,
            finish_reason="content_filter",
        )

        assert event.type == EventType.FINAL_ANSWER
        assert event.data == {
            "content": "partial",
            "degraded": True,
            "filtered": True,
            "finish_reason": "content_filter",
        }
