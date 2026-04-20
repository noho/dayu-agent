"""
异步 CLI Runner

通过异步 subprocess 调用外部命令行 LLM（主要面向 Codex CLI），解析 `--json`
模式输出的 JSONL 事件流，并将模型文本映射为 Engine 的 `StreamEvent` 事件流。

禁用说明：
- `AsyncCliRunner` 当前仅作为历史残留实现存在，已不再维护。
- 默认配置与 Host 主链路已禁止继续选择 CLI runner。
- 只有在清理历史代码或迁移旧实现时，才应阅读或修改本模块。

核心特性:
- Stdin 模式：总是通过 stdin 传入 prompt（命令末尾追加 `-`）
- 强制 JSON 输出：自动追加 `--json`，按 JSONL 逐行解析
- 输出净化：自动追加 `--color never`，避免 ANSI 转义码污染
- Streaming First：实时产出 `content_delta`，并在结束时产出 `content_complete`
- 事件顺序保证：`content_complete` 总是触发且先于 `done_event`
- 环境隔离：可指定 `working_dir` 与额外 `env`，并提供 `timeout` 控制

限制/约束:
- 不支持 Tool Calling：`supports_tool_calling` 固定为 False，`set_tools()` 为 no-op
- system role 自动写入：检测到 system role 时，会将其写入运行目录的 `AGENTS.md`
- 仅提取 `agent_message` 文本：其它 JSON item types（command/file_changes 等）当前忽略

配置示例（llm_models.json）：
{
  "codex_cli": {
    "runner_type": "cli",
    "command": ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only"],
    "working_dir": ".",
    "timeout": 3600,
    "env": {"OPENAI_API_KEY": "{{OPENAI_API_KEY}}"},
    "model": "gpt-4.1",
    "full_auto": false,
    "reasoning_effort": "medium"
  }
}

主要方法：

1. __init__(*, command, working_dir=None, env=None, timeout=3600, model=None,
            full_auto=False, reasoning_effort="medium")
   初始化 CLI Runner
   - command: CLI 命令（列表形式，如 ["codex", "exec"]）
   - working_dir: 子进程工作目录（影响相对路径读写）
   - env: 额外环境变量（合并到当前进程环境）
   - timeout: 超时时间（秒）
   - model: 模型名称（可选，通过 --model 传递）
   - full_auto: 是否启用低摩擦自动化模式（通过 --full-auto 传递）
   - reasoning_effort: 推理强度（通过 --config model_reasoning_effort="..." 传递）

2. set_tools(executor)
   CLI Runner 不使用工具调用机制；该方法为 no-op。

3. async call(messages, *, stream=True, **extra_payloads) -> AsyncIterator[StreamEvent]
   调用 CLI 并返回事件流（stream 参数保留以对齐 Runner 接口；CLI 默认视为流式）。
   - messages: OpenAI 格式消息列表（system role 将被忽略并触发 warning）
   - extra_payloads: 支持在调用级覆盖 full_auto/reasoning_effort

   事件类型（本 Runner 实际可能产出）：
   - content_delta: 实时文本增量（来自 JSONL 的 agent_message）
   - content_complete: 完整文本（总是触发，即使内容为空）
   - done_event: 回合完成（通常携带 usage 摘要）
   - error_event: CLI 失败/超时/JSON 结构错误等不可恢复错误
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from dataclasses import dataclass

from dayu.contracts.agent_types import AgentMessage
from .events import (
    EventType,
    StreamEvent,
    content_delta,
    content_complete,
    done_event,
    error_event,
    warning_event,
)
from .protocols import ToolExecutor
from dayu.log import Log

MODULE = "ENGINE.ASYNC_CLI_RUNNER"


@dataclass
class AsyncCliRunnerRunningConfig:
    """AsyncCliRunner 的运行时配置（空壳）。

    .. note::

        本 dataclass 当前无任何字段，整个 ``async_cli_runner`` 模块已标记弃用。
        后续应随模块整体删除而清理，不做单独修复。
        参见 code-review M23。
    """

    pass

class AsyncCliRunner:
    """
    异步 CLI Runner，通过 subprocess 调用外部 CLI 并实时 streaming 输出。

    禁用说明：
    - 本类仅保留为历史残留实现，已不再维护。
    - 默认配置与 Host 主链路已禁止继续使用本类。
    - 后续 code review 若发现本类问题，默认按已禁用旧路径处理。
    
    支持的 CLI 类型：
    1. Codex CLI: `codex exec -`
    
    配置示例（llm_models.json）：
    {
      "codex_cli": {
        "runner_type": "cli",
        "command": ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only"],
        "model": "gpt-4",
        "reasoning_effort": "high"
      }
    }
    
    注意：
    - 总是使用 stdin 模式传入 prompt（支持长文本、特殊字符、安全）
    - 不支持 tool calling 机制（Codex CLI 直接访问本地文件）
    """
    
    def __init__(
        self,
        *,
        command: List[str],
        working_dir: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 3600,
        model: Optional[str] = None,
        full_auto: bool = False,
        reasoning_effort: str = "medium",
        running_config: Optional[AsyncCliRunnerRunningConfig] = None,
    ):
        """
        Args:
            command: CLI 命令（列表形式，如 ["codex", "exec"]）
            working_dir: 工作目录
            env: 环境变量（补充到当前环境）
            timeout: 超时时间（秒）
            model: 模型名称（可选，通过 --model 传递）
            full_auto: 是否启用低摩擦自动化模式（默认 False）
            reasoning_effort: 推理强度（可选值：minimal | low | medium | high | xhigh，默认 medium）
        """
        self.command = command
        self.working_dir = working_dir or Path.cwd()
        Log.verbose(f"AsyncCliRunner: 使用工作目录 {self.working_dir}", module=MODULE)
        self.env = {**os.environ, **(env or {})}
        self.timeout = timeout
        self.model = model
        self.full_auto = full_auto
        self.reasoning_effort = reasoning_effort
        self.running_config = running_config or AsyncCliRunnerRunningConfig()
        Log.verbose(
            f"AsyncCliRunner: 模型={self.model}, full_auto={self.full_auto}, reasoning_effort={self.reasoning_effort}",
            module=MODULE,
        )
        # 不管配置里写什么，这里都强制设为 False
        self.supports_tool_calling = False  # CLI 不支持工具调用机制
    
    def set_tools(self, executor: Optional[ToolExecutor]) -> None:
        """
        设置工具执行器（CLI Runner 不需要，no-op）
        
        注意：AsyncCliRunner 直接访问本地文件，不使用 tool calling 机制。
        工具执行器被忽略。
        
        Args:
            executor: 工具执行器（被忽略，可为 None）
        """
        # No-op: CLI 不需要 tool calling
        Log.warn("AsyncCliRunner: set_tools() 被调用，但 CLI 不支持工具调用机制，忽略该操作", module=MODULE)

    def is_supports_tool_calling(self) -> bool:
        """是否支持 Tool Calling"""
        return False

    async def close(self) -> None:
        """关闭 Runner 资源。

        当前 CLI runner 不持有需要跨调用复用的异步资源，因此为 no-op。
        """

        return None
    
    async def call(
        self,
        messages: List[AgentMessage],
        *,
        stream: bool = True,
        **extra_payloads,
    ) -> AsyncIterator[StreamEvent]:
        """
        调用 CLI 并返回 streaming 事件流
        
        Args:
            messages: 消息列表（OpenAI 格式）
            stream: 是否启用 streaming（CLI 默认为 True）
            **extra_payloads: 额外参数（如 full_auto, reasoning_effort 等）
        
        Yields:
            StreamEvent: 流式事件
        """
        trace_context = extra_payloads.pop("trace_context", None)
        if not isinstance(trace_context, dict):
            trace_context = {}
        request_id = f"cli_{uuid.uuid4().hex[:8]}"
        run_id = trace_context.get("run_id") or f"run_{uuid.uuid4().hex[:8]}"
        iteration_id = trace_context.get("iteration_id") or f"{run_id}_iteration_{request_id}"
        trace_meta = {
            "run_id": run_id,
            "iteration_id": iteration_id,
            "request_id": request_id,
        }

        # 1. 构建完整命令
        full_command = self._build_command(**extra_payloads)
        
        # 2. 提取 prompt（将 messages 转换为单个文本）
        prompt_text, ok = self._format_messages(messages)
        if not ok:
            yield self._annotate_event(error_event(
                "Failed to write AGENTS.md for system prompt",
                recoverable=False,
                error_type="agents_md_write_failed",
            ), trace_meta)
            return
        
        # 3. 执行 CLI 并实时读取 stdout
        async for event in self._run_streaming(full_command, prompt_text):
            yield self._annotate_event(event, trace_meta)
    
    def _build_command(self, **extra_payloads) -> List[str]:
        """
        构建完整的 CLI 命令
        
        Args:
            **extra_payloads: 额外参数
                - full_auto: 是否启用低摩擦自动化模式（优先级高于 __init__ 参数）
                - reasoning_effort: 推理强度（minimal|low|medium|high|xhigh，优先级高于 __init__ 参数）
        """
        cmd = self.command.copy()
        
        # 强制启用 JSON 模式（结构化事件流）
        cmd.append("--json")
        
        # 强制禁用颜色输出（防止 ANSI 转义码污染）
        cmd.extend(["--color", "never"])
        
        # 添加 model 参数（来自 __init__）
        if self.model:
            cmd.extend(["--model", self.model])
        
        # 支持 --full-auto（优先使用 extra_payloads，回退到 self.full_auto）
        full_auto = extra_payloads.get("full_auto", self.full_auto)
        if full_auto:
            cmd.append("--full-auto")
        
        # 添加 reasoning_effort 参数（优先使用 extra_payloads，回退到 self.reasoning_effort）
        reasoning_effort = extra_payloads.get("reasoning_effort", self.reasoning_effort)
        if reasoning_effort:
            # 格式：--config model_reasoning_effort='"high"'
            cmd.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
        
        # 添加 stdin 标志作为 PROMPT 位置参数（总是使用 stdin 模式）
        # 文档：codex exec [FLAGS] -  (从 stdin 读取 PROMPT)
        cmd.append("-")
        
        return cmd

    def _annotate_event(self, event: StreamEvent, trace_meta: Dict[str, Any]) -> StreamEvent:
        metadata = dict(event.metadata) if event.metadata else {}
        for key in ("run_id", "iteration_id", "request_id"):
            if key in trace_meta:
                metadata.setdefault(key, trace_meta[key])
        event.metadata = metadata
        return event
    
    def _format_messages(self, messages: List[AgentMessage]) -> tuple[str, bool]:
        """
        将 OpenAI 格式的 messages 转换为单个文本 prompt
        
        注意：AsyncCliRunner 依赖 Codex CLI 的 AGENTS.md 机制处理 system prompt。
        Codex CLI 在启动时会自动读取以下位置的 AGENTS.md 文件：
        - 全局：~/.codex/AGENTS.md
        - 项目：从 Git 根目录到当前工作目录，逐层查找 AGENTS.md
        
        如果 messages 中包含 system role，会自动写入工作目录的 AGENTS.md 文件。
        详见：https://developers.openai.com/codex/guides/agents-md
        
        支持的角色：
        - user: 用户输入（提取）
        - assistant: 助手回复（用于多轮对话，提取）
        - system: 系统提示（自动写入 AGENTS.md）

        Returns:
            (prompt, ok): 组装后的 prompt 文本与写入状态
        """
        parts = []
        has_system_prompt = False
        system_messages = []
        success = True
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                parts.append(content)  # 用户消息不加前缀
            elif role == "assistant":
                parts.append(f"ASSISTANT: {content}")
            elif role == "system":
                has_system_prompt = True
                if content:
                    system_messages.append(content)
        
        if has_system_prompt:
            agents_path = self.working_dir / "AGENTS.md"
            try:
                agents_path.write_text("\n\n".join(system_messages), encoding="utf-8")
                Log.info(f"AsyncCliRunner: 已写入 system 指令到 {agents_path}", module=MODULE)
            except Exception as exc:
                Log.error(f"AsyncCliRunner: 写入 AGENTS.md 失败: {exc}", module=MODULE)
                success = False
        
        return "\n\n".join(parts).strip(), success
    
    def _parse_json_event(self, line_text: str) -> Optional[StreamEvent]:
        """
        解析 Codex CLI 的 JSON 事件行（JSONL 格式）
        
        Args:
            line_text: 单行 JSON 字符串
        
        Returns:
            StreamEvent 或 None（如果解析失败或为空行）
        
        Codex CLI --json 输出格式（来自官方文档）：
        - {"type":"thread.started","thread_id":"..."}
        - {"type":"turn.started"}
        - {"type":"item.started","item":{"id":"...","type":"agent_message",...}}
        - {"type":"item.completed","item":{"id":"...","type":"agent_message","text":"..."}}
        - {"type":"turn.completed","usage":{...}}
        - {"type":"turn.failed","error":{...}}
        - {"type":"error",...}
        
        item types: agent_message, command_execution, file_changes, reasoning, 
                    web_search, mcp_tool_call, plan_update
        """
        line_text = line_text.strip()
        if not line_text:
            return None
        
        try:
            event_data = json.loads(line_text)
            event_type = event_data.get("type")
            
            # 线程/回合级别事件（忽略，不转换为 StreamEvent）
            if event_type in ("thread.started", "turn.started"):
                return None
            
            # item.completed: 提取 agent_message 文本
            elif event_type == "item.completed":
                item = event_data.get("item", {})
                item_type = item.get("type")
                
                if item_type == "agent_message":
                    # 最终的 agent 输出文本
                    text = item.get("text", "")
                    if text:
                        return content_delta(text)
                
                # 其他 item 类型（command_execution, file_changes 等）暂不处理
                return None
            
            # turn.completed: 回合完成（带 token 使用统计）
            elif event_type == "turn.completed":
                usage = event_data.get("usage", {})
                return done_event(summary={"usage": usage})
            
            # turn.failed / error: 错误事件
            elif event_type in ("turn.failed", "error"):
                error_info = event_data.get("error", {}) if event_type == "turn.failed" else event_data
                error_message = error_info.get("message") or "未知错误"
                Log.error(f"CLI 返回错误事件: {error_message}（error_type=cli_error）", module=MODULE)
                return error_event(
                    error_info.get("message", "Unknown error"),
                    recoverable=False,
                    error_type="cli_error",
                    body=json.dumps(error_info, ensure_ascii=False),
                )
            
            # item.started: 可能需要处理进度（暂忽略）
            elif event_type == "item.started":
                return None
            
            else:
                # 未知事件类型：忽略（不是错误）
                return None
        
        except json.JSONDecodeError as e:
            # JSON 解析失败：记录警告（Codex CLI --json 模式应该只输出 JSONL）
            Log.warn(f"解析 JSON 事件失败: {e}。行内容: {line_text[:100]}...", module=MODULE)
            return error_event(
                    message="Failed to parse JSON event from CLI output",
                    recoverable=False,
                    error_type="response_error",
                    body=line_text,
                )
    
    async def _run_streaming(
        self,
        command: List[str],
        prompt_text: str,
    ) -> AsyncIterator[StreamEvent]:
        """
        执行 CLI 并实时读取 stdout（streaming 模式）
        """
        process = None
        try:
            deadline = time.monotonic() + self.timeout
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_dir),
                env=self.env,
            )
            
            # 写入 prompt 到 stdin
            if process.stdin:
                process.stdin.write(prompt_text.encode("utf-8"))
                await process.stdin.drain()
                process.stdin.close()
            
            # 实时读取 stdout（JSON 模式：JSONL 事件流）
            content_buffer = []  # 只累积 content_delta 文本（不包括 JSON 事件行）
            pending_done_event = None  # 暂存 done_event，等待 content_complete 先触发
            
            if process.stdout:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError

                    line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
                    if not line:
                        break

                    line_text = line.decode("utf-8", errors="ignore").strip()
                    if not line_text:
                        continue
                    
                    # 尝试解析 JSON 事件
                    event = self._parse_json_event(line_text)
                    if event:
                        # 累积文本内容（只存储 content_delta）
                        if event.type == EventType.CONTENT_DELTA:
                            content_buffer.append(event.data)
                            yield event
                        
                        # 暂存 done_event，确保 content_complete 先触发
                        elif event.type == EventType.DONE:
                            pending_done_event = event
                        
                        elif event.type == EventType.ERROR:
                            # 遇到 error_event，立即 yield 并停止处理
                            if process:
                                process.kill()
                            yield event
                            return
                        # 其他事件直接 yield
                        else:
                            yield event
            
            # 等待进程结束
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(process.wait(), timeout=remaining)
            
            # 检查退出码
            if process.returncode != 0:
                stderr = ""
                if process.stderr:
                    stderr = await process.stderr.read()
                    stderr = stderr.decode("utf-8", errors="ignore")
                
                # 不 raise，而是 yield error_event（保持与 AsyncOpenAIRunner 一致）
                yield error_event(
                    f"CLI failed with exit code {process.returncode}: {stderr}",
                    recoverable=False,
                    error_type="cli_error",
                    status=process.returncode,
                    body=json.dumps(
                        {
                            "exit_code": process.returncode,
                            "stderr": stderr,
                            "command": " ".join(command),
                        },
                        ensure_ascii=False,
                    ),
                )
                return  # 停止处理
            
            # 总是 yield content_complete（在 done_event 之前）
            # 即使内容为空也必须触发（与 OpenAIRunner 行为一致）
            full_content = "".join(content_buffer)
            yield content_complete(full_content)
            
            # 最后 yield done_event（如果收到了）
            if pending_done_event:
                yield pending_done_event
            else:
                yield warning_event(
                    "CLI stream ended without turn.completed; emitted synthetic done_event",
                    source="async_cli_runner",
                )
                yield done_event(summary={"usage": {}, "inferred": True})
            
        except asyncio.TimeoutError:
            # 先 kill 子进程（确保资源立即释放）
            if process:
                process.kill()
            # 再 yield error_event（调用者可能在处理后停止迭代）
            yield error_event(
                f"CLI timeout after {self.timeout}s",
                recoverable=False,
                error_type="timeout",
            )
        except Exception as e:
            # 先 kill 子进程（确保资源立即释放）
            if process:
                process.kill()
            # 再 yield error_event（调用者可能在处理后停止迭代）
            yield error_event(
                f"CLI execution error: {e}",
                exception=e,
                recoverable=False,
                error_type="unknown_error",
            )


def create_codex_runner(
    *,
    working_dir: Optional[Path] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> AsyncCliRunner:
    """
    创建 Codex CLI Runner 的便捷工厂函数
    
    Args:
        working_dir: 工作目录
        model: 模型名称（可选）
        reasoning_effort: 推理强度（可选）
    
    Returns:
        AsyncCliRunner 实例
    """
    command = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
    ]
    
    if model:
        command.extend(["--model", model])
    
    if reasoning_effort:
        command.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
    
    return AsyncCliRunner(
        command=command,
        working_dir=working_dir,
    )
