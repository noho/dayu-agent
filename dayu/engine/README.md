# dayu Agent 开发手册 - Engine 包

`dayu/engine` 是 Dayu 的通用执行原语层。它不理解业务语义，也不负责 scene 装配；它只负责把已经准备好的一次 Agent 消息交互稳定跑完。

本文档只写当前真实代码里的 Engine 责任、公共契约、事件流和扩展点。

## 1. Engine 负责什么

当前 Engine 负责：
- `AsyncAgent`
- `AsyncRunner` 协议与默认 Runner
- `ToolExecutor` / `ToolRegistry`
- `StreamEvent` 事件模型
- tool loop 与工具结果回填
- 上下文预算治理、截断续写、降级
- Tool Trace
- 进程内取消原语 `CancellationToken`

当前 Engine 不负责：
- 读取 `run.json` / `llm_models.json`
- scene 解析
- `system_prompt` 渲染
- `<when_tool>` / `<when_tag>` 条件块解析
- 模板变量替换与 `load_prompt()` 这类 prompt 渲染公共能力
- 全局日志接口；日志真源在 `dayu.log`，Engine 不再对 `Log` / `LogLevel` 做包级再导出
- `ticker`、写作、审计等业务语义
- session / run 生命周期治理

## 2. 主体关系

```text
Host / scene preparation
-> AgentInput
-> AsyncAgent
-> AsyncRunner
-> LLM API / CLI
```

其中：
- `AsyncAgent` 负责把多个 Runner 回合串成一次完整推理过程
- `AsyncRunner` 负责一次底层模型调用
- `ToolRegistry` 负责工具 schema、执行与 middleware

术语约束：

- Engine 内部把一次 LLM 调用加工具闭环称为 `agent iteration`。
- transcript / memory 中的 `turn` 指的是 `conversation turn`，不是 Engine iteration。
- `AsyncAgent`、Tool Trace 与 trace 分析脚本都以 `iteration_id` 作为 Engine 内部轮次唯一真源。

## 3. 稳定契约

### 3.1 AsyncAgent

当前主入口：
- `AsyncAgent.__init__(runner=..., running_config=..., ...)`
- `run(...)`
- `run_messages(...)`
- `run_and_wait(...)`

但主链已经不再使用旧的“配置直读式”构造入口。

关键约束：
- `AgentInput` 只在 Host / scene preparation 内部流转
- `AsyncAgent` 只消费最终 `messages`、工具执行器、显式 trace 身份和显式传入的 `run_id`
- `run_id` 不进入模型 payload
- 同一个 `AsyncAgent` 实例不支持并发运行

当前 Host -> Engine 输入口径：
- `messages` 使用共享 `AgentMessage` TypedDict 联合，而不是无约束 `dict[str, Any]`
- `tools` 使用 `ToolExecutor` 协议
- `trace_identity` 使用固定字段的 `AgentTraceIdentity`
- `runtime_limits` 当前只显式承载 `timeout_ms`，它仍是 Host -> Engine 的超时数据契约；tool 级预算与取消观察则通过 runner 配置和单次 `ToolExecutionContext` 进入 Engine
- Engine 级取消观察保持可选：`AsyncAgent`、`AsyncOpenAIRunner`、`SSEStreamParser` 都允许没有 `CancellationToken` 独立运行；一旦显式注入令牌，Runner 必须在模型请求进入、响应体读取、重试退避等待与 SSE 分块等待这些阻塞边界及时抛出 `dayu.contracts.cancellation.CancelledError`
- 当上层不是通过 `CancellationToken`，而是直接对外层 asyncio task 做 `cancel()` / `wait_for()` 超时时，Runner / Parser 在这些阻塞边界创建的内部子任务也必须被同步取消并等待收口，不能把 HTTP 建连、响应体读取或分块读取留在后台继续运行

### 3.2 AgentCreateArgs

`AgentCreateArgs` 是构造 Agent / Runner 的完整参数对象。Engine 假设它已经解析完成，不再回头读配置文件。

关键字段：
- `runner_type`
- `model_name`
- `max_turns`
- `max_context_tokens`
- `temperature`
- `runner_params`
- `runner_running_config`：跨层快照，不再使用未约束的配置袋子
- `agent_running_config`：跨层快照，不再使用未约束的配置袋子

跨 Service / Host 传递的 runtime config 快照（`RunnerRunningConfigSnapshot` / `AgentRunningConfigSnapshot`）稳定真源在 `dayu.contracts.runtime_config_snapshot`；execution 层运行配置值对象与快照转换函数仍归属 `dayu.execution.runtime_config`。

Host 负责最后一跳把这些纯配置快照恢复并转换成 engine 内部实现对象；`AgentRunningConfig` 与 `AsyncOpenAIRunnerRunningConfig` 只作为 engine 定义模块内的实现细节存在，不属于 `dayu.engine` 的包级公共 API。

### 3.3 AsyncRunner

最小 Runner 契约：

```python
class AsyncRunner(Protocol):
    def call(messages, *, stream=True, **extra_payloads) -> AsyncIterator[StreamEvent]: ...
    def set_tools(executor: ToolExecutor | None) -> None: ...
    def is_supports_tool_calling() -> bool: ...
    async def close() -> None: ...
```

其中 `messages` 的稳定类型是 `list[AgentMessage]`。

`extra_payloads` 只用于 provider 扩展参数，不能覆盖 `messages`、`model`、`temperature`、`stream`、`tools` 等 Runner 显式字段。

当前默认实现：
- `AsyncOpenAIRunner`

`AsyncOpenAIRunner` 当前稳定行为补充：
- `AsyncRunner.close()` 已成为稳定生命周期契约：Runner 如果持有 HTTP session、子进程句柄或其它异步资源，必须通过该入口显式收口；`AsyncAgent` 会在单次 `run/run_messages/run_and_wait` 生命周期结束时统一调用它
- `AsyncOpenAIRunner` 当前把 `aiohttp.ClientSession` 提升为 Runner 实例级资源：同一 `AsyncAgent` run 内的多轮 iteration 会复用同一个 session/connector；只有 run 结束或显式 `runner.close()` 时才关闭，下一次调用再按需重建
- 取消信号不是只在 iteration 首尾观察；Runner 必须在 `session.post(...)` 建连等待、`response.json()` / `response.text()` 读取、429/5xx 重试 sleep，以及 SSE 流的下一块等待期间协作式响应取消
- 取消一旦命中，上层看到的稳定事实是抛出 `dayu.contracts.cancellation.CancelledError`；不能把这类路径降级成 `error_event`、普通超时重试或吞掉后继续产出 `final_answer`
- Runner 为取消观察临时注册到 `CancellationToken` 的回调必须在本轮调用结束后注销；复用同一 token 的多轮调用不允许累积历史 loop/future 闭包

历史残留实现：
- `AsyncCliRunner`：已禁用，仅保留源码以便迁移旧实现，不允许再通过配置或 Host 主链路使用；已从 `dayu.engine` 包级公共导出移除，测试等内部使用方须通过 `dayu.engine.async_cli_runner` 直接导入

### 3.4 ToolExecutor

最小工具执行契约：

```python
class ToolExecutor(Protocol):
    def execute(name, arguments, context=None) -> dict[str, Any]: ...
    def get_schemas() -> list[dict[str, Any]]: ...
    def clear_cursors() -> None: ...
    def get_tool_display_info(name) -> tuple[str, list[str] | None]: ...
```

当前默认实现是 `ToolRegistry`。

当前工具执行契约补充：
- `context` 的稳定真源已经收敛为强类型 `ToolExecutionContext`，包含 `run_id / iteration_id / tool_call_id / index_in_iteration / timeout_seconds / cancellation_token`
- `ToolExecutionContext` 只保留属性访问，不再提供 `dict` / `Mapping` 兼容桥接；调用方应直接构造并透传强类型对象
- `ToolRegistry` 与 `TruncationManager` 必须共同消费同一份 `ToolExecutionContext`；不要在分页续读链路把上下文签名退回旧的 `dict[str, Any]` 专用接口
- 只有显式声明 `execution_context_param_name` 的工具才会收到 execution context 注入；未声明的旧工具继续走兼容路径
- `ToolRegistry` / `TruncationManager` 只负责工具结果契约级截断：按工具 schema 的 `truncate_spec` 处理单次返回，并在需要时生成 `fetch_more` 续读信息；它们不感知 Agent 主循环的全局上下文预算
- `ToolsetRegistrationContext.registry` 依赖结构化 `ToolRegistryProtocol`；协议方法签名必须与 `ToolRegistry` 的公开方法保持同名同参，尤其是 `register_allowed_paths(paths)` 这类会被 Host 直接注入的注册入口
- toolset registrar 从 `ToolsetConfigSnapshot.payload` 恢复 `DocToolLimits / FinsToolLimits / WebToolsConfig` 时，应优先复用 `dayu.contracts.tool_configs` 的共享 helper；底层数值收口仍统一走 `dayu.contracts.toolset_config` 的 coercion helper，不能在 adapter 内部直接对 `ToolsetConfigValue` 做裸 `int()` / `float()`
- doc 工具白名单解析属于 doc toolset 自身边界，当前收口在 `dayu.engine.doc_access_policy`；`Host` 只能传通用 `workspace + execution_permissions`，不能上推解释 doc domain 规则
- Agent 级预算闸门的真源已经收口到 `dayu.engine.context_budget`：当工具结果已经序列化、准备注入下一轮 `tool` message 时，`AsyncAgent` 会基于 `ContextBudgetState` 做预测性预算裁剪
- `AsyncAgent` 会在提交 `final_answer` 前再次检查取消，避免同一轮同时对外落出“已回答”和“已取消”两种事实
- `DefaultHostExecutor` 会在写 transcript 前再次检查取消，取消 run 不再把本轮回答持久化进会话 transcript
- `SSEStreamParser` 的取消观察边界与 Runner 对齐：流式解析在 heartbeat 空转等待和下一块分片读取期间也必须继续观察同一令牌，不能等到整段流结束后才统一发现取消
- `SSEStreamParser` 使用的取消等待回调同样必须是“按轮注册、按轮注销”；成功解析、异常退出和外层任务取消都不能留下悬空回调或后台 chunk 读取 task

当前 web tools 的内部边界：
- `search_web` 的 provider 选择、API 调用与结果组装真源已经下沉到 `dayu.engine.tools.web_search_providers`
- `search_web` 的公开返回形状必须继续对齐 `SearchWebOutput` / `SearchResultRow` 这组稳定 TypedDict；`preferred_result` 只能是同结构单条结果或 `None`
- requests Session/timeout 与内容编码/字符集解析真源已经下沉到 `dayu.engine.tools.web_http_session` 和 `dayu.engine.tools.web_http_encoding`
- `fetch_web_page` 的 requests 主路径编排真源已经下沉到 `dayu.engine.tools.web_fetch_orchestrator`，集中承载 warmup、content-type probe、HTML/Docling 路由与浏览器升级判定
- Playwright 浏览器单例、子进程 worker、storage state 解析与回退执行真源已经下沉到 `dayu.engine.tools.web_playwright_backend`
- `dayu.engine.tools.web_tools` 当前主要保留 tool registration，以及兼容现有测试锚点的薄包装

当前 `fetch_web_page` 的机械恢复边界也在 Engine 内部：
- requests 抓取只声明当前运行时真的能解码的 `Accept-Encoding`，不会继续固定宣称 `br/zstd`
- 进入 HTML 抽取前会先按响应头 charset、HTML meta charset 与 `apparent_encoding` 纠正文本解码，避免旧站点 `gb2312/gbk` 页面直接乱码
- HTML 壳页里的立即 `meta refresh` 由抓取编排层有限跟随，避免把壳页直接送进正文抽取；每一跳都要重新按剩余 tool budget 计算 timeout，不能复用首跳超时
- requests 主路径会在 warmup、content-type probe、请求下载、meta refresh 跟随与正文转换这些阶段边界执行协作式取消检查；取消意图不会只在 iteration 首尾才被观察
- requests 侧 timeout、部分 SSL/TLS 握手失败、当前运行时不支持的内容编码、以及一小组更像浏览器/网关差异的 HTTP 状态（如 `412/521`），都会优先尝试既有 Playwright 浏览器回退
- 对未类型化但已暴露“正文为空/主体抽取失败”的 HTML 壳页，只要原始响应带有客户端渲染特征，也会升级到 Playwright，而不是直接判成换来源
- challenge 检测使用“正文模式 + vendor 响应头 + 错误状态下的复合信号”三类强信号；单独的 vendor 清理 cookie 不能直接判成 blocked，避免把正常正文页误伤成挑战页
- 对按 host 命中的 Playwright storage state，requests 主路径也会机械导入其中的 cookie；如果人工浏览已拿到可复用会话态，后续抓取不必先失败一次再退回浏览器
- Playwright 回退会复用按 host 命中的 storage state（含 `www` 变体），先做一次同域首页预热，再做有上限的页面稳定化等待，并在浏览器上下文里忽略证书错误以贴近人工浏览器可继续访问的场景；回退执行优先放到子进程边界内，在 timeout 或取消时由父进程硬终止
- 父进程等待 Playwright 回退结果时，会优先轮询 result queue，再处理子进程退出；不能要求 worker 先完全退出才读取结果，否则大页面结果可能因为 Queue flush 时序被误判成 timeout
- HTML pipeline 只处理已经拿到的文本，不负责再次联网导航

### 3.5 Processor contract

处理器相关稳定契约补充：
- `DocumentProcessor` 通过 `get_parser_version()` 暴露 parser version；协议层不再要求上层直接读取类变量。
- `Source` 是只读结构化协议，约束的是 `uri/media_type/content_length/etag/open/materialize` 这组能力，而不是要求实现类显式继承协议。
- 处理器、仓储返回的具体 `Source` 实现和测试桩都应依赖结构兼容，不要把 `Protocol` 当作运行时父类使用。
- `DoclingProcessor` 的文档装配主链路与私有 helper 当前以 `DoclingDocument / NodeItem / TableItem` 这组 docling 真类型作为静态契约；如果测试需要覆盖异常分支，应让测试桩通过 typed fake / 显式 cast 对齐该边界，而不是反向放宽生产代码签名去兼容旧测试对象。
- `BSProcessor` 只负责通用 HTML 读取、DOM 清洗、章节/表格抽取与业务域扩展钩子；Engine 默认读取链不得再内置 EDGAR SGML 剥离、SEC 封面页识别之类领域预处理。
- `text_utils.clean_page_header_noise()` 与 `infer_caption_from_context()` 当前属于跨处理器复用的通用 HTML/Text 基元；若业务域需要更强规则，应在上层包做包装，不要把领域增强直接塞回 Engine。
- 跨处理器复用的 HTML 表格 DataFrame 解析真源在 `dayu.engine.processors.table_utils.parse_html_table_dataframe()`；其它包若需要共享这项能力，应依赖该公共入口，不能反向 import `bs_processor.py` 之类具体处理器模块的私有 helper。
- 页面级 `sections/tables` 摘要必须复用 `build_section_summary()` / `build_table_summary()` 这一组稳定工厂，不能从处理器 helper 直接返回未声明字段的私有字典结构。
- `SearchEvidence` 的稳定边界只包含 `matched_text` 与 `context`；命中偏移等搜索内部细节必须在进入 `build_search_hit()` 前收口，不能直接外泄到公共命中契约。

## 4. 事件流

### 4.1 事件类型

当前稳定事件包括：
- `content_delta`
- `content_complete`
- `reasoning_delta`
- `tool_call_start`
- `tool_call_delta`
- `tool_call_dispatched`
- `tool_calls_batch_ready`
- `tool_call_result`
- `tool_calls_batch_done`
- `iteration_start`
- `metadata`
- `warning`
- `error`
- `done`
- `final_answer`

### 4.2 事件语义

需要区分两层：
- Runner 事件：一次底层模型回合的事件
- Agent 事件：多个 Runner 回合收敛后的事件

其中：
- `done` 只表示当前 Runner 回合结束
- `final_answer` 只由 `AsyncAgent` 产出
- `iteration_start` 由 `AsyncAgent` 在每轮迭代开始时产出，携带 `{iteration, run_id}`；Host 透传为 `AppEventType.ITERATION_START`，供 UI 层展示"第 N 轮思考..."
- `tool_call_dispatched` 和 `tool_call_result` 的 payload 中携带 `display_name`（工具展示名，fallback 到原始 name）和 `param_preview`（关键参数截断预览），由 Engine 在事件构造时从 `@tool` 声明的 `display_name` / `summary_params` 自动填充

`done` 的 `summary.finish_reason` 只承载底层模型本轮结束原因，当前要重点区分两类：
- `length`：表示本轮回答被截断。`AsyncAgent` 可在 continuation 预算允许时继续续写。
- `content_filter`：表示本轮命中内容过滤。`AsyncAgent` 不再走 continuation，而是保留当前已生成的 partial content，并额外发出 `warning`，随后产出带 `degraded=true`、`filtered=true` 与 `finish_reason="content_filter"` 的 `final_answer`。

因此，跨层是否发生“受过滤完成态”，应以 `final_answer` payload 里的 `filtered` 为真源；`warning` 只负责面向人的提示，不承担稳定契约语义。

### 4.3 元数据

`AsyncAgent` 会统一补这些元数据：
- `run_id`
- `iteration_id`
- 对工具事件补 `tool_call_id`

Host 提供的是权威 `run_id`。Engine 不负责生成宿主级 run，只负责消费并在事件链路里延续它。
Host 传入的固定身份信息使用 `AgentTraceIdentity`，Engine 只做透传和标准化，不在内部重新拼装身份袋子。

## 5. 状态机

`AsyncAgent` 当前大致遵循这条状态机：

```text
PrepareIteration
-> CallRunner
-> HandleToolBatch / ContinueAnswer / Finalize
-> PrepareNextIteration
```

关键分支：
- 工具调用闭环：收集工具调用 -> 执行工具 -> 回填 tool messages -> 下一轮
- 失败保护：连续多个 agent iteration 的工具批次全部失败时，按 `fallback_mode` 提前进入 `raise_error` 或 `force_answer`
- 续写：收到 `truncated` 后追加 continuation prompt
- 受过滤完成：收到 `content_filter` 后停止续写，保留 partial content，并把最终答案标记为 `filtered`
- 压缩：上下文超限或接近软上限时，对消息做压缩后重试
- 降级：达到最大工具轮次后移除工具能力并要求直接回答

## 6. 上下文预算治理

预算治理当前由 `AsyncAgent` 编排，并由 `dayu.engine.context_budget` 承载其中的预算原语，主要包括：
- 软上限触发的主动压缩
- 硬上限触发的压缩重试
- 工具结果预测性截断（只作用于“已序列化、待注入下一轮消息”的工具结果，不替代 ToolRegistry 的 schema 驱动截断）
- 截断续写

预算参数的最终真源在 `AgentCreateArgs`，不是 Runner 自己去读模型配置文件。

## 7. Tool Trace

当前 trace 责任分层：
- Host 提供 `run_id`
- Host 提供固定字段的 `AgentTraceIdentity`
- Engine 在每次 agent iteration 生成 `iteration_id`
- `ToolTraceRecorderFactory` 为本次 run 提供 recorder
- Runner / Agent / ToolRegistry 在执行过程中写 trace

Engine 自己不做 run registry，也不做跨进程取消桥接。

## 8. 当前默认 Runner

### 8.1 AsyncOpenAIRunner

定位：
- OpenAI compatible HTTP runner

负责：
- 组装请求 payload
- 处理 SSE / 非 SSE 响应
- 按 `debug_sse` / `debug_sse_sample_rate` / `debug_sse_throttle_sec` 做 SSE 调试日志采样与节流
- 工具调用批次执行
- 错误分类与可恢复重试
- 为单次 tool call 生成 linked `ToolExecutionContext`，并在 tool timeout / run cancel 时先触发 linked cancellation token

### 8.2 AsyncCliRunner

定位：
- 外部 CLI runner，当前主要服务 Codex CLI 一类本地命令行模型

负责：
- 异步 subprocess 执行
- JSONL 事件解析
- system role 写入工作目录 `AGENTS.md`

约束：
- 不支持 tool calling

## 9. 扩展方式

### 9.1 新增 Runner

需要满足：
- 实现 `AsyncRunner` 协议
- 产出稳定 `StreamEvent`
- 正确处理 `trace_context`

### 9.2 新增工具

推荐路径：
- 在 Host / Service 显式装配的工具注册模块里向 `ToolRegistry` 注入 schema
- 工具返回值遵守统一结果信封
- 建议在 `@tool()` 装饰器中声明 `display_name`（中文展示名）和 `summary_params`（摘要参数列表），使工具在 CLI / Web 进度展示中自动获得可读描述；不填时 fallback 到英文 `name`，已有工具行为不变

### 9.3 修改事件流

任何对事件顺序、字段或语义的修改，都必须同步更新：
- `tests/engine/test_async_agent.py`
- `tests/engine/test_async_openai_runner*.py`
- `tests/engine/test_context_budget.py`

## 10. 代码阅读顺序

推荐阅读顺序：

1. [async_agent.py](async_agent.py)
2. [protocols.py](protocols.py)
3. [events.py](events.py)
4. [async_openai_runner.py](async_openai_runner.py)
5. [async_cli_runner.py](async_cli_runner.py)
6. [tool_registry.py](tool_registry.py)
7. [tool_trace.py](tool_trace.py)
8. [cancellation.py](cancellation.py)
