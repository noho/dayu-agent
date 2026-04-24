# Dayu 开发手册总览

`dayu/` 是 Dayu 的开发入口。本文档不复述包内实现细节，而是面向两类开发者说明当前系统的稳定边界、数据流转和扩展方式：

- 上层接入开发者：编写 `CLI / GUI / Web / FastAPI / WeChat` 适配层的开发者
- 扩展开发者：新增 `Service` 或工具的开发者

本文档以当前代码为准，只写：

- 设计目标
- 总体架构与主链时序
- 模块边界
- 核心契约与数据流转
- Host、多轮会话、Scene 机制
- 常见扩展入口

相关文档：

- [../docs/architect.md](../docs/architect.md)
- [engine/README.md](engine/README.md)
- [fins/README.md](fins/README.md)
- [config/README.md](config/README.md)

## 0.1 开发环境安装

开发环境以 Python 3.11 为基准。源码安装只面向开发者，不作为最终用户官方交付路径。建议直接使用受控 constraints 安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,dev,browser]" -c constraints/lock-macos-arm64-py311.txt
```

说明：

- macOS Intel 开发环境改用 `constraints/lock-macos-x64-py311.txt`
- Linux 开发环境改用 `constraints/lock-linux-x64-py311.txt`
- Windows 开发环境改用 `constraints/lock-windows-x64-py311.txt`


如果为了专门验证最低支持边界，请使用：
```bash
pip install -e ".[test,dev,browser]" -c constraints/min-py311.txt
```

浏览器回退抓取是开发环境默认必备能力，因为它会直接影响 `web tools` 中 `fetch_web_page` 的表现；完成依赖安装后，还需要执行：

```bash
playwright install chromium
```

如需 PDF 渲染，还需要安装 `pandoc`。此外，渲染 HTML / PDF 仍建议安装 Google Chrome：

- macOS：`brew install pandoc`
- Ubuntu / Debian：`sudo apt-get install pandoc`
- Windows：`choco install pandoc` 或从 [pandoc 官网](https://pandoc.org/installing.html) 下载安装

## 0. 如果你想参与项目
- 定性分析模板 读起来机械感还很强，还没写出差异化：
  - 同一章节里，不同行业公司写出明显不同的判断路径。
  - 同一行业里，不同公司写出公司自己的特殊结构变量。
- 位于 Engine 的 web tools 现在的对抗challenge能力很弱，很多网站无法访问。
- 位于 Fins 的港股、A股财报下载功能尚未实现。
- GUI 尚未实现；Web UI 目前仍只有 FastAPI 骨架。
- WeChat UI 仅支持文本消息首版，还可添加更多好玩的功能。
- 财报电话会议记录音频转录文字后信息提取（起码要区分信息来自提问还是回答）尚未实现。
- 财报presentation信息提取尚未实现。
- 欢迎围绕以下方向提交 issue 或 PR：
  - 普通文件（非财报文件）信息提取还需要优化。
  - 优化 Fins 里的港股/A股/美股财报信息提取。
  - Anthropic 原生 API 支持。
  - Durable memory / Retrieval layer（ Memory只实现了working memory 和 episode summary ）。
  - FMP 工具（调研工作已做，见 [../docs/fmp_integration_research.md](../docs/fmp_integration_research.md) ）尚未实现。
  - 更多LLM 工具。

## 1. 设计目标

Dayu 的设计目标来自 [architect.md](../docs/architect.md) 的“设计目标”，当前可以压缩为四点：

- Dayu 采用“宿主强约束下的 `LLM in the loop`”，而不是把系统做成 `LLM on the loop` 的工作流编排器。
- 宿主必须显式拥有执行生命周期、资源边界和治理能力，LLM 只在宿主给定的边界内执行消息交互。
- 同一套宿主治理能力需要同时支撑普通 Agent 与金融专门 Agent。
- 系统设计目标是“稳定托管 Agent 运行”，而不是“把一次推理请求尽快发出去”。

因此，后文所有边界都服务于同一个北极星：

- 让 `Service` 只做业务解释
- 让 `Host` 只做托管执行
- 让 `Agent` 只做消息交互
- 让这三者之间通过稳定契约协同，而不是通过隐藏装配逻辑耦合

## 2. 总体架构

Dayu 当前稳定架构只有四层：

```mermaid
flowchart LR
    UI[UI] --> Service[Service]
    Service --> Host[Host]
    Host --> Agent[Agent]
```

如果把执行过程展开，当前代码里的实际链路是：

```mermaid
flowchart LR
    UI[UI] --> Startup[startup preparation]
    Startup --> Service[Service]
    Service --> Contract[Contract preparation]
    Contract --> Host[Host]
    Host --> Scene[scene preparation]
    Scene --> Agent[Agent]
```

这里要区分“层次”和“装配过程”：

- `UI -> Service -> Host -> Agent` 是稳定分层。
- `startup preparation` 不是新层，它是 `UI` 在启动期使用的 public 模块。
- `Contract preparation` 不是新层，它是 `Service` 内部使用的 public 模块。
- `prompting/` 不是新层，它是 `Host / Service` 复用的 prompt 渲染与装配公共模块。
- `scene preparation` 不是新层，它是 `Host` 内部使用的 public 模块。

几个关键判断：

- `UI` 决定调用哪个 `Service`，并把结果渲染给宿主用户。
- `UI` 可以在启动期一次准备稳定依赖，但不应该为一次 CLI 请求预先创建所有 `Service`；当前 CLI 按命令分支惰性创建所需 `Service`。
- `startup preparation` 只把启动期原始来源收敛成稳定依赖。
- `Service` 决定“这次业务上要做什么”。
- `Contract preparation` 只把 Service 已做出的决策收敛成 `Execution Contract`。
- `Host` 决定“这次执行如何被托管、追踪、取消、恢复和治理”。
- `Host` 默认内部子组件的装配权属于 `Host` 自己，而不属于 `UI`。
- prompt 模板条件块解析与变量替换归 `prompting/`，不归 `Engine`。
- `scene preparation` 只把 `Execution Contract` 与 Host 自有状态收敛成 `AgentInput`。
- `Agent` 不理解业务语义，只执行已经准备好的消息交互。

### 2.1 组件简要说明

- `UI`
  - 负责接入宿主入口，例如 `CLI / Web / FastAPI / WeChat`
  - 在启动期通过 `startup preparation` 拿稳定依赖
  - `dayu.cli` 当前固定拆成三层：`arg_parsing.py` 只负责参数定义，`main.py` 只负责顶层命令分发，`commands/` 负责各子命令执行；CLI 共享运行时装配真源继续集中在 `dependency_setup.py`
  - `dayu.wechat` 当前也固定拆成四层：`arg_parsing.py` 只负责参数定义与上下文解析，`runtime.py` 只负责 WeChat 运行时装配与 service helper，`commands/` 负责 `login / run / service` 子命令执行，`main.py` 只负责顶层分发
  - 调用 `dayu.services.startup_preparation` / `dayu.host.startup_preparation` 暴露的启动期 public API，收敛 `Host` 级稳定依赖
  - 不复制 `Host` 装配链，也不显式构造 `SQLiteSessionRegistry`、`SQLiteRunRegistry`、`SQLiteConcurrencyGovernor`、`DefaultScenePreparer`、`DefaultHostExecutor`
  - 显式 `new Service(...)`
  - 宿主管理类 UI 命令也只消费窄 `Service`（如 `HostAdminService`），不在请求期直接调用 `Host` 方法
  - interactive / web / wechat 这类 UI 适配层只消费各自稳定 `ServiceProtocol` 已声明的方法，不保留 `hasattr` 兼容分支去探测旧接口；对多轮 Chat 入口，CLI interactive、Web 和 WeChat 统一只走 `submit_turn()` / `list_resumable_pending_turns()` / `resume_pending_turn()` 这组公开契约
  - Host 事件订阅只依赖稳定事件包络，而不是把事件总线钉死为某个具体业务事件类；因此管理面 / SSE 可以同时转发 `AppEvent` 与 direct operation 的流式事件；当 direct operation 事件自带 `command` 判别字段时，SSE 也必须一并透传，不能压扁成只有 `type/payload` 的旧 `AppEvent` 形状
  - 对需要可靠出站交付的渠道路径，可显式 `new ReplyDeliveryService(...)`
  - 只为当前请求路径创建所需 `Service`，不维护覆盖所有命令的大型 runtime bundle
  - 在请求期只向 `Service` 传 `Request DTO`
- `Service`
  - 是唯一允许理解业务语义的一层
  - 把请求解释成一个明确业务动作
  - 产出 `Execution Contract`，再把它交给 `Host`
  - 在拿到 `Host` 终态结果后，决定当前路径是否显式使用 `Host` 的 `reply outbox` 能力
  - 只能依赖 `Host` 暴露的稳定能力协议与对外接口（public API），不能直接读取 `executor / session_registry / run_registry / concurrency_governor` 这类内部子组件
- `Host`
  - 是通用托管执行层，不是“Agent 专属壳”
  - 同时托管 Agent 子执行和 direct operation
  - 拥有 session、run、并发、取消、事件发布、多轮会话状态
  - 可选托管 `reply outbox` 真源与状态机，但不会在 internal success 时自动把 answer 写入 outbox
- `Agent`
  - 只关心 messages、工具、预算、取消信号和 trace 上下文
  - 不理解 `ticker`、场景语义、配置文件结构或业务流程

### 2.2 `dayu.cli prompt` 时序图

`dayu.cli prompt` 是最简单的一条 Agent 路径。当前时序如下：

```mermaid
sequenceDiagram
    participant UI as CLI
    participant Startup as startup preparation
    participant Service as PromptService
    participant Contract as Contract preparation
    participant Host as Host
    participant SP as scene preparation
    participant Agent as AsyncAgent

    UI->>Startup: resolve_startup_paths / ConfigLoader / prepare_host_runtime_dependencies(...)
    Startup-->>UI: workspace/model/prompt/default execution options/Host 稳定依赖
    UI->>Service: new PromptService(...)
    UI->>Service: submit(PromptRequest(user_text, ticker, session_id?, session_resolution_policy, execution_options))
    Service->>Host: resolve session(create_session() / touch_session())
    Service-->>UI: PromptSubmission(session_id, event_stream)
    Service->>Service: SceneExecutionAcceptancePreparer.prepare(...)
    Service->>Contract: prepare_execution_contract(...)
    Contract-->>Service: ExecutionContract
    Service->>Host: run_agent_stream(execution_contract)
    Host->>Host: register run / cancel bridge / deadline watcher / concurrency
    Host->>SP: prepare(execution_contract, run_context)
    SP->>SP: 读取 scene definition / 组 tools / prompt / messages
    SP->>SP: 基于 accepted_execution_spec 装配 agent_create_args
    Host->>Agent: build_async_agent(...) + run_messages(...)
    Agent-->>Host: StreamEvent
    Host-->>Service: AppEvent
    Service-->>UI: AsyncIterator[AppEvent]
```

这条链路里最重要的事实是：

- `ticker` 不会直接传给 `Agent`
- `ticker` 也不会进入 `Host Session` / `Host Run` 的结构化字段
- `model_name` 也不会原样传给 `Agent`
- `Service` 先把它们解释或收敛，再通过 `Execution Contract` 交给 `Host`
- `prompting/` 在 `Host / Service` 侧完成 prompt 渲染，`Engine` 只消费最终可执行的 prompt 文本

## 3. 模块边界

### 3.1 UI

`UI` 是 composition root。

它负责：

- 选择调用哪个 `Service`
- 通过 `startup preparation` 拿到稳定依赖
- 显式 `new Service(...)`
- 作为 composition root，把窄依赖显式注入各个 Web router / CLI 命令入口
- 在请求期构造 `Request DTO`
- 观察并渲染 `Service` 返回的事件流或结果

它不负责：

- 解释业务语义
- 直接驱动 `Agent`
- 解析 prompt scene
- 管理 run 生命周期

### 3.2 Service

`Service` 是唯一允许理解业务语义的一层。

它负责：

- 解释用户请求与领域参数
- 决定本次请求应该走哪条业务路径
- 决定使用哪个 `scene`
- 生成 `Prompt Contributions`
- 基于启动期稳定依赖与当前请求收敛出 `Execution Contract`
- 把 `Execution Contract` 提交给 `Host`
- 仅通过 `Host` 暴露的稳定协议与对外接口交互，不依赖 `Host` 具体实现或内部子组件属性

它不负责：

- 自己管理 run、session、取消、并发
- 自己拼最终 `messages`
- 自己构造 `AsyncAgent`
- 让 `Agent` 理解业务参数

### 3.3 Host

`Host` 是通用托管执行层，承担 Session / Run / 并发治理 / 事件发布 / timeout / cancel / resume / 多轮会话托管 / reply outbox 九项能力，把 `Execution Contract` 收敛为 `Agent` 可执行输入。

本总览只给出能力边界；Host 的能力契约、状态机、启动恢复顺序、清理分支等机制细节全部收口在 [host/README.md](host/README.md)，总览不重复。

UI / Service 的**消费者视角使用指南**（调用序、稳定接口、必须处理的错误、禁止越界动作、装配/测试约束）见 [host/README.md §14 作为 Host 的上游：UI / Service 使用指南](host/README.md#14-作为-host-的上游uiservice-使用指南)。

它不负责：

- 理解 `ticker`、写作、审计、修复等业务语义
- 决定这次业务“要做什么”
- 回头理解 `Service` 私有的业务规则

### 3.4 Agent

`Agent` 是最低层消息执行器。

它负责：

- 消费最终 `messages`
- 在受限工具集合内执行工具调用
- 产出流式事件
- 记录 trace / tool trace

它不负责：

- 解释 `Request DTO`
- 解释 `Execution Contract`
- 理解 scene、ticker、文档范围、写作阶段
- 读取 `run.json`、`llm_models.json`

### 3.5 startup preparation

`startup preparation` 是启动期 public surface 的统称，当前分布在 `startup/`、`dayu.services.startup_preparation` 和 `dayu.host.startup_preparation`；它只服务于启动期，不进入请求期调用链。

它负责：

- 接收 `workspace_root`、`config_root`、默认 `ExecutionOptions`
- 解析路径与配置来源
- 准备 `ConfigLoader`、`PromptAssetStore`、`WorkspaceResources`、`ModelCatalog`
- 准备默认 `ResolvedExecutionOptions`
- 准备金融领域专用 `FinsRuntime`
- 调用 `Service` 暴露的 startup preparation API，收敛 `SceneExecutionAcceptancePreparer` 与共享 Host runtime 依赖
- 调用 `Host` 暴露的 startup preparation API，收敛 `HostStore path`、`lane config`
- 支持 UI 先准备稳定依赖，再按命令分支惰性创建所需 `Service`

它不负责：

- 构造 `Host`
- 构造 `Service`
- 直接实例化 `SceneDefinitionReader`、Conversation Policy 解析器等 Service 内部实现
- 直接读取 `DEFAULT_LANE_CONFIG` 这类 Host 内部常量
- 生成 `ExecutionContract`
- 生成 `AgentInput`

### 3.6 Contract preparation

`Contract preparation` 是 `Service` 内部使用的 public 模块。

它负责：

- 接收 Service 已经做出的业务决策
- 把这些决策收敛成 `ExecutionContract`
- 准备 `host_policy`
- 准备 `ScenePreparationSpec`
- 准备 `message_inputs`
- 写入 `accepted_execution_spec`

它不负责：

- 再次解释业务语义
- 组装最终 `ToolRegistry`
- 组装最终 `system_prompt` / `messages`
- 构造 `AsyncAgent`

### 3.7 scene preparation

`scene preparation` 是 `Host` 内部使用的 public 模块。

它负责：

- 按 `scene_name` 加载 scene 定义
- 按 scene manifest 的 `conversation.enabled` 判断是否进入多轮 transcript / memory 组装
- 基于 scene manifest、`selected_toolsets`、`execution_permissions` 组装最终工具集合
- 基于 `accepted_execution_spec` 与模型目录装配 `agent_create_args`
- 组装 `system_prompt`
- 组装单轮或多轮 `messages`
- 维护会话 transcript、memory 和 trace 上下文
- 返回 `AgentInput`

它不负责：

- 接受或拒绝 execution options
- 决定 scene
- 决定 prompt contributions
- 创建 Session 或 Run

### 3.8 workspace migrations

`workspace migrations` 是 `dayu-cli init` 在启动期针对**旧工作区**执行的一次性修复脚本集合，集中在 `dayu/cli/workspace_migrations/`。它与 §3.5 `startup preparation` 并列——都只服务于启动期、不进入请求期调用链——但职责是一次性的"把旧工作区就地升级到当前 schema"，而不是"为本次运行准备依赖"。

它负责：

- 向后扫描 `workspace/config/run.json`，在缺少新 schema 要求的 key 时补齐默认值，已有取值一律保留
- 向后扫描 `.dayu/host/dayu_host.db` 中的 `pending_conversation_turns.resume_source_json`，按当前 schema 原地改名旧 JSON key
- 每条规则一个模块、一个幂等函数，通过 `apply_all_workspace_migrations` 统一调度
- 只在规则实际生效时打印一行，供用户感知

它不负责：

- 维护数据库 schema 本身（`CREATE TABLE` 仍在 `dayu.host.host_store`）
- 保留旧 schema 的兼容读取路径——规则按 CLAUDE.md "全新 schema 起库"约束，**只向前迁移**
- 进入请求期——`dayu-cli` 任何非 `init` 命令都不会触发该目录

扩展约束：新增一次性迁移时，只在 `dayu/cli/workspace_migrations/` 下新增模块 + 登记到 runner，禁止把规则写回 `dayu/cli/commands/init.py`。

## 4. 核心契约

Dayu 在 Agent 路径上稳定使用五类数据契约：

- `UI -> Service` 请求契约：`Request DTO`
- `Service -> UI` 流式输出契约：`AppEvent`
- `Service -> UI / Host -> Service` 结果契约：`AppResult`
- `Service -> Host` 执行契约：`Execution Contract`
- `Host -> Agent` 最低可执行输入：`Agent Input`

按边界分组来看：

- `UI <-> Service`
- `Service -> Host`
- `Host -> Agent`

### 4.1 `UI <-> Service`

#### 4.1.1 `Request DTO`

`Request DTO` 只回答一个问题：用户这次显式提交了什么。

典型字段包括：

- 用户输入文本
- 显式业务参数，例如 `ticker`
- 通用执行显式参数，例如 `model_name`、`temperature`、`max_iterations`

这里要区分两类参数：

- 领域显式参数
  - 例如 `ticker`
  - 只能由需要该领域语义的 `Service` 理解
- 通用执行显式参数
  - 例如 `model_name`
  - 也必须先进入 `Service`
  - 由具体 `Service` 决定是否接受、如何覆盖默认策略

当前 `Request DTO` 族的 schema 如下：

```python
@dataclass(frozen=True)
class ChatTurnRequest:
    user_text: str
    session_id: str | None = None
    ticker: str | None = None
    execution_options: ExecutionOptions | None = None
    scene_name: str | None = None
    session_resolution_policy: SessionResolutionPolicy = SessionResolutionPolicy.AUTO


@dataclass(frozen=True)
class PromptRequest:
    user_text: str
    ticker: str | None = None
    session_id: str | None = None
    execution_options: ExecutionOptions | None = None
    session_resolution_policy: SessionResolutionPolicy = SessionResolutionPolicy.AUTO


@dataclass(frozen=True)
class WriteRequest:
    write_config: WriteRunConfig
    execution_options: ExecutionOptions | None = None
```

其中：

- `session_id` 在 `chat` / `prompt` 请求里是可选的
- 首轮请求可不传 `session_id`，由 `Service` 在内部创建 Host session
- `Service.submit_*()` 必须先完成请求级同步校验，再决定是否创建 Host session；像空输入、非法 scene、direct operation 的空 `ticker` 这类客户端错误，必须在返回 submission 句柄前失败，不能先受理再让后台任务首轮消费时报错
- `session_resolution_policy` 只声明“这次请求希望怎样解析 session”
- `source` 不再作为请求字段暴露；它是 `Service` 构造时持有的固定上下文，用于写入 Host session provenance
- `chat` / `prompt` / `fins` 对 UI 返回的是 `*Submission(session_id, event_stream|execution)` 句柄，而不是让 UI 先去创建 Host session

其中最常被 UI 显式覆盖的 `ExecutionOptions` schema 如下：

```python
@dataclass(frozen=True)
class ExecutionOptions:
    model_name: str | None = None
    temperature: float | None = None
    debug_sse: bool = False
    debug_tool_delta: bool = False
    debug_sse_sample_rate: float | None = None
    debug_sse_throttle_sec: float | None = None
  tool_timeout_seconds: float | None = None
    max_iterations: int | None = None
    fallback_mode: str | None = None
    fallback_prompt: str | None = None
    max_consecutive_failed_tool_batches: int | None = None
    max_duplicate_tool_calls: int | None = None
    duplicate_tool_hint_prompt: str | None = None
    web_provider: str | None = None
    trace_enabled: bool | None = None
    trace_output_dir: Path | None = None
    toolset_configs: tuple[ToolsetConfigSnapshot, ...] = ()
    toolset_config_overrides: tuple[ToolsetConfigSnapshot, ...] = ()
    doc_tool_limits: DocToolLimits | None = None
    fins_tool_limits: FinsToolLimits | None = None
    web_tools_config: WebToolsConfig | None = None
```

  其中 `toolset_configs` 是唯一的跨层工具配置真源，`toolset_config_overrides` 是请求层对该真源的稀疏覆盖表达；`doc_tool_limits / fins_tool_limits / web_tools_config` 之所以仍出现在 `ExecutionOptions` 上，只是为了保留请求入口的人体工学简写。它们在 `ExecutionOptions` 构造时就会立即被收敛进 `toolset_configs`，后续 `Service -> Host -> snapshot` 链路不再把这些旧字段当作独立跨层契约继续传播。

#### 4.1.2 `AppEvent`

`AppEvent` 是 `Service -> UI` 的流式事件契约。

它回答的问题是：

- 这次执行产生了什么增量
- 当前是内容、推理、工具、告警还是错误
- UI 应如何逐步渲染本轮输出

它的 schema 如下：

```python
class AppEventType(Enum):
    CONTENT_DELTA = "content_delta"
    REASONING_DELTA = "reasoning_delta"
    FINAL_ANSWER = "final_answer"
    TOOL_EVENT = "tool_event"
    WARNING = "warning"
    ERROR = "error"
    METADATA = "metadata"
    DONE = "done"


@dataclass
class AppEvent:
    type: AppEventType
    payload: Any
    meta: dict[str, Any] = field(default_factory=dict)
```

`Service` 对 `UI` 的典型返回类型不是单个 `AppEvent`，而是：

```python
AsyncIterator[AppEvent]
```

对需要先拿到会话句柄再异步消费事件的 UI（例如 Web SSE、CLI 多轮），`Service` 返回的稳定 public contract 是：

```python
@dataclass(frozen=True)
class ChatTurnSubmission:
    session_id: str
    event_stream: AsyncIterator[AppEvent]


@dataclass(frozen=True)
class PromptSubmission:
    session_id: str
    event_stream: AsyncIterator[AppEvent]
```

#### 4.1.3 `AppResult`

`AppResult` 是聚合后的结果契约。

它主要用于：

- `Host -> Service` 的 `run_agent_and_wait(...)`
- 写作 pipeline 等需要一次性消费最终结果的场景
- 不适合逐 token 渲染的同步收口路径

它的 schema 如下：

```python
@dataclass
class AppResult:
    content: str
    errors: list[str]
    warnings: list[str]
    degraded: bool = False
```

### 4.2 `Service -> Host`

#### 4.2.1 `Execution Contract`

`Execution Contract` 是 `Service -> Host` 的执行决策。

它回答的问题是：

- 这次执行要跑哪个 `scene`
- 宿主应如何托管这次执行
- scene preparation 需要哪些已解析的装配信息
- Host 应基于哪些已接受执行规格继续机械装配

完整 schema 如下：

```python
@dataclass(frozen=True)
class ExecutionWebPermissions:
    allow_private_network_url: bool = False


@dataclass(frozen=True)
class ExecutionDocPermissions:
    allowed_read_paths: tuple[str, ...] = ()
    allow_file_write: bool = False
    allowed_write_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionPermissions:
    web: ExecutionWebPermissions = field(default_factory=ExecutionWebPermissions)
    doc: ExecutionDocPermissions = field(default_factory=ExecutionDocPermissions)


@dataclass(frozen=True)
class ScenePreparationSpec:
    selected_toolsets: tuple[str, ...] = ()
    execution_permissions: ExecutionPermissions = field(default_factory=ExecutionPermissions)
    prompt_contributions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AcceptedExecutionSpec:
  model: AcceptedModelSpec
  runtime: AcceptedRuntimeSpec = field(default_factory=AcceptedRuntimeSpec)
  tools: AcceptedToolConfigSpec = field(default_factory=AcceptedToolConfigSpec)
  infrastructure: AcceptedInfrastructureSpec = field(default_factory=AcceptedInfrastructureSpec)


@dataclass(frozen=True)
class AcceptedModelSpec:
  model_name: str
  temperature: float | None = None


@dataclass(frozen=True)
class AcceptedRuntimeSpec:
  runner_running_config: dict[str, Any] = field(default_factory=dict)
  agent_running_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcceptedToolConfigSpec:
  toolset_configs: tuple[ToolsetConfigSnapshot, ...] = ()


@dataclass(frozen=True)
class AcceptedInfrastructureSpec:
  trace_settings: Any | None = None
  conversation_memory_settings: Any | None = None


@dataclass(frozen=True)
class ExecutionHostPolicy:
    session_key: str | None = None
    business_concurrency_lane: str | None = None
    timeout_ms: int | None = None
    resumable: bool = False


@dataclass(frozen=True)
class ExecutionMessageInputs:
    user_message: str | None = None

@dataclass(frozen=True)
class ExecutionContract:
    service_name: str
    scene_name: str
    host_policy: ExecutionHostPolicy
    preparation_spec: ScenePreparationSpec
    message_inputs: ExecutionMessageInputs
    accepted_execution_spec: AcceptedExecutionSpec
    execution_options: Any | None = None
    metadata: ExecutionDeliveryContext = field(default_factory=empty_execution_delivery_context)
```

  `AcceptedToolConfigSpec` 现在只把通用 `toolset_configs` 作为稳定真源；Host / scene preparation 不再继续传递专用工具配置字段。

其中：

- `host_policy`
  - 描述 Host Session、并发 lane、超时、是否可恢复
  - `timeout_ms=None` 表示该参数已被 Service 受理，但当前不启用 deadline；真正的超时取消语义由 Host 执行
  - 对 `conversation.enabled=true` 的 scene，Service 当前默认写入 `resumable=True`；Host 仍会在 scene gate 上做最终校验
- `preparation_spec`
  - 描述 scene preparation 需要的机械装配信息
- `message_inputs.user_message`
  - 当前轮用户输入
- `accepted_execution_spec`
  - 描述 Service 已接受的四组执行结果：模型选择、运行时快照、工具域配置、基础设施配置
  - 其中工具域配置以 `accepted_execution_spec.tools.toolset_configs` 为真源；若某个 toolset 需要专用配置对象，由对应 registrar adapter 在包内自行反解
  - 如果 web toolset config 中声明了 `allow_private_network_url`，`Contract preparation` 仍必须显式同步写入 `execution_permissions.web.allow_private_network_url`；动态权限收窄以 `execution_permissions` 为准，不能把它继续表述成 `AcceptedToolConfigSpec` 上的独立旧字段

### 4.3 `Host -> Agent`

#### 4.3.1 `Agent Input`

`Agent Input` 是 `Host` 内部的最低可执行输入。

它回答的问题是：

- 送给 Agent 的最终 prompt 和 messages 是什么
- 这次执行最终能用哪些工具
- Agent 应按什么运行参数被创建
- Host 要把哪些会话状态、trace 与取消上下文一起交给 Agent

完整 schema 如下：

```python
@dataclass(frozen=True)
class AgentCreateArgs:
    runner_type: str
    model_name: str
    max_turns: int | None = None
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    runner_params: dict[str, Any] = field(default_factory=dict)
    runner_running_config: dict[str, Any] = field(default_factory=dict)
    agent_running_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentInput:
    system_prompt: str
    messages: list[dict[str, Any]]
    tools: Any | None = None
    agent_create_args: AgentCreateArgs = field(
        default_factory=lambda: AgentCreateArgs(runner_type="", model_name="")
    )
    session_state: Any | None = None
    runtime_limits: dict[str, Any] = field(default_factory=dict)
    cancellation_handle: Any | None = None
    tool_trace_recorder_factory: Any | None = None
    trace_identity: dict[str, Any] | None = None
```

它只在 `Host / scene preparation` 内部流转，不作为 Engine 对外公共接口。

## 5. 数据流转重点

### 5.1 `--ticker` 怎么“传到” LLM

`--ticker` 不会原样传给 `Agent`。当前稳定链路是：

```text
CLI/Web/WeChat
-> Request DTO.ticker
-> Service 解释 ticker 的业务含义
-> Service 生成 Prompt Contributions["fins_default_subject"]
-> scene preparation 按 scene context slot 追加到 system_prompt 尾部
-> Agent 只看到最终 system_prompt
```

对 `Agent` 而言，不存在“收到一个 ticker 参数”这件事。它只收到已经降解后的 prompt 文本。
对 `Host` 而言，也不存在“结构化保存一个 ticker 字段”这件事；`ticker` 的跨轮保持与恢复属于上层自己的状态职责。

### 5.2 `--model-name` 怎么传到 Agent

`--model-name` 也不会原样传给 `Agent`。

当前稳定链路是：

```text
UI 显式参数
-> Request DTO.execution_options.model_name
-> Service 调用 SceneExecutionAcceptancePreparer
-> Service 产出 accepted_execution_spec
-> Host 在 scene preparation 内部机械装配 agent_create_args
-> Host 使用 agent_create_args 构造 AsyncAgent / AsyncRunner
```

因此：

- `--ticker` 最终降解成 prompt 文本
- `--model-name` 最终降解成 `accepted_execution_spec`，再由 Host 落成 `AgentCreateArgs`

两者都不会以原始请求字段形态进入 `Agent`

### 5.3 Fins 两类 dispatch：ticker → pipeline 与 document → processor

Fins 包里存在两条互不相同的 dispatch：一条决定**长事务走哪条 pipeline**（download / upload / process），另一条决定**读取某份文档时用哪个 processor**（tool calling 与 pipeline 内部都会触发）。它们的键、真源、消费者都不同，不能互相代替。

#### 5.3.1 ticker → pipeline（按 market 分派）

进入点有两条：

- **CLI `--ticker`**（`dayu.cli` 的 `download / upload_filings_from / upload_filing / upload_material / process / process_filing / process_material`）：`cli_support.py::_build_pipeline_for_ticker` → `normalize_ticker(ticker)` → `get_pipeline_from_normalized_ticker(...)`。
- **Tool calling 触发的长事务**（LLM 调用 ingestion 工具由 `IngestionJobManager` 调度）：`toolset_registrars.py` → `FinsRuntime.build_ingestion_service_factory()` 返回 `ticker -> FinsIngestionService` 闭包；闭包内部对每个 job 的 `ticker` 执行 `normalize_ticker`，再通过 `dayu/fins/pipelines/factory.py::build_ingestion_service_from_normalized_ticker` 拿到对应 pipeline 的长事务服务。

两条路径最终落在同一条分派规则上：

```text
NormalizedTicker.market == "US"            → SecPipeline
NormalizedTicker.market in {"HK", "CN"}    → CnPipeline
其它                                        → ValueError("不支持的 market: …")
```

分派真源是 `dayu/fins/pipelines/factory.py::get_pipeline_from_normalized_ticker`。CLI 路径直接复用该真源；tool calling 路径通过 `build_ingestion_service_from_normalized_ticker` 间接复用同一真源，不再维护独立的 `if US … if CN/HK … else raise` 分支，新增 market 时只需改一处。

关键事实：

- pipeline 分派只看 `NormalizedTicker.market`，不看 ticker 字面量或字母数字模式。
- 同一 `FinsRuntime` 实例里，CLI 与 tool 两路共享同一份 `ProcessorRegistry` 与仓储实例，pipeline 构造时把它们逐项注入。
- `SecPipeline` 已不再接受识别器注入参数；ticker 的市场识别统一由 `dayu/fins/ticker_normalization.py::normalize_ticker` 真源完成。

#### 5.3.2 document → processor（按 source + form_type + media_type 分派）

这条 dispatch 发生在**读取某份已入库文档**时，由 `FinsToolService._create_processor_for_document(ticker, document_id)`（tool calling 路径）或 pipeline 内部 `resolve_processor_*`（长事务路径）触发，调用的都是同一张 `ProcessorRegistry`。

触发链路（tool 视角）：

```text
LLM tool call
  → FinsToolService._create_processor_for_document(ticker, document_id)
  → SourceRepository.get_primary_source(ticker, doc_id, source_kind)   ← 拿到 Source 抽象
  → SourceRepository.get_source_meta(...)                              ← 读 form_type
  → ProcessorRegistry.create_with_fallback(
        source=source,
        form_type=form_type,
        media_type=source.media_type,
    )
```

分派由三键组合决定：

- `source`：仓储提供的 `Source` 抽象，其 `supports()` 判定 SEC filing 结构、本地文件、扩展名等。
- `form_type`：`source_meta["form_type"]`，驱动 SEC 表单专项处理器（10-K / 10-Q / 20-F / 8-K / 6-K / SC13 / DEF 14A）。
- `media_type`：`source.media_type`，驱动 Docling（PDF）/ Markdown / BS（HTML）等通用处理器。

`ProcessorRegistry.resolve_candidates()` 遍历全部注册项调用 `processor_cls.supports(source, form_type, media_type)`，按 `priority` 降序产出候选；`create_with_fallback` 取第一个实例化成功的候选，**构造器抛异常**时自动回退到下一候选并触发 `on_fallback` 回调。

`build_fins_processor_registry()` 当前的优先级锚点：

| Priority | 角色 | 代表处理器 |
|---:|---|---|
| 200 | SEC 表单 BS 主路径 | `BsSc13FormProcessor` / `BsSixKFormProcessor` / `BsDef14AFormProcessor` / `BsEightKFormProcessor` / `BsTenKFormProcessor` / `BsTenQFormProcessor` / `BsTwentyFFormProcessor` |
| 190 | SEC 表单 edgartools 回退 | `Sc13FormProcessor` / `Def14AFormProcessor` / `EightKFormProcessor` / `TenKFormProcessor` / `TenQFormProcessor` / `TwentyFFormProcessor` |
| 120 | SEC 通用兜底 | `SecProcessor` |
| 100 | 文档格式通用 | `FinsDoclingProcessor`（PDF）、`FinsMarkdownProcessor`（Markdown） |
| 80 | HTML 通用 | `FinsBSProcessor` |
| —  | engine 基座 | `build_engine_processor_registry()` 注入的通用处理器 |

关键事实：

- processor 分派**不看 market、不看 ticker**；CN / HK 文档当前直接走通用 Docling / Markdown / BS 处理器，没有 CN 特化专项处理器——若未来需要 A 股年报专项路径，应在注册表中新增按 form_type + source 判定的注册项，而不是回退到按 market 硬分支。
- tool calling 路径**不经过 `FinsIngestionService` / `SecPipeline`**；它直接复用共享 `ProcessorRegistry`。因此 tool 路径读取文档时不会触发 pipeline 分派，但**会**受 processor 分派规则的全量约束。
- 同一 `FinsRuntime` 内 tool 与 pipeline **共享同一份 `ProcessorRegistry`**（在 `DefaultFinsRuntime.create` 时 `build_fins_processor_registry()` 一次构建），注册顺序与优先级在 runtime 初始化时锁定。
- 若仓储 `form_type` 字段缺失，SEC 专项处理器会在 `supports()` 阶段全部拒绝，实际命中 `SecProcessor(120)` 或更低优先级的通用处理器——这是当前分派的一个已知脆弱点，依赖 ingestion 期正确写入 `form_type`。

## 6. Host

Host 是 Dayu 的通用托管执行层。它的价值不在于“帮 Service 调一下 Agent”，而在于把一次执行真正收敛成可治理、可观察、可取消、可恢复、可补投递的宿主能力。

当前 Host 稳定能力可归纳为：

- Session 能力：统一长期会话身份。
- Run 能力：统一一次执行尝试的生命周期。
- 并发治理能力：为不同 lane 提供限流与资源隔离。
- 事件发布能力：把执行事件稳定暴露给 UI。
- timeout 能力：在宿主侧收敛 deadline，而不是让 Agent 自己猜超时。
- cancel 能力：把“取消意图”和“取消终态”分离并统一收口。
- resume 能力：把 conversation turn 恢复真源固定在 Host。
- 多轮会话托管能力：统一 transcript、memory、compaction 调度。
- reply outbox 能力：把出站交付真源作为可选宿主能力托管。

本章只给出上述能力的**对外边界**：`Service` 只依赖 Host public API 与协议契约，不构造内部子组件（session/run registry、并发治理器、pending turn / reply outbox 仓储、event bus、默认执行器等均由 Host 自行拥有）。

### 6.1 三套真源边界

聊天主链简化为：`User -> 渠道 UI -> Service -> Host -> Service -> 渠道 UI -> User`。其中三套真源必须保持分离：

- **入站交付真源**：回答“用户消息是否已经被系统可靠拿到”，属于 `UI` / 渠道适配层。
- **执行真源**：回答“这条请求在 Host 内是否还未完成、是否可恢复、是否已经 internal success”，属于 `Host`。
- **出站交付真源**：回答“final answer 是否已经可靠送达用户”，可以作为 `Host` 的一项可选通用能力被托管。

resume 属于执行真源；reply outbox 属于出站交付真源。Host internal success 不会自动入 outbox，是否补投递由 `Service` 显式决定（例如 `ReplyDeliveryService`）。

### 6.2 能力机制

Host 九项能力的具体机制（包括 Session / Run 状态机、pending turn 与 reply outbox 状态机与 CAS 契约、并发治理 lane 合并顺序、cancel 两层语义、启动恢复顺序、分层记忆裁剪与 compaction 乐观锁等）统一收口到 [host/README.md](host/README.md)。本总览不复述。

### 6.3 sub-agent 扩展边界

当前第一版没有 `parent_run_id`。

如果后续需要支持 sub-agent / child-run，扩展点应放在 Host 侧，而不在 `Execution Contract` 或 `Agent Input`：

- 在 Host Run 上增加 lineage 关系
- 让一个 Host Run 可以托管多个子 run
- 保持 `Service -> Host -> Agent` 主链不变

这样可以避免让 `Agent` 反向感知更高层的编排结构。

## 7. Scene

Scene 是 `Host` 托管链路中的声明式执行策略，不是业务解释层。

Scene 当前负责声明：

- 这个场景的 prompt 片段装配计划
- 允许哪些 context slot
- 默认模型与 allowlist
- 默认 temperature profile
- 默认最大迭代次数
- 工具候选集合策略

### 7.1 Prompt Contributions

动态 prompt 片段当前统一通过 `Prompt Contributions` 机制进入 scene。

规则固定为：

- `Service` 负责提供动态片段文本
- scene manifest 用 `context_slots` 声明允许的 slot
- `Service` 在 contract preparation 阶段按 `context_slots` 收口为当前 scene 的 exact set
- `Host` 只机械消费已收口的结果；若仍收到未声明 slot，只打 warning 并忽略
- system prompt 尾部的动态片段顺序始终由 `context_slots` 决定

当前公共 slot 主要有：

- `base_user`
- `fins_default_subject`

### 7.2 工具装配

最终工具集合由三层约束求交得到：

- scene manifest 的工具候选集合
- `selected_toolsets`
- `execution_permissions`

当前可以把它理解为：

- scene manifest 决定“哪些工具有资格进入候选集合”
- `selected_toolsets` 决定“这次执行实际启用哪些工具集合”
- `execution_permissions` 决定“这些工具在本次执行中拿到哪些动态权限”

这三层语义求交之后，`Host` 只做一件机械动作：按 `toolset_name -> registrar import path` 的安装清单加载对应 registrar，并把通用注册上下文交给它。当前安装清单来自 `toolset_registrars.json`，它不是第四层执行决策，也不负责放宽 scene / Service 已经收窄过的工具集合。

这里的“通用注册上下文”已经进一步收紧为：`Host` 只把当前 `toolset_name` 命中的单个 `toolset_config` 快照、动态 `execution_permissions` 与工作区稳定资源交给 registrar；registrar 负责在 adapter 边界把这些通用输入适配成叶子 `register_*_tools(...)` 所需参数。共享的 limits/config 反序列化统一收口在 `dayu.contracts.tool_configs`，而像 doc 白名单解析这类 domain 规则则留在对应 toolset 的边界模块内部，不能再让 `Host` 直接解释。

因此需要明确两点：

- `toolset_registrars.json` 只回答“某个 toolset 由哪个 registrar 实现”
- 如果某个 toolset 已经被前三层约束明确启用，但安装清单里没有对应 registrar，scene preparation 必须直接失败，而不是静默跳过

### 7.3 `selected_toolsets` / `execution_permissions`

这两个字段是 `Execution Contract` 的关键扩展点。

当前代码里它们的架构定位已经固定，但启用状态不同：

- `selected_toolsets`
  - 用于让 `Service` 按业务意图选择这次实际启用哪些工具集合
  - 当前主链路显式传 `()`
- `execution_permissions`
  - 用于描述本次执行的动态权限收窄
  - 当前已经真实参与 doc/web 工具注册

这两类扩展都不应让 `Agent` 自己理解业务原因；它们只应由 `Host` 在 scene preparation 阶段落实。

## 8. 多轮会话机制

多轮会话当前采用分层记忆模型，而不是简单回放所有历史消息。

当前分层记忆模型由三个层次组成：

```text
┌──────────────────────────────────────────────────────────┐
│  Pinned State（钉住状态）                                 │
│  ↳ 不可压缩的会话级最小状态，跨 episode 保留              │
│  ↳ 内容：当前主任务、已确认对象、用户约束、未决问题        │
├──────────────────────────────────────────────────────────┤
│  Episodic Memory（阶段摘要）                              │
│  ↳ 已压缩的 episode 结构化摘要                            │
│  ↳ 由 conversation_compaction scene 调用 LLM 生成         │
│  ↳ 按 token 预算从最近向前回溯选入                        │
├──────────────────────────────────────────────────────────┤
│  Working Memory（工作记忆）                               │
│  ↳ 最近的未压缩 turn，高保真回放                          │
│  ↳ 从后向前选择，受轮数上限和 token 预算双重约束           │
├──────────────────────────────────────────────────────────┤
│  Raw Transcript（原始 transcript）                        │
│  ↳ 所有原始 turn 的完整记录，以文件系统落盘               │
│  ↳ 含用户文本、助手回复、工具调用摘要、警告、错误          │
└──────────────────────────────────────────────────────────┘
```

当前固定机制概述：当前轮用户输入由 `ExecutionContract.message_inputs.user_message` 带入，`system_prompt` 由 scene preparation 生成；Host 在本轮开始前执行同步压缩，再装配历史消息、memory block 与当前轮输入；compaction 预算从最终模型上下文窗口推导。具体裁剪规则、触发阈值、乐观锁写回、默认配置表见 [host/README.md](host/README.md)。

### 8.1 为什么多轮会话属于 Host

多轮会话涉及：

- session 持久化
- transcript 读写
- working memory 裁剪
- episode summary 压缩
- compaction 后台任务
- 取消与恢复

这些都属于托管执行能力，因此必须归 `Host`，而不是归 `Agent`。分层记忆模型（Pinned State / Episodic Memory / Working Memory / Raw Transcript）、裁剪顺序、compaction 触发条件、同步/后台压缩的乐观锁语义、配置默认值等机制细节收口于 [host/README.md](host/README.md)，总览不复述。

### 8.2 Durable Memory / Retrieval 扩展边界

当前 memory 只实现了：

- working memory
- episode summary

当前代码中 `DurableMemoryStoreProtocol` 和 `ConversationRetrievalIndexProtocol` 已定义协议接口，但使用的是空实现（`NullDurableMemoryStore` / `NullConversationRetrievalIndex`）。

如果后续要扩展：

- durable memory
- retrieval layer

扩展点仍应落在 `Host` 的会话与 scene preparation 机制中，而不是让 `Agent` 自己感知更多状态层次。稳定方向应保持为：

```text
Host Session
-> transcript / summaries / retrieval state
-> scene preparation
-> Agent messages
```

## 9. `dayu.cli interactive` 总结性时序图

`dayu.cli interactive` 是当前最能体现四层协同方式的一条路径。

当前实现里，interactive UI 会把本地会话绑定持久化在
`<workspace>/.dayu/interactive/state.json`。默认路径只保存 UI 自己拥有的
`interactive_key`，并在每次启动时把它确定性映射为 `session_id`。带
`--label` 的 CLI conversation 另走独立的 UI 层 label registry：
`<workspace>/.dayu/cli-conversations/<label>.json`。registry 只保存
`label -> session_id + scene_name` 的映射，不进入 Host schema。因此：

- 默认重启 interactive 会续接上一条多轮会话。
- 显式传 `--new-session` 时，UI 会删除旧绑定并生成新的 `interactive_key`。
- 显式传 `--label` 时，UI 会先在 label registry 中恢复或创建会话：`interactive --label` 首次创建 `scene=interactive`，`prompt --label` 首次创建 `scene=prompt_mt`；后续无论从 `prompt` 还是 `interactive` 入口恢复，都沿用 registry 已记录的 `session_id + scene_name`，不按入口覆盖 scene。
- CLI 对 labeled conversation 额外维护 label 独占锁：`prompt --label` 在本轮完成前持锁，`interactive --label` 在整个 REPL 生命周期内持锁；同一个 label 被占用时，其它 CLI 进程只能显式失败，不能并发复用同一条 conversation。
- 带 label 的 scene 必须继续满足 `conversation.enabled=true`；若 workspace 本地覆写的 manifest 关闭了多轮模式，CLI 会在 `prompt --label` / `interactive --label` 入口直接拒绝执行，而不是静默退化成伪多轮。
- 若 label registry 命中的 Host session 已经是 `closed`，CLI 会先删除旧 record，再以同名 label 重建一条全新的 conversation，并向用户打印“旧对话已关闭，现创建新对话”的提示；只有用户显式执行 `conv remove --label` 后的重新创建不会额外提示。
- 带 label 的 interactive 进入 REPL 前，CLI 会通过 `HostAdminService.list_session_recent_turns()` 读取最近一轮 conversation 摘录并打印恢复分隔提示。
- `dayu-cli conv list/status/remove` 管理的是 CLI label registry；`dayu-cli sessions --source/--scene` 管理的是 Host session。两者职责分离：前者回答“哪些 label 可恢复/释放”，后者回答“Host 当前有哪些 session”。当 label registry 指向的 Host session 已不存在时，CLI 会在 `conv` / `prompt --label` / `interactive --label` 入口先清理漂移 record，再按正常恢复或新建路径继续；`conv remove --label` 则会在 label 未被占用时关闭底层 session 并删除 registry record。
- `dayu-cli sessions` 当前统一通过 `HostAdminService.list_sessions(state/source/scene)` 列出 Host session digest，并展示 `SCENE / TURNS / OVERVIEW`；`OVERVIEW` 首版只按 `first_question -> last_question -> "-"` 选择文本。
- Host 不负责决定“上一次 interactive 是哪个 session”，它只接收 UI 已解析并校验过的 `session_id`。
- CLI / WeChat 在完成 Host runtime 装配后，会先统一执行一次 Host-owned 启动恢复：清理 orphan run（写入独立的 `RunState.UNSETTLED` 吸收态，不再与业务 `FAILED` 混用）、stale permit、以及陈旧的 `reply_outbox.DELIVERY_IN_PROGRESS`；interactive 进入 REPL 前只负责恢复上一轮 pending turn，本身不再直接持有宿主管理依赖。Owner 自愈判据是 `state == UNSETTLED && owner_pid == os.getpid()`，不再依赖 `error_summary` 字符串。
- CLI 主入口会在 Host 装配阶段一次性注册 SIGTERM / SIGHUP / `atexit` 优雅退出 hook：进程收到信号或正常退出时主动调用 `Host.shutdown_active_runs_for_owner()`，把本 pid 仍活跃的 run 主动收敛为 `CANCELLED`，避免把下次启动的 orphan 判定窗口暴露给用户。
- CLI interactive 与 WeChat daemon 都按各自 `state_dir` 持有单实例锁，避免同一 workspace / label 下两个前台进程并发共享本地状态目录。
- WeChat daemon 的自动恢复还会额外绑定当前 `state_dir` 派生的 runtime identity，避免不同 daemon 实例在相同 `scene=wechat` 下串恢复彼此的 pending turn；同时同一个 `state_dir` 只允许一个 daemon 持锁运行，只有在单实例前提下，daemon 才会把上一进程遗留的 `delivery_in_progress` reply outbox 记录回收为可重试状态，再进入 pending turn 恢复 / 补发 / 长轮询主循环。
- 如果某个微信会话的旧 pending turn 无法恢复，daemon 不会再静默吞掉错误并继续收新消息；它会在该会话下一条入站消息到达时显式拒绝处理，并返回固定错误提示，避免同一 `session_id + scene` 槽位被坏 pending turn 永久卡死后又退化成通用错误回复。

```mermaid
sequenceDiagram
    participant UI as CLI interactive
    participant Startup as startup/*
    participant Admin as HostAdminService
    participant Service as ChatService
    participant Contract as Contract preparation
    participant Host as Host
    participant SP as scene preparation
    participant Agent as AsyncAgent

    UI->>Startup: resolve_startup_paths / ConfigLoader / prepare_host_runtime_dependencies(...)
    Startup-->>UI: workspace/model/runtime/default execution options/Host 稳定依赖
    UI->>Service: new ChatService(host, scene_execution_acceptance_preparer, ...)
    alt 带 --label
      UI->>UI: 读取/创建 CLI label registry record
      UI->>UI: 解析 session_id + scene_name
      UI->>Admin: list_session_recent_turns(session_id)
      Admin->>Host: list_conversation_session_turn_excerpts(session_id)
    else 无 --label
      UI->>UI: 从 interactive state 解析默认 session_id
    end
    UI->>Service: list_resumable_pending_turns(session_id, scene_name)
    Service->>Host: list_pending_turns(resumable_only=True)
    alt 存在可恢复 pending turn
      UI->>Service: resume_pending_turn(session_id, pending_turn_id)
      Service->>Host: resume_pending_turn_stream(pending_turn_id, session_id)
      Host->>Host: register run / cancel bridge / deadline watcher / concurrency
      Host->>SP: restore_prepared_execution(prepared_turn, run_context)
      SP->>SP: 重建 tools / session_state
      SP-->>Host: AgentInput
      Host->>Agent: build_async_agent(...) + run_messages(...)
      Agent-->>Host: StreamEvent
      Host-->>Service: AppEvent
      Service-->>UI: AsyncIterator[AppEvent]
      Host->>Host: 写回 transcript / 调度 compaction
    end
    loop 每一轮输入
        UI->>Service: submit_turn(ChatTurnRequest(session_id, user_text, ticker?, scene_name="interactive", session_resolution_policy=ENSURE_DETERMINISTIC, execution_options?))
        Service->>Service: 解释业务语义，生成 Prompt Contributions
        Service->>Host: ensure_session()
        Service-->>UI: ChatTurnSubmission(session_id, event_stream)
        Service->>Service: SceneExecutionAcceptancePreparer.prepare(...)
        Service->>Contract: prepare_execution_contract(...)
        Contract-->>Service: ExecutionContract
        Service->>Host: run_agent_stream(execution_contract)
        Host->>Host: register run / cancel bridge / deadline watcher / concurrency
        Host->>SP: prepare(contract, run_context)
        SP->>SP: 读取 scene definition / transcript
        SP->>SP: 装配 tools / system_prompt / messages / agent_create_args
        SP-->>Host: AgentInput
        Host->>Agent: build_async_agent(...) + run_messages(...)
        Agent-->>Host: StreamEvent
        Host-->>Service: AppEvent
        Service-->>UI: AsyncIterator[AppEvent]
        Host->>Host: 写回 transcript / 调度 compaction
    end
```

这条路径也总结了前面几章的分工：

- `UI` 负责装配和展示
- `Service` 负责业务解释和执行决策
- `Host` 负责托管和会话状态
- `Agent` 负责在给定输入下完成消息交互

## 10. 扩展点

### 10.1 新增 Service

新增一个 Agent 路径 `Service` 时，推荐按下面的顺序做：

1. 定义 `Request DTO`
2. 明确这个 `Service` 需要理解哪些业务语义
3. 选择或新增 `scene`
4. 基于 startup 提供的稳定依赖与 Service 自身规则形成执行决策
5. 产出 `Execution Contract`
6. 调用 `Host`

新增 `Service` 时最重要的边界是：

- 不要让 `Service` 自己构造 `AsyncAgent`
- 不要让 `Host` 反向理解你的业务参数
- 不要把动态 prompt 做成新的事实投影系统

### 10.2 新增工具

新增工具时，需要先回答两个问题：

- 这是 Agent augmentation 工具，还是 direct operation？
- 它应该被哪些 scene 允许、被哪些 `Service` 选中、受哪些动态权限约束？

当前稳定扩展方式是：

- 在对应包内新增工具实现
- 在 `toolset_registrars.json` 中为该 toolset 配置 registrar，并由 registrar 在所属包内完成叶子工具注册
- 通过 scene manifest、`selected_toolsets`、`execution_permissions` 控制它是否真的进入当前执行

不要把工具的业务启用逻辑放进 `Agent`。

如果你要新增一个全新的 toolset，最小接入清单固定为：

1. 在所属包内实现叶子工具注册函数，保持工具参数面只面向该包自己的能力边界。
2. 在所属包内新增一个 `toolset registrar` adapter，签名固定为 `registrar(context)`；它负责消费当前 toolset 的 `toolset_config` 快照，并把通用 `ToolsetRegistrationContext` 适配成叶子 `register_*_tools(...)` 所需参数。
3. 在 `dayu/config/toolset_registrars.json` 中新增 `toolset_name -> registrar import path` 映射；如果需要工作区覆盖，则在 `workspace/config/toolset_registrars.json` 中写同名映射。
4. 在对应 scene manifest 的 `tool_selection` 中声明该 toolset 是否进入候选集合；如果 scene 不声明，它不会被注册。
5. 在对应 `Service` 的 contract preparation 中决定是否通过 `selected_toolsets` 进一步收窄；若不需要额外收窄，保持 `()` 即可。
6. 如果工具需要执行期动态权限，只通过 `execution_permissions` 扩展该权限域，并由 registrar 在 adapter 内消费；不要让 `Host` 直接理解某个工具包的细节参数。
7. 补贯通测试，至少覆盖“scene manifest 命中 toolset 后会调用 registrar”“registrar 能把通用上下文正确适配到叶子注册函数”“缺失 registrar 时 scene preparation 显式失败”。
8. 同步更新开发文档，至少说明这个 toolset 属于哪个包、由哪个 registrar 暴露、受哪些 scene 和权限约束控制。

判断一个实现是否走在正确边界上的快速标准只有三条：

- 新增 toolset 时不需要修改 `Host` 的工具装配分支。
- `Host` 不需要新增对该领域 runtime 或细节参数的直接依赖。
- 业务启用逻辑仍然只落在 scene manifest、`selected_toolsets` 与 `execution_permissions` 上。

### 10.3 新增 scene

新增 scene 时，需要维护两类内容：

- prompt 与 manifest
- 模型与工具选择策略

一个 scene 应该只回答执行期声明式问题，例如：

- prompt 结构是什么
- 允许哪些动态 slot
- 默认模型与 allowlist 是什么
- 允许哪些工具进入候选集合

不要把业务解释放进 scene。

## 11. 建议的阅读顺序

对上层接入开发者，建议按这个顺序读代码：

1. `startup/`
2. `services/`
3. `host/`
4. `cli/arg_parsing.py` -> `cli/main.py` -> `cli/commands/`
5. `wechat/arg_parsing.py` -> `wechat/runtime.py` -> `wechat/commands/` -> `wechat/main.py`
6. `web/`

对扩展开发者，建议按这个顺序读代码：

1. [../docs/architect.md](../docs/architect.md)
2. `contracts/agent_execution.py`
3. `services/`
4. `host/`
5. `prompting/`
6. `engine/README.md`
7. `fins/README.md`

如果你在扩展时发现某个设计需要让：

- `Host` 理解业务
- `Agent` 理解 `ticker`
- `UI` 直接操纵 run 生命周期
- `Service` 直接构造 `AsyncAgent`

那通常说明边界已经偏离当前架构。
