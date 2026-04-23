# 测试手册

本文档只说明测试分层、运行方式和维护规则。具体业务背景与代码架构请看各包 README。

## 1. 目录分层

当前测试按五组组织：

- `tests/application/`
  - 当前目录名保留，但内容实际覆盖 `Service` 与 `Host` 的稳定边界
  - 重点覆盖 `ExecutionContract`、contract preparation、Host Session、Host run、HostExecutor、scene preparation
- `tests/engine/`
  - Engine、prompt 资产、写作流水线、多轮记忆测试
  - 重点覆盖 `AsyncAgent`、Runner、ToolRegistry、PromptComposer、write pipeline
  - `tests/engine/test_log.py` 负责守住全局日志双流路由边界：`ERROR` 以下只写 stdout，`ERROR` 及以上只写 stderr，不能再把同一条失败日志同时打到两个流里造成 CLI 重复显示
  - `tests/engine/test_prompt_assets.py` 与 `tests/engine/test_prompt_composer.py` 共同守住 scene manifest 的当前默认模型边界：推理/交互类 scene 默认应对齐 `mimo-v2-pro-thinking-plan`，写作类 scene 默认应对齐 `mimo-v2-pro-plan`；若测试仍断言旧 `pro` 默认，会把真实 prompt 配置漂移误报成回归
  - `tests/engine/test_write_pipeline.py` 还要守住 `scene_executor.py` 的共享重试边界：write/overview/infer/decision/fix/regenerate/raw prompt、confirm 和 repair 都必须复用同一套 LLM 执行失败重试语义；取消不得重试，解析失败只在 confirm/repair 这类声明了解析契约的路径重试，且最终错误消息必须保留各 scene 自身语义
  - `tests/engine/test_context_budget.py` 负责守住 `dayu.engine.context_budget` 的真源边界，包括 `ContextBudgetState`、工具结果预测性预算裁剪和相关 warning 语义
  - `tests/engine/test_web_tools.py` 负责守住 web search provider 回退、web tools 的请求头、内容编码、自刷新壳页跟随、challenge 检测、storage state 解析与浏览器回退边界
  - `tests/engine/test_web_playwright_backend.py` 负责守住 `dayu.engine.tools.web_playwright_backend` 的真源边界，包括浏览器单例、子进程 worker、资源路由与关闭/终止收口
- `tests/fins/`
  - Fins direct operation、窄仓储、processor、tool service 测试
  - `tests/integration/fins/test_fins_tools_ground_truth.py` 依赖仓库内 `workspace/` 的真实 source 文档样本与 `tests/fixtures/fins/ground_truth/` 基线；当 CI 或干净环境里缺少这批本地样本时，应显式 `skip`，不能把“样本未准备”误报成产品回归
  - `tests/fins/test_sec_pipeline_helpers.py` 与 `tests/fins/test_sec_rebuild_workflow.py` 共同守住 source fiscal 真源边界：download/source 层禁止从 `report_date`/`filing_date` 硬猜 fiscal 字段，尤其是 6-K / 6-K/A 不得猜 `fiscal_year/fiscal_period`，rebuild 在新推断为空时也不得继续沿用 previous_meta 中的旧猜测值
  - `tests/fins/test_fins_tools_service.py` 与 `tests/fins/test_fins_tools_service_helpers_coverage.py` 共同守住消费侧 fiscal 边界：`list_documents()` 不得再仅凭 `report_date` 为 10-Q 回填季度，也不得为其他 source 文档回填空的 `fiscal_year`；当前只允许保留表单内生且不依赖日期猜测的低风险回退（如 `10-K/20-F -> FY`）
  - `tests/fins/test_fins_runtime_tool_service.py` 还要守住 `FinsToolService` 的构造期实例不变量：runtime 返回的真实 service 必须在 `__init__` 中声明实例级缓存（如 `_meta_cache`），不能把属性存在性延后到读路径里的 `hasattr` 懒初始化
  - `tests/fins/test_virtual_section_table_assignment.py` 负责守住 `_VirtualSectionProcessorMixin` 的稳定边界：虚拟章节表格映射要保持同源重分配，未启用虚拟章节时 `list_sections/read_section/list_tables/get_section_title/search` 也必须原样透传到底层 processor，而不是靠 `type: ignore[misc]` 放任下一跳协议漂移
  - 其中 `tests/fins/test_storage_batch_recovery.py` 负责守住 storage batch 真源边界：隐藏工作目录统一位于 `workspace/.dayu/`，并且 orphan staging / backup 必须由 storage recovery 自动收口，不能把清理逻辑散落到上层入口；batch 锁必须按 ticker 粒度保持跨进程互斥，同一 ticker 不能双写，不同 ticker 仍可并发，recovery 看到活跃锁时只能跳过；这套文件锁语义在 POSIX 必须继续走 `fcntl.flock()`，在 Windows 必须改用 `msvcrt.locking()`，不能再退化成无锁，而且 blocking recovery 锁在 Windows 上也必须保持“持续等待直到拿到锁”的语义，不能偷换成有限次重试；一旦 ticker 锁已经释放，即使原 owner 进程仍然存活，也必须允许 recovery 清理遗留 token，缺少 journal 且无法判定 ticker 的 token 目录同样必须保守跳过，不能误删 live staging；如果 recovery 在枚举 token 后目录被 owner 正常删除，也必须把它视为已消失 token 并跳过，不能因 `FileNotFoundError` 中断整批恢复；rollback 即使 journal 写失败也必须继续清理并释放锁，不能把同 ticker 后续 batch 永久卡死
  - 其中 `tests/fins/test_llm_ci_scripts.py` 负责守住 `utils/llm_ci_process.py` 与 `utils/llm_ci_score.py` 的最小增量调度、同一 ticker 子批次串行化、failed filings 识别、form 归一化、summary 聚合与输入解析边界，避免 CI 优化辅助脚本在执行层漂移
  - 其中 `tests/fins/test_sec_6k_primary_document_repair.py` 与 `tests/fins/test_sec_pipeline_download.py` 共同守住 6-K active source 主文件 reconcile 边界：下载/历史修正都必须以当前 `BsSixKFormProcessor` 对同 filing HTML 候选的核心报表可提取性为真源，把最能稳定提取 `income + balance_sheet` 的文件写回 `primary_document`，不能退回 cover-only 保护或 filename 启发式主导
  - 其中 `tests/fins/test_tool_snapshot_export.py` 负责守住 legacy snapshot 导出入口与当前窄仓储契约的一致性，尤其是快照导出完成后必须能同步回写 processed manifest，避免“文件已落盘但文档不可发现”的测试漂移
- `tests/integration/`
  - 端到端集成测试
- `tests/architecture/`
  - 依赖边界与架构守护测试

另外：
- `tests/fixtures/` 放测试数据
- `tests/` 根目录下的少量 `test_*.py` 用于承接项目级工具脚本与通用辅助模块的轻量回归；这类测试应优先守住稳定输入输出边界，不把临时脚本细节固化进测试
- `tests/engine/test_docling_processor_integration.py`、`tests/fins/test_docling_upload_service_integration.py`、`tests/engine/test_web_fetch_docling_integration.py` 是问题 2 第一批真实集成测试，必须直接走真实 Docling 执行链，不允许通过 monkeypatch `DocumentConverter` 或 fake `DoclingDocument` 伪造通过
- 仓库根 `tests/` 明确作为本地测试包维护，避免干净虚拟环境里第三方同名 `tests` 包抢占导入解析，导致 `pyright` 或测试辅助模块引用漂移到 `site-packages`

## 2. 运行方式

运行测试前，先安装项目依赖：

```bash
pip install -r requirements.txt
```

其中 `requirements.txt` 已包含异步测试所需的 `pytest-asyncio`，以及覆盖率报告所需的 `pytest-cov`。

本仓库默认在项目自带的 `.venv` 中运行测试和类型检查。

常用命令：

```bash
.venv/bin/pytest tests/application -q
.venv/bin/pytest tests/engine -q
.venv/bin/pytest tests/fins -q
.venv/bin/pytest tests/integration -q
.venv/bin/pytest tests/architecture -q
.venv/bin/pytest tests -q
```

查看覆盖率：

```bash
.venv/bin/pytest tests --cov=dayu --cov-report=term --cov-branch
```

类型检查：

```bash
.venv/bin/pyright --pythonpath .venv/bin/python
```

## 3. 维护规则

### 3.1 测试跟着边界迁移

当实现边界变化时，优先迁移测试，不要用生产代码兼容旧测试。

例如这次主链收口后，已经删除的旧装配层、旧 runtime facade、旧能力注入链，都不应继续保留对应测试。

真实 Docling/PDF 集成测试也遵守同一规则：

- 必须使用固定 fixture
- 必须走真实第三方执行链
- 表格退化不可接受，不能只断言“返回非空字符串”
- 当前 Dayu 已将 upload / web tool 等 Docling PDF 转换入口统一收口到 `dayu.docling_runtime`；默认设备为 `auto`，若 `auto` 转换阶段失败则统一回退到 `cpu` 重试一次；若需要显式验证其他设备，可通过环境变量 `DAYU_DOCLING_DEVICE=auto|cpu|cuda|mps|xpu` 覆盖
- 与此对应，unit test 若只想隔离 Docling 转换结果，应优先 patch `run_docling_pdf_conversion()` 这类项目内真源 seam；若只测装配参数，再 patch `build_docling_pdf_converter()`，不要继续直接 patch 第三方 `DocumentConverter` 类

### 3.2 Service / Host / Agent 路径守护

涉及 Agent 路径改动时，优先补这些测试：
- `tests/application/test_prompt_service.py`
- `tests/application/test_chat_service.py`
- `tests/application/test_write_service.py`
- `tests/application/test_host_executor.py`
- `tests/application/test_host_reply_outbox.py`
- `tests/application/test_reply_delivery_service.py`
- `tests/application/test_reply_outbox_web_integration.py`
- `tests/application/test_reply_outbox_store.py`
- `tests/application/test_web_routes.py`
- `tests/application/test_wechat_outbox_integration.py`
- `tests/application/test_host_commands.py`
- `tests/engine/test_prompt_composer.py`
- `tests/engine/test_conversation_memory.py`
- `tests/engine/test_async_agent.py`
- `tests/engine/test_web_tools.py`

其中：
- `test_web_routes.py` 负责守住 Web 依赖装配已经收口到 `fastapi_app` 的显式窄依赖注入，router 工厂不再回退到全局 service locator 或旧 `Application` API。
- `test_chat_service.py`、`test_fins_service.py` 与 `test_web_routes.py` 还要共同守住请求受理时机：`Service.submit_*()` 必须在返回 submission 前完成同步校验，校验失败时不得创建新的 Host session，Web 入口也不得返回 `202 Accepted` 或启动后台消费任务。
- `tests/application/test_console_output.py` 负责守住 CLI / WeChat / render 入口的标准流容错边界：在非 UTF-8 终端里打印中文 help 或错误文案时不得因 `UnicodeEncodeError` 崩溃。
- `tests/cli/test_init_command.py` 还要守住 `dayu-cli init` 的交互输入边界：测试不得隐式依赖开发机已有的 `SEC_USER_AGENT` 等环境变量短路交互流程，凡是验证完整 `run_init_command()` 路径的用例，都应显式 `delenv/setenv` 相关变量并提供完整输入序列；`--reset` 这类破坏性开关还必须验证二次确认语义，且直接回车默认按 `N` 取消。
- `tests/cli/test_init_command.py` 还要守住已有 workspace 的增量补齐语义：未传 `--overwrite` 时，`init` 只能补齐缺失的 `config/prompts/**` 资产，不能覆盖用户本地已经改过的 manifest、scene 或其他配置文件。
- `tests/application/test_cancellation_bridge.py` 与 `tests/engine/test_async_openai_runner_utils.py` 这类并发/超时测试，不得把 0.08s、0.2s 这类固定 sleep 当成必然成立的完成信号；应使用带超时的轮询等待来守住语义，避免 CI runner 时序抖动造成伪失败。
- `test_web_routes.py` 还要守住 Web 的客户端错误语义：像 `PromptService.submit()`、`ChatService.resume_pending_turn()` 这类已经在 Service 边界同步抛出的 `ValueError/KeyError`，router 必须映射成对应 `4xx`，且 `/api/chat/resume` 只能在 resume 成功后才创建后台消费任务，不能漏成 `500` 或先受理后失败。
- `test_web_routes.py` 还要守住 `/api/write` 的未支持语义：当 Web 当前不支持在线写作时，route 必须显式返回 `501`，不能再用 `202` / `accepted=true` 伪装成已受理。
- `test_fins_service.py`、`test_sec_process_workflow.py`、`test_sec_pipeline_process_filing_source.py`、`test_cn_pipeline_process.py` 与 `test_tool_snapshot_export.py` 还要共同守住 Fins 取消传播链路：`Host` 的取消状态只能以窄 `cancel_checker` 形式从 `FinsService -> FinsRuntime -> Pipeline` 下传；同步 `process_filing/process_material` 与已支持取消协作的流式 `download/process` 都必须在阶段边界及时抛出 `CancelledError` 或收口为 cancelled，不能等 direct operation 整体结束后才统一收口。
- `test_sec_pipeline_process_filing_source.py` 还要继续守住 snapshot 导出后的仓储可发现性：当 `tool_snapshot_*.json` 已成功写入 `processed/{document_id}` 时，`FsProcessedDocumentRepository.list_processed_documents()` 必须能同步发现该文档，不能再出现“目录里有 snapshot、manifest 里没有登记”的漂移。
- `test_host_admin_service.py` 负责守住宿主管理面已经收口到 `HostAdminService`，避免 Web / CLI 管理面重新直接触碰 Host。
- `test_event_bus.py`、`test_host_admin_service.py` 与 `test_web_routes.py` 还要共同守住 Host 事件订阅边界：event bus / 管理订阅 / SSE 只依赖稳定事件包络，不再把运行事件硬编码成 `AppEvent`；direct operation 的 `FinsEvent` 也必须能被原样转发，Web SSE 除了规范化 dataclass payload 外，还必须保留 `command` 判别字段，避免不同 direct operation 在 `type=progress/result` 下失去反序列化依据。
- `test_ui_host_boundary.py` 负责守住 UI 请求期不再直接访问 `dependencies.host`、WeChat daemon 不再自持 `ensure_chat_session`、CLI interactive 不再自己创建 Host session。
- `test_cli_interactive_coverage.py` 与 `test_interactive_resume.py` 还要共同守住 interactive UI 的服务消费边界：chat 只走 `submit_turn()/resume_pending_turn()/list_resumable_pending_turns()`，prompt 只走 `submit()`，不能再回退到 `stream_turn()` / `stream()` 或 `hasattr` 兼容分支；interactive 请求执行阶段仍只消费统一 Host 启动恢复后的 runtime，但 CLI 入口可以注入 `HostAdminService` 用于打印恢复摘录与清理漂移 label record，不能回退成直接读取 transcript 或直接操作 Host。
- `test_host_commands.py` 负责守住 startup 链路是否继续吃到 `run.json` 里的 `host_config`，例如 `host_config.store.path`、`host_config.lane` 与 `host_config.pending_turn_resume.max_attempts`；同时 CLI 宿主管理命令在请求期只能消费 `HostAdminService`，不能再保留 `runtime.host` 兜底或直接调用 Host 方法。
- `test_cli_interactive_coverage.py` 负责守住 interactive 的两条恢复路径：无 label 时继续使用 `interactive_key -> session_id` 的本地状态绑定并支持 `--new-session`，带 `--label` 时改走 CLI label registry；恢复时仍要展示上一轮对话提示，且 interactive 请求继续走 `ENSURE_DETERMINISTIC`。另外同一个 label 在 REPL 生命周期内必须保持独占，直到双 `Ctrl+D` 完整退出后才允许其它 prompt / interactive 复用。
- `test_cli_running_config.py` 与 `test_cli_interactive_coverage.py` 还要守住 `sessions close` 与 labeled conversation 的衔接语义：如果 label 命中的底层 Host session 已是 `closed`，CLI 必须先提示“旧对话已关闭”，再按同名 label 新建一条全新 conversation；若 label 之前是通过 `conv remove --label` 主动释放，则不应再额外提示“旧对话已关闭”。
- `test_host_commands.py` 与 `test_host_admin_service.py` 负责守住会话管理面边界：CLI 只能消费 `HostAdminService.list_sessions()` / `list_session_recent_turns()` 这组通用 digest 接口，并通过 `--source/--scene` 过滤；Service 只能通过 Host 公开协议读取 conversation digest / turn excerpt，不能让 UI 直接读取 transcript 文件。
- `test_conv_commands.py` 负责守住 CLI label registry 的管理面：`conv list/status/remove` 只管理 label 到 Host session 的映射，不直接替代 Host session 列表，也不把普通 one-shot `scene=prompt` 会话纳入可恢复 conversation。`conv list` 默认只输出 active 的 label 视角摘要，`conv list --all` 才补充已关闭对话，并且不再暴露 `SESSION_ID`；若命中 registry 漂移到不存在 Host session 的脏 record，CLI 必须先清理而不是展示伪 `missing` 状态。`conv remove --label` 必须在 label 未被占用时关闭底层 session 并删除 registry record，使同名 label 下次重新从新会话开始。若需要诊断底层 `session_id`，应通过 `conv status --label <label>` 或 `sessions` 查看。
- `test_cli_running_config.py` 还要守住 interactive 的 `state_dir` 单实例锁：同一 workspace 下第二个 interactive 进程必须显式失败，不能并发共享 `.dayu/interactive`。
- Host 默认 SQLite、conversation transcript、interactive 绑定、SEC cache/throttle 与 WeChat 默认状态目录都属于 workspace 内部运行时状态，默认路径应统一落在 `workspace/.dayu/` 下；涉及这些默认值的测试不要再把 `.host`、`.session`、`.interactive`、`.sec_cache`、`.sec_throttle`、`.wechat` 当成真源。
- `test_dependency_boundaries.py` 还要守住 `startup` 只能依赖 `services/host` 暴露的 public preparation API，不能再直接 import `SceneDefinitionReader`、`ConversationPolicyReader`、`SceneExecutionAcceptancePreparer` 或 `DEFAULT_LANE_CONFIG`。
- `test_dependency_boundaries.py` 也要守住 `Service` 只能依赖 `dayu.host.protocols` 与 `dayu.host.host_execution` 暴露的稳定边界，不能再 import `dayu.host.host.Host`，也不能直接访问 `host.executor`、`host.session_registry`、`host.run_registry`、`host.concurrency_governor`、`host.event_bus`，以及加下划线后的 `host._executor`、`host._session_registry`、`host._run_registry`、`host._concurrency_governor`、`host._event_bus`、`host._pending_turn_store`。
- `test_dependency_boundaries.py` 还要守住 `Service` 不得直接 import `dayu.host.pending_turn_store`，pending turn 的上层公开边界只能来自 `dayu.host.protocols` 中的稳定 DTO 与协议。
- `test_dependency_boundaries.py` 还要守住 `contracts/` 与 `execution/` 作为跨层稳定契约层，不得反向 import `dayu.engine`；运行时配置、取消原语与跨层协议必须定义在 contracts/execution 自己的真源模块里。
- `test_dependency_boundaries.py` 还要守住 UI 请求入口不得直接 import `DocToolLimits`、`FinsToolLimits`、`WebToolsConfig` 这类具体 limits/config 类型；CLI / WeChat 只能传原始 override，由 execution 真源在 merge/snapshot 阶段完成规范化。
- prompt 渲染相关测试现在以 `dayu.prompting.prompt_renderer` 为真源，不应再从 `dayu.engine.prompts` 导入。
- `test_contract_preparation.py` 负责守住 Service 侧 contract preparation 的机械装配边界，尤其是 `accepted_execution_spec` 与 `execution_permissions` 的同步关系。
- `test_contract_preparation.py`、`test_prompt_composer.py`、`test_scene_execution.py` 与 `test_write_pipeline.py` 还要共同守住 Prompt Contributions 边界：Service 必须先按 scene manifest 的 `context_slots` 收口为 exact set，Host 若收到未声明 slot 只能 warning + ignore，不能再把冗余 slot 升级成 scene 执行失败。
- 涉及 `AcceptedExecutionSpec` 的新测试，优先按 `model/runtime/tools/infrastructure` 四个分组构造与断言；不要再新增新的平铺 bag 夹具。
- `test_chat_service.py`、`test_prompt_service.py` 与 `test_scene_execution.py` 需要守住 `conversation.enabled=true` 的 scene 默认写入 `host_policy.resumable=True`，而非 conversation scene 继续保持 `False`。
- `test_prompt_service.py` 这类 Service contract 测试里，若只验证 `PromptService -> ExecutionContract` 的装配，不要再用 `SimpleNamespace` 伪造 `AcceptedSceneExecution` 内部字段；应直接构造真实 `SceneDefinition`、`ModelConfig` 与 `ResolvedExecutionOptions`，并通过测试 helper 把 `host` 收窄回 `StubHostExecutor` / `StubSessionRegistry` 后再断言内部记录。
- `test_scene_execution.py` 还需要守住 scene allowlist / max_iterations / max_consecutive_failed_tool_batches / temperature profile 的共享真源在 `dayu.execution.options`，避免 Service 与 Host 再各自维护一份规则。
- `test_scene_execution.py` 与配置加载相关测试还需要守住 CLI runner 已彻底禁用：`runner_type=cli` 应在模型加载或 Host 装配阶段显式失败，而不是继续被当成可用路径。
- 直接构造 `ResolvedExecutionOptions`、`AcceptedToolConfigSpec` 或 `PreparedAgentTurnSnapshot` 的测试，工具配置应通过 `toolset_configs` 进入契约；不要再把 `doc_tool_limits` / `fins_tool_limits` / `web_tools_config` 当成跨层字段夹带到 Host 或 contract snapshot。
- `test_scene_execution.py` 还要守住内部 `RunnerType` / `FallbackMode` 可提升为枚举承接，但配置快照和 JSON 边界仍保持原字符串值，不要把外部输入边界改造成枚举字面量协议。
- execution 运行配置的测试真源现在拆成三层：`dayu.contracts.runtime_config_snapshot` 负责跨层快照 TypedDict（`RunnerRunningConfigSnapshot` / `AgentRunningConfigSnapshot`）；`dayu.execution.options` 负责合并规则；`dayu.execution.runtime_config` 负责纯运行配置模型与快照转换函数。凡是直接构造 `ResolvedExecutionOptions` 的测试，应优先使用 `AgentRuntimeConfig`、`OpenAIRunnerRuntimeConfig`、`CliRunnerRuntimeConfig` 这组 execution 侧纯模型，不要再把 engine 实现类直接塞回 execution 层对象。
- `test_host_executor.py`、`test_run_registry.py`、`test_host_admin_service.py` 与 `test_web_routes.py` 需要共同守住 Host timeout 取消链路。`request_cancel()` 只记录 `cancel_requested_at` / `cancel_requested_reason`，终态 `cancel_reason` 必须等 Host 在 run 生命周期边界统一收敛后才写入；管理视图与 API 需要同时透出这两层信息。
- `test_host_executor.py` 与 `test_run_registry.py` 还要继续守住 orphan run 修复语义：startup cleanup 只能清理超过 grace period 的 dead-pid 活跃 run，不能误伤刚启动或仍在收尾的前台 direct operation；若同一 owner 的 run 被外部误判成 `orphan: owner process terminated`，后续成功收口必须允许修复回 `SUCCEEDED`，而异常路径也不能再被重复 `fail_run()` 掩盖成状态机错误。
- `test_host_executor.py` 还要守住 transcript 落库闸门：即使 Agent 本轮已经产出 `final_answer`，只要 run 在 transcript 持久化前被取消，`persist_turn()` 也必须跳过，不能把“已取消”run 的回答写进会话真源。
- `test_host_executor.py` 还要守住 `run_agent_and_wait()` 的同步聚合路径直接比较 `AppEventType`，不能退回依赖 `final_answer` / `warning` / `error` 这类字符串 value，否则应用层事件枚举改值时会出现静默失配。
- `test_host_executor.py`、`test_cli_running_config.py`、`test_reply_outbox_web_integration.py`、`test_wechat_daemon.py` 与 `test_write_pipeline.py` 还要共同守住取消消费语义：`Host.run_agent_and_wait()` 遇到 `AppEventType.CANCELLED` 必须抛出 `CancelledError`，不能伪装成空 `AppResult`；CLI 写作模式应返回显式取消退出码，Web reply outbox 不得提交取消前的 partial reply，WeChat 不得发送空兜底或残留 partial 内容，write pipeline 也不得对取消做重试或降级继续。
- `test_host_reply_outbox.py` 与 `test_reply_outbox_store.py` 需要共同守住三条边界：reply outbox 是独立于 pending turn 的第三套真源、Host internal success 不会自动入队 outbox、reply outbox 的状态流转必须只由显式提交与显式 delivery 回执驱动；其中 SQLite `submit_reply` 必须对同一 `delivery_key` 提供数据库级原子幂等提交，不能因为唯一键竞争把补投递/重试路径打成异常流程，`claim` 也必须是数据库级原子迁移，不能因为陈旧读取把同一条记录重复领取。
- 上述 reply outbox 状态机测试还要继续守住 `ack` 边界：`mark_delivered` 只能接受已 `claim` 的 `delivery_in_progress` 记录，不能绕过 `claim` 直接确认 `pending_delivery` / failed 状态；`delivery_attempt_count` 只能在 `claim` 时增长，不能被 `ack` 污染。
- `test_reply_delivery_service.py` 负责守住渠道层只能通过 `ReplyDeliveryService` 使用 reply outbox，不能重新把 Web / WeChat 之类的 UI 适配层直接耦合回 `Host` 门面或底层 store。
- `test_reply_outbox_web_integration.py` 与 `test_web_routes.py` 需要共同守住 Web 路径的边界：chat 最终答案先回到 Service 语义，再显式写入 reply outbox；FastAPI 只注入窄 Service，不回退到全局应用对象或直接管理 Host；对外 reply outbox API 必须暴露 `claim`，让 worker 按 `list -> claim -> ack/nack` 的顺序独占发送权。
- `test_web_routes.py` 还要守住 Web 路由的 Service 返回值契约：`create_chat_router()` / `create_fins_router()` 只能接受稳定的 `ChatTurnSubmission` / `FinsSubmission` DTO，且异步端点必须在创建后台任务前先拒绝空 submission、空 session_id 或非流式 execution/event_stream，并把这种内部契约破坏映射成 500。
- `test_web_routes.py` 这类 FastAPI 组合根测试，若通过 `sys.modules` 注入假的 `fastapi` / `pydantic` 模块，应在 `ModuleType` 边界集中 `cast(Any, module)` 后再挂 `FastAPI`、`APIRouter`、`BaseModel`；若依赖对象只用于验证 router 工厂接收的是哪一组窄 Service，可在 `create_fastapi_app()` 调用边界做一次协议收窄，不要散落 `SimpleNamespace` 假协议。
- `test_wechat_outbox_integration.py` 与 `test_wechat_daemon.py` 需要共同守住 WeChat 路径的边界：daemon 通过 `ReplyDeliveryService` 提交/claim/ack/nack 交付状态，reply 补投递不再复用 pending turn，也不能绕开 Service 直接写 Host reply outbox；同一个 `state_dir` 还必须保持 daemon 单实例锁，避免并发实例错误回收 `delivery_in_progress` 并造成重复交付；`.daemon.lock` 记录的 owner PID 也必须始终覆盖为当前实例，不能因为 append 模式把旧 PID 或锁区填充字节残留在文件里。
- `test_wechat_daemon.py` 还要继续守住 delivery worker 级策略：WeChat 只对网络/服务暂时性错误做有限次重试，`iLink ret != 0` 这类显式业务失败必须首轮收口为 `failed_terminal`，不能被误判成可无限补发。
- `test_wechat_main.py` 与 `test_wechat_service_manager.py` 需要共同守住 WeChat 用户级托管命令面的稳定语义，尤其是实例标签 `--label -> workspace/.dayu/wechat-<label>` 的映射、`service list` 仅枚举当前 workspace 下已安装实例，以及 `service start/restart/stop/status/uninstall` 与 launchd/systemd 后端的启动/重启边界不能混淆。
- `test_wechat_daemon.py` 还要守住 WeChat daemon 的状态文件 I/O 已通过 daemon 自持的串行适配器搬离 event loop，并且 shutdown 会等待已提交持久化请求收口完成。
- `test_wechat_daemon.py` 还要守住 iLink 载荷里的 `message_type` / `item.type` 解析边界：非数字字符串只能被安全视为非文本或跳过坏 item，不能因为裸 `int()` 崩溃打断 daemon 主循环。
- `test_wechat_daemon.py` 与 `test_wechat_outbox_integration.py` 还要共同守住 WeChat 的 ChatService 消费边界：daemon 只允许走 `submit_turn()`、`list_resumable_pending_turns()`、`resume_pending_turn()` 这组稳定 public contract，不能再回退到 `stream_turn()` 或 `hasattr`/`type: ignore` 兼容分支。
- `test_async_openai_runner.py`、`test_async_agent.py`、`test_host_executor.py`、`test_reply_outbox_web_integration.py`、`test_wechat_daemon.py` 与 `test_interactive_resume.py` 还要共同守住 `finish_reason=content_filter` 的稳定语义：Runner done summary 必须显式标记 `content_filtered` 与 `finish_reason`，Agent 不得继续续写但要保留 partial content，并通过 `final_answer.filtered` 把“受过滤完成态”透传到 Host、reply metadata 与 CLI/WeChat 可见提示。
- `test_async_openai_runner_call_paths.py`、`test_sse_parser.py` 与 `test_async_agent.py` 还要共同守住 Engine 取消传播链路：显式注入 `CancellationToken` 后，`AsyncOpenAIRunner` 必须在建连等待、响应体读取、重试 sleep 与 SSE 分块等待期间及时抛出 `CancelledError`；`AsyncAgent` 收到 runner 侧取消后不得再补发 `final_answer`。
- 上述取消传播测试还要继续守住两条资源清理边界：外层 asyncio task 被 `cancel()` / `wait_for()` 中止时，Runner / Parser 内部创建的等待子任务必须同步取消并收口；同一 `CancellationToken` 被多轮复用时，成功路径和异常路径都不能残留历史 `on_cancel` 回调。
- `test_async_openai_runner.py`、`test_async_openai_runner_call_paths.py`、`test_async_agent.py` 与相关集成测试还要共同守住 Runner 生命周期边界：`AsyncRunner` 测试桩必须实现异步 `close()`；`AsyncOpenAIRunner` 在同一 Agent run 的多轮 `call()` 中应复用实例级 `aiohttp.ClientSession`，只有 `runner.close()` 后才关闭并允许后续按需重建。
- `test_pending_turn_store.py`、`test_host_executor.py`、`test_chat_service.py`、`test_interactive_resume.py` 与 `test_wechat_daemon.py` 需要共同守住 pending `conversation turn` 才是 V1 恢复真源，而且真源必须是 Host-owned accepted/prepared snapshot；interactive / wechat 的恢复只能通过 Service 暴露的 `resume/list` 入口完成，不能回退到 transcript 或 UI 直连 Host 内部仓储。
- `test_pending_turn_store.py` 还需要守住 SQLite 同一 `session_id + scene_name` 槽位的并发 upsert 原子性；重复首写必须收敛为单条记录，不能因为唯一键竞争把恢复真源打成异常路径。
- `test_pending_turn_store.py` 还要守住 `record_resume_attempt` 达上限时必须在同一 `BEGIN IMMEDIATE` 事务里原子 `DELETE` 超限记录并抛 `ValueError`，不能让超限 pending turn 卡死 session/scene 槽位；并发多方调用只能有一方 UPDATE 成功、其他方抛 `ValueError` 且读不到残留。
- pending `conversation turn` 的测试还必须覆盖六条语义：resumable turn 被 Host 接受后就必须生成 `accepted_by_host` 真源；scene preparation 成功后要升级为 `prepared_by_host`，只有 transcript 成功持久化后才可推进 `sent_to_llm`；恢复 gate 只能放行 `FAILED` 或 `CANCELLED(timeout)` 的 source run；成功完成的 run 不得继续残留 pending turn；恢复请求必须显式匹配 pending turn 自身 `session_id`；WeChat daemon 的自动恢复只能处理当前 `state_dir` 对应 runtime identity 的记录。runtime 装配阶段还必须统一执行 Host-owned 启动恢复，然后才允许 interactive / wechat 继续做 pending turn 恢复；如果某个微信会话的 pending turn 仍无法恢复，daemon 必须显式拒绝该会话的新消息，而不能静默吞错后让同一槽位永久卡死；但若恢复失败后 Host 已经原子清理掉该 pending turn，interactive / wechat 必须放行当前会话继续前进，不能再用过期的 blocked 状态卡死 UI。daemon 侧还要把同实例遗留的 `delivery_in_progress` reply outbox 回收到可重试态，避免单条坏记录把整个微信服务启动打死。恢复 attempt 自增也必须由 Host 仓储原子守住 `max_attempts`，不能依赖 UI/Service 先读后判。
- 上述 pending `conversation turn` 原子性测试还要守住一条额外边界：只要取消发生在 transcript 持久化前，或 `persist_turn()` 自身失败，pending turn 都必须保持 `PREPARED_BY_HOST`，不能谎称 `SENT_TO_LLM`。
- 上述 pending `conversation turn` 收口测试还要守住成功提交边界：一旦 transcript 已成功持久化，source run 必须先 durable 地收口为 `SUCCEEDED`，后续 `sent_to_llm` / `delete_pending_turn` 清理失败最多留下不可恢复的脏 pending 记录，不能再把 run 打回 `FAILED` 并允许重复 resume。
- 上述 pending `conversation turn` 关键状态迁移与恢复分支还应保持 Host 侧 verbose 可观测性，至少覆盖 accepted/prepared/sent_to_llm 三阶段，以及 accepted/prepared 两条恢复入口。
- Host SQLite schema 发生非兼容变更时，测试要守住“启动期直接失败并提示删库重建”，不要把旧库残留留到运行中第一次读写某张表时才以 SQL 异常形式延迟爆炸。
- Host 与多轮会话的关键生命周期日志也属于稳定可观测性契约，至少要守住 session create/ensure/close、run register/start/cancel、transcript load/save，以及 conversation compaction 的调度与写回日志。
- 涉及 `AsyncAgent` 轮次命名时，Engine 测试要区分 `agent iteration` 与 `conversation turn`：`AsyncAgent` 事件、Tool Trace 与 `utils/analyze_tool_trace.py` 都应以 `iteration_id` / `iteration_*` 字段作为真源，禁止再把 Engine 内部轮次写成 `turn_id`。
- `test_cli_running_config.py` 负责守住 CLI 显式参数、`run.json`、`llm_models.json`、scene manifest、`toolset_registrars.json` 与 prompt assets 能否真正贯通到 `Service -> Host -> scene preparation -> Agent`，包括 `accepted_execution_spec`、最终 `AgentCreateArgs`、system prompt、`execution_permissions` 以及 toolset registrar adapter 到叶子 `register_*_tools` 参数的落地；这类测试应优先在 `build_async_agent` 边界替换 `MockAgent`，并在 registrar adapter 边界观察 `context.toolset_config` 与真实工具注册，而不是重新 mock 掉整条 runtime。
- `test_write_pipeline.py` 还要守住 write pipeline 的 scene 级环境变量闸门：模型环境变量校验必须收口在 `SceneContractPreparer` 这类 Service 内部真源，并按实际创建的 scene 惰性执行；未使用的 scene 不得提前阻断轻量模式。
- `test_cli_running_config.py` 还要守住 CLI 参数校验的退出契约：`setup_write_config()` 在缺失模板文件或 `--write-max-retries < 0` 时必须显式 `raise SystemExit(2)`，不能回退到站点级 `exit()`，以免嵌入环境与测试捕获语义漂移。
- `test_cli_running_config.py` 还要守住 CLI 写作薄入口的参数边界：`dayu.cli.main.run_write_pipeline()` 只接受真实执行所需的 `write_config` 与 `write_service`，不能继续保留 `workspace_config`、`running_config`、`model_config` 这类未消费的兼容关键字误导调用方。
- `test_toolset_registrar_coverage.py` 负责守住 toolset registrar 的数值反序列化边界：`ToolsetConfigSnapshot.payload` 中的字符串数字必须先通过统一 coercion helper 收口，再进入 doc/web/fins 专用 limits/config 对象。
- `test_cli_running_config.py`、`tests/fins/test_storage_split_repositories.py` 与 `tests/fins/test_sec_pipeline_download_stream.py` 还要共同守住财报存储边界：CLI 对“是否已有本地 filings”的判断，以及 SEC pipeline 对“是否已有 instance XBRL”的判断，都必须通过 `SourceDocumentRepositoryProtocol` 的仓储事实接口完成，不能回退到上层拼 `portfolio/...` 路径或自行扫目录。
- `tests/fins/test_storage_batch_recovery.py`、`tests/fins/test_storage_split_repositories.py`、`tests/engine/test_cli_running_config.py` 与 `tests/fins/test_sec_pipeline_helpers.py` 还要共同守住 batch/recovery 边界：隐藏工作目录只认 `workspace/.dayu/`，活动 batch 的 staging 读视图必须继续可见，而异常退出后的 `repo_batches` / `repo_backups` 残留只能由 storage 层按 journal 和文件状态自动恢复；恢复器的活跃性真源是 ticker 锁本身，不能把“owner 进程还活着”误提升成 batch 仍活跃的证据，也不能回退到 UI startup 的 `try/finally` 清理；ticker 锁与 recovery 锁的跨平台实现必须收敛到同一套文件锁真源，POSIX/Windows 都要保持真实互斥；`create_directories=False` 的只读探测不得触发 recovery 或写入 `.dayu`。
- `tests/application/test_host_admin_service.py` 与 `tests/application/test_web_routes.py` 还要守住 session source 受理边界：`HostAdminService.create_session()` 遇到非法 `source` 必须抛出 `ValueError`，`/api/sessions` 必须把它映射成 `400`；不能再把错误来源静默降级成 `web` 并污染 session 来源统计。
- `tests/fins/test_ground_truth_baseline.py` 与 `tests/fins/test_score_sec_ci.py` 还要继续守住 processed snapshot 边界：离线基线/评分工具只能通过 `ProcessedDocumentRepositoryProtocol` 发现样本、通过 `DocumentBlobRepositoryProtocol` 读取 `tool_snapshot_*.json`，不能在生产代码里回退到直接拼 `workspace/portfolio/{ticker}/processed/...` 或 `glob` 扫描目录。
- `tests/fins/test_score_sec_ci.py` 还要守住新的可喂性评分口径：active filing 缺少 processed，或 processed 文档缺少/破坏 `tool_snapshot_meta.json` 时，必须落到 completeness hard gate；其它 truth 快照缺失只能让对应维度记 0 分，不能再把整批评分直接打成异常退出。
- `test_conversation_memory.py` 还需要守住 Host conversation compaction runtime contract 的公开边界：conversation memory 只能调用显式公开的 compaction scene / agent 准备接口，不能再通过 `object` 请求或跨模块受保护方法拼装 Agent。
- `test_conversation_store.py` 还要守住 `dayu.host._coercion` 是 Host 内部共享字符串元组规范化真源，避免 `conversation_memory.py` 与 `conversation_store.py` 再各自复制 `_coerce_string_tuple()` 逻辑。
- `test_write_pipeline.py` 与相关 prompt/runtime 装配测试里的 fake 不应再用宽 `SimpleNamespace` 冒充 `WorkspaceResourcesProtocol`、`ResolvedExecutionOptions`、`SceneExecutionAcceptancePreparer` 或 `HostExecutorProtocol`；优先复用强类型 helper，并在测试边界显式 `cast` 到稳定协议视图。
- `test_write_pipeline.py` 还要继续守住 `confirm -> repair contract -> repair executor` 这条处置链：`confirmed_missing` 必须被代码侧收口成 `delete_claim`，repair 不得再把这类 claim 弱化后保留；对应 patch 也不得使用 `substring` 只删半句，至少要命中完整 line / bullet / paragraph。删除整段或整行后若只留下缩进空白行、空 bullet 或多余空行，也必须由 executor 侧 cleanup 统一收口，不能把结构残片留给模型善后。
- `test_write_pipeline.py` 还要继续守住 `E3` 的收口边界：当 evidence line 自己指向不存在、不可定位或明显不属于当前 ticker 的来源时，这不是可 confirm 的 `E1/E2` 缺证，也不是可局部补锚的 `S7`；代码侧必须强制把它收口成 `regenerate_chapter` / `chapter_regenerate`，不能让自定义 `repair_contract` 再把它降回局部 patch。
- `test_write_pipeline.py` 还要守住证据锚点轻量修复的可观测性边界：`maybe_rewrite_evidence_anchors()` 即使在后验校验失败回退原文时，也必须把 `attempted/applied/failure_reason` 显式写入 `process_state.latest_anchor_rewrite` 与历史记录，不能再让调用方只能靠正文是否变化去猜测“未尝试”和“尝试失败”。
- `test_write_pipeline.py` 还要守住 manifest 文件锁的跨平台边界：`artifact_store._manifest_file_lock()` 在 POSIX 必须继续走 `fcntl.flock()`，在 Windows 必须改用 `msvcrt.locking()` 提供真实互斥；其中 blocking 语义必须显式轮询直到拿到锁，不能退回 `LK_LOCK` 那种有限次重试；如果当前平台两种实现都不可用，也必须显式失败，不能静默跳过锁操作。
- `test_write_pipeline.py` 还要守住审计模块拆分边界：`audit_formatting.py` 只放 Markdown 文本操作（标题/证据行/内容提取/patch 匹配），`audit_rules.py` 只放审计决策真源（解析/规则/修复合同/复核合并），`audit_evidence_rewriter.py` 只放证据锚点重写。三个模块之间的依赖方向是 `formatting ← rules` 和 `formatting ← evidence_rewriter → rules`，不允许反向。
- `test_cli_running_config.py` 和 `test_cli_interactive_coverage.py` 还要守住 CLI 模块拆分边界：`arg_parsing.py` 只放 argparse 参数定义和解析器构建，`dependency_setup.py` 只放数据类型定义、配置解析和 Service 构建，`commands/` 承担各子命令执行，`main.py` 只做命令分发。monkeypatch 路径必须指向被调用函数所在的实际模块命名空间（例如 `_build_fins_ops_service` 在 `dayu.cli.commands.fins` 被调用，patch 应指向 `dayu.cli.commands.fins._build_fins_ops_service`），不允许通过包级 re-export 或旧 `main.py` seam 绕行。
- `test_wechat_main.py` 还要守住 WeChat 模块拆分边界：`arg_parsing.py` 只负责参数解析与上下文收口，`runtime.py` 只负责 WeChat 运行时装配与 service/runtime helper，`commands/` 承担 `login / run / service` 子命令执行，`main.py` 只做命令分发。monkeypatch 必须 patch 到真实被调用模块（例如 `_has_persisted_wechat_login` 在 `dayu.wechat.commands.service` 被调用时，应 patch `dayu.wechat.commands.service._has_persisted_wechat_login`，不能回退去 patch 旧 `main.py` seam 或其他转发层）。
- `test_entrypoints.py` 还要守住 CLI / WeChat 的冷启动边界：导入 `dayu.cli.main` 时不得抢先导入 `dayu.cli.commands.*`、`dependency_setup` 这类重运行时模块，导入 `dayu.wechat.main` 时也不得提前拉起 `dayu.wechat.daemon`；`--help` 与 `dayu-cli init` 这类薄入口必须继续保持“主入口只分发、命令模块按需导入”的结构。
- `test_chat_service.py` 这类 Service/Host 边界测试里，如果断言必须观察 Host 内部 stub（如 `_executor`、`_run_registry`、`_pending_turn_store`），应先通过测试 helper 把 `ConversationalExecutionGatewayProtocol` 显式收窄到测试用具体 `Host` / stub 类型；不要把私有属性访问直接散落在测试正文里。
- `tests/application/conftest.py` 这类共享夹具文件必须直接跟随 `HostExecutorProtocol` / `SessionRegistryProtocol` 演进；像 `StubHostExecutor.run_operation_stream/run_operation_sync` 这类泛型返回签名、以及 `StubSessionRegistry.close_idle_sessions()` 这类新协议方法，应优先在共享夹具真源补齐，而不是在每个下游测试里重复 `cast` 或 `type: ignore`。
- 项目级 `tests/conftest.py` 这类 import 兼容夹具若需要预加载模块，也要先显式断言 `ModuleSpec/loader` 非空，再在 `ModuleType` 边界集中 `cast(Any, module)` 后补动态别名；不要把 `spec.loader` 的可空分支或动态属性赋值散落到正文。
- `test_cli_interactive_coverage.py` 这类 CLI UI 测试里的 fake session 应直接实现 `ChatServiceProtocol` / `PromptServiceProtocol` 公开方法（如 `submit_turn`、`resume_pending_turn`、`list_resumable_pending_turns`、`submit`），不要继续只靠旧的 `stream/stream_turn` 形态再由测试把对象硬塞进 `interactive_ui`。
- `test_sse_parser.py` 与 `test_sec_pipeline_download.py` 这类 Engine/Fins 装配测试，遇到 `ClientResponse`、`AsyncOpenAIRunnerRunningConfig`、`SecDownloader` 等生产签名时，优先在测试装配边界用 helper 做一次显式类型收窄；不要为了消除 pyright 报错去让测试 stub 继承不匹配的生产实现类，尤其不要把同步 stub 硬继承到异步 downloader 真源上。ticker 市场识别一律对齐 `dayu.fins.ticker_normalization.normalize_ticker` 真源，需要 stub 时在消费者模块路径上 `monkeypatch.setattr(module, "normalize_ticker", ...)` 返回自造 `NormalizedTicker`，不要重造识别类。
- Engine 内部测试如果必须直接构造 `AgentRunningConfig` 或 `AsyncOpenAIRunnerRunningConfig`，应从各自定义模块导入，不要再依赖 `dayu.engine` 的聚合导出面；跨层 runtime config 快照真源在 `dayu.contracts.runtime_config_snapshot`，纯运行配置值对象与转换函数在 `dayu.execution.runtime_config`。
- `tests/fins/test_bs_report_form_common.py` 这类 processor 单测里，`Source` dummy 应保持普通类做结构兼容；`XBRL`、`_TableBlock` 等实现类型应在测试装配边界集中 `cast` / helper 收窄，不要把 `SimpleNamespace` 直接写进处理器字段或方法签名。
- `tests/engine/test_bs_processor.py` 这类基础 processor 搜索测试里，`SearchHit.section_ref/snippet` 等非必填字段也应通过 helper 安全读取；不要在基础回归里裸下标访问，否则很容易和 `search_utils` 一类文件重复产生同源 TypedDict 错误。
- `tests/engine/test_bs_processor_coverage.py` 这类覆盖补丁测试里，不要复用同名测试函数；传给 `_safe_table_text()` 的 `table_tag` 也要先断言非空，再把 BeautifulSoup 的可选返回值传进内部 helper。
- `tests/engine/test_html_processing_pipeline.py` 这类 HTML 管线测试若通过 `ModuleType` 注入假的 `trafilatura/readability/html2text` 模块，应像 FastAPI 组合根测试一样在模块边界集中 `cast(Any, module)` 后再挂 `extract/Document/HTML2Text`，不要直接对 `ModuleType` 动态赋属性。
- `tests/engine/test_docling_processor_helpers.py` 这类 Docling helper 测试里的 source 桩，应直接补齐 `Source` 协议要求的 `uri/media_type/content_length/etag/open/materialize` 最小集合；不要为了只测 `_sniff_docling_json()` 就继续传半截 source 对象。
- `tests/engine/test_markdown_processor_coverage_supplement.py` 这类 Markdown 搜索覆盖测试里，`SearchHit.section_title` 等可选字段也应通过 helper / `.get()` 安全读取，不要把搜索结果里的可选键当成必填键直接下标访问。
- `tests/fins/test_sec_pipeline_process_filing_source.py` 这类 pipeline + processor registry 测试里的 fake processor 应直接对齐 `DocumentProcessor` 稳定协议；只有故意验证“缺失 parser version”这类负例时，才允许在 `registry.register(...)` 边界做一次显式 `cast(type[DocumentProcessor], ...)`。市场识别一律通过 `normalize_ticker` 真源，在消费者模块路径上 monkeypatch 返回自造 `NormalizedTicker`，不要再引入 fake resolver 类。
- `tests/engine/test_processor_registry.py` 这类 registry 单测里的 dummy processor，也要补齐 `DocumentProcessor` 公共基元（`get_parser_version`、`list_sections`、`list_tables`、`read_section`、`read_table`、`get_section_title`、`search`、全文接口）；不要再用只实现 `supports/__init__` 的半截 fake 去注册。
- `tests/engine/test_processor_registry_builder.py` 这类 registry builder 测试里，若 `list_processors()` 返回的是 `dict[str, object]` 视图，就在断言边界单次 `cast` 后再做 `int(...)` 之类窄化；不要把 `object` 直接传给数值转换函数。
- `tests/engine/test_tools_base.py` 这类 tools/base 负例测试里，若故意给 `build_tool_schema()` 或 `@tool(..., truncate=...)` 传错误类型，只在该负例调用边界单次 `cast(Any, ...)`；不要让非法参数的弱类型输入污染其它测试分支。
- `tests/engine/test_truncation_manager.py` 这类 TruncationManager 负例测试里，若故意构造包含非字符串 key 的嵌套字典以覆盖失败分支，只在该单个调用边界做一次 `cast(Any, ...)`；不要把非契约输入扩散到其它正常路径测试。
- `tests/engine/test_argument_validator.py` 这类参数校验负例测试也应沿用同一原则：故意传非法 schema/object 形状时，只在该负例调用边界单次 `cast(Any, ...)`，不要让错误类型输入扩散到正常路径断言。
- `tests/engine/test_conversation_memory.py` 这类 Host conversation 测试里，`PreparedSceneState` 装配应优先使用真实 `ModelConfig` / `AgentCreateArgs` / `ToolRegistry`，并通过单次 helper 收窄 `PromptAssetStoreProtocol`；`AgentMessage` 断言则应通过 helper 读取 `role/content`，不要继续裸下标访问联合 TypedDict。
- `tests/engine/test_conversation_memory.py` 与 `tests/application/test_scene_execution.py` 这类 Host compaction / scene 装配测试，如果需要观察 `PreparedSceneState.tool_registry` 或 `ConversationCompactionAgentHandle.agent`，应在测试边界收窄到 Host 自有协议，而不是把 `AsyncAgent`、`ToolRegistry` 具体实现类型继续塞进 Host 共享协议位置。
- `tests/fins/legacy_repository_adapters.py` 这类测试兼容层模块也要跟随窄仓储协议演进：`Protocol` 方法体要显式 `...`，适配器签名要对齐 `SourceDocumentRepositoryProtocol` / `ProcessedDocumentRepositoryProtocol`，不能继续留 `object -> object` 的旧壳子。
- `tests/engine/test_search_utils_coverage.py` 这类搜索工具测试里，`SearchHit` 输入和断言都应通过 helper 在测试边界统一收窄 / 读取；不要直接把 `list[dict[str, object]]` 传进 `enrich_hits_by_section()`，也不要裸下标访问 `section_ref` / `snippet` / `page_no` 这类非必填字段。
- `tests/engine/test_search_utils_evidence.py` 这类 evidence/token-fallback 测试同样应把 `hits_raw` 先统一收窄为 `list[SearchHit]`，然后用 helper 读取 `section_ref` 等非必填字段，避免同一类 TypedDict 漂移在相邻测试文件重复出现。
- `tests/engine/test_search_utils.py` 也应沿用同一套 `SearchHit` helper 规则；不要在基础行为测试里重新退回 `list[dict[str, object]]` 或裸访问非必填字段，否则同一错误簇会在 coverage / evidence / base 三个文件来回出现。
- `tests/fins/test_upload_company_meta.py` 这类 company meta 写入测试里的仓储桩，应直接实现 `CompanyMetaRepositoryProtocol` 需要的 `scan_company_meta_inventory` / `resolve_existing_ticker` / `get_company_meta` / `upsert_company_meta`，并让 `captured` 保持 `CompanyMeta | None`，不要再退回 `object`。
- `tests/application/test_fins_service.py` 这类 Service runtime 测试里的 fake fins runtime，应直接补齐 `FinsRuntimeProtocol` 的公开方法；如果断言需要观察 `Host` 内部 executor，也要先通过 helper 收窄到测试 stub，再读 `last_spec` 这类测试字段。
- `tests/fins/test_docling_upload_service.py` 这类上传服务测试里，`build_fs_storage_test_context()` 的返回值应保持强类型 `FsStorageTestContext`，初始化负例参数与动态注入模块属性则在测试边界单次 `cast` 收窄，不要把 helper 返回值退回 `object`。
- `tests/fins/test_fins_tools_registry.py` 这类 legacy adapter + registry 装配测试里的 fake processor registry，应直接对齐 `LegacyProcessorRegistryProtocol` 的稳定签名，尤其是 `create()` / `create_with_fallback()` 的 `source: Source` 参数与 fallback 回调；不要再把私有 `DummySource` 写进公开协议位置。
- `tests/fins/test_sec_downloader.py` 这类 downloader 回调测试里的 `store_file` 测试桩，应按生产真源声明 `BinaryIO` 输入；只想覆盖 `_safe_header()` 这类 helper 时，也优先传真实 `httpx.Response`，不要手写不满足库类型的伪响应对象。
- `tests/engine/test_async_openai_runner_utils.py` 这类 Runner 辅助测试里，`ClientResponse` 输入、`AgentMessage` 列表与 `ToolExecutor` 测试桩都应在测试边界统一收窄；不要继续把 `SimpleNamespace` / `list[dict[str, str]]` / 缺半截协议方法的 executor 直接传给 `_calculate_backoff()`、`call()` 或 `set_tools()`。
- `tests/engine/test_async_agent_utils.py` 这类 `AsyncAgent` 辅助方法测试里的 runner 桩，也要补齐 `AsyncRunner.call()` 最小签名；即使测试本身不消费事件流，也不要传缺半截协议方法的 runner。
- `tests/engine/test_async_openai_runner_call_paths.py` 这类 Runner 调用路径测试里的 executor 桩，也应直接补齐 `ToolExecutor` 的稳定公开方法，尤其是 `execute(..., context=...)`、`clear_cursors()`、`get_dup_call_spec()`、`get_execution_context_param_name()` 与 `register_response_middleware()`；不要再把只够当前断言路径的半截 executor 传进 `set_tools()`。
- `tests/engine/test_async_openai_runner.py` 这类 Runner 主路径测试也应沿用同一边界：dummy executor 直接补齐 `ToolExecutor` 稳定方法，SSE 假响应通过 helper 在测试边界收窄为 `ClientResponse`，而传给 `runner.call()` 的消息列表则先统一收窄为 `list[AgentMessage]`；不要把半截 executor、裸 `FakeSSEResponse` 或 `list[dict[str, str]]` 直接送进公开入口。
- `tests/engine/test_async_cli_runner_utils.py` 这类 CLI Runner 辅助测试里，`_parse_json_event()` 的结果要先通过 helper 断言非空再传入 `_annotate_event()`；`_format_messages()` 输入应显式收窄为 `list[AgentMessage]`，而 `set_tools()` 若只验证 warning 语义，也应传满足 `ToolExecutor` 协议的最小桩，不要直接塞 `object()`。
- `tests/engine/test_async_cli_runner.py` 这类 CLI Runner 主路径测试也应沿用同一条边界：`set_tools()` 传最小 `ToolExecutor` 桩，`call()` / `_format_messages()` 的输入先统一收窄为 `list[AgentMessage]`，不要在基础行为测试里重新退回 `object()` 或 `list[dict[str, str]]`。
- `tests/engine/test_async_cli_runner_streaming.py` 这类 CLI Runner 流处理测试里，传给 `runner.call()` 的消息列表也应统一通过 helper 收窄为 `list[AgentMessage]`，不要继续直接传 `list[dict[str, str]]`。
- `tests/engine/test_tool_registry_prompt.py` 这类 prompt/tool snapshot 测试里，给函数挂 `__tool_tags__` 之类动态属性时，应在测试边界集中 `cast(Any, func)` 后再赋值，不要直接在 `FunctionType` 上裸写未知属性。
- `tests/engine/test_cancellation.py` 这类并发/取消原语测试里，不要依赖 `threading.atomic` 这类不存在或不可移植的 API；计数或同步需求统一用 `Lock + list/int` 这类标准库可移植原语表达。
- `tests/engine/test_docling_processor.py` 这类 processor 返回值测试里，`SectionSummary` / `TableContent` / `SearchHit` 的可选字段应通过 `.get()` 或 helper 读取；只有故意制造非法 `page_range` 之类负例时，才允许在测试边界对内部 block 做一次显式 `cast(Any, ...)` 注入异常值。
- `tests/engine/test_tool_registry_v2.py` 这类 ToolRegistry 边界测试里，给工具函数挂 `__tool_extra__` 这类动态属性时，应集中通过 helper 在测试边界做一次 `cast(Any, func)`；若故意验证 `execute()` 的非法 arguments 分支，也只在该负例调用点做单次收窄，不要把错误类型输入扩散到其它测试夹具。
- `tests/fins/test_sec_pipeline_process.py` 这类 pipeline 离线导出测试应通过 `monkeypatch.setattr(workflow_module, "normalize_ticker", ...)` 在消费者模块路径上替换市场识别真源；测试仓储 core 上的 `portfolio_root` / `create_filing()` 访问则通过 helper 在边界集中收窄，不要在 helper 签名里继续裸用 `object` 后直接点属性。
- `tests/fins/test_cn_pipeline_helpers.py` 这类 CN pipeline helper 测试同样应把 `build_storage_core()` 返回值先收窄到最小仓储协议，再访问 `portfolio_root/create_filing/create_material`；不要在 helper 签名里继续保留 `object` 后直接点属性。
- `tests/fins/test_sec_pipeline_process_material.py` 这类 material 处理测试也应复用同一原则：市场识别统一对齐 `normalize_ticker` 真源（在 workflow 模块路径上 monkeypatch），fake processor 直接补齐 `DocumentProcessor` 的全文/章节/表格稳定接口，测试仓储则通过最小协议 helper 收窄 `portfolio_root` / `create_material()`，不要把 `build_storage_core()` 返回值继续当成 `object` 使用。
- `tests/fins/test_cn_pipeline_process.py` 这类 CN pipeline 离线处理测试里的 fake processor，也应像 SEC 对应测试一样直接补齐 `DocumentProcessor` 的章节/表格/搜索/全文稳定接口；源文档仓储访问则继续通过最小协议 helper 收窄，避免 `processor_cls` 和 `repository` 两头同时漂移。
- `tests/fins/test_bs_def14a_processor.py` 这类专项 processor 单测里，marker 列表应显式按 `list[tuple[int, str | None]]` 构造；`Source` dummy 也要补齐 `content_length/etag` 等稳定字段，避免为了偷懒用半截 source 桩反复触发协议不匹配。
- `tests/fins/test_tool_snapshot_export_helpers.py` 这类 tool snapshot helper 测试里，company/blob 仓储桩应直接补齐 `CompanyMetaRepositoryProtocol` / `DocumentBlobRepositoryProtocol` 的稳定方法与签名；只有故意喂非法 `financial_statement_calls` 或 concrete service 参数时，才在单个负例调用边界做一次 `cast`，不要把半截仓储桩散落到多个 helper 断言里。
- `tests/fins/test_tool_snapshot_export.py` 这类 legacy snapshot 导出测试里的 fake repository，`store_file()` 句柄签名也要对齐 `SourceHandle | ProcessedHandle` 的联合稳定边界；不要只按 `ProcessedHandle` 写死，否则 legacy adapter 测试会反复与主仓储协议脱节。
- `tests/fins/test_fins_tools_service_helpers_coverage.py` 与 `tests/fins/test_search_mode_and_scale.py` 这类通过 legacy adapter 初始化 `FinsToolService` 的测试里，fake repository / fake processor registry 也要直接补齐 `LegacyReadRepositoryProtocol` / `LegacyProcessorRegistryProtocol` 所需的最小方法，不要只实现当前断言路径碰巧会走到的半截仓储或半截 registry。
- `tests/fins/test_search_mode_and_scale.py` 这类搜索测试中的命中列表，应显式收窄为 `list[SearchHit]`，而不是 `list[dict[str, ...]]`；若处理器桩记录查询词，也应让 `search_calls` 直接保持非空 `list[str]`，避免 `Optional[list[str]]` 在断言里反复触发无意义的 `len(...)` 报错。
- `tests/fins/test_ingestion_service.py` 这类流水线事件测试里，`DownloadEvent` / `ProcessEvent` 的 `event_type` 应统一使用 `DownloadEventType` / `ProcessEventType` 枚举成员，不要再传裸字符串常量。
- `tests/fins/test_cn_pipeline_coverage_extra.py` 这类 `CnPipeline` 覆盖测试里，即使当前断言不关心 registry 行为，也应传真实 `ProcessorRegistry` 或最小兼容桩，不要再把 `object()` 直接塞进构造器。
- `tests/fins/test_processing_helpers.py` 这类处理辅助函数测试里，若 helper 参数要求具体 `ProcessorRegistry` 或 `ProcessedDocumentRepositoryProtocol`，优先让测试桩直接继承真注册表或补齐仓储稳定方法与返回类型，而不是继续把 `create_processed/update_processed` 写成半截 `None` 返回。
- `tests/fins/test_html_financial_statement_common.py` 这类 HTML 财务报表共享层测试里，`FinancialStatementResult.statement_locator` 及其 `row_labels` 这类可选/宽类型字段，应通过 helper 安全读取并收窄，不要在断言里直接把可选键和 `object` 视图当成强类型结构使用。
- `tests/fins/test_financial_enhancer_coverage.py` 这类财务增强负例测试里，若故意验证 `headers` 含 `None` 的兼容分支，只在该调用边界单次 `cast(Any, ...)`；不要把 `list[str | None]` 直接扩散到正常签名路径。
- `tests/fins/test_bs_ten_q_processor_constructor.py` 这类专项处理器构造测试里的 source 桩，也要补齐 `content_length/etag` 等 `Source` 稳定字段，不要因为只验证构造代理逻辑就传半截 source。
- `tests/fins/test_six_k_section_processor_coverage.py` 这类 6-K 覆盖测试里，`FinancialStatementResult.reason`、`table.section_ref` 等非必填字段应通过 helper / `.get()` 安全读取；不要在断言里直接用裸下标把 TypedDict 可选键当必填键访问。
- `tests/fins/test_local_file_store.py` 这类本地文件存储测试里，自定义 `BytesIO` 子类应保持与基类一致的 `read(size: int | None = ...)` 签名，不要把标准库 IO 基类方法缩窄成更强约束。
- `tests/fins/test_sec_report_form_processors_coverage.py` 这类专项处理器覆盖测试里，`processor_cls` 参数应显式收窄到真实报告处理器类联合，而不是 `Type[object]`；传给 `_has_minimum_twenty_f_item_quality()` 的标题列表也应按 `list[str | None]` 构造，避免在 20-F 质量门禁测试里反复触发 list 不变性报错。
- `tests/fins/test_processor_registry_builder.py` 这类 fins registry builder 测试，也应沿用 engine 对应文件的同一规则：`list_processors()` 返回的 `priority` 若仍是 `object` 视图，就在断言边界单次 `cast` 后再做数值比较。
- `tests/fins/test_sec_6k_rule_diagnostics.py` 这类诊断脚本测试里，若 `kwargs` 被故意声明成宽 `object` 以覆盖入口边界，真正消费 `tickers/document_ids` 这类字段时应在该读取点单次收窄，不要直接把 `object` 传给 `list(...)` 或其它 iterable API。
- `tests/fins/test_sec_pipeline_http_cache.py` 与 `tests/fins/test_sec_pipeline_rejection_registry.py` 这类 `SecPipeline` 测试里的市场识别，应通过 monkeypatch 在消费者 workflow 模块路径上替换 `normalize_ticker`，不要再引入 fake resolver 类，也不要向 `SecPipeline(...)` 构造器传入任何识别器注入参数。
- `tests/fins/test_sec_pipeline_download_stream.py` 这类下载流测试同样要沿用真实构造边界：市场识别对齐 `normalize_ticker` 真源（在 workflow 模块路径上 monkeypatch），fake downloader 直接继承 `SecDownloader` 并对齐 `download_files_stream()` 的 `store_file` 签名；不要把“长得像下载器”的普通类直接塞进 `SecPipeline(...)`。
- `tests/fins/test_section_semantic.py` 这类 section semantic 测试里，`resolve_section_semantic()` 返回的标题是 `str | None`，在做子串断言前要先显式断言非空，不要直接对可空字符串做 `in` 判断。
- `tests/application/test_execution_runtime_config.py` 这类运行配置快照测试里，`build_runner_running_config_from_snapshot()` 的返回值在断言 OpenAI 字段前要先显式收窄到 `OpenAIRunnerRuntimeConfig`；构造 `AgentRuntimeConfig` 时也应直接使用 `FallbackMode` 枚举，而不是再传裸字符串。
- `tests/application/test_cancellation_bridge.py` 这类 Host polling 测试里的 mock run registry，如果只实现了 `get_run()` 这一条被测路径，可在 `CancellationBridge` 构造边界单次 `cast(RunRegistryProtocol, ...)`；不要为消除 pyright 报错把整套 `RunRegistryProtocol` 在测试里机械复刻一遍。
- `tests/engine/test_tool_registry_extra.py` 这类 ToolRegistry 附加分支测试里，动态 `__tool_extra__` 元数据与故意无效的 truncate spec 都应在 helper / 负例边界集中收窄；response middleware 的 `context` 参数也要直接按 `ToolExecutionContext | None` 声明，避免再次退回宽 `dict | None`。
- `tests/engine/test_markdown_processor.py` 这类 MarkdownProcessor 基础行为测试，也应沿用同一套 `SearchHit` helper 规则：对 `section_ref`、`snippet` 这类非必填字段统一通过 helper / `.get()` 安全读取，不要在基础断言里退回裸下标访问可选键。
- `tests/application/test_reply_outbox_web_integration.py` 这类 Web reply outbox 集成测试里，若用假 `fastapi` / `pydantic` 模块装配 router，应像 `test_web_routes.py` 一样在 `ModuleType` 边界集中 `cast(Any, module)` 后挂属性；Host 注入与 route handler 返回值也要分别在构造点、`asyncio.run(...)` 调用点收窄到真实协议 / `ReplyDeliveryView`，不要把 `object` / 宽 `Coroutine` 类型一直带到断言阶段。
- `tests/application/test_wechat_daemon.py` 与 `tests/application/test_wechat_outbox_integration.py` 这类 WeChat 路径测试里的 fake chat service，也应直接补齐 `ChatServiceProtocol` 的 `submit_turn` / `resume_pending_turn` / `list_resumable_pending_turns`；对 `ExecutionDeliveryContext.filtered`、`wechat_runtime_identity` 这类非必填字段则通过 helper / `.get()` 安全读取，不要在断言里裸下标访问。
- `tests/integration/test_config_loader_e2e.py` 这类 ConfigLoader 端到端测试里，联合模型配置上的 `runner_type`、`headers` 等非必填键应通过 `.get()` 与 helper 收敛后再断言；不要把 `CliModelConfig | OpenAICompatibleModelConfig` 的联合直接当成所有键都存在的字典来裸下标访问。
- `tests/application/test_write_service.py` 这类写作服务测试不应再用 `SimpleNamespace` 伪造 `WorkspaceResourcesProtocol` 或 `SceneExecutionAcceptancePreparer`；优先复用真实 `WorkspaceResources`，并用最小强类型 fake 实现 config loader / prompt asset store / scene acceptance preparer。
- `tests/application/test_scene_execution.py` 这类 Host scene 装配测试里，`workspace` 应直接使用真实 `WorkspaceResources`；如果要覆盖 `config_loader` 行为，就提供最小 `ConfigLoaderProtocol` 桩，不要在协议属性位上回填 `SimpleNamespace` 或 `object`。
- `tests/engine/test_write_pipeline_contracts.py` 这类写作流水线 contract 测试里的 helper，应优先接真实 `TemplateChapter` 之类稳定模型，而不是用 `object` 再直接点属性访问 `chapter_contract`。
- `tests/application/test_host_commands.py` 这类宿主管理测试若需要断言默认 `Host` 内部组件，应通过 helper 在测试边界把 `_executor` / `_session_registry` 收窄到 `DefaultHostExecutor` / `SQLiteSessionRegistry` / `HostStore`，不要把协议视图上的私有属性访问直接散落在正文。
- `tests/application/test_event_bus.py` 这类 EventBus 测试里，若要覆盖内部慢消费者/队列溢出分支，应先把 `subscribe()` 返回值通过 helper 收窄到内部 subscription 实现，再操作 `_queue/try_put()`；`run_registry` 若只为 session 过滤提供 `get_run()`，也应在 `AsyncQueueEventBus(...)` 构造边界单次收窄为 `RunRegistryProtocol`。
- `tests/engine/test_async_openai_runner_tools.py` 这类 Runner 工具批次测试里的 dummy executor，应直接补齐 `ToolExecutor` 协议缺失的方法；若需要故意返回非法工具结果，也应在 `execute()` 返回点做单次 `cast`，而不是保留与协议冲突的返回签名。
- `tests/engine/test_async_agent_coverage_extra.py` 这类 `AsyncAgent` 补充覆盖测试也应沿用同一原则：runner 桩的 `call()` 入参显式写成 `list[AgentMessage]`，tool executor 桩补齐 `execute/get_dup_call_spec/get_execution_context_param_name/register_response_middleware` 等稳定方法；`run_messages()` 输入统一通过 helper 构造强类型消息。
- `tests/engine/test_sec_processor.py` 这类 SEC processor 测试里，`TableSummary` / `SearchHit` / `FinancialStatementResult` / `XbrlFactsResult` 的可选字段访问都应通过 helper 收口；fake XBRL query provider 若只想覆盖 `_infer_units_from_xbrl_query()` / `_query_facts_rows()`，应在测试边界单次收窄为 `XBRL`，不要散落 `type: ignore[arg-type]`。
- `tests/engine/test_sec_processor.py` 在 `SecProcessor` 拆分或装配链调整时，还要继续守住全文回退语义：单全文章节预加载与运行时 `get_full_text()` 两条路径都必须在 `document.text()` 为空时统一回退到原始 HTML 抽文，避免 `20-F/iXBRL` 文档重新产出空全文。
- `test_web_tools.py` 需要守住 web fetch 的机械恢复规则：requests 侧 `Accept-Encoding` 只能声明当前运行时真的支持的编码；立即 `meta refresh` 要在抓取层继续跟随，且每一跳都要按剩余 budget 重算 timeout；不支持的内容编码、timeout、部分 SSL/TLS 握手失败、一小组浏览器更容易恢复的 `412/521` 类状态，以及带前端壳页特征的 challenge/抽取失败，都要优先导向既有浏览器回退，而不是把脏 HTML 继续送进正文抽取。
- `test_web_tools.py` 还要守住取消语义：requests 主路径要在 warmup、probe、下载、meta refresh 跟随与正文转换的阶段边界观察 linked `CancellationToken`；走到浏览器回退时，同一执行令牌也必须继续透传。
- `test_web_tools.py` 还需要守住第二阶段下沉后的 HTTP 真源：timeout/session 相关 helper 的真源在 `dayu.engine.tools.web_http_session`，内容编码与字符集解析真源在 `dayu.engine.tools.web_http_encoding`；新增 monkeypatch 或直接导入时不要再默认绑回 `web_tools.py` 本地实现。
- `test_web_tools.py` 还需要守住第二阶段继续下沉后的 fetch 编排真源：warmup、content-type probe、HTML/Docling 路由、meta refresh 跟随与 requests->browser 升级判定的真源在 `dayu.engine.tools.web_fetch_orchestrator`；`web_tools.py` 里的同名函数现在只是兼容测试锚点的薄包装。
- `test_web_tools.py` 与 `test_web_playwright_backend.py` 还需要共同守住第三阶段下沉后的 Playwright 真源：浏览器单例、子进程 worker、storage state 解析、deadline 预算与浏览器回退执行的真源在 `dayu.engine.tools.web_playwright_backend`；`web_tools.py` 中对应函数现在主要承担兼容包装职责，而 backend 真源分支应优先在专门测试文件中直接覆盖。
- `test_web_tools.py` 还需要守住 Playwright 的硬收口：默认回退执行应优先在子进程边界内完成，timeout 或取消后父进程必须终止 worker，而不是继续接受“后台线程可能仍在运行”的软超时语义。
- `test_web_tools.py` 还要守住 Playwright 结果回传时序：父进程必须先消费 result queue，再等待 worker 完全退出；同时要允许“进程已退出但队列结果稍后才可见”的短暂 drain 窗口，避免把成功抓取误报成 timeout 或 worker exited without result。
- `test_web_tools.py` 还需要守住 `search_web` 的内部边界：provider 选择、API key 缺失回退、DuckDuckGo URL 解析与结果摘要组装的真源在 `dayu.engine.tools.web_search_providers`，而不是重新塞回 `web_tools.py`。
- `test_web_tools.py` 还需要守住 challenge 误判边界：DataDome 这类 vendor header 仍是强挑战信号，但单独的清理型 vendor cookie 不能直接把 200 正常正文页判成 blocked。
- `test_web_tools.py` 还需要守住 storage state 的双路径收敛：按 host 命中的 Playwright storage state 不仅要传给浏览器回退，也要能把其中 cookie 注入 requests 主路径，避免像 Bloomberg 这类站点仍然卡在 403 robot page。
- `test_web_tools.py` 还需要守住浏览器回退画像：Playwright context 要继续带现代导航头、支持按 host 及 `www` 变体命中 storage state，并在受控范围内做首页预热与页面稳定化等待。
- `test_web_tools.py` 还需要守住旧站点字符集纠偏：当响应头缺失 charset、但 HTML meta 或 `apparent_encoding` 已暴露 `gb2312/gbk/gb18030` 等编码时，正文抽取前必须先修正解码，不能直接信任 `response.text` 的默认拉丁编码结果。
- `test_web_tools.py` 的测试桩也要跟随真源边界：requests 会话桩应直接继承 `requests.Session` 并保持 `get/head` 与基类兼容的签名；故意喂非法 `domains`、`MaxRetryError(pool=None)`、函数 `__tool_extra__` 或 `ModuleType` 动态属性时，只在该负例/注入边界单次 `cast(Any, ...)`，不要把宽类型和动态属性写入散落到正文断言里。
- `test_log.py` 需要守住两条日志边界：全局默认日志分流语义仍是 stdout 保留全量日志、`ERROR` 及以上同时走 stderr；日志真源仍是 `dayu.log`，不要把 `Log` / `LogLevel` 再包装回 `dayu.engine`。
- 这类 `build_async_agent` 边界测试里的 mock 需要跟随公共契约升级：当前 `trace_identity` 是 `AgentTraceIdentity`，如果测试要断言字段值，应显式转成 `to_metadata()` 后再比较，而不是继续把它当成可迭代字典输入。
- 对 CLI 装配测试，优先 patch `_prepare_cli_host_dependencies` 与按需 `Service builder` 这类稳定边界；不要重新引入“一次性创建所有 Service”的 bundle runtime 测试夹具。
- 当前这类贯通测试至少要覆盖三段：
  - `run.json` 到 `accepted_execution_spec` 的真实传递
  - `accepted_execution_spec` 到 `AgentCreateArgs` 的真实传递
  - scene manifest / `toolset_registrars.json` / `execution_permissions` / `toolset_configs` 到 registrar adapter，再到 `register_doc_tools / register_fins_read_tools / register_web_tools` 的真实传递
  - `ExecutionOptions.toolset_config_overrides` 对上述运行参数的真实覆盖；如果入口来自 CLI/WeChat，就要验证这些入口已先把显式参数收敛成 `toolset_config_overrides`
  - scene manifest / prompt assets / `context_slots` / `llm_models.json.temperature_profiles` 到最终 `system prompt` 与 `AgentCreateArgs.temperature` 的真实传递
- Host session/run 的 Web/CLI 展示只允许使用 Host 自己的通用字段，不应重新依赖 `ticker` 这类领域字段。
- `test_session_registry.py` 与 `test_run_registry.py` 需要守住 SQLite schema 不再回退到 `ticker` 这类领域列。

### 3.3 Prompt 资产变更

当 scene manifest、scene prompt、task prompt 或 `context_slots` 变化时，至少同步更新：
- `tests/engine/test_prompt_assets.py`
- `tests/engine/test_prompt_composer.py`
- `tests/engine/test_prompts_tags.py`
- 必要时的 `tests/engine/test_write_pipeline.py`

重点守护：
- scene `model.default_name`
- `model.allowed_names`
- `model.temperature_profile`
- `runtime.agent.max_iterations`
- `runtime.agent.max_consecutive_failed_tool_batches`
- `runtime.runner.tool_timeout_seconds`
- `conversation.enabled`
- `context_slots`
- `tool_filters`
- 测试替身里的 `load_task_prompt()` 必须继续返回模板字符串；不要把 task prompt 文本和 task contract schema 混成同一返回类型。
- prompt asset store 的测试替身应继续返回稳定 prompt asset schema；scene 优先对齐 `SceneManifestAsset`，task contract 优先对齐 `TaskPromptContractAsset`，不要再退回宽泛的 `dict[str, Any]` 或 `object`。
- prompt asset 断言优先依赖 `load_scene_definition()`、`parse_task_prompt_contract()` 等解析后稳定对象；只有在验证文件边界本身时才直接深层索引 raw manifest / contract。
- `tests/engine/test_prompt_assets.py` 这类文件边界测试需要继续守住 repair task prompt 的低熵执行口径：`resolution_mode`（至少 `delete_claim`）和 `target_kind`（`substring|line|bullet|paragraph`）必须在 prompt 文本里被显式解释成下一步动作，同时 `patches` 不允许再回到空数组口径。

另外，若实现涉及 `resume`，测试必须守住 `conversation.enabled=true` 是 V1 恢复的硬门槛；`conversation.enabled=false` 的 scene 不得静默进入恢复路径。

### 3.4 写作链路变更

写作链路调整时，至少同步更新：
- `tests/application/test_write_service.py`
- `tests/engine/test_write_pipeline.py`
- `tests/engine/test_report_assembler.py`
- `tests/engine/test_execution_summary_builder.py`
- `tests/engine/test_write_pipeline_contracts.py`

另外：
- 写作流水线的 regenerate / repair 路径必须有非空 `audit_decision` invariant；若状态机调整，这个 guard 需要在 `tests/engine/test_write_pipeline.py` 继续守住，不能退回“靠流程顺序默认成立”。
- write pipeline 现在已把单章状态机与审计协调下沉到内部协调器，并把最终报告组装、运行摘要聚合拆到独立协作者；`tests/engine/test_write_pipeline.py` 需要继续守住 `WritePipelineRunner` 只做 orchestration/delegation，不要把审计、rewrite、报告组装或摘要聚合重新塞回 Runner。
- write pipeline 的审计规则码现在统一收口到 `dayu/services/internal/write_pipeline/enums.py`；`tests/engine/test_write_pipeline.py` 在构造 `Violation`、`EvidenceConfirmationEntry`、`RemediationAction` 等内部模型时，应直接使用 `AuditRuleCode` 真源，而不是继续散落规则字符串；只有 prompt / JSON payload 这类外部协议表面才保留字符串形态。
- `tests/engine/test_write_pipeline.py` 对 audit / rewrite 的细粒度断言要直接命中 `ChapterAuditCoordinator` 与 `ChapterExecutionCoordinator` 真源，不能再通过 `WritePipelineRunner` 保留纯透传 facade 测试入口。
- `tests/engine/test_write_pipeline.py` 还要守住章节状态机的 rewrite 计数边界：`retry_count` 只表示已成功提交的重写次数，`PREPARE_REWRITE` 只能生成待执行序号；若 regenerate / repair 在本轮执行中抛错，`retry_count` 与 `rewrite_history` 都不能提前加一。
- `tests/application/test_execution_runtime_config.py` 负责守住运行配置快照恢复语义，包括字符串数字快照仍按当前规则恢复成数值；不要把这类恢复逻辑散落到调用方测试里。

### 3.5 Fins 相关变更

Fins 相关改动时，至少同步更新：
- `tests/fins/test_fins_tools_service.py`
- `tests/fins/test_fins_tools_registry.py`
- `tests/fins/test_ingestion_tools.py`
- `tests/fins/test_storage_split_repositories.py`
- 受影响的 pipeline / processor 测试
- `tests/fins/test_cli_formatters_coverage.py` 需要继续守住 Fins pipeline 事件即使内部收口为枚举，也保持原有小写字符串值兼容 CLI 输出与 fixture 构造。
- `tests/fins/test_search_mode_and_scale.py`、`tests/fins/test_fins_tools_service.py` 与 `tests/fins/test_fins_tools_service_helpers_coverage.py` 还要守住 Fins tools 拆分边界：搜索模型、搜索引擎和通用 helper 的测试必须分别直接绑定 `search_models.py`、`search_engine.py`、`service_helpers.py` 真源，不能再经由 `service.py` 兼容出口导入私有符号。
- `tests/engine/test_cli_running_config.py` 与 `tests/fins/test_fins_runtime_tool_service.py` 还要继续守住 direct operation 的 CLI 规范化真源：`_build_fins_command()` 必须先经过 `prepare_cli_args`，runtime 在进入 pipeline 前也必须消费规范化后的 ticker、alias、公司名与上传进度事件名，不能把原始 CLI god-bag 重新透传下去；同时 `upload_filing/upload_material --action delete` 必须允许 `files=None` 进入 runtime，而不是在命令构建阶段提前报错。
- `tests/fins/test_sec_pipeline_download_stream.py`、`tests/fins/test_sec_pipeline_upload_*_stream.py`、`tests/fins/test_cn_pipeline.py` 与 `tests/fins/test_upload_progress_helpers.py` 需要共同守住 Fins pipeline 事件类型边界：pipeline 对外事件应断言对应 `StrEnum` 成员，upload 内部 `UploadFileEventPayload` 若仍使用窄字面量类型，必须通过显式映射 helper 转成 `UploadFilingEventType` / `UploadMaterialEventType` 后再向外发射；`process` 相关桥接测试同时要守住 `document_ids` 不会在 ingestion backend/service 协议层丢失。
- `tests/fins/test_cli_helpers_coverage.py` 这类 CLI 辅助测试里的 pipeline 桩，应直接补齐 `PipelineProtocol` 的稳定同步/流式方法签名；构造 download/process/upload 事件时优先使用 `DownloadEventType` / `ProcessEventType` / `Upload*EventType` 枚举真源，不要一边用半截 pipeline stub、一边把字符串字面量直接塞进事件 dataclass。
- `tests/fins/test_sec_6k_rule_diagnostics.py` 需要守住 SEC 6-K 诊断遍历 active filings 的容错边界：active source 目录若缺失 `meta.json`，诊断应跳过坏目录继续收集样本，而不是让单个残缺 filing 中断整批规则分析。
- `tests/fins/test_sec_6k_primary_document_diagnostics.py` 需要守住另一类同源 root cause：当同一 6-K filing 里 `primary_document` 指向的是治理/持股/会议类 exhibit，但目录中另一个 exhibit 才是当前真源 `_classify_6k_text()` 会保留的季度正文时，诊断必须把它识别成“选文错位”而不是继续误归因为规则缺陷；若当前 `primary_document` 自身已经是季度结果，则不能误报。
- `tests/fins/test_sec_downloader.py` 与 `tests/fins/test_sec_pipeline_helpers.py` 还要共同守住 6-K 候选收集与提升边界：下载器必须把 `TYPE=6-K` 的 HTML cover 与 `EX-99.x` exhibit 一起带进预筛选，避免季度正文挂在非 exhibit cover 时被候选集漏掉；同时预筛选不能让旧 `primary_document` 的 `EXCLUDE_NON_QUARTERLY` 结果短路压过同 filing 里已命中季度结果的其它 candidate。`tests/fins/test_sec_downloader.py` 还要额外守住主 `6-K` cover 的补链行为：若 cover 内存在同 filing、同目录、相对路径的 HTML 附件链接，下载器必须把这些链接补进远端文件集合，避免像 `d940644dex1.htm` / `d11619606_ex1.htm` 这类真实财务附件因为 index 列表缺项而根本没被下载。
- `tests/fins/test_sec_download_source_upsert.py` 负责守住下载成功后的 source meta upsert 真源边界：`first_ingested_at/created_at` 的稳定性、create/update 分支选择、以及 `processed` 的 `reprocess_required` 只在“首次重建已有 processed”或“历史 source_fingerprint 发生变化”两类场景下被标记。
- `tests/fins/test_sec_company_meta.py` 负责守住 SEC company meta 真源边界：主 ticker 置顶、alias 大写去重、SEC submissions alias 提取降噪、以及公司级 meta 写入时的 `company_name` 回退规则。
- `tests/fins/test_sec_safe_meta_access.py` 负责守住 source/company/processed meta 的安全读取收口与 document version 计算边界：不存在时返回 `None`、filing staging 读取失败时不向上冒泡、版本号只在历史 `source_fingerprint` 变化时递增。
- `tests/fins/test_sec_sc13_filtering.py` 负责守住 SC13 专项真源边界：方向判定只保留“别人持股我”，同一 filer 只保留最新一份，且缺失 warning 只在显式请求 SC13 且结果为空时触发。
- `tests/fins/test_sec_rebuild_workflow.py` 负责守住 rebuild 真源边界：本地过滤条件的 form/date 收敛、以及单 filing 重建时 `document_version/source_fingerprint` 保持稳定并通过 replace-source-meta 清理历史脏字段。
- `tests/fins/test_rejected_6k_rescue.py` 还要守住 `.rejections/` 枚举容错边界：若个别 rejected artifact 目录缺失 `meta.json` 或元数据损坏，rescue 必须跳过坏目录继续处理其它候选，不能让单个脏目录中断整批误拒救回。
- `tests/fins/test_active_6k_retriage.py` 还要守住 active filing 枚举容错边界：若个别 active `6-K` 目录缺失 `meta.json` 或元数据损坏，retriage 必须跳过坏目录继续处理其它 active 样本，不能让单个脏目录中断整批误收剔除。
- `tests/fins/test_pipeline_cli.py` 与 `tests/fins/test_ingestion_job_manager.py` 还要继续共同守住 `document_ids` 透传链路：CLI `process` 子命令、job manager 和 ingestion backend 必须把显式文档过滤以等价语义传到 `process_stream(...)`；允许做去空、去重和稳定排序，但不能改变过滤结果，也不能让等价请求绕过去重复用。
- `tests/fins/test_fins_runtime_tool_service.py` 与 `tests/fins/test_cli_formatters_coverage.py` 现在还要继续守住 direct operation 的强类型消费边界：`runtime.execute()` 的测试必须先显式收窄 `FinsResult | AsyncIterator[FinsEvent]`，formatter 测试构造 pipeline 事件时优先使用 `DownloadEventType` / `ProcessEventType` / `Upload*EventType` 枚举真源，未知事件分支才允许显式 cast 到窄类型做兜底覆盖。

另外，processor/source 相关测试现在要守住两条契约：
- `Source` 是只读结构化协议，测试里的 dummy source 应写成普通类做结构兼容，不要显式继承 `Source`。
- 处理器版本契约以 `get_parser_version()` 为准；若需要断言 parser version，优先校验方法返回值，而不是要求协议声明类变量。
- Engine 与 Fins 的 processor 测试要继续守住 HTML 规则边界：`tests/engine/` 只验证通用 HTML/Text 基元和扩展钩子，EDGAR SGML 剥离、SEC 封面页 layout 识别、section heading 横线表识别等领域规则必须放在 `tests/fins/`，不能再由 Engine 测试钉死。
- direct operation 公共契约测试还要守住 `dayu.contracts.fins` 已按命令拆成强类型 payload/result/progress dataclass；`tests/application/test_fins_service.py`、`tests/fins/test_fins_runtime_tool_service.py`、`tests/engine/test_cli_running_config.py` 与 `tests/fins/test_cli_formatters_coverage.py` 不应再回退到构造跨层 `Dict[str, Any]` god-bag。
- 共享 processor helper 的测试要绑定公共真源而不是具体处理器私有实现；例如 HTML 表格 DataFrame 解析应锚定 `dayu.engine.processors.table_utils.parse_html_table_dataframe()`，不要再 monkeypatch `bs_processor.py` 内部下划线函数。
- source storage 相关测试还要继续守住共享真源分工：`tests/fins/test_storage_split_repositories.py` 负责仓储事实接口，`tests/fins/test_sec_pipeline_download_stream.py` / `tests/fins/test_sec_pipeline_download.py` 负责 pipeline 只能消费仓储结论，`tests/engine/test_cli_running_config.py` 负责 CLI startup 只能消费 `has_local_filings` 这类上层事实，而不是直接读取 `portfolio/` 目录布局。
- Engine / processor 测试若覆盖页面摘要或搜索 evidence，断言应对齐稳定 TypedDict 契约，不要继续验证 helper 时代遗留的私有字段，例如页面 `start_on_page/end_on_page` 或 evidence `match_position`。
- SEC processor / virtual section 测试还要守住两个稳定输出：`SectionContent.children` 只承载直接子章节导航摘要，`FinancialStatementResult` 在失败分支也必须保持完整必填字段形状。
- `tests/fins/test_sec_form_section_common_coverage.py` 这类 virtual section 测试若直接构造 marker，应显式对齐 `list[tuple[int, str | None]]` 真源；不要继续把 `list[tuple[int, str]]` 直接传给 `_build_virtual_sections()` 或作为 `_build_markers()` override 返回值。
- `tests/fins/test_virtual_section_table_assignment.py` 这类 virtual section 表格映射测试，构造底层表格与章节桩时应优先使用 `build_table_summary()` / `build_section_summary()` / `build_section_content()` 等 TypedDict helper，并让 `_build_markers()` override 显式返回 `list[tuple[int, str | None]]`；不要再用半截 `TableSummary(...)` 或宽 `dict[str, object]` 返回值碰运气过类型检查。
- processor/search 测试若断言 `SectionContent.children`、`SearchHit.section_ref`、`SearchHit.snippet` 这类非必填 TypedDict 字段，应优先通过测试 helper 做一次安全读取，不要在断言里散落裸下标访问。
- `tests/fins/test_sec_report_form_common.py` 这类 report form fallback 单测里，写入处理器 `_tables` 时应优先构造真实 `_TableBlock`，不要把 `SimpleNamespace` 直接塞进处理器内部字段；对 `FinancialStatementResult.reason` 这类非必填字段则继续通过 helper / `.get()` 安全读取。
- 20-F 相关测试还需要继续守住三类专项回归：年报样式 `Item 18` 页码 locator/front matter 污染替换、句中引号包裹的 `Item 5` 交叉引用过滤，以及无 XBRL 报告类表单通过宽松 row-signal 进入 HTML 财报 fallback；对应优先更新 `tests/fins/test_bs_twenty_f_processor.py`、`tests/fins/test_sec_form_section_common_coverage.py`、`tests/fins/test_report_form_financial_statement_common.py`。
- 20-F 相关测试还要继续守住 `Item 18` 的 ToC 误判边界：真实标题后紧跟 `page F-1` 等正文 locator 时，key-heading re-search 不能把它误杀成目录行；但 `Item 18. Financial Statements 180 Item 19. Exhibits 185` 这类单行 ToC stub 仍必须被过滤。
- 20-F 相关测试还要守住一条性能短路约束：`Item 18` 正文白名单与 locator guide 判定前，必须先用局部 probe / standalone heading 过滤非相关候选；新增 regression case 时，优先补 `tests/fins/test_bs_twenty_f_processor.py` 中“非 Item 18 不进入逐行邻域扫描”“正文句内命中不进入 guide 判定”这两类断言。
- 20-F BS 路径测试还要继续守住初始化阶段的性能契约：当默认 DOM 抽文与正式 virtual-section 切分使用的是同一份全文时，`BsTwentyFFormProcessor` 不能重复重建同一份 marker；同时，front matter 命中的 `标题 + 页码` 行也不能再进入 annual-report page-heading 的重判定。
- 20-F key-item 回归还要额外守住一类顺序修复缺陷：当 `tests/fins/test_bs_twenty_f_processor.py` 中已经能从正文 heading 拼出完整且单调的 `Item 3 < 5 < 18` fallback 主链时，晚期 `Not applicable` / guide / 交叉引用 marker 不能再把这条主链顶掉，否则 `Item 5` 会在最终 monotonicity 修正里被错误删除。
- 20-F key-item 回归还要继续守住另一类 annual-report-style 缺陷：如果只先恢复出干净 `Item 5`，而前面保留下来的 `Item 1-4A` 全是 guide/front-matter 污染位，最终 monotonicity 修正也不能把这个 `Item 5` 再删掉；否则会把 `UL/SNN/BP` 一类文档重新打回 `缺失关键 Item（Item 5）`。
- 20-F key-item 回归还要继续覆盖两类新缺陷：partial guide reconstructed 若只能补出 `Item 3/4`，测试必须断言处理器会把它与现有 `Item 5/18` 主链合并，而不是在 merge 前提前短路；同时，当文末 `Item 1/2/3` 全部聚成尾部目录簇时，测试也必须断言 synthetic `Item 3` 不会再被这些污染尾标当作顺序下界顶回文末。
- 20-F key-heading repair 的回归还要继续守住“末尾 monotonicity 不得重新删 key item”：若 `repaired` 阶段已经恢复出完整 `Item 3/5/18`，但后续 `Item 6/7/9/10/...` 非关键链与之冲突，测试必须断言处理器优先保留 `Item 5/18`，而不是在最终单调化后重新退回 `缺失关键 Item（Item 5）` 或 `缺失关键 Item（Item 18）`。
- 20-F BS 路径回归还要守住 DOM 抽文边界：当 `tests/fins/test_bs_twenty_f_processor.py` 中 HTML 的 Item 标题由多个相邻节点组成时，`BsTwentyFFormProcessor` 不能再依赖 `strip=True` 的基类抽文结果；否则 `CYBR` 一类文档会在主路径就丢掉 `Item 5` 并错误回退到旧处理器。
- 20-F key-item 回归还要覆盖 `BAP` 这类 repair 顺序缺陷：若重搜得到的 `Item 5` fallback 早于现有 `Item 4A`，测试必须断言处理器保留当前顺序内的 `Item 5`，而不是接受这个更早伪命中后再在 monotonicity 阶段把 `Item 5` 删除。
- 20-F guide repair 的回归还要单独守住 `report-suite cover` 误命中：当 locator phrase 命中真实 `Form 20-F` 之前的 `Integrated Annual Report` / `Annual Financial Report` 封面或封面目录页时，测试必须断言处理器把它识别成无效锚点，而不是把 `Item 8/9/10` 边界提前到册子封面。
- 20-F guide/snippet 回归还要继续守住真实锚点白名单：`Form 20-F references` 与 `SEC Form 20-F cross reference guide` 都必须能进入 locator repair；同时 `Item 5` 的 key-heading 回归也要覆盖 `Financial performance` 这类年报样式标题，避免 `UL/NWG` 一类文档再次退回 `缺失关键 Item`。
- 20-F 大文档回归还要继续守住性能边界：对 `Item 18` 白名单、guide 过滤和 inline cross-reference 过滤，测试至少要覆盖“明显不可能命中时不会再触发昂贵判定”的场景，避免 `BILI/GFI` 这类超大 annual-report-style 文档再次在 BS 主路径初始化阶段超时。
- 20-F / 10-K 标题恢复的性能回归测试还要继续守住“先保真、后短路”的顺序：annual-report-style `20-F` 中，真实 page heading 即使前面紧邻 `Form 20-F caption` / `location in this document` 也不能被 front matter 过滤抢先丢掉；`10-K` 的 ToC 局部 probe 也不能因为窗口太短而放过带长导点或长空白的目录 stub。
- 10-K fallback 回归也要覆盖性能护栏：当命中附近没有 `Table of Contents`、页码引用或其它 `Item` 编号时，测试必须断言不会进入完整的 ToC 正文探测；对应优先更新 `tests/fins/test_bs_ten_k_processor.py`。
- 6-K 相关测试还需要继续守住 image+OCR exhibit 回归：即使原始 HTML 完全没有 `<table>`，只要白色 1px / 1pt 隐藏文本页中存在 statement title、期间表头和成组数值，`BsSixKFormProcessor.get_financial_statement()` 也必须能回退提取 `income` / `balance_sheet`，避免 `core_available=[]` 的系统性 GateFail 重新出现。
- 同一组 6-K 覆盖测试还要继续守住 OCR 回退的两类扩展 source 形态：`Page1/Page2` 这类 fixed-layout HTML 容器，以及 PDF/演示稿转 HTML 后形成的“伪表格页文本”。只要页文本里存在 statement title、`1Q25/4Q24/FY24` 这类期间 token 和成组数值，测试就要断言处理器能通过统一 OCR 回退链抽出核心报表，而不是把这类样本重新误判成普通 `low_confidence_extraction`。
- 同一组覆盖测试还要额外守住 Workiva / slide 版式：若隐藏 OCR 文本挂在图片页容器内的 `<font style="font-size:1pt;color:white">`，测试也必须断言处理器能按图片页重新聚合文本，而不是只认 `<p style="font-size:1px;color:white">`。
- 同一组覆盖测试还要继续守住 hidden OCR 的样式简写边界：若页内只留下 `font: 0.01px/115% ...; color: White` 这类 `font:` shorthand 或更小字号变体，测试也必须断言处理器能通过统一样式归一化识别它，而不是因为不是字面 `font-size:1px` 就把真实财报页重新漏掉。
- 同一组覆盖测试还要继续守住 `page-break` 分页文本回退：若 6-K 正文没有标准 `<table>`，而是靠 `page-break-before/after: always` 把 statement 文本切成多页段落，测试也必须断言处理器能按分页边界重建页级文本，并继续从中提取 `income` / `balance_sheet` / `cash_flow`，而不是把这类 results release 重新误判成 `statement_not_found`。
- 同一组覆盖测试还要继续守住 `Profit & Loss` OCR 摘要页：若 6-K / earnings deck 只提供 `Profit & Loss` 标题下的当前期间金额摘要，而不再具备标准多期间表头，测试也必须断言处理器仍能回退提取单期间 `income`，避免 `BBVA` 这类真实 results summary 重新掉回 `core_available=[]`。
- `tests/fins/test_html_financial_statement_common.py` 这类共享 HTML 财报解析测试还要继续守住期间表头回归：真实期间行即使被空行、标题行和币种/`Unaudited` 行压到更深位置，也必须继续向后探测；`31-Dec-24 / 31-Mar-25`、`Fourth Quarter 2024`、`Quarter endedSep 2025` 这类表头 token 也必须被解析成稳定的 `period_end / fiscal_period`，不能让共享层因为表头更深或日期写法变化把 `mean_row_count` 又打回 `0`。
- `tests/fins/test_html_financial_statement_common.py` 还要继续守住单期间摘要表回归：若 caption/header 不再显式重复日期，但表前 context 已写明 `for the three-month period ended ...`，而表内只剩一个数值列，共享层也必须能补出单期间 `period_end / fiscal_period`，不能把 `YPF/EDN` 这类本地交易所结果摘要重新打回 `None`。
- `tests/fins/test_six_k_section_processor_coverage.py` 还要继续守住本地交易所单期间摘要表：当 `6-K` 主文件只给出 `Net profit for the period` 这类单期间 income summary 时，处理器至少要稳定返回 `core_available=['income']`，避免真实季度结果重新掉回 `core_available=[]` 的 HGF。
- `tests/fins/test_six_k_section_processor_coverage.py` 还要继续守住“标题 table + 数据 table”拆分版式：若 statement title 独立放在一个 3 行标题 table 中，而期间列和数值行落在紧邻的未命名 table，测试也必须断言 `BsSixKFormProcessor` 会把后继数据 table 拼回同一 statement 候选，避免 `WB` 这类真实季度结果因为只命中标题 table 而重新掉回 `core_available=[]`。
- SEC 6-K 规则改动还要同时更新三类测试：
  - `tests/fins/test_sec_pipeline_download.py` 与 `tests/fins/test_sec_pipeline_rejection_registry.py`
    这里负责守住 policy reject 会同时写 skip index 和 `.rejections/` artifact，且 rejected artifact 不污染 active filings
  - `tests/fins/test_sec_pipeline_helpers.py`
    这里负责守住 `_classify_6k_text()` 的文本同源回归，优先用最小化 false positive / false negative 样本覆盖“强排除 / 强保留 / 中性信号”和封面页与 exhibit 的优先级；同时还要守住 6-K 预筛选的主文件提升边界：若 `EX-99.1` 只是董事会结果或治理附件，而 `EX-99.2/99.3`、6-K cover，或根本没有 `EX-99/XBRL` 但 `primary_document` 自身才是季度正文时，测试必须断言预筛选会把真正命中的 candidate 保留/提升为 `preferred_primary`，而不是继续按文件名顺序固化错选，也不能让旧 `primary_document` 的 `EXCLUDE_NON_QUARTERLY` 结果或 `NO_EX99_OR_XBRL` 早退提前短路整个 filing；新增样本时要优先补齐 `Date of Board / Audit Committee Meeting`、`Financial Results Announcement Date`、被动语态 `results will be released`、`profit warning`、`operating update ... financial results are only provided on a six-monthly basis`、资本市场条款调整公告（如 `adjustment to exercise price ... call spread`）、`trading statement and production update`、战略展示 deck（如 `Our strategy / Agenda / Q&A`）、`Reports Third/Fourth-Quarter [and Full-Year] Financial Results` 这类真实季报标题，以及“真实中报里引用 `annual report` / `annual financial statements` / `annual general meeting`”这类容易误杀的上下文样本
    还要继续守住一组最新的 hard-boundary 误收回归：`Results for Announcement to the Market ... filed the following documents with the ASX` 这类 ASX 转发壳、`Response to ASX Aware Letter`、`operating results for June/September/March 20xx` 这类月度经营数据、`Announcement Regarding ... Financial Results Calendar`、`Results for Production and Volume Sold per Metal`、`Quarterly Activities Report`、`Quarterly Update of Resumption Progress`、`Bitcoin Production, Mining Operation Updates, and Preliminary ... Financial Results` 这类经营更新公告、`Shell ... update note ... scheduled to be published on <date>` 这类结果预告说明、`Minutes of the Board of Directors / Fiscal Council` 或 `Opinion of the Fiscal Council` 且只是在审议 `quarterly financial report / interim financial statements / annual financial statements` 的治理材料、纯治理通知型 `Dear Sirs ... approved the condensed interim financial statements ...` 本地交易所函、`Request for clarification / Official Letter / news published in the media` 这类监管问询回复函、`annual report and form 20-F / AGM statements / voting results / financial calendar / corporate calendar` 这类 AGM 与日历公告、`will present at ... conference / host investor meeting / ahead of scheduled Investor Day` 且只承诺提供 `slides / transcript / outlook / strategic update` 的会议预告，以及带 `closing procedures not yet complete / financial statements not yet available / ranges have been provided` 限制语的 `preliminary / estimated results`；同时还要补齐几类更晚出现的预估快讯写法，比如 `preliminary expected financial results`、`preliminary unaudited ... results`、`subject to the Company’s detailed quarter-end closing procedures`、`subject to revision`、`subject to change upon the completion of the audit process`、`estimated to be in the range of`、`tentative consolidated revenue`、`final result ... will be provided by our annual report`、`preliminary internal data available as of the date of this announcement`、`independent registered accounting firm has not reviewed or audited`；这些样本都应该稳定落到 `EXCLUDE_NON_QUARTERLY`，而不能再留在 active 里制造假 HGF
    同时还要守住八条新增边界：排除词如 `Convertible Senior Notes` 只能在标题/前缀语境里生效，不能因为正文后段的资本结构说明误杀季度结果；季度标题还要覆盖 `4Q24 Results`、`1Q 2025 Results`、`Q3 Results`、`first-quarter results` 这类紧凑写法，以及 `reported its financial results for the three and twelve month periods ending ...` 这类 `4Q + FY` 新闻稿句式；若 primary document 是压缩的季度 XBRL instance 头部，测试也要守住“fiscal quarter code + taxonomy token + 季度/半年日期区间”应保留，而 `FY` / 年报型 XBRL 附件不能被同一规则误抬；同时还要覆盖 `Financial Management Review 4Q24 / Quarterly YTD Report`、`Exhibit 99.1 ... interim report for the quarter ended ...`、`Full Year Financial Results + Fourth Quarter Business Update` 这些在放大样本里确认过的季度财报变体；以及 `UNAUDITED CONDENSED CONSOLIDATED FINANCIAL STATEMENTS ... FOR THE THREE AND SIX/NINE MONTHS ENDED ...`、`Half Year Financial Results + Second Quarter Business Update` 这类在更大样本中确认过的半年报/季度封面变体；还要新增覆盖三类 future-notice 误收回归：`board meeting / trading window closure` 预告将审议结果、`will report ... results on <date>` / `conference call to discuss results` / `results release scheduled` / `will hold ... earnings conference` / `earnings release date` / `Q1-Q4 results call` / `financial results and corporate update webcast` / `earnings release zoom meeting` / `earnings release conference and blackout period` / `financial results calendar` / `notice of announcement of ... interim results` 这类发布时间通知，以及 `group reporting changes / data pack / in advance of earnings release` 这类口径重列说明；对这类显式 schedule notice，测试还要守住“尾部 About 段里的历史 `revenues / net income` 弱指标不能抵消前缀 notice 语义”，只有 `financial highlights / statements of income / balance sheets / unaudited condensed consolidated financial statements / reported its financial results for ...` 这类强结构信号才允许保留；还要继续覆盖 `trading statement / update note / operating statistics / trading update / quarterly activities report / production and volume sold / resumption progress / preliminary vehicle deliveries`、`audio recording of the earnings call`、`results conference call transcript`、`investor presentation referred during the earnings call`、`presentation materials related to ... financial results`、`materials and a webcast replay are available`、`analyst Q&A session transcript`、`financial statements already available on the Investor Relations website`、`newspaper advertisement regarding ... financial results`、`minutes of the board of directors / audit and control committee`、`ER Investor Presentation` / `earnings release investor presentation` 这些显式误收材料；资本回报误收也要显式覆盖 `Transaction in Own Shares`、`Announcement of Share Repurchase Programme`、`Shell announces commencement of a share buyback programme`、`Adjustment to Cash Dividend Per Share`、`Stockholder Remuneration Policy (Dividends and Interest on Capital)`、`Public Announcement Post-Buyback`、`Renews Its ... Share Buyback Program` 这些标题家族，确保这类材料不会再因为提到 `results announcement`、历史季度现金分红或股东回报政策而误留在 active；分红日期通知也要显式覆盖 `Appendix 3A.1 - Notification of dividend / distribution`、`Key Dates ... Ex-dividend date / results announcement` 这类澳交所/港交所日期表，确保它们不会再被误判成季度结果披露；最后还要覆盖 `NOTICE AND INFORMATION CIRCULAR`、年度审计财务报表附件、`Investor Conference` 业务更新、`Mining Forum ... Presentation` 这些显然应当明确排除而不是停留在 `NO_MATCH` 的非季报材料，同时要保留 `OMA Announces ... Operating and Financial Results` 这类“标题里同时出现 operating 和 financial”的真实 results release，防止规则把强披露正文误扫出 active
  - `tests/fins/test_sec_6k_rule_diagnostics.py`
    这里负责守住诊断闭环口径：20-F 公司必须从 active filings 识别，`process --ci --overwrite` 必须显式透传 `--base {workspace_root}` 且只重跑 active `6-K` document_ids`；HGF 只排除单个 filing，不排除整家公司，其它 hard-gate fail 不得误计为 HGF；false negative 必须关联 `.rejections/`，但如果同一个 `document_id` 已经恢复到 active filings，即使 stale rejected artifact 还在，也绝不能再重复计入 false negative；并发上限固定为 `26`；若支持 `--tickers` 子集运行，测试也要断言 `process` 与 `score_batch` 只看到目标 ticker，且 `score_batch` 不会把已脱离 active source 的陈旧 `processed/fil_*` 再算回去
  - `tests/fins/test_sec_6k_primary_document_diagnostics.py`
    这里负责守住主文件选文诊断口径：只能通过 source/blob 仓储读取 active filing 的 `meta/files` 与候选正文，逐个候选重新走 `_classify_6k_text()`，并且只在“当前 `primary_document` 非季度、同 filing 存在季度 exhibit”时报告样本；不能因为普通多文件 6-K 或候选里存在额外非季度材料就误报
  - `tests/fins/test_rejected_6k_rescue.py`
    这里负责守住“规则修复后不重下载、直接本地救回 `.rejections/` artifact”的边界：`dry-run` 只能识别候选不能写回 active filings；`apply` 必须通过仓储把 rejected 文件复制回 active source、保留 `selected_primary_document` 作为主文件，并同步清理 `_download_rejections.json` 对应 skip 记录；若 active 侧只残留 deleted meta、实体文件已缺失，rescue 也必须回退到 create 重建 active source；非季报 reject 不得被误救回
  - `tests/fins/test_active_6k_retriage.py`
    这里负责守住与 rescue 对称的“误收 active 6-K 退出 active”边界：`dry-run` 只能标出当前规则下的误收候选，不能修改 active source；`apply` 必须先把 active filing 通过仓储写回 `.rejections/` 并写入 `_download_rejections.json`，再把 active source 逻辑删除；真实季度结果不得被误移出 active

## 4. 当前重点测试文件

### 4.1 Service / Host

- `test_host_executor.py`
- `test_prompt_service.py`
- `test_chat_service.py`
- `test_write_service.py`
- `test_fins_service.py`
- `test_host_commands.py`
- `test_host_admin_service.py`
- `test_ui_host_boundary.py`
- `test_web_routes.py`
- `test_host_reply_outbox.py`
- `test_reply_outbox_store.py`

### 4.2 Engine

- `test_async_agent.py`
- `test_async_openai_runner.py`
- `test_async_openai_runner_call_paths.py`
- `test_async_cli_runner.py`
- `test_context_budget.py`
- `test_prompt_assets.py`
- `test_prompt_composer.py`
- `test_tool_registry_prompt.py`
- `test_conversation_store.py`
- `test_conversation_memory.py`
- `test_write_pipeline.py`
- `test_report_assembler.py`
- `test_execution_summary_builder.py`

### 4.3 Architecture

- `tests/architecture/test_dependency_boundaries.py`

它负责守护：
- 层间依赖不反向穿透
- `Service` 不得反向 import `Engine`
- 主链不再出现 `application/runtime/capabilities`
- `startup preparation` 不得创建 `Host`
- `startup preparation` 只能调用 `services/host` 的公开 preparation API，不能触碰下层内部实现
- `prompting` 不得反向依赖已删除的 `dayu.engine.prompts`
- `Engine` 不得重新出现 `load_prompt`、`parse_when_*`、`PromptParseError` 这类 prompt 渲染实现
- `ExecutionContract` 的业务装配只能在 `contract_preparation` 中收口构造；`contracts` 层的 snapshot restore 属于机械恢复例外
- `Service` 不得直接构造 `AgentCreateArgs`
- `services/host/fins` 等内部实现层不得再通过 `dayu.prompting`、`dayu.fins` 这类 package root 获取实现符号；若需要实现对象，必须直接导入具体子模块
- `Host` 只消费 `ExecutionContract`
- `Agent` 不理解上层聚合对象
- `fins/tools` 与 `fins/pipelines` 不得重新 import 旧 `DocumentRepository / FsDocumentRepository`
- 旧总仓储 public 模块必须保持已删除状态，测试若需要共享底层文件系统 core，应通过 `tests/fins/storage_testkit.py` 装配

## 5. 新增测试时的约束

新增测试时，遵循这些原则：
- 优先写单元测试，只有边界跨层时再写集成测试
- 测试名直接表达行为，不写历史实现细节
- fixture 只做最小装配，不偷偷复刻生产链路
- 20-F / 报告类表单的 regression case，若依赖 source text 恢复 marker，至少要区分三类输入形态：显式 `Item` 标题、`cross-reference guide`、以及无 guide 的 `annual-report-style` 页标题（含页码且后续紧跟正文）；不要只覆盖其中一种文本形态。另需单独守住 `Item 3` narrative 中“重复 `Annual Report` + `Not applicable` 但无 locator/note”的误判回归，避免 child split 被过宽的 guide 判定提前吞掉。
- 若 20-F regression case 同时包含“早期真实 heading fallback”和“晚期伪 marker”，断言必须覆盖最终修复结果优先采用完整单调 fallback 主链，而不是只验证 helper 能不能找到 fallback 候选。
- 对 Fins 读工具 / runtime 的旧 fake repository，优先通过 `tests/fins/legacy_repository_adapters.py` 适配到 `company/source/processed/blob` 窄仓储；不要为了测试回加生产代码里的 `repository=` 兼容参数
- 若某层已删除，对应旧测试也应删除，不做“保留纪念”
