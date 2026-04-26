# 代码仓库结构总览

本文档面向开发者，给出 `dayu-agent` 仓库的"鸟瞰地图"：把每个 `dayu/` 子包、`tests/` 子目录、以及 `pyproject.toml` 入口对齐到 [docs/architect.md](architect.md) 固定的四层模型 `UI -> Service -> Host -> Agent`，方便新读者在不读 1k+ 行 README 的前提下定位代码。

读完本文档你应该能回答四个问题：

- 当前仓库有哪些公开包，每个包属于四层模型的哪一层？
- 每个包对外的稳定入口/契约是什么？
- 对应测试落在哪里？
- 包内引入的关键外部依赖有哪些？

更深的设计判断请继续读：

- 架构基线：[architect.md](architect.md)
- 包级开发手册总览：[../dayu/README.md](../dayu/README.md)
- 各层细节：[../dayu/host/README.md](../dayu/host/README.md) / [../dayu/engine/README.md](../dayu/engine/README.md) / [../dayu/fins/README.md](../dayu/fins/README.md) / [../dayu/web/README.md](../dayu/web/README.md)

## 1. 顶层布局

```
dayu-agent/
├── dayu/              # 实现包（按四层 + 公共模块组织）
├── tests/             # 测试树（按 application/engine/fins/integration/architecture/cli/contracts/services 分组）
├── docs/              # 设计文档（架构、CI、code review、专题调研、TODO）
├── docker/            # Docker 镜像/部署脚本
├── constraints/       # 受控依赖锁文件（按平台 × Py 版本）
├── utils/             # 命令行式辅助脚本（不是包）
├── CHANGELOG.md       # 发布日志
├── CONTRIBUTING.md    # 贡献指南
├── README.md          # 用户手册（终端用户视角）
├── AGENTS.md          # Agent 协作约定
├── pyproject.toml     # 打包元数据 + 命令行入口
├── pytest.ini         # 测试配置
├── pyrightconfig.json # 类型检查配置
└── requirements.txt   # 生产依赖（与 pyproject 保持一致）
```

`dayu/` 的子目录与四层模型映射如下（详见第 2 节）：

| 四层 / 模块                         | 子包                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| UI                                  | `dayu/cli`、`dayu/web`、`dayu/wechat`、`dayu/gui`、`dayu/render` |
| Service                             | `dayu/services`                                              |
| Host                                | `dayu/host`                                                  |
| Agent (Engine + Domain)             | `dayu/engine`、`dayu/fins`                                   |
| Public 模块 / 跨层基础设施          | `dayu/contracts`、`dayu/prompting`、`dayu/startup`、`dayu/execution`、`dayu/config`、`dayu/assets` |
| 顶层工具模块                        | `dayu/log.py`、`dayu/file_lock.py`、`dayu/state_dir_lock.py`、`dayu/process_liveness.py`、`dayu/console_output.py`、`dayu/presenters.py`、`dayu/prompt_template_rendering.py`、`dayu/tool_limits.py`、`dayu/workspace_paths.py`、`dayu/docling_runtime.py` |

> 本文中"层"严格按 [architect.md](architect.md) 第 1 节的固定结论；如 `startup preparation` / `Contract preparation` / `scene preparation` 不被列为新层，而归入"public 模块"。

## 2. 子包对照表

下表每行四列：**包路径**、**层 / 职责**、**对外稳定入口或契约**、**对应测试**。最后一列只列直接外部依赖大类，详细版本见 `pyproject.toml` 与 `requirements.txt`。

### 2.1 UI 适配层

| 包                | 职责                                                                                                                                       | 对外入口                                                                                                                                                                                                                                                                                                | 对应测试                                                                                                                                                                                            | 关键外部依赖                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `dayu/cli`        | 终端用户主入口；按 `arg_parsing → main → commands/` 三层拆分；`dependency_setup.py` 统一收口共享运行时；`workspace_migrations/` 仅服务 `init` | `dayu-cli`（`dayu.cli.main:main`）；子命令：`init / download / upload / process / write / interactive / prompt / host / conv` 等                                                                                                                                                                       | `tests/cli/`（`test_init_command.py`、`test_fins_commands.py`、`test_conversation_labels.py` 等）；`tests/application/` 内 `test_cli_*` 系列；`tests/cli/workspace_migrations/`                       | `argparse`、`prompt_toolkit`  |
| `dayu/web`        | Streamlit + FastAPI 双入口；`streamlit_app.py` 是用户页面，`fastapi_app.py` 是 HTTP API；`routes/` 注册子路由；`streamlit/` 含 page/component | `dayu-web`（`dayu.web.__main__:main`）；FastAPI app 工厂；Streamlit 启动器 `run_streamlit()`                                                                                                                                                                                                            | `tests/application/test_web_routes.py`、`test_reply_outbox_web_integration.py`、`test_streamlit_*` 等                                                                                                | `fastapi`、`streamlit`        |
| `dayu/wechat`     | 企微/iLink 适配；按 `arg_parsing → main → commands/` 三层 + `runtime.py` / `daemon.py` / `service_manager.py`                                | `dayu-wechat`（`dayu.wechat.main:main`）；子命令：`login / run / service`                                                                                                                                                                                                                              | `tests/application/test_wechat_*`、`test_wechat_outbox_integration.py`                                                                                                                              | `requests`                    |
| `dayu/render`     | Markdown → HTML/PDF/Word；纯命令行渲染                                                                                                       | `dayu-render`（`dayu.render.render:main`）                                                                                                                                                                                                                                                             | 由 `tests/application/test_console_output.py`、`tests/test_build_offline_bundle.py` 等覆盖                                                                                                          | `pandoc`（外部进程）、Chrome  |
| `dayu/gui`        | 占位包，GUI 尚未实现                                                                                                                          | （无）                                                                                                                                                                                                                                                                                                 | （无）                                                                                                                                                                                              | -                             |

UI 共同约束（沿用 [dayu/README.md](../dayu/README.md) §3.1）：

- 只在启动期通过 `startup preparation` 拿稳定依赖，请求期不复制 `Host` 装配链；
- 显式 `new Service(...)`，按命令分支惰性创建；
- 多轮会话只走 `submit_turn / list_resumable_pending_turns / resume_pending_turn` 这套稳定 ChatService 契约。

### 2.2 Service 业务层

| 包               | 职责                                                                                                                                                                                                                                          | 对外入口 / 公共契约                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 对应测试                                                                                                                                                                                                | 关键外部依赖     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `dayu/services`  | 唯一允许"理解业务语义"的层；产出 `ExecutionContract` 提交给 `Host`；包含 GeneralChat / Prompt / Write / Fins / HostAdmin / ReplyDelivery 等服务，以及 `contract_preparation.py`、`scene_execution_acceptance.py`、`startup_preparation.py` | 通过 `dayu.services` 包级 `__init__.py` 显式导出：`ChatService` / `PromptService` / `WriteService` / `FinsService` / `HostAdminService` / `ReplyDeliveryService` 及对应 `*ServiceProtocol`；`Request DTO`：`ChatTurnRequest` / `PromptRequest` / `WriteRequest` / `FinsSubmitRequest` 等；`Submission` 句柄：`ChatTurnSubmission` / `PromptSubmission` / `FinsSubmission`；启动期 API：`prepare_host_runtime_dependencies`、`prepare_scene_execution_acceptance_preparer`、`recover_host_startup_state` | `tests/application/test_chat_service.py`、`test_prompt_service*.py`、`test_write_service*.py`、`test_fins_service.py`、`test_host_admin_service.py`、`test_reply_delivery_service.py`、`test_contract_preparation.py` 等 | -                |
| `dayu/services/internal/write_pipeline` | 写作复合流水线的内部实现：`pipeline.py` 编排，`scene_executor.py` 共享重试，`chapter_*.py` / `repair_*.py` / `audit_*.py` 等阶段                                                                                                                                                                                                                                            | 仅供 `WriteService` 内部使用，不属于 Service 公共导出                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `tests/engine/test_write_pipeline*.py`、`tests/application/test_write_service*.py`                                                                                                                       | -                |

Service 与 Host 之间的跨层契约由 `dayu/contracts` 提供（见 §2.5）。

### 2.3 Host 托管执行层

| 包          | 职责                                                                                                                                                                                                                                                              | 对外入口 / 公共契约                                                                                                                                                                                                                                                              | 对应测试                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 关键外部依赖    |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `dayu/host` | 通用托管执行层；承担 Session / Run / 并发 / 事件 / Timeout / Cancel / Resume / Memory / Reply outbox / Replay 共九项稳定能力（详见 [host/README.md](../dayu/host/README.md)）；内含 `host.py`（门面）、`executor.py`、`scene_preparer.py`、`*_registry.py`、`*_store.py`、`event_bus.py`、`pending_turn_store.py`、`reply_outbox_store.py` 等 | `dayu.host` 包级仅导出三项：`Host`（门面）、`HostExecutorProtocol`（执行器协议）、`ResolvedHostConfig` + `resolve_host_config(...)`（配置规范化入口）。其余子组件（registry、governor、store）由 Host 自己拥有 | `tests/application/test_host*.py`（含 `test_host_executor*.py`、`test_host_admin_service.py`、`test_host_store.py`、`test_host_reply_outbox.py`、`test_host_logging.py`、`test_host_shutdown.py`、`test_host_executor_lane_stacking.py`、`test_host_executor_replay.py`、`test_pending_turn_store.py` 等）；`tests/architecture/test_dependency_boundaries.py`（守 Service 不直接访问 Host 内部子组件） | `sqlite3`（标准库） |

### 2.4 Agent / Engine + 领域包

| 包           | 职责                                                                                                                                                                                                                                                                                                                                                | 对外入口 / 公共契约                                                                                                                                                                                                                                                                                                  | 对应测试                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | 关键外部依赖                                                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `dayu/engine` | 通用执行原语层；不理解业务；负责 `AsyncAgent` / `AsyncRunner` / `ToolRegistry` / 事件 / 上下文预算 / 取消 / Tool Trace；`processors/` 是文档解析基座；`tools/` 是 doc/web/utils 内置工具；`tool_registry.py`、`tool_contracts.py`、`truncation_manager.py`、`context_budget.py` 等                                                                                                            | 包级 `__init__.py` 导出：`AsyncAgent` / `AgentResult` / `ToolRegistry` / `register_doc_tools` / `register_web_tools` / `register_utils_builtin_tools` / `WebToolsConfig` / `DocToolLimits` / 全套事件构造函数 / `EngineError` 系列 / `AsyncRunnerProtocol`；`AsyncCliRunner` 已禁用，仅留源代码                | `tests/engine/`（75+ 文件）：`test_async_agent*.py`、`test_async_openai_runner*.py`、`test_async_cli_runner*.py`、`test_tool_registry*.py`、`test_context_budget.py`、`test_conversation_memory.py`、`test_web_tools.py`、`test_web_playwright_backend.py`、`test_docling_processor_integration.py`、`test_processor*.py`、`test_log.py`、`test_prompt_*` 等；架构守护 `tests/engine/test_architecture_boundaries.py` | `aiohttp`、`docling`、`docling-core`、`pandas`、`lxml`、`beautifulsoup4`、`requests`、`trafilatura`、`readability-lxml`、`markdownify`、`html2text`、`tabulate`、`playwright`（可选） |
| `dayu/fins`  | 证券财报领域包；两条路径：Agent augmentation（工具注入）+ direct operation（download/upload/process）；内部分五层：Service-Adapter → Pipeline/Ingestion → Tool Service → Processor → Repository（详见 [fins/README.md](../dayu/fins/README.md) §7）；含 SEC 6-K 治理脚本（active retriage / rejected rescue / primary document repair）                                                                            | `FinsRuntimeProtocol`（`service_runtime.py`）：`execute(command)` / `validate_command(command)` / `get_processor_registry()` / `get_tool_service()` / `build_ingestion_service_factory()` / `get_ingestion_manager_key()` / `get_company_name(ticker)` / `get_company_meta_summary(ticker)`；direct operation 公共契约定义在 `dayu.contracts.fins`；toolset registrar：`register_fins_read_toolset(context)` / `register_fins_ingestion_toolset(context)`；ticker 真源：`normalize_ticker / try_normalize_ticker / ticker_to_company_id` | `tests/fins/`（82+ 文件，包含 SEC pipeline、processors、storage batch 锁、rule diagnostics、CI 脚本边界等）；`tests/integration/fins/`                                                                                                                                                                                                                                                                                                                          | `edgartools`、`docling`、`docling-core`、`pandas`、`lxml`、`beautifulsoup4`、`requests`、`PyYAML` |

### 2.5 Public 模块 / 跨层基础设施

这些包不构成新层，但承载稳定契约或启动期/请求期收敛逻辑：

| 包                 | 职责                                                                                                                                                                            | 关键导出                                                                                                                                                                                                                                                                                                                                                                                          | 对应测试                                                                                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dayu/contracts`   | 跨层稳定数据契约：Agent/Host/Run/Session/Reply outbox/Tool/Toolset/Prompt assets/Execution metadata 等；包级 `__init__.py` 通过 `__getattr__` 懒加载，避免循环导入                                                                                                | `AgentInput`、`AgentCreateArgs`、`ExecutionContract`、`ScenePreparationSpec`、`AcceptedExecutionSpec`、`ExecutionHostPolicy`、`AppEvent` / `AppEventType` / `AppResult`、`SessionRecord` / `SessionSource` / `SessionState`、`RunRecord` / `RunState`、`ReplyOutboxRecord` / `ReplyOutboxState` / `ReplyOutboxSubmitRequest`、`FinsCommand` / `FinsEvent` / `FinsResult`、`CancellationToken` / `CancelledError`、`ToolExecutor`、`ModelConfig` 等 | `tests/contracts/test_agent_execution.py`、`test_run.py`、`test_session.py`、`test_protocols_extra.py`、`test_toolset_config_extra.py`                                                                                            |
| `dayu/prompting`   | Prompt 渲染与装配公共能力（条件块解析、变量替换、scene definition reader、prompt composer、tool snapshot、contribution slots、prompt plan）                                                                | `prompt_composer.py`、`prompt_plan.py`、`prompt_renderer.py`、`scene_definition.py`、`prompt_contribution_slots.py`、`tool_snapshot.py`                                                                                                                                                                                                                                                          | `tests/engine/test_prompt_composer.py`、`test_prompt_assets.py`、`test_prompt_plan.py` 等                                                                                                |
| `dayu/startup`     | 启动期 public 模块：把 workspace path / config path 等原始来源收敛成稳定依赖（`WorkspaceResources` / `ConfigLoader` / `PromptAssetStore` / `ModelCatalog`）                                                          | `paths.py`、`workspace.py`、`config_loader.py`、`config_file_resolver.py`、`prompt_assets.py`、`model_catalog.py`                                                                                                                                                                                                                                                                                | `tests/engine/test_config_loader_*.py`、`tests/integration/test_config_loader_e2e.py`、`tests/test_workspace_paths.py`                                                                  |
| `dayu/execution`   | 跨层运行配置值对象：`ExecutionOptions`、`DocToolLimits`、`WebLimits`、`RuntimeConfig` 转换 helper                                                                                                          | `options.py`、`runtime_config.py`、`doc_access.py`、`doc_limits.py`、`web_limits.py`、`cli_execution_options.py`                                                                                                                                                                                                                                                                                | `tests/application/test_execution_runtime_config.py`、`test_doc_access.py`、`tests/engine/test_cli_running_config.py`                                                                  |
| `dayu/config`      | 包内默认配置资源：`run.json`、`llm_models.json`、`toolset_registrars.json`、`prompts/**`；用 `package-data` 随包分发                                                                                              | （静态资源）                                                                                                                                                                                                                                                                                                                                                                                       | `tests/integration/test_prompts_e2e.py` 等                                                                                                                                              |
| `dayu/assets`      | 包内默认写作模板（Markdown 基线）                                                                                                                                                                          | （静态资源）                                                                                                                                                                                                                                                                                                                                                                                       | （间接覆盖）                                                                                                                                                                              |

### 2.6 顶层工具模块（`dayu/*.py`）

这些是包根直接持有的小型 utility，跨层可以引用，不属于任何层：

| 文件                                | 职责                                                                                          |
| ----------------------------------- | --------------------------------------------------------------------------------------------- |
| `dayu/log.py`                       | 全局日志真源；`ERROR` 以下走 stdout，`ERROR+` 走 stderr，禁止双流写同一条                      |
| `dayu/file_lock.py`                 | 跨平台文件锁原语（POSIX `fcntl` / Windows `msvcrt`）                                          |
| `dayu/state_dir_lock.py`            | `workspace/.dayu/` 状态目录锁                                                                 |
| `dayu/process_liveness.py`          | 进程存活探测；用于 Host orphan 检测                                                            |
| `dayu/console_output.py`            | 中文/非 UTF-8 终端输出兜底                                                                    |
| `dayu/presenters.py`                | UI 共享的格式化/展示 helper                                                                   |
| `dayu/prompt_template_rendering.py` | prompt 模板渲染入口（被 `prompting/` 复用）                                                   |
| `dayu/tool_limits.py`               | 工具限制公共值对象                                                                            |
| `dayu/workspace_paths.py`           | workspace 内的标准路径常量与解析                                                              |
| `dayu/docling_runtime.py`           | Docling PDF 转换统一入口（封装 backend × device 二维回退）；其他模块禁止直接 patch `DocumentConverter` |

## 3. 测试树到代码层的映射

`tests/` 下八个子目录对齐到层与场景：

| 测试目录                          | 对应代码层                              | 关键守护点                                                                                           | 文件数（粗略） |
| --------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------- |
| `tests/application/`              | Service + Host 主链                     | `ExecutionContract`、contract preparation、Host Session/Run/HostExecutor/scene preparation/reply outbox | 47             |
| `tests/engine/`                   | Engine + prompting + write pipeline + memory | `AsyncAgent`、Runner、ToolRegistry、PromptComposer、context_budget、log、web tools、processors、docling 集成 | 75             |
| `tests/fins/`                     | Fins direct operation + 仓储 + processor + tool service | SEC pipeline、storage batch 锁、6-K 规则诊断、processor 真实文档回归、CI 脚本边界                     | 82             |
| `tests/integration/`              | E2E 跨层                                | `test_agent_integration.py`、`test_doc_tools_e2e.py`、`test_prompts_e2e.py`、`test_tool_registry_e2e.py`、`fins/` | 7              |
| `tests/architecture/`             | 架构守护                                | `test_dependency_boundaries.py`：Service 不得直接 import Host 内部子组件                               | 1              |
| `tests/cli/`                      | CLI UI 层                               | `init` 命令交互、`fins` 子命令、conversation labels、workspace migrations                              | 4 + migrations |
| `tests/contracts/`                | 跨层契约                                | `agent_execution`、`run`、`session`、protocols、toolset config                                       | 7              |
| `tests/services/`                 | Service 协议补充                        | `concurrency_lanes` 等                                                                              | 2              |
| `tests/fixtures/`                 | 测试数据                                | config / doc_tools / docling / fins / prompts / registry                                            | -              |
| `tests/test_*` (根目录)           | 项目级 utility                          | `test_build_offline_bundle.py`、`test_process_liveness.py`、`test_workspace_paths.py`               | 3              |

约束与运行规则详见 [tests/README.md](../tests/README.md)。CI 主线（详见 [docs/ci.md](ci.md) 与 `.github/workflows/`）按 fast / integration / e2e / lock-smoke 等多个 gate 拉起这些目录。

## 4. 命令行入口（`pyproject.toml`）

打包安装后暴露的 console scripts：

| 命令          | 入口                          | 用途                                       |
| ------------- | ----------------------------- | ------------------------------------------ |
| `dayu-cli`    | `dayu.cli.main:main`          | 终端用户主入口（init / download / upload / process / write / interactive / prompt / host / conv / fins） |
| `dayu-web`    | `dayu.web.__main__:main`      | Streamlit 用户页面 + FastAPI HTTP API     |
| `dayu-wechat` | `dayu.wechat.main:main`       | WeChat 渠道入口（login / run / service） |
| `dayu-render` | `dayu.render.render:main`     | Markdown → HTML/PDF/Word 渲染             |

## 5. 包外辅助资源

- `utils/` — 命令行式辅助脚本（不打进 wheel），主要用于 SEC 6-K 诊断/治理、CI 评分、离线 bundle 构建等：
  - `sec_6k_rule_diagnostics.py` / `sec_6k_primary_document_diagnostics.py`
  - `rescue_rejected_6k_filings.py` / `retriage_active_6k_filings.py`
  - `reconcile_active_6k_primary_documents.py`
  - `llm_ci_process.py` / `llm_ci_score.py`
  - `build_offline_bundle.py`
- `constraints/` — 受控依赖锁文件：`lock-{macos-arm64,macos-x64,linux-x64,windows-x64}-py311.txt`、`min-py311.txt`
- `docker/` — Docker 镜像与部署脚本
- `docs/` — 设计文档：`architect.md`（架构基线）、`ci.md`、`code_review*.md`、`conversation_memory_optimization.md`、`fmp_integration_research.md`、`github_main_workflow.md`、`TODO.md`，以及本文件
- `requirements.txt` — 生产依赖（与 `pyproject.toml` 的 `dependencies` 段保持一致；额外的 dev/test/browser/web 走 `pip install -e ".[test,dev,browser,web]"` 并配合 `constraints/`）
- `pytest.ini` — 测试配置（marker、覆盖率、超时）
- `pyrightconfig.json` — pyright 类型检查配置

## 6. 阅读顺序建议

新读者建议按以下顺序进入：

1. [README.md](../README.md) — 了解使用形态与四类工作（数据管线 / 投研问答 / 写作 / 渲染）
2. [docs/architect.md](architect.md) — 固定四层模型、`Request DTO` / `ExecutionContract` / `AgentInput` 三类对象的稳定契约
3. 本文档（`docs/repo_structure.md`） — 把上面两份抽象映射到代码
4. [dayu/README.md](../dayu/README.md) — 总体开发手册（含主链时序图、`ticker → prompt` 数据流转）
5. 按需深入：
   - 托管执行 → [dayu/host/README.md](../dayu/host/README.md)
   - 推理引擎 → [dayu/engine/README.md](../dayu/engine/README.md)
   - 财报领域 → [dayu/fins/README.md](../dayu/fins/README.md)
   - Web 适配 → [dayu/web/README.md](../dayu/web/README.md)
   - 配置打包 → [dayu/config/README.md](../dayu/config/README.md)
   - 测试规则 → [tests/README.md](../tests/README.md)

## 7. 维护规则

- 当 `dayu/` 新增子包、删除子包、或更换公共导出时，必须同时更新本文档；
- 新增/变更不得绕过 [architect.md](architect.md) 的四层结论；如确需引入新层级，先改 `architect.md`，再改本文；
- 包级 `__init__.py` 的公共导出应保持窄而稳定；本文表格里写出的"对外入口/公共契约"是该承诺的具体落点；
- 测试目录与代码层的映射只在测试结构整体调整时改本文，不为单个 `test_*.py` 文件抖动而改。
