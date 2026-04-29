# dayu Agent 开发手册 - Fins 包

`dayu/fins` 是 Dayu 的证券财报领域包。它不是系统基础设施，也不是架构层级；它提供的是一组围绕证券文档的领域实现。

本文档只写 Fins 的两条执行路径、对外接口与内部层次。

## 1. Fins 在系统中的位置

Fins 当前参与两条稳定路径：

### 1.1 Agent augmentation path

```text
UI
-> Service
-> Host
-> scene preparation
-> Agent
```

在这条路径上，Fins 提供的是：
- 财报工具注册
- 公司名称 / 公司摘要等 Service 辅助查询
- 公司 / source / processed / blob / maintenance 窄仓储与工具服务真源

关键约束：
- `ticker` 的业务解释发生在具体 `Service`
- Fins 不把 `ticker` 直接塞进 Host 或 Agent
- 动态 prompt 文本由 Service 侧公共函数生成，再通过 `Prompt Contributions` 下传

### 1.2 Direct operation path

```text
UI
-> FinsService
-> Host
-> FinsRuntime / pipeline
```

这条路径不经过 Agent。当前主要覆盖：
- `download`
- `upload_filing`
- `upload_filings_from`
- `upload_material`
- `process`
- `process_filing`
- `process_material`

当前上传链路还固定遵守三条实现边界：
- `upload_filing` 的 `document_id` 继续由财年/财期规则稳定生成，不跟文件内容绑定。
- `upload_material` 的 `document_id` 由 `form_type + material_name + fiscal_year? + fiscal_period?` 稳定生成；当前 material 场景下 `document_id` 与 `internal_document_id` 恒等，显式传入时也只能与这套规则一致，不能覆盖它。
- `upload_filing` / `upload_material` 未显式传 `action` 时统一按 source meta 自动解析 `create/update`；删除动作必须显式传入 `delete`。相同原始上传文件指纹会在 Docling convert 前直接 `skip`；`overwrite` 只重置当前 `document_id`，不会做 ticker 级清空。

## 2. 设计意图

### 2.1 为什么文档读取统一经过 FinsToolService

`FinsToolService` 是财报文档读取能力的统一入口。

它统一收敛：
- company / source / processed 窄仓储
- processor 路由
- 工具返回语义

这样可以保证：
- 在线工具调用
- 离线快照导出

使用同一套文档存取与处理真源。

当前还要守住两条读取契约：
- `Processor.read_section()` 的稳定返回真源仍是 `SectionContent`；当处理器支持父子章节导航时，只能通过可选 `children` 字段附加直接子章节摘要，且子项结构必须对齐 `SectionSummary`。
- 财务报表处理器的所有返回路径都必须满足完整 `FinancialStatementResult` 契约；早退分支也要显式给出稳定必填字段，并用 `reason` 标识失败原因，而不是回退为未注解字典。
- `FinsToolService` 的实例级缓存属性必须在 `__init__` 中一次性声明；像文档 meta 这类读取缓存可以按实例复用，但不能靠 `hasattr` 在业务路径里懒创建，避免实例不变量漂移到调用分支。
- SEC processor 的标题恢复只能把性能优化建立在同源正文信号之上：annual-report-style `20-F` 需要先保留真实 page heading，再过滤 front matter / guide 污染；`10-K` 的 ToC 短路探针也不能因为局部窗口过小而放过长导点目录行。
- `dayu/fins/tools/service.py` 只承担 `FinsToolService` 及其直接服务逻辑；搜索数据模型、搜索引擎和通用 helper 的真源分别位于 `search_models.py`、`search_engine.py`、`service_helpers.py`。调用方若需要这些私有符号或常量，必须直接从对应模块导入，禁止再把 `service.py` 当作兼容超级入口。

### 2.2 为什么 direct operation 与 Agent augmentation 分开

Agent augmentation 关注的是：
- 把工具暴露给 LLM
- 为 Service 提供最小辅助查询

direct operation 关注的是：
- 下载
- 上传
- 预处理

这两条路径共享底层仓储和处理器，但不是同一个执行单元，也不共享 Agent 契约。

## 3. 对外接口

### 3.1 FinsRuntimeProtocol

当前运行时协议定义在 [service_runtime.py](service_runtime.py)。

direct operation 的公共命令/事件/结果契约定义在 [../contracts/fins.py](../contracts/fins.py)。

对外主要暴露：
- `execute(command)`
- `get_processor_registry()`
- `get_tool_service()`
- `build_ingestion_service_factory()`
- `get_ingestion_manager_key()`
- `list_source_filings(ticker)`
- `get_company_name(ticker)`
- `get_company_meta_summary(ticker)`

其中：
- `execute(command)` 用于 direct operation
- `get_tool_service()` 用于 Agent augmentation 的读工具注入
- `build_ingestion_service_factory()` / `get_ingestion_manager_key()` 用于长事务工具注入
- `list_source_filings(ticker)` 返回 `list[FilingSummary]`，用于上层读取 source 财报摘要（不承载 UI 渲染副作用）
- 公司信息接口用于 Service 辅助查询

当前 direct operation 还需要守住一条稳定契约边界：
- `FinsCommand` / `FinsResult` / `FinsEvent` 只保留 envelope 角色，内部载荷必须是按命令和事件类型拆开的强类型 dataclass；UI、Service、Runtime 不得重新把这些跨层对象退化回 `Dict[str, Any]` 的公共 god-bag。
- `FinsEvent` 在 `type=RESULT` 分支必须复用与 `FinsResult` 相同的结果 dataclass 校验，不允许把 progress payload 作为 result event 透传到跨层边界。
- `DownloadFilingResultItem.status` 的跨层真源是 `DownloadFilingResultStatus` 枚举；UI 与 runtime 只能依赖该枚举语义，禁止再以裸字符串拼接 `downloaded/skipped/failed` 分支。

### 3.2 FinsService

`FinsService` 是 direct operation 的服务层入口。

它负责：
- 命令语义
- submit 返回前的同步受理校验
- session / run 入口描述
- 把 direct operation 提交给 Host 托管

它不负责：
- 自己管理 run registry
- 自己桥接取消
- 自己处理并发 lane

这些都由 `Host` 统一收口。

当前 direct operation 还要守住一条请求期边界：
- `FinsService.submit()` 必须先调用 `FinsRuntime` 的同步 preflight，再创建 Host session 和 run 规格；像空 `ticker`、payload 类型不匹配、CLI 规范化失败、不支持流式执行的命令被设置 `stream=True` 这类请求级错误，必须在返回 `FinsSubmission` 之前抛出，不能先返回可执行句柄、再让后台流式消费阶段失败。
- 对同步 `process_filing` / `process_material`，以及已支持取消协作的流式 `download` / `process` 这类 direct operation，`FinsService` 只能把 `Host` 提供的取消状态收敛成窄 `cancel_checker` 继续下传；取消真源仍归 `Host`，但 runtime / pipeline 必须在长事务阶段边界协作停止，不能等整个 direct operation 完成后才被动收口。

## 4. Agent 路径中的 Fins

### 4.1 工具注入

Fins 当前通过两个 toolset registrar 向 Agent 路径注入工具：

- `register_fins_read_toolset(context)`
- `register_fins_ingestion_toolset(context)`

它们由 `toolset_registrars.json` 映射到对应 toolset 名称，`Host` 只负责在 scene preparation 阶段加载 registrar，并把当前 `fins` / `ingestion` toolset 命中的单个 `toolset_config` 快照与动态权限传入通用上下文，不直接 import Fins 叶子注册函数。

在 registrar 内部，Fins 仍然落到两个显式叶子注册入口：

- `register_fins_read_tools(...)`
- `register_fins_ingestion_tools(...)`

其中：
- registrar 内部自行准备 `FinsRuntime` / `FinsToolService` / `service_factory + manager_key`
- 财报读取 toolset 的限制配置由 registrar 自己从 `context.toolset_config.payload` 反解，不再由 Host 传专用 `fins_tool_limits`
- registrar 反解 `context.toolset_config.payload` 中的数值限制时，必须走 `dayu.contracts.toolset_config` 的统一 coercion helper，不能直接对 `ToolsetConfigValue` 做裸 `int()`
- 财报读取工具只接受预构建 `FinsToolService`
- ingestion 工具只接受 `service_factory + manager_key`
- `Host` 不再从 `FinsRuntime` 拉总仓储对象，也不再持有 Agent 工具注入所需的 Fins runtime

### 4.2 Prompt Contributions

当前与财报对象有关的动态文本由 Service 侧公共函数生成：
- `build_fins_default_subject_contribution(...)`
- `build_base_user_contribution(...)`

其中：
- `fins_default_subject` 会被多个 Fin Service 复用
- `base_user` 会被 Fin Service 和 General Service 共同复用

这两类文本都不属于 Fins 的 Host 注入职责。

## 5. Direct operation

direct operation 当前由 `FinsRuntime` 和对应 pipeline 实现。

`SecPipeline` 已按高内聚职责拆分为独立子模块，`sec_pipeline.py` 仅保留 `SecPipeline` 类定义与版本常量，不再承担 facade 或兼容导出：

- `dayu/fins/pipelines/sec_pipeline.py` — `SecPipeline` 类定义、版本常量（`SEC_PIPELINE_DOWNLOAD_VERSION` / `SEC_PIPELINE_PROCESS_SCHEMA_VERSION`）与依赖装配
- `dayu/fins/pipelines/sec_form_utils.py` — 表单类型解析、日期计算、常量（`DEFAULT_FORMS_US` / `LOOKBACK_YEARS_BY_FORM` / `SUPPORTED_FORMS`）等纯函数工具
- `dayu/fins/pipelines/sec_filing_collection.py` — `FilingRecord` 数据类、submissions 表解析、6-K 远程候选分类
- `dayu/fins/pipelines/sec_download_event_mapping.py` — 下载事件/结果的类型映射与 payload 构建
- `dayu/fins/pipelines/sec_download_diagnostics.py` — filing 数量/XBRL 覆盖率的不足告警检测
- `dayu/fins/pipelines/sec_6k_rules.py` — 6-K 分类、候选评分与文本规则
- `dayu/fins/pipelines/sec_fiscal_fields.py` — fiscal year / fiscal period 推断与财务载荷
- `dayu/fins/pipelines/sec_download_state.py` — 拒绝注册表、SEC HTTP 缓存与重建快照比较
- `dayu/fins/pipelines/sec_download_workflow.py` — `download/download_stream_impl` 的主流程编排
- `dayu/fins/pipelines/sec_download_filing_workflow.py` — 单个 filing 下载、6-K 预筛选分支后的落盘编排
- `dayu/fins/pipelines/sec_6k_primary_document_repair.py` — 6-K active source 主文件 reconcile；按当前 `BsSixKFormProcessor` 真源重排同 filing HTML 候选并写回最终 `primary_document`
- `dayu/fins/pipelines/sec_download_persistence.py` — download 阶段的文件条目构建、rejected artifact 落盘与 reprocess 标记
- `dayu/fins/pipelines/sec_download_source_upsert.py` — 下载成功后的 source meta payload、create/update upsert 与 reprocess 判定
- `dayu/fins/pipelines/sec_company_meta.py` — SEC download/upload 初始化阶段的 ticker alias 规范化、SEC alias 提取、alias 合并与公司级 meta 写入
- `dayu/fins/pipelines/sec_safe_meta_access.py` — source/processed/company meta 的安全读取与 document version 计算
- `dayu/fins/pipelines/sec_sc13_filtering.py` — SC13 的方向过滤、browse-edgar 补拉、空结果回溯重试与同 filer 去重
- `dayu/fins/pipelines/sec_rebuild_workflow.py` — rebuild 模式的本地过滤条件、单 filing canonical meta 重建与 replace-source-meta 覆盖
- `dayu/fins/pipelines/sec_process_workflow.py` — process/process_filing/process_material 的单文档决策与批量工作流编排
- `dayu/fins/pipelines/sec_upload_workflow.py` — upload_filing/upload_material 的事件编排、稳定上传身份、自动动作解析与单文档 overwrite reset 收口

当前 source fiscal 字段还需要守住一条稳定边界：source/download/rebuild/list 四条链路都不能仅凭 `report_date` 或 `filing_date` 编造 fiscal 事实。`6-K` 与 `6-K/A` 在没有同源 fiscal 证据时都不得再猜 `fiscal_year/fiscal_period`；`10-Q` 也不得在 `list_documents()` 阶段仅凭 `report_date` 推断季度；其他表单同样不得在消费侧把空的 `fiscal_year` 从日期回填出来。当前仅保留表单内生、且不依赖日期猜测的低风险回退，例如 `10-K/20-F -> FY`。`download --rebuild` 走同一套真源，并且会清理历史上遗留在 source meta / manifest 中的 6-K / 6-K/A 猜测值。这个修复只影响 fiscal 字段，不改变 6-K 现有的 `document_type` 返回语义。

边界划分：
- `FinsService`：接收命令、声明宿主执行需求
- `Host`：托管 run 生命周期
- `FinsRuntime`：命令路由
- pipeline：具体执行下载、上传、预处理

### 5.1 并发与宿主约束

当前直接受 Host 约束的典型点：
- `download` 走 `sec_download` lane，必须串行
- 流式 direct operation 事件可以双写到 EventBus
- 取消通过 `CancellationBridge` 收口在 Host，而不是散在 Fins 内部
- direct operation 不得自己持有 `RunRegistry` 或 Host 内部桥接器；若需要及时响应取消，只能沿 `Service -> Runtime -> Pipeline` 透传窄 `cancel_checker`，并在已支持取消协作的阶段边界主动停机

### 5.2 direct operation 公共契约

`download/upload/process` 这条 direct operation 链路当前按命令拆成独立载荷和结果类型：

- `DownloadCommandPayload` / `DownloadResultData` / `DownloadProgressPayload`
- `UploadFilingCommandPayload` / `UploadFilingResultData` / `UploadFilingProgressPayload`
- `UploadFilingsFromCommandPayload` / `UploadFilingsFromResultData`
- `UploadMaterialCommandPayload` / `UploadMaterialResultData` / `UploadMaterialProgressPayload`
- `ProcessCommandPayload` / `ProcessResultData` / `ProcessProgressPayload`
- `ProcessFilingCommandPayload` / `ProcessSingleResultData`
- `ProcessMaterialCommandPayload` / `ProcessSingleResultData`

实现约束：
- CLI / Web 入口负责把外部参数收敛成这些强类型 payload，再交给 `FinsService`
- `FinsRuntime` 只按 `command.name -> payload dataclass` 做机械路由，不再靠 `payload.get(...)` 猜字段
- `FinsRuntime` 需要同时提供同步 `validate_command()` 与执行期 `execute()` 两个入口；前者用于 `Service` 受理阶段的 preflight，后者用于真实执行，两者必须复用同一套 namespace builder / ticker 校验真源，不能各自维护一份规则
- `FinsRuntime.execute()` 与流式桥接必须先按 `command.name` / 事件流来源把 `FinsCommandPayload`、pipeline 事件联合显式收窄到对应 dataclass，再进入 namespace builder、pipeline 调用和 progress/result builder；不允许靠“各分支刚好共享字段名”穿透联合类型
- `ProcessCommandPayload` 支持可选 `document_ids`；当调用方显式指定文档 ID 时，Runtime 必须把这组文档 ID 规范化后传给 pipeline，不能在桥接层丢失过滤语义
- CLI `process` 命令和 ingestion job manager 一样必须把 `document_ids` 继续传到 `process_stream(...)`；允许在桥接层做去空、去重和稳定排序，但不能静默丢失文档过滤语义
- pipeline 对外 `download/process/upload` 事件对象的 `event_type` 必须直接使用各自的 `StrEnum` 真源；允许内部 helper/service 先产出更窄的文件级字面量事件，但必须在 pipeline 边界显式映射到 `DownloadEventType`、`ProcessEventType`、`UploadFilingEventType`、`UploadMaterialEventType`，不能再把裸字符串直接塞进公开事件对象
- 对仍需沿用 CLI 规范化规则的 direct operation 命令，payload 里要显式保留 `infer` / `ticker_aliases` 等语义字段；`FinsRuntime` 在进入 pipeline 前会重建 namespace 并复用 `prepare_cli_args` / `validate_*`，保证 ticker、alias、公司名和 `form_type` 与 CLI 真源一致；这条链路必须支持“canonical ticker + 已归并 ticker_aliases”的二次 prepare，不得在 runtime 重放时把 alias 再解析丢失
- CLI formatter 可以兼容当前 pipeline 直出字典结果，但要先在 formatter 边界把原始字典收敛成 `dayu.contracts.fins` 中的强类型 result dataclass，再按命令名分发到对应 `_format_*_result`；不能把 `FinsResultData` 联合或 `dict.get(...)` 宽访问继续穿透到展示逻辑深处

### 5.2 SEC source filings 的 active / rejected 边界

SEC 下载链路当前把 source filings 分成两类持久化结果：

- active filings
  - 由 source 文档仓储维护，进入正常 `filings` 文档集合
  - 会写入 active source meta，并参与默认 `process` / `process --ci` 主链
  - 对 `6-K`，下载链路会先按 `_classify_6k_text()` 决定是否保留 filing，再在 active source 内用当前 `BsSixKFormProcessor` 评估全部 HTML 候选，把最能稳定提取 `income + balance_sheet` 的文件写回 `primary_document`，避免 D3 因错误主文件稳定失分
- rejected artifacts
  - 由 filing maintenance 仓储维护，统一落到 `workspace/portfolio/{ticker}/filings/.rejections/`
  - 当前只承载 policy reject 证据，例如 `6-K` 预筛选 miss 与 `SC 13D / SC 13D/A / SC 13G / SC 13G/A` direction mismatch
  - rejected artifact 保留完整 source-doc 形态：原始文件 + `meta.json`
  - rejected artifact 不进入正常 `filings/manifest.json`，也不进入默认 `process` 主链

`6-K` 预筛选的远端候选集合当前不再只看 `primary_document + EX-99.x exhibit`；若
`index-headers` 中存在 `TYPE=6-K` 的 HTML cover，也必须一并下载并交给同一个
`_classify_6k_text()` 真源做重排，避免像 FER 这类“季度正文挂在 cover、而 primary/exhibit
都只是包装材料”的漏选回归。除此之外，下载器还要继续从主 `6-K` cover 中补链同 filing、
同归档目录下的相对 HTML 链接；像 VIST / FRO 这类“真正财务报表挂在 cover 里的相对
`dex1/ex1` HTML 附件、而 index.json / index-headers 都没有列出来”的样本，若补链缺失，
会直接表现成 source 缺文件而不是 processor 不会抽取。

与之配套，`_download_rejections.json` 只保留轻量 skip index 语义；它负责“当前版本规则下可以跳过重复下载”，不承担诊断样本保存职责。

### 5.3 6-K 规则诊断闭环

当前 `dayu/fins/sec_6k_rule_diagnostics.py` 提供可复用的诊断闭环，`utils/sec_6k_rule_diagnostics.py` 只负责参数解析和调用。

当前还新增了与之互补的 `dayu/fins/sec_6k_primary_document_diagnostics.py`：

- 规则诊断回答“当前 `_classify_6k_text()` 是否把这份 6-K 判错了”
- 主文件诊断回答“这份 6-K 当前是不是选错了 `primary_document`，把真正的季度正文 exhibit 漏掉了”
- 当前下载链路按两段真源收口：`SecPipeline._precheck_6k_filter()` 先把 `primary_document` 与同 filing 的 HTML candidate 一起重新交给 `_classify_6k_text()` 分类，只要存在季度结果 candidate 就必须保留该 filing；落盘后再用同一份 active source 和当前 `BsSixKFormProcessor` 评估全部 HTML 候选，把最能稳定提取核心报表的文件写回 `primary_document`。也就是说，`_classify_6k_text()` 决定“这份 filing 要不要留”，而处理器 reconcile 决定“留下后谁是最终 primary”

第二条诊断只报告一种严格场景：当前 active filing 的 `primary_document` 重新分类后不是季度结果，但同一 filing 中另一个 HTML candidate 被同一真源 `_classify_6k_text()` 判成 `RESULTS_RELEASE` / `IFRS_RECON`。也就是说，它只抓“选文问题伪装成规则问题”的样本，不会把普通多文件 6-K 全部打出来。

当需要先做小样本验证时，可以通过 `python utils/sec_6k_rule_diagnostics.py --tickers AAA,BBB,CCC` 只对指定 ticker 子集运行同一套诊断口径。

这条闭环的固定口径是：

- 20-F 公司识别真源：当前 workspace active `filings` 中至少有一份 `20-F`
- 遍历 active filings 时若遇到“目录仍在、但 `meta.json` 缺失”的残缺 source 目录，诊断闭环会直接跳过该样本；这类坏数据不应阻断整批规则诊断
- 重跑快照：只对 active `6-K` 文档并发执行 `python -m dayu.cli process --ci --overwrite --base {workspace_root} --ticker {ticker} --document-id {document_id}`，并发上限 `26`
- false positive 证据：active `6-K` 中，CI 评分 `hard_gate.passed = false`
- false negative 证据：只排除 `hard_gate_reasons` 中明确标记 `HGF` 的 filing 后，active `6-K` 数量 `< 3`，并关联 `.rejections/` 中的 `6-K` 样本；其它 hard-gate fail 仍保留在 active 计数里
- false negative 证据只允许统计“当前仍停留在 `.rejections/` 的样本”；若某个 `document_id` 已经回到 active filings，即使同名 rejected artifact 还残留，也必须在诊断口径里忽略，不能把已救回样本再次误算成 false negative

使用这批证据迭代 `_classify_6k_text()` 时，必须先区分“筛选规则误判”和“下游抽取 / HGF 失败”两类问题：

- 像 `Date of Board Meeting`、`Financial Results Announcement Date`、`results will be released`、`profit warning`、`operating update ... financial results are only provided on a six-monthly basis`、`adjustment to exercise price of equity linked securities and call spread`、`trading statement / update note / operating statistics`、`audio recording of the earnings call`、`results conference call transcript`、`investor presentation referred during the earnings call`、`results release scheduled` / `will report results and host a conference call` / `earnings release date` / `results call` / `financial results and corporate update webcast` / `earnings release zoom meeting` / `earnings release conference and blackout period` 这类 future notice、`live webcast of the conference and the presentation materials will be available` 这类 earnings conference 通知、`presentation materials related to ... financial results` / `materials and a webcast replay are available` / `analyst Q&A session transcript` / `ER Investor Presentation` / `earnings release investor presentation` 这类结果附属材料、`we are enclosing herewith the presentation on the ... financial results` / `newspaper advertisement regarding ... financial results` 这类本地交易所同步材料、`Transaction in Own Shares` / `Announcement of Share Repurchase Programme` / `Shell announces commencement of a share buyback programme` / `Adjustment to Cash Dividend Per Share` / `Stockholder Remuneration Policy (Dividends and Interest on Capital)` / `Public Announcement Post-Buyback` 这类资本回报材料、`Appendix 3A.1 - Notification of dividend / distribution` / `Key Dates ... Ex-dividend date / results announcement` 这类分红日期通知、带 `preliminary expected financial results` / `preliminary unaudited ... results` / `tentative consolidated revenue` / `final result ... will be provided by our annual report` / `subject to ... closing procedures` / `subject to revision` / `independent registered accounting firm has not reviewed or audited` 限制语的预估快讯，以及带有 `Our strategy / Agenda / Q&A` 结构的战略 deck，这类文本同源信号属于筛选规则应该直接排除的样本
- 像已经包含 `unaudited financial results`、`interim consolidated financial statements`、`letter to shareholders containing quarterly results`、或 `Reports Third/Fourth-Quarter [and Full-Year] Financial Results` 这类真实季报标题家族的样本，更可能是季度财报正文或下游处理器 / GateFail 问题，不能再靠收紧 `_classify_6k_text()` 去“修”
- `_classify_6k_text()` 的排除信号应继续只基于标题附近前缀做判断；像 `annual report`、`Convertible Senior Notes` 这类词如果只出现在真实季度结果正文后段，不应覆盖前缀里已经成立的季度披露主信号
- 季度主信号还需要覆盖国际发行人常见的紧凑标题写法，例如 `4Q24 Results`、`1Q 2025 Results`、`Q3 Results`、`first-quarter results`，以及 `reported its financial results for the three and twelve month periods ending ...` 这类 `4Q + FY` 新闻稿句式
- 若 primary document 直接是季度 XBRL instance HTML，`_extract_head_text()` 可能只剩压缩的 fiscal quarter code、taxonomy token 与 `2025-04-012025-06-30` 这类 context 日期区间；这类样本也必须由 `_classify_6k_text()` 在真源层直接保留，而不是等治理脚本特判
- 放大样本后还要继续守住三类常见季度正文/封面变体：`Financial Management Review 4Q24 / Quarterly YTD Report`、`Exhibit 99.1 ... interim report for the quarter ended ...`、以及 `Reports Full Year ... Financial Results and Provides Fourth Quarter Business Update`；这几类虽然不总是直接写成 `Q1 Results`，但同源语义仍是季度财报披露
- 再继续放大样本时，还要守住三条新增边界：半年报 XBRL instance 不能只因 `2025Q2` + `2025-01-012025-06-30` 这类压缩区间而漏判；`UNAUDITED CONDENSED CONSOLIDATED FINANCIAL STATEMENTS ... FOR THE THREE AND SIX/NINE MONTHS ENDED ...` 这类财务报表封面必须直接保留；`Reports Half Year ... Financial Results and Provides Second Quarter Business Update` 这类已披露结果、同时附带季度 business update 的标题，不能再被 future-announcement 规则误杀
- 当前 future-announcement 规则还要继续守住四类“看起来像 results release、其实只是预告/说明材料”的误收边界：`Board Meeting / Trading Window Closure` 里预告将审议季度财报、`will report ... results on <date>` / `will host conference call to discuss results` / `earnings release date` / `Q1-Q4 results call` / `financial results and corporate update webcast` / `earnings release zoom meeting` / `earnings release conference and blackout period` / `notice of announcement of ... interim results` 这类发布时间与电话会通知、`results conference call` / `conference call and live audio webcast presentation` / `performance report dates` / `financial statements filing will be postponed` 这类结果日期与会议安排公告，以及 `Group Reporting Changes / data pack / in advance of the publication of earnings release` 这类口径重列说明；但若同一前缀里已经出现 `financial highlights / statements of income / balance sheets / unaudited condensed consolidated financial statements / reported its financial results for ...` 这类强披露结构，仍必须继续保留；公司介绍段里的历史 `revenues / net income` 之类弱指标不能单独把明确的 schedule notice 重新抬回 active
- 在继续收紧 future-notice 边界时，还要显式守住八类“当前 active 中常见、但原文并非季度结果正文”的误收材料：`Results for Announcement to the Market ... filed the following documents with the ASX` 这类 ASX 转发壳、`Response to ASX Aware Letter` 说明函、`operating results for June/September/March 20xx` 这类月度经营数据、`Announcement Regarding ... Financial Results Calendar` / `Results for Production and Volume Sold per Metal` / `Quarterly Activities Report` / `Quarterly Update of Resumption Progress` / `Bitcoin Production, Mining Operation Updates, and Preliminary ... Financial Results` 这类经营更新与产销数据公告、`Shell first quarter/fourth quarter update note ... results are subject to finalisation and scheduled to be published on <date>` 这类 outlook/update note、`Minutes of the Board of Directors / Fiscal Council` 或 `Opinion of the Fiscal Council` 且正文只是在审议/确认 `quarterly financial report / interim financial statements / annual financial statements` 的治理纪要、只包含治理说明的 `Dear Sirs ... approved the condensed interim financial statements ...` 本地交易所通知函，以及 `Request for clarification / Official Letter / news published in the media` 这类监管问询回复函；这些都应退出 active，而不是继续要求 processor 抽三大表
- 对巴西/阿根廷本地交易所摘要函要额外守住“治理通知”和“结果摘要”两层语义：若函件正文已经列出 `net profit for the period`、`statement of financial position`、`statement of comprehensive income`、`statement of cash flows` 或 `relevant information ... follows` 这类当期财务摘要，它就属于本地同步披露的季度结果，必须保留为 active，不能再按纯治理函件误杀
- 对 `trading update / operating highlights / operational highlights` 这类最容易误伤的标题，真源规则必须坚持“双向约束”：只有当前缀里同时出现 `guidance / deliveries / mineral resource estimate / resumption progress / production per metal / volume sold per metal / key metrics` 这类经营更新证据，且没有 `reported its unaudited consolidated financial results / statements of income / balance sheets / unaudited condensed consolidated financial statements` 这些强披露信号时，才允许退出 active；像 `OMA Announces ... Operating and Financial Results` 这种真实 results release 必须继续保留
- 当放大样本已经收敛到少数明显非季报材料时，`_classify_6k_text()` 还应把它们从 `NO_MATCH` 收紧成显式排除：`NOTICE AND INFORMATION CIRCULAR / NOTICE OF MEETING / ANNUAL GENERAL MEETING OF SHAREHOLDERS` 这类治理材料、`Consolidated Financial Statements For the years ended ...` 且带审计报告的年度财务报表附件、`Investor Conference / Investor Day` 业务更新材料，以及 `Mining Forum ... Presentation` 这类论坛展示材料

诊断输出统一写到 `workspace/tmp/sec_6k_rule_diagnostics/`，至少包含：

- `summary.json`
- `summary.md`
- `false_positive_6k.json`
- `false_negative_6k.json`

当规则修复后需要把 `.rejections/` 中被误拒的 `6-K` 本地救回到 active filings 时，当前真源是 `dayu/fins/rejected_6k_rescue.py`，`utils/rescue_rejected_6k_filings.py` 只负责参数解析和打印结果。它的固定边界是：

- 不重新下载 SEC 文档，只复用 `.rejections/` 中已有 artifact
- 只扫描 `rejection_reason == "6k_filtered"` 的 `6-K` artifact，并用当前 `_classify_6k_text()` 重新分类
- 只有当前分类重新落到 `RESULTS_RELEASE` / `IFRS_RECON` 的样本才会回灌 active filings
- 回灌必须通过 source/blob/maintenance 仓储完成，自动更新 active source meta 与 manifest，不能直接 move 目录或手改 manifest
- 遍历 `.rejections/` 时若个别 `document_id` 目录缺失 `meta.json` 或元数据损坏，仓储会跳过坏目录继续处理其它 artifact，不能让单个坏目录中断整批 rescue
- 会同步清理 `_download_rejections.json` 中对应 `document_id` 的 skip 记录，但 `.rejections/` artifact 会保留为审计痕迹
- 脚本默认 `dry-run`；只有显式传 `--apply` 才会实际写回
- 若 active 侧只残留 `is_deleted=true` 的历史 meta、但实际 `meta.json` 已缺失，rescue 也必须回退到 create 路径重建 active source，不能因为半残 deleted state 让误拒样本卡在无法恢复的中间态

与之对称，若规则修复后需要把当前 active filings 中“旧规则误收”的 `6-K` 重新退出 active，当前真源是 `dayu/fins/active_6k_retriage.py`，`utils/retriage_active_6k_filings.py` 只负责参数解析和打印结果。它的固定边界是：

- 只扫描 active filings 中当前仍未删除的 `6-K`
- 复判真源仍然只有 `_classify_6k_text()`，不在治理脚本里复制第二套规则
- 只有当前重新分类为 `EXCLUDE_NON_QUARTERLY` / `NO_MATCH` 的样本才会进入误收候选
- 遍历 active filings 时若个别 `document_id` 目录缺失 `meta.json` 或元数据损坏，复判会跳过坏目录继续处理其它 active 样本，不能让单个脏目录中断整批 retriage
- `--apply` 时会先把 active filing 对称写回 `.rejections/`、写入 `_download_rejections.json`，再把 active source 逻辑删除，并尽力删除同 `document_id` 的 processed 产物，避免继续污染下游统计
- 脚本默认 `dry-run`；只有显式传 `--apply` 才会实际把误收样本移出 active

### 5.4 这几条 6-K 治理脚本怎么用

这套 6-K 治理当前涉及 6 个相关文件，但日常真正需要手动运行的是 `utils/` 下的 4 个入口：

- `utils/sec_6k_rule_diagnostics.py`
  - 用途：先看当前规则哪里误收、哪里误拒
  - 是否手动运行：是
  - 参数口径：`--workspace-root`、`--output-dir`、`--tickers`
- `utils/sec_6k_primary_document_diagnostics.py`
  - 用途：识别 active `6-K` 是否把 `primary_document` 指到了非季度 exhibit，导致真正的季度正文被同 filing 里的其它文件遮住
  - 适用场景：排查历史 active 数据，或在放大样本时把“旧数据残留错选”和“当前规则本体问题”分层
  - 是否手动运行：是
  - 参数口径：`--workspace-root`、`--output-dir`、`--tickers`、`--document-ids`
- `utils/reconcile_active_6k_primary_documents.py`
  - 用途：对当前 active `6-K` 按现有 `BsSixKFormProcessor` 真源重排 `primary_document`，一次性修正历史错选 source meta
  - 适用场景：确认问题在历史 active 数据，而不是当前下载规则本体时，批量收口旧 primary 误选
  - 是否手动运行：是
  - 参数口径：`--base`、`--tickers`、`--document-ids`
- `utils/rescue_rejected_6k_filings.py`
  - 用途：把 `.rejections/` 中按新规则应保留的 `6-K` 救回 active filings
  - 是否手动运行：是
  - 参数口径：`--base`、`--tickers`、`--document-ids`、`--apply`
- `utils/retriage_active_6k_filings.py`
  - 用途：把 active filings 中按新规则应排除的 `6-K` 重新退出 active
  - 是否手动运行：是
  - 参数口径：`--base`、`--tickers`、`--document-ids`、`--apply`
- `dayu/fins/rejected_6k_rescue.py`
  - 用途：误拒救回的核心实现模块
  - 是否手动运行：否；由 `utils/rescue_rejected_6k_filings.py` 调用
- `dayu/fins/active_6k_retriage.py`
  - 用途：误收剔除的核心实现模块
  - 是否手动运行：否；由 `utils/retriage_active_6k_filings.py` 调用

日常排查和治理时，推荐按下面顺序使用这些命令：

1. 先跑诊断，确认当前 false positive / false negative 样本

```bash
python utils/sec_6k_rule_diagnostics.py --workspace-root workspace
```

如果怀疑某批样本的问题根本不在 `_classify_6k_text()`，而在同一 filing 多 exhibit 下 `primary_document` 选错，可先跑主文件诊断：

```bash
python utils/sec_6k_primary_document_diagnostics.py --workspace-root workspace --tickers JHX,ITUB
```

也可直接缩到 document_id 粒度：

```bash
python utils/sec_6k_primary_document_diagnostics.py --workspace-root workspace --document-ids fil_0001159152-25-000045
```

如果诊断确认问题是历史 active source 的 `primary_document` 残留错选，而不是当前下载规则误判，可直接批量 reconcile：

```bash
python utils/reconcile_active_6k_primary_documents.py --base workspace --tickers JHX,ITUB
```

需要先做小样本验证时，可加 ticker 子集：

```bash
python utils/sec_6k_rule_diagnostics.py --workspace-root workspace --tickers ASML,FMX
```

可选参数：

- `--workspace-root` 指定 workspace 根目录，默认是 `workspace`
- `--output-dir` 指定诊断输出目录；不传时默认写到 `workspace/tmp/sec_6k_rule_diagnostics/`
- `--tickers AAA,BBB` 只扫描指定 ticker
- `--max-concurrency` 控制 `process --ci --overwrite` 并发上限，默认 `26`

2. 对 `.rejections/` 中疑似误拒样本做 dry-run；确认无误后再 `--apply`

```bash
python utils/rescue_rejected_6k_filings.py --base workspace --tickers ASML,FMX
python utils/rescue_rejected_6k_filings.py --base workspace --tickers ASML,FMX --apply
```

可选参数：

- `--tickers AAA,BBB` 只扫描指定 ticker
- `--document-ids fil_xxx,fil_yyy` 只扫描指定 document_id
- 不传 `--apply` 时永远是 dry-run

3. 对 active filings 中疑似误收样本做 dry-run；确认无误后再 `--apply`

```bash
python utils/retriage_active_6k_filings.py --base workspace --tickers ALC,ASM,NVA
python utils/retriage_active_6k_filings.py --base workspace --tickers ALC,ASM,NVA --apply
```

可选参数：

- `--tickers AAA,BBB` 只扫描指定 ticker
- `--document-ids fil_xxx,fil_yyy` 只扫描指定 document_id
- 不传 `--apply` 时永远是 dry-run

诊断脚本和治理脚本的 workspace 参数名当前有意保持与各自入口一致：

- 诊断脚本沿用 `--workspace-root`
- rescue / retriage 脚本统一使用 `--base`

建议工作顺序固定为：

1. 先用 `sec_6k_primary_document_diagnostics.py` 排除“选文问题伪装成规则问题”的样本
2. 再改 `_classify_6k_text()`
3. 跑 `sec_6k_rule_diagnostics.py` 看样本是否收敛
4. 对误拒样本跑 `rescue_rejected_6k_filings.py`
5. 对误收样本跑 `retriage_active_6k_filings.py`

其中第 2 到第 5 步都必须把 `_classify_6k_text()` 当作唯一真源；主文件诊断也只是把同 filing 下各候选 exhibit 重新交给同一真源分类，再报告“当前主文件是否被别的季度 exhibit 支配”，不会额外复制另一套规则。

## 6. Ticker 归一化真源

所有 ticker 的归一化都统一收敛到 `dayu/fins/ticker_normalization.py`：

- 公共 API：`normalize_ticker(raw) -> NormalizedTicker`（非法抛 `ValueError`）、`try_normalize_ticker(raw) -> Optional[NormalizedTicker]`（非法返回 `None`）、`ticker_to_company_id(ticker) -> str`。
- `NormalizedTicker` 包含 `canonical`、`market`、`exchange`、`raw` 四个字段。
- Canonical 约定：
  - 港股 4 位补零（`0700`）；港交所新发 5 位代码原样保留（`89988`）；`exchange="HKEX"`、`market="HK"`。
  - 沪股 6 位裸码，首位 `6`（主板 / 科创板）；`exchange="SSE"`、`market="CN"`。
  - 深股 6 位裸码，首位 `0` / `3`（主板 / 创业板）；`exchange="SZSE"`、`market="CN"`。
  - 美股保留原字母（`AAPL` / `BRK.B` / `BF.B`）；`exchange=None`、`market="US"`（当前不区分 NYSE/NASDAQ）。
- 使用规则：
  - CLI / service / 仓储 / downloader / pipeline / prompt contribution 一律调用该真源，不再在各自模块重造 normalize 实现。
  - `FinsToolService._resolve_canonical_ticker` 中真源识别失败时会回退到 `strip().upper()` 作为查询候选，保留"公司名当 ticker 传"的既有行为。
  - CLI `--ticker` CSV 中**每个 token 都走真源归一化**，再整体去重：首个归一化结果作为 canonical，其余作为显式 alias。业务动机是把同一公司的跨市场 ticker（如 `BABA,9988`）整体作为 alias 写入 meta，让工具查询无论传哪种变形都能命中。
- `ticker_to_company_id` 当前直接返回 `ticker.canonical`，属"稳定契约、实现可演进"：保留该接口以便后续接入跨市场上市折叠、CIK、统一社会信用代码等更精细的公司主体映射。

## 7. 内部分层

当前更合理的内部理解方式是五层：

| 层 | 责任 |
| --- | --- |
| Service / Runtime Adapter | direct operation 与 Host 对接 |
| Pipeline / Ingestion | 下载、上传、处理等长事务编排 |
| Tool Service | 文档读取、搜索、表格与财务查询能力 |
| Processor | 单文档解析、切 section、读表、抽报表 |
| Repository | 公司 / source / processed / blob / maintenance 文档存取与文件系统落盘 |

其中：
- `Tool Service` 回答“给上层什么读取能力”
- `Processor` 回答“如何解析单份文档”
- `Repository` 回答“不同职责簇下文档如何存与取”
- 旧的 `DocumentRepository / FsDocumentRepository` 已被删除，不再保留总仓储 facade
`Repository` 层的文件系统实现当前按职责拆分为 mixin 组合：

| 模块 | 职责 |
| --- | --- |
| `_fs_storage_utils` | 模块级工具函数与常量 |
| `_fs_storage_infra` | 共享基础设施（batch / path / manifest / handle） |
| `_fs_company_meta_core` | 公司级元数据操作 |
| `_fs_source_document_core` | 源文档 CRUD / 查询 / 文件访问 |
| `_fs_processed_core` | 解析产物操作 |
| `_fs_blob_core` | Blob / 文件条目操作 |
| `_fs_maintenance_core` | 拒绝注册表与清理 |
| `_fs_storage_core` | 组合入口（mixin 钻石继承，对外保持 `FsStorageCore` 不变） |

当前文件系统仓储还新增了一条稳定实现边界：
- batch 暂存、提交备份与 crash recovery 的隐藏工作目录统一收敛到 `workspace/.dayu/repo_batches` 与 `workspace/.dayu/repo_backups`；上层不应直接读写这些目录。
- batch recovery 必须由 storage core 在可写装配时自动执行，并通过 journal + 文件锁按同源文件状态恢复；ticker 锁是“该 ticker 是否仍有活跃 batch” 的权威边界，不能把 `transaction.json` 中仍然存活的 `owner_pid` 误当成充分条件，否则会把已释放锁但遗留 staging 的真 orphan 永久保留下来；这套文件锁语义必须跨平台成立：POSIX 走 `fcntl.flock()`，Windows 走 `msvcrt.locking()`，同一 ticker 的非阻塞 batch 锁与 recovery 锁都不能在 Windows 上退化成无锁；其中 recovery 这类 blocking 锁在 Windows 上也必须继续等待到真正拿到锁，不能偷换成 `LK_LOCK` 的有限次重试；recovery 在枚举 live token 后，如果对应 token 目录已被 owner 正常提交并删除，也必须把它视为已消失 token 并安静跳过，不能让单个 `FileNotFoundError` 中断整批恢复；`create_directories=False` 的只读探测必须保持无副作用，CLI / Web / WeChat / pipeline 不能各自复制清理逻辑。
- rollback 属于补偿路径：即使 `rolled_back` journal 写入失败，也必须继续清理 staging 并释放 ticker 锁；在 auto batch 写路径里，rollback 失败只能作为附注暴露，不能覆盖原始写操作异常。

当前还有两条需要持续守住的实现约束：
- 处理器侧的文档来源依赖 `Source` 结构化协议；具体实现与测试桩保持结构兼容即可，不应把只读 `Protocol` 当成运行时基类。
- `processors/sec_form_section_common.py` 中的 `_VirtualSectionProcessorMixin` 只能叠加在已经实现标准 `list_sections/read_section/list_tables/get_section_title/search` 接口的底层 processor 上；mixin 对下一跳的所有透传都以这组稳定协议为边界，禁止再用 `type: ignore[misc]` 掩盖 MRO 下一跳不明确的问题。
- SEC/EDGAR HTML 预处理规则的真源在 `processors/sec_html_rules.py`；凡是 exhibit SGML 信封剥离、SEC 封面页 layout 表识别、section heading 横线表识别，都必须复用这一个模块，不能把规则散落在 Engine 或多个 Fins 处理器私有 helper 中。
- `FinsBSProcessor` 通过覆写 `BSProcessor` 的 HTML 读取钩子承接 EDGAR SGML 预处理；Engine 默认 HTML 读取链只做通用文件读取，不负责财报领域规范化。
- 报告类表单与 6-K 处理器若需要复用 Engine 的 HTML 表格 DataFrame 解析能力，只能依赖 `dayu.engine.processors.table_utils.parse_html_table_dataframe()` 这类公共入口；禁止直接 import `bs_processor.py` 等具体处理器模块里的私有 helper。
- 20-F 处理器的顶层 marker 恢复必须同时覆盖三类真源：显式 `Item` 标题、`cross-reference guide` 反查，以及无 guide 的 `annual-report-style` 页标题；后一类要以 source text 中同源可见的“页脚导航/栏目标题 + 标题短语 + 页码 + 后续正文”组合为准，不能只依赖独立换行标题。`Item 3` 这类 narrative section 中反复出现的 `Annual Report` / `Not applicable`，若没有页码 locator 或 `Note X` 证据，不能被误判成 `cross-reference guide` 并提前短路 child split。
- 20-F key-item 修复当前还必须同时过滤三类伪命中：正文句内 bare phrase、句中引号包裹的 `Item 5 ...` 交叉引用，以及夹在 annual-report 页码 locator 块中的财报页标题；对用于替换 front matter 污染位的 `Item 18`，即使物理位置早于后续 `Item 5`，也要保留其真实财报边界。
- 20-F key-item re-search 还必须区分两类外观相似的 `Item 18` 文本：真正的 `Item 18` 标题后紧跟 `starting on page F-1` 之类正文 locator 时，应保留该标题作为财报边界；但若同一行已经混入页码和后继 `Item 19`，则仍应视为 ToC / locator stub，不能当作正文标题。
- 20-F locator / key-item re-search 的性能护栏也必须持续生效：在做 `Item 18` 正文白名单或 guide 判定前，先用局部 probe / standalone heading 过滤掉绝大多数无关候选，避免 `UBS` 一类 annual-report-style 20-F 在 `Financial Statements` 全文重扫阶段把时间耗在非 `Item 18` 命中的逐行邻域分析上。
- 20-F 的 BS 主路径在同一份 `full_text` 上不能重复重建 marker：默认 DOM 抽文的最低质量判断与正式 virtual-section 切分若命中同一文本，必须复用同一份 `marker` 结果，避免 `UBS` 一类超长 guide 文档在 Processor 初始化阶段重复支付整套 locator / key-item 回查成本。
- 20-F 对 `标题 + 页码` 形态命中的 annual-report page heading 判定也必须先让 front matter 短路：若同一上下文已经出现 `Form 20-F caption`、`location in this document` 等 cover/guide 信号，就不应再进入昂贵的 annual-report page-heading 邻域分析。
- 20-F key-item repair 若已经拿到同源、单调的 `Item 3 < 5 < 18` fallback 主链，仍要先确认当前 `Item 3/4/5/18` 是否已经是干净、顺序安全的正文锚点：只有当前位置缺失、污染或明显更晚时，才允许用 fallback 主链回填；若当前 marker 已是正文标题，就不能再被更早的 ToC / annual-report guide fallback 反向覆盖，否则 `NVS` 一类文档会从真实 `Item 3` 重新退回目录区。
- 对只有 `Item 5` 先恢复出来、而前面暂存的 `Item 1-4A` 仍全是 guide/front-matter 污染位的 20-F，最终 monotonicity 修正也必须保留这个干净 `Item 5`，不能因为它物理位置早于污染前缀就再次删除；这类文档允许先保住 key item，再交给按位置排序的虚拟切分处理。
- 20-F 的最终 monotonicity 修正还必须再守住一个稳定约束：若当前候选已经是干净正文锚点，而 `result` 尾部只是连续的 ToC / guide 污染 marker，应先清掉这些污染尾巴，再决定是否回滚 moved marker；否则 `AEG/CCEP` 一类文档里的真实 `Item 3` 会被污染前缀错误挡掉。
- 20-F 的 guide repair 还必须允许“partial reconstructed + 现有 key-item 主链”合并：如果 guide 只能补出更早的 `Item 3/4`，而 `Item 5/18` 已由 key-heading fallback 找回，就不能再要求 guide 自身独立覆盖完整 `3/5/18` 才放行；否则 `AMX/ASML` 一类文档会在 merge 前被提前短路。
- 20-F 的 key-item monotonicity 还必须优先保留 `Item 3/5/18`：若 guide merge 或 clustered-tail repair 引入的非关键尾部 marker 与关键项冲突，应优先丢弃这些非关键尾标，而不是把关键 `Item 18` 或 synthetic `Item 3` 再删掉；同理，当原始 `Item 1/2/3` 整体聚在文末目录簇时，也不能再让这些污染尾标充当 synthetic `Item 3` 的顺序下界。
- 20-F 的 key-heading repair 末尾也必须复用同一条“关键项优先”约束：如果 `repaired` 阶段已经恢复出完整 `Item 3/5/18`，但统一 monotonicity 因为后续 `Item 6/7/9/10/...` 非关键链冲突又删掉 `Item 5` 或 `Item 18`，就必须回退到关键项优先的单调链；否则 `PHI/TRMD/TIGR/BCS` 一类文档会在最后一步被重新打回 `缺失关键 Item`。
- 20-F 的 BS 主路径在默认 DOM 抽文时也必须保留节点间换行边界，不能对每个节点做 `strip=True` 压平；否则像 `CYBR` 这类目录/正文都依赖节点边界维持 Item 结构的文档，会把 `Item 5` 再次挤回 ToC 形态并误触发回退路由。
- 20-F key-item re-search 还必须守住另一类顺序缺陷：如果按“最近干净前序 Item”重搜出来的 `Item 5` fallback 反而跳回当前 `Item 4A` 之前，就不能用这个更早伪命中替换掉已经处在合法顺序里的现有 `Item 5`；否则 `BAP` 一类 annual-report-style 文档会在 repair 阶段先把 `Item 5` 改早，再在最终顺序修正里把它彻底删掉。
- 20-F 的 guide-based locator repair 还必须过滤另一类 annual-report-style 伪命中：若 locator phrase 命中了真实 `Form 20-F` 之前的 `Integrated Annual Report` / `Annual Financial Report` 等报告套件封面或封面目录页，不能把这些 report-suite cover 当成顶层 Item 锚点；否则会把 `Item 8/9/10` 等边界提前到册子封面，直接制造超大 section。
- 20-F key-item / guide repair 的性能约束也属于稳定实现边界：`Item 18` 正文白名单、guide 判定与句内 cross-reference 过滤必须先做低成本局部 probe，再进入逐行邻域分析；否则 `BILI/GFI` 一类超大 annual-report-style 文档会在 BS 主路由初始化阶段被高频 fallback 重扫拖回 fallback processor。
- 20-F guide snippet 抽取还必须接受 `Form 20-F references` 与 `SEC Form 20-F cross reference guide` 这类真实锚点，不能只认 `Form 20-F caption` / `Location in this document`；否则 `UL/NWG` 一类年报样式 20-F 根本进不了 locator repair。
- 20-F 的 `Item 5` re-search 还必须覆盖 `Financial performance` 这种年报样式标题，而不只识别 `Financial review` / `Operating results`；否则 `NWG` 一类文档即使 `Item 3/18` 已回收，仍会卡在 `缺失 Item 5`。
- 报告类表单在 `xbrl_not_available` 等允许回退的场景下，HTML 财报候选筛选要走三层收敛：先做 `is_financial` + caption/header/context 分类，再做严格 row-signal，最后再做提高阈值的宽松 row-signal；不能把无 XBRL 直接等同于“无可用财报表”。
- 10-K heading fallback 的性能护栏同样不能回退：只有当命中附近已经出现 `Table of Contents`、页码引用或其它 `Item` 编号时，才进入完整的 ToC 语境分析；不能让 `SO/ETR` 一类正文句内短语在每次 fallback 命中时都重复跑页码统计和正文探测。
- 6-K processor 对 exhibit 年报这类 image+OCR HTML 还必须保留“无 `<table>` 时的隐藏文本财报回退提取”；若原文同源只暴露图片页和白色 1px/1pt OCR 文本，不能因为 `list_tables=[]` 就把 `income` / `balance_sheet` 直接判成 `statement_not_found`。
- 6-K processor 对本地交易所同步摘要函还要保留“单期间摘要表”回退：若 caption/context 已明确给出 `for the three-month period ended ...`、`net profit for the period` 这类结果摘要，但表内只剩单数值列，处理器也必须能补出单期间 `income`，把它归类为真实 processor 抽取问题，而不是再倒逼规则层把样本 reject 掉。
- 6-K processor 还要保留“标题 table + 数据 table 拆开”的 press release 版式：若前一个 table 只有 `WEIBO CORPORATION / UNAUDITED CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS` 这类报表标题，后一个紧邻 table 才是真正的期间列和数值行，处理器也必须把两者拼成同一 statement 候选，不能只命中标题 table 后又把 `WB` 一类真实季度结果重新打回 `core_available=[]`。
- 同一条 fallback 还必须覆盖 Workiva / slide 风格的图片页容器：隐藏 OCR 文本不一定挂在 `<p style="font-size:1px;color:white">`，也可能挂在与页图同层的 `<div><font size="1" style="font-size:1pt;color:white">...` 中；`BsSixKFormProcessor` 必须按图片页容器把这些文本重新聚合成页级 OCR 输入，不能因为 DOM 形态不同就再次漏掉 `BSAC / GGAL / VEON` 一类 statement 页。
- 同一条 hidden OCR 回退还必须识别 CSS `font:` 简写和极小字号变体，而不只认字面上的 `font-size:1px`：像 `font: 0.01px/115% Tahoma, Helvetica, Sans-Serif; color: White`、`font-size:0.5pt;color:white` 这类图片页隐藏文本，只要同源仍然暴露了 statement title、期间列和数值组，就必须被收进同一条页级 OCR 聚合链，不能因为样式写法不同再次掉回 `core_available=[]`。
- 同一条 6-K fallback 还必须覆盖 `page-break-before/after: always` 版式：有些季度结果 HTML 既没有标准 `<table>`，也没有 `Page1/Page2` fixed-layout 容器，而是把整页 statement 文本拆成若干段落，再用 `page-break` 样式分页。`BsSixKFormProcessor` 必须按这些分页边界重建页级文本，并继续交给 `six_k_form_common` 的 OCR 解析器，不能因为原文是“分页文本块”而稳定漏掉 `TGS / NMR / BMA / SAN / UL` 一类 results release。
- 同一条 6-K fallback 还必须覆盖 `Profit & Loss` OCR 摘要页：若正文不是完整三大表，而是银行/券商 results summary 里常见的 `Profit & Loss` 摘要块，行项目后直接跟随当前期间金额和 delta 指标，`BsSixKFormProcessor` 也必须能回退提取单期间 `income`，不能让 `BBVA` 这类真实 results summary 因为没有标准多期间表头而重新掉回 `core_available=[]`。
- 同一条 6-K OCR 回退链还必须覆盖两类历史高频 source 形态：`Page1/Page2` 这类 fixed-layout HTML 容器，以及 PDF/演示稿转 HTML 后落成“每页一张伪表”的页面文本表。当前 `BsSixKFormProcessor` 会把隐藏 OCR 文本、fixed-layout 页文本和表格 caption/header/context 中的长页文本统一交给 `six_k_form_common` 的 OCR 解析器；只要原文同源确实包含 statement title、期间列和数值组，就不能因为标准二维 `<table>` 缺失而稳定掉进 `core_available=[]`。
- 6-K / 10-K / 10-Q / 20-F 共用的 HTML 财务表结构化层还必须接受“真实期间行不在前 3 行”的 header 版式，并覆盖几类高频期间写法：`31-Dec-24 / 31-Mar-25` 这类连字符日期、`Fourth Quarter 2024` 这类文本季度标题、`Quarter endedSep 2025` 这类粘连月份 token。只要原文同源表格已经包含期间列和数值列，共享层就必须继续向后探测表头并解析出 period metadata，不能因为期间行更深或日期 token 变体不同，就把 `GRVY / ASR / TME / SIFY` 一类样本重新打回 `core_available=[]`。
- 6-K 的可喂性评分集合必须先通过 source 仓储与 active filing 对齐，只评分当前仍存在于 active source 的 processed 文档；不能把已经被替换或删除的陈旧 `processed/fil_*` 继续算进 `six_k_release_core` 的 D3 门禁。`_classify_6k_text()` 是待优化规则本身，只能作为诊断输出和下载筛选逻辑，不能在评分入口被当作最终裁判形成循环论证。
- `_classify_6k_text()` 的年报排除信号必须继续保持“上下文优先”而不是词面一票否决：像 `annual report`、`annual financial statements`、`annual general meeting` 这类字样若只是中报附注或治理章节里的引用，不能压过同源可见的 `interim report`、`unaudited condensed consolidated statements`、`June quarter financial results` 等强季度财报信号。
- SEC 规则诊断与 reject 样本读取也必须继续走仓储协议：active source doc 通过 source 仓储读取，`.rejections/` 样本通过 filing maintenance 仓储读取；上层禁止自行拼接 `portfolio/{ticker}/filings` 路径。
- 源文档读取必须继续经过 `company/source/processed/blob` 窄仓储协作完成；上层只能通过源文档仓储拿 `handle/meta/source`，不能绕过仓储自己拼文件路径。
- `ground_truth_baseline.py`、`score_sec_ci.py` 这类离线基线/评分工具同样属于这条边界：processed 文档发现必须通过 `ProcessedDocumentRepositoryProtocol`，snapshot 文件枚举与字节读取必须通过 `DocumentBlobRepositoryProtocol`；不能再直接扫描 `workspace/portfolio/{ticker}/processed/*` 或按路径打开 `tool_snapshot_*.json`。
- `score_sec_ci.py` 还要继续守住两层评分语义：active source filing 缺少 processed，或 processed 文档缺少/破坏 `tool_snapshot_meta.json` 时，必须记为 completeness hard gate 失败；其它 `tool_snapshot_*.json` 缺失则只能把对应评分维度记 0 分，不能直接把整批评分流程抛异常中断。
- `source` 仓储还是真实存储事实的唯一入口：诸如“某 ticker 是否已有本地 filings 根目录”“某 filing 是否存在 instance XBRL”这类判断，必须通过 `SourceDocumentRepositoryProtocol` 获取；CLI、pipeline 和 tool service 都不能自己拼 `portfolio/...` 路径或重新扫描目录。
- SEC/XBRL 关联文件发现规则的共享真源在 `xbrl_file_discovery.py`；凡是仓储实现、SEC pipeline 或 processor 需要判断 instance/auxiliary XBRL 文件，都必须复用这一真源，而不是各自复制文件名规则。
- `blob` 仓储只负责文件枚举、字节读写与文件对象落盘；凡是需要返回 `Source` 或物化本地路径的流程，必须走 `SourceDocumentRepositoryProtocol`，不能在 blob 仓储协议上临时补 source 能力。
- Docling 上传链路若需要访问第三方 stub 未声明但运行时存在的字段，必须把适配逻辑收口在 pipeline 单点 helper，不要把第三方具体字段要求向 Tool Service、Processor 或上层调用方扩散。

## 8. 代码阅读顺序

推荐从这里进入：

1. [../services/fins_service.py](../services/fins_service.py)
2. [service_runtime.py](service_runtime.py)
3. [tools/fins_tools.py](tools/fins_tools.py)
4. [tools/service.py](tools/service.py)
5. [processors/registry.py](processors/registry.py)
6. [storage/repository_protocols.py](storage/repository_protocols.py)
7. [storage/fs_source_document_repository.py](storage/fs_source_document_repository.py)
