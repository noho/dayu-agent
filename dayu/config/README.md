# dayu Agent — 配置说明手册

本文档面向两类读者：
- 最终用户。
- 需要扩展配置的开发者。

本文档只讨论 `dayu/config/` 与 `workspace/config/` 的配置体系，不展开 Engine / Fins 内部实现细节。

## 1. 配置体系总览

Dayu 有两层配置：

1. 包内默认配置：`dayu/config/`
2. 运行时覆盖配置：`workspace/config/`

优先级规则：
- 优先读取 `workspace/config/*`
- 若工作区缺失对应文件，再回退到 `dayu/config/*`

这意味着：
- 你不需要修改仓库内的默认配置才能跑通系统
- 推荐把所有项目级修改都放在 `workspace/config/`

## 2. 最常改的三个位置

大多数用户只需要关注这三个位置：

| 路径 | 用途 |
|------|------|
| `workspace/config/llm_models.json` | 模型配置、API Key 占位符、endpoint |
| `workspace/config/run.json` | Agent 行为、宿主 Host 配置、tool trace、budget、工具 limits |
| `workspace/config/prompts/` | prompt 资产 |

只有在“安装或替换某个 toolset 的 registrar”时，才需要额外关注 `workspace/config/toolset_registrars.json`。它是开发者扩展入口，不是日常运行参数入口。

最小修改路径：
- 想换模型：改 `llm_models.json`
- 想新增自定义模型：改 `llm_models.json`、对应 scene manifest 的 `model` 配置；如果它要长期跑 `interactive` 多轮，再补该模型的 `runtime_hints.conversation_memory`
- 想调 Agent 行为：改 `run.json`
- 想改提示词：改 `prompts/`

当前包内也提供了几组可直接参考的官方模型入口示例，包括 `gpt-5.4`、`claude-sonnet-4-6`、`gemini-2.5-flash`。
其中 `claude-sonnet-4-6` 由于当前 Runtime 只支持 OpenAI 兼容 runner，所以走的是 Anthropic 官方 OpenAI compatibility 入口，而不是原生 `v1/messages`。

## 3. 目录结构

```text
dayu/config/
├── README.md
├── llm_models.json
├── run.json
├── toolset_registrars.json
└── prompts/
    ├── base/
    │   ├── agents.md
    │   ├── fact_rules.md
    │   ├── soul.md
    │   └── tools.md
    ├── manifests/
    │   ├── audit.json
    │   ├── confirm.json
    │   ├── decision.json
    │   ├── fix.json
    │   ├── infer.json
    │   ├── interactive.json
    │   ├── overview.json
    │   ├── prompt.json
    │   ├── regenerate.json
    │   ├── repair.json
    │   ├── wechat.json
    │   └── write.json
    ├── scenes/
    │   ├── audit.md
    │   ├── confirm.md
    │   ├── decision.md
    │   ├── fix.md
    │   ├── infer.md
    │   ├── interactive.md
    │   ├── overview.md
    │   ├── prompt.md
    │   ├── regenerate.md
    │   ├── repair.md
    │   ├── wechat.md
    │   └── write.md
    └── tasks/
        ├── audit_facts_tone_json.contract.yaml
        ├── audit_facts_tone_json.md
        ├── confirm_evidence_violations.contract.yaml
        ├── confirm_evidence_violations.md
        ├── fill_overview.contract.yaml
        ├── fill_overview.md
        ├── fix_placeholders.contract.yaml
        ├── fix_placeholders.md
        ├── infer_company_facets.contract.yaml
        ├── infer_company_facets.md
        ├── repair_chapter.contract.yaml
        ├── repair_chapter.md
        ├── regenerate_chapter.contract.yaml
        ├── regenerate_chapter.md
        ├── write_research_decision.contract.yaml
        ├── write_research_decision.md
        ├── write_chapter.contract.yaml
        └── write_chapter.md
```

### 3.1 `toolset_registrars.json`

这个文件只负责声明“某个 toolset 由哪个 registrar adapter 实现”，不负责决定某次执行是否启用这个 toolset。

最小格式如下：

```json
{
  "utils": "dayu.engine.toolset_registrars.register_utils_toolset",
  "doc": "dayu.engine.toolset_registrars.register_doc_toolset",
  "web": "dayu.engine.toolset_registrars.register_web_toolset",
  "fins": "dayu.fins.toolset_registrars.register_fins_read_toolset",
  "ingestion": "dayu.fins.toolset_registrars.register_fins_ingestion_toolset"
}
```

新增一个 toolset 时，只需要遵守这四条：

- key 使用 scene manifest / `selected_toolsets` 里会出现的同一个 toolset 名称。
- value 必须是可导入的 `module.attr` 路径，目标对象必须可调用。
- registrar 负责把通用上下文适配成叶子 `register_*_tools(...)` 所需参数。
- 如果某个 toolset 在本次执行中已经被判定为启用，但这里缺少对应映射，scene preparation 会直接失败。

### 3.2 prompts 分层约定

- `manifests/*.json`：定义某个 scene 的装配方式、对话模式、工具选择和 fragment 列表。
- `scenes/*.md`：定义该 scene 的执行契约，只回答“这一类任务怎么执行”。
- `tasks/*.md`：定义某个具体任务要输入什么、输出什么。
- `wechat` scene：服务微信通道的多轮交互，工具集合与 `interactive` 相同，但输出约束更窄，要求回复为适合微信下发的纯文本；其多轮行为由 manifest 的 `conversation.enabled=true` 显式声明。
- 对 `conversation.enabled=true` 的 scene，Service 当前会默认把 `host_policy.resumable=True` 写入 `ExecutionContract`；Host 仍会继续做硬门槛校验，只有稳定 `session_key` 存在时才允许进入 V1 恢复路径。
- V1 恢复的真源不是 transcript，而是 Host 持久化的 pending `conversation turn`：`accepted_by_host -> prepared_by_host -> sent_to_llm`。它表示“Host 内尚未完成的当前 turn”：accepted 阶段可以重新进入 scene preparation，prepared / sent 阶段可以直接恢复执行；`interactive` 与 `wechat` 依赖它在启动时先补完上一轮，但恢复请求仍必须显式带上所属 `session_id`，其中 WeChat 启动恢复还会再限定到当前 `state_dir` 派生的 runtime identity。

Prompt 装配还遵循一条 Prefix Cache 导向的顺序约束：
- 稳定、跨 ticker / 跨日期不变的 fragments（如 `base_agents`、`base_soul`、`base_fact_rules`、`base_tools`、各 `scene`）应尽量排在前面。
- 动态 Prompt Contributions（如 `fins_default_subject`、`base_user`）不再作为 fragment 资产存在，而由 Service 在运行时提供；Service 会先按 scene manifest 的 `context_slots` 收口，再由 Host 按该顺序统一追加到 system prompt 尾部。
- `directories` 现在通过独立的 `base/directories.md` 注入，不再嵌入 `base/tools.md`；只有 scene 实际注册了 doc 工具、确实需要暴露可访问路径时，才应挂这个 fragment。
- `base_user` 不只服务写作；凡是会核实或修复相对时长、任职年限、最近几年等相对时间判断的 scene，也应声明这个 slot。
- `tool_call_budget_threshold` 不再进入 prompt context，预算感知统一依赖工具结果里的 `tool_calls_remaining`。

分层原则：
- `base/fact_rules.md`：只放通用事实/前瞻/证据格式边界。
- `scenes/write.md`：只放写作场景的执行契约与证据锚点动作规则。
- `scenes/regenerate.md`：只放整章重建场景的执行契约与重建边界。
- `tasks/write_chapter.md`、`tasks/regenerate_chapter.md`：只放各自任务特有的动作与输入输出要求，不重复 system prompt 中的通用边界；优先保留“当前这轮必须知道的输入与局部约束”，不要把 scene 已声明的通用规则再讲一遍。
  - 其中 `ITEM_RULE` 的成功标准不是“多写一点行业标签”，而是把行业共性变量和特殊结构变量写成本章真正改变判断的内容。
- `tasks/fix_placeholders.md`：只放占位符补强特有规则，不重复 `fix` scene 已声明的结构边界。
- `tasks/infer_company_facets.md`：只放“公司业务类型与关键约束”判断任务的输出规则与必要背景说明；要向模型讲清这一步不是做行业分类展示，而是为后续模板写作确定“判断分岔点”，让不同行业公司写出明显不同的判断路径、同一行业内不同公司写出各自的特殊结构变量。主业务类型候选与关键约束候选的固定词表不写死在代码里，而是来自写作模板前导区的 `COMPANY_FACET_CATALOG`。
- `tasks/audit_facts_tone_json.md`：只放审计 JSON schema、判定步骤与规则定义；`audit` 是否用工具、是否只做疑似裁决，由 `scenes/audit.md` 决定；不要把 scene 已声明的 JSON-only 或无工具边界重复堆进 task。当前 `E3` 要明确只用于“证据条目指向的来源本身不可定位或不属于当前 ticker/公司”的场景，例如 document id、accession、filing、URL 或标题路径不存在；若来源仍可追溯、只是锚点不够细，应降到 `S7`，不要误报 `E3`。
- `tasks/repair_chapter.md`：只放 patch 生成规则；是否允许研究、是否允许补新事实、无法完全修复时如何收手，由 `scenes/repair.md` 与 task schema 一致约束。`repair_chapter` 需要先把代码侧 repair contract 翻译成低认知负担的下一步动作：`remediation_actions[*].resolution_mode` 会显式解释给模型，至少包含 `delete_claim`、`rewrite_with_existing_evidence`、`anchor_fix_only` 三种处置模式；其中 `delete_claim` 表示该 claim 已被 confirm 证实缺证，repair 不得再做“弱化但保留”，而必须删除其所在的完整语义单元。`target_kind` 也要在 task 中显式解释为 `substring|line|bullet|paragraph` 四种命中粒度，其中 `delete_claim` 不得使用 `substring`。`target_section_heading` 只能填写当前正文中真实存在的可见标题，不能把输出项名称、bullet 标签或问句当成标题。`repair_chapter` 只修正文，不修改 `### 证据与出处`；evidence line 的细化、补锚和同一 filing 内 anchor 修复统一交给代码基于 `anchor_fix` 处理。若“是否允许补入新事实”为 true 且修复合同存在 evidence 缺口，可做最小必要检索，但若最终仍需要改证据区才能闭环，只能在 `notes` 说明并收回到删 / 弱化正文中的 unsupported 细节。task 输出的 `patches` 当前要求至少 1 条，不允许返回空数组。
- `tasks/confirm_evidence_violations.md`：`supported_but_anchor_too_coarse` / `supported_elsewhere_in_same_filing` 优先返回结构化 `anchor_fix`；若不能稳定给出可执行定位，应省略该字段，不要输出空对象 `{}`。写作流水线会优先消化 `anchor_fix`，仅在旧产物或缺少结构化定位时才退回 `S7` 兜底。
- `base/tools.md`：只放跨工具的最小操作指引。对于 fins 工具，这里需要明确“先 `list_documents` 再读文档”“ticker alias 交给工具自动收敛”“若 `list_documents` 返回 `not_found` 就立即切 web，而不是继续穷举 ticker 变体”。
- `base/directories.md`：只放当前可访问路径；它是动态 runtime context，不承担工具工作流规则，只在 scene 实际需要 doc 工具路径时挂载。

当前写作流水线中：
- `write` scene：只服务初稿写作。
- `infer` scene：只服务“公司业务类型与关键约束”判断。
  - 当前只注册 `fins` 工具，不注册 `web` / `doc`。
  - scene 要向模型讲清：这一步的下游是固定模板逐章写作，所以输出的 `business_model_tags` / `constraint_tags` 本质上是在为模板写作确定“判断分岔点”，而不是做行业分类展示。
  - 这是高确定性的受控路由任务，不是开放式头脑风暴；默认先求稳，再求全，优先选最会改变后续章节判断路径的 1 个主业务类型，只有漏掉它会明显让后续章节走错路时才补第 2 / 3 个。
  - 默认输出一个受控 JSON，对应 `business_model_tags`、`constraint_tags`、`judgement_notes`。
- `decision` scene：只服务第10章“是否值得继续深研与待验证问题”的研究决策综合。
  - 对应 task `write_research_decision` 默认要求：先决定当前研究动作，再压缩出最小判断链、最大反证和最小验证计划；若不足可写“当前更倾向于...”；无论结论为何，都要写清什么变化会升级、降级或终止当前动作，不写“`不适用`”。
  - 第10章默认继续收口为：最小判断链优先只写 `1` 条主判断链（必要时才补第 `2` 条）、主卡点默认只写 `1` 个、最小验证计划默认先写 `1` 个最关键问题、动作改变条件默认收成 `1` 个升级阈值和 `1` 个降级或终止阈值，并尽量避免把单个字段再拆成子 bullet。
- `regenerate` scene：只服务整章重建。
- `fix` scene：只服务占位符补强。
- `repair` scene：只服务局部 patch 修复。
- `audit` scene：只服务疑似违规检查。
- `confirm` scene：只服务证据复核。
- `overview` scene：只服务第0章“投资要点概览”的封面页回填，默认无工具。
- `fill_overview` task：只服务第0章“投资要点概览”的封面页回填。
  - 当前口径不是“摘要汇总”，而是把传入的前文章节结构化输入收成买方初筛封面。
  - 默认不再拆成“结论要点 / 详细情况 / 证据与出处”三段结构，也不承载证据溯源。
  - 当前默认层级是：先用“一眼看懂”给出“这是什么生意”“公司简介”和“当前研究动作”，再回答“为什么现在是这个动作”，最后回答“下一步怎么验证”。
  - 第0章增加“公司简介”，但它不是旧式公司介绍，而是帮助第一次接触这家公司的读者快速建立公司画像的极简背景入口。
  - 第0章不只是动作卡片；还要补一条“当前经营与财务处在什么状态”，但只保留最能解释当前动作的经营、盈利、现金流或资产负债状态。
  - 默认预设读者是第一次接触或第一次系统性了解这家公司，因此要主动解释最必要的背景，不假设读者已经知道公司历史、行业术语或既有争议。
  - 其中“先抓哪个变量”与“最大卡点”若本质是同一件事，变量写成需要持续观察的观测点，卡点写成一旦验证失败会如何推翻当前动作，避免重复复述。
  - 当前最大卡点必须写成“现在最阻止升级或维持当前动作的现实阻碍”，而不是“如果未来变好，当前动作可能过于保守”的反事实句。
  - “下一步先验证什么”默认只写 `1` 个最关键问题；只有两个问题彼此独立且任一单独成立都会改变当前动作时，才允许并列。
  - 运行时不再复用 `write` scene，而是使用独立的 `overview` scene；该 scene 禁用工具，第0章也默认跳过 `audit / confirm / repair`。

## 4. `llm_models.json`

`llm_models.json` 定义可选模型配置。顶层每个键对应一套模型运行参数。

### 4.1 常用字段

| 字段 | 含义 |
|------|------|
| `runner_type` | 运行器类型，当前只允许 `openai_compatible` |
| `name` | 配置名称 |
| `endpoint_url` | API 地址 |
| `model` | 模型 ID |
| `headers` | HTTP 请求头，支持 `{{ENV_VAR}}` 占位符 |
| `timeout` | 单次模型请求总超时 |
| `stream_idle_timeout` | 流式响应空闲读超时 |
| `stream_idle_heartbeat_sec` | 流式响应空闲心跳日志间隔 |
| `supports_stream` | 是否支持流式输出 |
| `supports_tool_calling` | 是否支持工具调用 |
| `supports_usage` | 是否支持 usage 采集 |
| `supports_stream_usage` | 是否支持流式 usage 采集 |
| `max_context_tokens` | 最大上下文 token |
| `extra_payloads` | Provider 扩展请求参数；禁止放入 `model`、`messages`、`temperature`、`stream`、`tools` 等显式字段 |

### 4.2 CLI runner 状态

CLI runner 已彻底禁用，不再允许通过 `llm_models.json` 配置或使用 `runner_type=cli`。

如果工作区残留旧的 CLI 模型配置，系统会在模型加载阶段显式报错，而不是继续进入 Host 主链路。

### 4.3 当前内置配置键

- `deepseek-chat`
- `deepseek-thinking`
- `gpt-5.4`
- `claude-sonnet-4-6`
- `gemini-2.5-flash`
- `mimo-v2-flash`
- `mimo-v2-flash-thinking`
- `mimo-v2-pro`
- `mimo-v2-pro-thinking`
- `qwen3`
- `qwen3-thinking`
- `qwen3:30b-thinking`

### 4.4 最小修改示例

如果你只想把 `deepseek-thinking` 的 API Key 改成环境变量读取，通常只需要保持：

```json
"headers": {
  "Authorization": "Bearer {{DEEPSEEK_API_KEY}}",
  "Content-Type": "application/json"
}
```

然后在 shell 中设置：

```bash
export MIMO_API_KEY="sk-xxxxxxxx"
```

### 4.5 想新增自定义模型

新增一个模型，至少要同时考虑三层配置：

1. `llm_models.json`
2. 对应 scene manifest 的 `model`
3. 如需模型特例 memory policy，再补 `llm_models.json -> runtime_hints.conversation_memory`

推荐顺序如下。

#### 第一步：在 `llm_models.json` 注册模型

先在 `workspace/config/llm_models.json` 顶层新增一个模型键，例如：

```json
"my-provider-chat": {
  "runner_type": "openai_compatible",
  "name": "my-provider-chat",
  "endpoint_url": "https://api.example.com/v1/chat/completions",
  "model": "example-chat",
  "headers": {
    "Authorization": "Bearer {{MY_PROVIDER_API_KEY}}",
    "Content-Type": "application/json"
  },
  "timeout": 3600,
  "stream_idle_timeout": 120.0,
  "stream_idle_heartbeat_sec": 10.0,
  "supports_stream": true,
  "supports_tool_calling": true,
  "supports_usage": true,
  "supports_stream_usage": true,
  "max_context_tokens": 131072,
  "runtime_hints": {
    "temperature_profiles": {
      "write": {"temperature": 0.7},
      "audit": {"temperature": 0.3}
    }
  }
}
```

这里需要你自己提供的关键字段是：

- `endpoint_url`
- `model`
- `headers`
- `supports_tool_calling`
- `max_context_tokens`

其中：
- `max_context_tokens` 是 Runtime 计算 memory 预算的重要输入。
- `runtime_hints.temperature_profiles` 是 scene temperature 的唯一真源。
- 如果模型不支持工具调用，就不能直接替换当前依赖 tool calling 的 scene。

#### 第二步：把模型接入对应 scene manifest

只把模型加进 `llm_models.json` 还不够。  
某个 scene 想合法使用它，还必须在对应 manifest 的 `model.allowed_names` 里声明该模型名。

例如把它接到审计链路：

```json
{
  "scene": "audit",
  "model": {
    "default_name": "mimo-v2-flash-thinking",
    "allowed_names": ["mimo-v2-flash-thinking", "my-provider-chat"],
    "temperature_profile": "audit"
  }
}
```

若这个 scene 还需要单独预算，再额外声明：

```json
{
  "runtime": {
    "agent": {
      "max_iterations": 32
    }
  }
}
```

若希望它成为该 scene 的默认模型，再把：

- `model.default_name`

改成这个新模型名。

固定约束：

- `model.default_name` 必须存在于 `model.allowed_names` 中。
- scene manifest 的 Agent 预算位于 `runtime.agent.max_iterations`；优先级为 `CLI/request override > scene.runtime.agent.max_iterations > run.json.agent_running_config.max_iterations`。
- scene manifest 的 Runner 工具超时位于 `runtime.runner.tool_timeout_seconds`；优先级为 `CLI/request override > scene.runtime.runner.tool_timeout_seconds > run.json.runner_running_config.tool_timeout_seconds`。
- scene manifest 的 `model` 只解决“这个 scene 默认用谁、允许谁、使用哪种 decoding profile”。预算与执行超时放在 `runtime`，不再放在 `model`。
- 它不重复声明 `max_context_tokens` 这类模型客观能力；这些仍只来自 `llm_models.json`。

#### 第三步：如需模型特例 memory policy，再补 `runtime_hints.conversation_memory`

如果这个模型主要用于 `prompt` 或 `write` 单轮，前两步通常就够了。  
如果它要长期用于 `interactive` 多轮会话，且默认 memory 公式不合适，再在模型配置里补特例：

```json
"runtime_hints": {
  "temperature_profiles": {
    "interactive": {"temperature": 0.6}
  },
  "conversation_memory": {
    "working_memory_token_budget_cap": 20000,
    "episodic_memory_token_budget_floor": 4000,
    "episodic_memory_token_budget_cap": 4000
  }
}
```

Runtime 的选择规则是：

- 先用 `run.json.conversation_memory.default` 作为全局公式
- 再合并 `llm_models.json[{model_name}].runtime_hints.conversation_memory`
- 然后结合该模型的 `max_context_tokens` 计算最终 working / episodic budget

#### 最小检查清单

新增模型后，至少检查这几件事：

- 环境变量占位符已配置，例如 `{{MY_PROVIDER_API_KEY}}`
- `supports_tool_calling` 与实际模型能力一致
- `max_context_tokens` 填写正确
- 目标 scene 的 `model.allowed_names` 已加入该模型
- 目标 scene 的 `temperature_profile` 已在该模型的 `runtime_hints.temperature_profiles` 中配置
- 如果要做多轮 `interactive`，已确认默认 memory 公式足够，或已补 `runtime_hints.conversation_memory`

## 5. `run.json`

`run.json` 定义运行期行为配置。

### 5.1 `runner_running_config`

Runner 调试与调用控制：

- `debug_sse`
- `debug_tool_delta`
- `debug_sse_sample_rate`
- `debug_sse_throttle_sec`
- `tool_timeout_seconds`
- `stream_idle_timeout`
- `stream_idle_heartbeat_sec`

### 5.2 `agent_running_config`

Agent 行为控制：

- `max_iterations`
- `fallback_mode`
- `fallback_prompt`
- `duplicate_tool_hint_prompt`
- `continuation_prompt`
- `compaction_summary_header`
- `compaction_summary_instruction`
- `max_consecutive_failed_tool_batches`
- `max_duplicate_tool_calls`
- `budget_soft_limit_ratio`
- `budget_hard_limit_ratio`
- `max_continuations`
- `max_compactions`

最常改的字段通常是：
- `max_iterations`
- `max_consecutive_failed_tool_batches`
- `tool_timeout_seconds`
- `fallback_mode`

其中 `agent_running_config` 的默认值来自 `run.json`，但 scene manifest 的 `runtime.agent.max_iterations` 与
`runtime.agent.max_consecutive_failed_tool_batches` 可以对其做 scene 级覆盖；Runner 的
`tool_timeout_seconds` 则可由 `runtime.runner.tool_timeout_seconds` 做 scene 级覆盖；若请求显式传入 CLI / request override，
则显式参数优先级最高。

### 5.3 `doc_tool_limits`

文档工具参数上限：

- `list_files_max`
- `get_sections_max`
- `search_files_max_results`
- `read_file_max_chars`
- `read_file_section_max_chars`

### 5.4 `fins_tool_limits`

财报工具参数上限：

- `processor_cache_max_entries`
- `list_documents_max_items`
- `get_document_sections_max_items`
- `search_document_max_items`
- `list_tables_max_items`
- `read_section_max_chars`
- `get_page_content_max_chars`
- `get_table_max_items`
- `get_financial_statement_max_items`
- `query_xbrl_facts_max_items`

### 5.5 `web_tools_config`

联网工具配置：

- `provider`
- `request_timeout_seconds`
- `max_search_results`
- `fetch_truncate_chars`
- `allow_private_network_url`
- `playwright_channel`
- `playwright_storage_state_dir`

说明：
- `run.json` 不决定 web 工具是否进入候选集合；候选资格由 scene manifest 的 `tool_selection` 决定，最终还要命中已安装的 registrar。
- `web_tools_config` 只负责传递联网执行参数，例如 provider、超时、结果上限与 URL 安全策略。
- 默认只允许访问公开互联网 URL。
- 当你处在代理、内网文档站点、开发机本地服务等场景，确实需要访问 `localhost`、私网 IP、私网 DNS 解析目标时，可把 `allow_private_network_url` 设为 `true`。
- 这个开关不会放开 `file://` 等非 HTTP/HTTPS scheme；它只放开内网/本地网络目标。
- `playwright_channel` 只影响 `fetch_web_page` 的浏览器回退画像；默认值 `chrome` 会让 Playwright 使用本机 Chrome channel，空字符串表示退回 Playwright 自带 Chromium 默认画像。
- `playwright_storage_state_dir` 是可选的 Playwright storage state 目录。`fetch_web_page` 会按 host 自动查找 `<host>.json`。它适用于“fresh 浏览器过不去，但人工浏览器能过”的站点：先用诊断脚本在 headed 模式人工通过验证并导出 storage state，再让 `fetch_web_page` 复用对应 host 的状态文件。

### 5.6 `host_config`

宿主运行配置：

- `store`
- `lane`
- `pending_turn_resume`
- `pending_turn_retention`

说明:
- `host_config.store.path` 是宿主层 SQLite 数据库文件路径。相对路径按 `workspace` 根目录解析；绝对路径则直接使用。当前默认值是 `.dayu/host/dayu_host.db`。
- 宿主层的 `session`、`run`、并发 `permit` 与 pending turn 恢复状态都共享这一个数据库文件。
- `host_config.lane` 是宿主层并发 lane 配置，内容为 `lane_name: max_concurrent` 键值对；未显式填写的 lane 回退到 Host 默认（`llm_api`）与 Service 业务默认（`write_chapter`、`sec_download`）的合并值。`llm_api` 是 Host 自治 lane，用户可在此下调/上调运维抓手；`write_chapter` 控制同时在写的章节数（写作 pipeline 的 in-process ThreadPoolExecutor worker 上限与本值同源）；`sec_download` 控制 SEC 下载的跨进程串行度。
- `host_config.pending_turn_resume.max_attempts` 控制单条 pending turn 的最大恢复次数。当前默认值是 `3`；达到上限后 Host 会删除该 pending turn，避免同一 `session + scene` 槽位被坏记录永久卡死。
- `host_config.pending_turn_retention.retention_hours` 控制 pending turn 在 `ACCEPTED_BY_HOST` / `PREPARED_BY_HOST` 状态下的最长保留时间（小时）。当前默认值是 `168`（7 天）；`Host.cleanup_stale_pending_turns` 会在启动 / 维护阶段把超过保留期的记录兜底删除，避免 UI 层长期未询问用户"是否重发"时库无限累积。该值是 UI 询问窗口的上限，典型 UI 会在 <72 小时内完成询问。

### 5.7 `tool_trace_config`

工具调用追踪配置：

- `enabled`
- `output_dir`

用户最常见的配置动作：
- 打开 trace：把 `enabled` 改成 `true`
- 改 trace 输出目录：调整 `output_dir`

### 5.8 `conversation_memory`

interactive 多轮会话的分层记忆配置：

- `default`

当前语义：
- `default` 是所有模型共享的 memory policy 默认值。
- 模型客观能力，例如 `max_context_tokens`，仍然只来自 `llm_models.json`；`conversation_memory` 不重复声明模型窗口大小。
- 若某个模型需要特例 policy，配置在 `llm_models.runtime_hints.conversation_memory`，不再写在 `run.json`。

可以把它理解成：

- `llm_models.json` 决定模型“客观能做多少”
- `conversation_memory` 决定多轮会话里“愿意拿多少上下文来放历史”

`default` / `runtime_hints.conversation_memory` 内可配置的字段：

- `working_memory_max_turns`
- `working_memory_token_budget_ratio`
- `working_memory_token_budget_floor`
- `working_memory_token_budget_cap`
- `episodic_memory_token_budget_ratio`
- `episodic_memory_token_budget_floor`
- `episodic_memory_token_budget_cap`
- `compaction_trigger_turn_count`
- `compaction_trigger_token_ratio`
- `compaction_tail_preserve_turns`
- `compaction_context_episode_window`
- `compaction_scene_name`

#### 5.7.1 working memory 和 episodic memory 是什么

当前多轮会话上下文分成两层：

- `working memory`
  - 最近的高保真原始历史
  - 主要服务“继续追问上一轮”这种局部连续性
- `episodic memory`
  - 更早历史的结构化阶段摘要
  - 主要服务“别把更早的目标、约束、结论完全忘掉”

当前轮送模时，Runtime 会把这两层和当前问题一起编译进 prompt：

- 当前 scene 的 `system_prompt`
- `[Conversation Memory]` 摘要块
- 最近原始历史
- 当前用户输入

#### 5.7.2 关键字段是什么意思

最重要的 4 个字段是：

- `working_memory_token_budget_ratio`
  - 先按当前模型 `max_context_tokens * ratio` 计算 working memory 预算
  - 作用是让大上下文模型自然拿到更大的历史预算
- `working_memory_token_budget_floor`
  - working memory 的最小保底
  - 防止小上下文模型或某些配置下，预算小到几乎什么都放不下
- `working_memory_token_budget_cap`
  - working memory 的最大封顶
  - 防止大上下文模型把太多最近历史塞进 prompt，淹没当前问题
- `episodic_memory_token_budget_ratio`
  - 先按当前模型 `max_context_tokens * ratio` 计算 episode summaries 的预算
- `episodic_memory_token_budget_floor`
  - episodic memory 的最小保底
- `episodic_memory_token_budget_cap`
  - episodic memory 的最大封顶

可以把 working memory 的预算理解成：

```text
working_budget = clamp(
  max_context_tokens * working_memory_token_budget_ratio,
  working_memory_token_budget_floor,
  working_memory_token_budget_cap
)
```

例如：

- 模型 `max_context_tokens = 131072`
- `working_memory_token_budget_ratio = 0.08`
- 原始计算值约为 `10485`
- 如果 `floor = 1500`、`cap = 20000`
- 那最终 working budget 就是 `10485`

如果模型是 1M 上下文，而 ratio 算出来特别大，最终仍会被 `cap` 封顶。

#### 5.7.3 其余字段的作用

- `working_memory_max_turns`
  - 最近原始历史按 turn 回放时的最大轮数上限
  - 它和 token budget 一起起作用；先满足预算，再受 turn 数限制
- `compaction_trigger_turn_count`
  - 未压缩 raw turns 超过多少轮后，允许触发 episode compaction
- `compaction_trigger_token_ratio`
  - 未压缩 raw history 的估算体量，超过 working budget 的多少倍后触发 compaction
- `compaction_tail_preserve_turns`
  - 压缩时保留最近多少轮 raw turns 不压
- `compaction_context_episode_window`
  - 生成新 episode summary 时，带入多少个最近 episodes 作为邻近上下文
- `compaction_scene_name`
  - 执行结构化压缩时使用的专用 scene 名，当前默认 `conversation_compaction`

当前 working memory 策略：
- 正常情况下按 turn 为单位，从最近历史向前回放。
- 若最新一轮 turn 过长，不会整轮丢弃；Runtime 会保留完整 `user_text`，并对 `assistant` 内容做降级裁剪。
- 裁剪顺序固定为：先丢历史工具摘要，再截断 `assistant` 文本前缀，并附加 `...<truncated>` 标记。
- episodic budget 也按 `ratio/floor/cap` 公式计算，不再使用固定绝对值。
- `compaction_*` 控制什么时候把更早的 raw turns 压成结构化 episode summary。
- `compaction_scene_name` 指向 Runtime 使用的专用无工具压缩 scene，默认是 `conversation_compaction`。

#### 5.7.4 什么时候应该改这些值

常见调参信号：

- 如果你发现 follow-up 经常“忘记上一轮刚说过什么”
  - 先检查 `working_memory_token_budget_cap` 是否过小
  - 再检查是否给该模型配置了合适的 `runtime_hints.conversation_memory`
- 如果你发现 prompt 被旧内容塞得太满、当前问题不聚焦
  - 优先降低 `working_memory_token_budget_cap`
  - 或适当降低 `working_memory_token_budget_ratio`
- 如果你只是新增了一个大上下文模型
  - 不需要在 `conversation_memory` 里重复填写它的上下文窗口
  - 只需要按需要给它补一个 `runtime_hints.conversation_memory` policy
  - 最佳实践不是按 `max_context_tokens` 线性放大 memory；长上下文优先留给财报材料、检索结果和当前章节上下文，memory 只建议小步上调 `working_memory_token_budget_cap`

#### 5.7.5 当前配置示意

当前配置形状类似：

```json
"conversation_memory": {
  "default": {
    "working_memory_max_turns": 6,
    "working_memory_token_budget_ratio": 0.08,
    "working_memory_token_budget_floor": 1500,
    "working_memory_token_budget_cap": 12000,
    "episodic_memory_token_budget_ratio": 0.02,
    "episodic_memory_token_budget_floor": 2000,
    "episodic_memory_token_budget_cap": 12000
  }
}
```

实际选择规则始终是：

- 先读取 `default`
- 再合并当前模型的 `runtime_hints.conversation_memory`
- 然后再结合该模型在 `llm_models.json` 里的 `max_context_tokens` 算预算

## 6. `prompts/`

`prompts/` 目录分成两类资产：

### 6.1 system 级 scene manifest 资产

- `base/agents.md`：全局行为规范与输出约束；包括 JSON/Markdown 输出硬约束、禁止过程性自述，以及工具调用轮和最终回答都必须“直接行动、直接从目标格式起始字符开始输出”的规则
- `base/soul.md`：分析人格
- `base/fact_rules.md`：事实与分析区分、前瞻性表述、会计数字呈现、证据与出处统一格式（含 section path、`Financial Statement:{statement_type}`、`XBRL Facts` 三类合法定位格式）
- `base/tools.md`：工具使用原则，不承载工具 schema
- `scenes/*.md`：interactive / write / repair / audit / confirm 等场景覆盖片段
- `scenes/conversation_compaction.md`：多轮会话阶段摘要压缩场景，只生成严格 JSON，不调用工具
- `manifests/*.json`：场景装配清单，声明当前 scene 应加载哪些静态 fragment、默认模型名、允许模型集合、decoding profile、是否启用多轮对话，以及允许追加哪些 `context_slots`
- `tasks/*.md`：任务级 prompt。它们必须保持章节无关，只承载通用写作/修复/审计规则；当前写作/审计链路的重要边界来自章节 `CHAPTER_CONTRACT`，而不是写进共享 task prompt 的章节专用约束。
  - task prompt 中对输入块的可见标签应统一使用中文 reader-facing 名称，例如“章节标题”“全文目标”“读者画像”“本章回答”“章节结构”“章节合同”“条件写作规则”“当前输入口径”；正文里引用这些输入块时，优先使用与可见标签完全同名的中文引号，不要再用反引号把它们写成内部字段名。

scene manifest 当前有两类职责：
- 装配 prompt fragment。
- 作为该 scene 的工具注册唯一真源。
- 作为该 scene 的默认模型真源。
- 作为该 scene 的对话模式真源。

补充约定：
- 共享 `base/*` 只放通用静态片段，不放证券域专用提示。
- 类似 `fins_default_subject`、`base_user` 的动态上下文必须通过 `context_slots` 显式声明；Service 负责生成文本并在 contract preparation 阶段按声明收口，Host 只负责按声明顺序机械追加，若收到未声明 slot 仅告警并忽略。
- 多轮会话开关必须通过 `conversation.enabled` 显式声明；Host 只机械读取该字段决定是否加载 transcript / conversation memory，不再根据 scene 名称硬编码判断。

`manifests/*.json` 顶层可选声明 `conversation`：

```json
{
  "conversation": {
    "enabled": true
  }
}
```

固定约束：
- `conversation` 缺省时默认单轮。
- `conversation.enabled` 必须是布尔值。

`manifests/*.json` 顶层必须显式声明 `model`：

```json
{
  "scene": "write",
  "model": {
    "default_name": "mimo-v2-pro-plan",
    "allowed_names": ["mimo-v2-flash", "mimo-v2-pro", "mimo-v2-pro-plan", "mimo-v2-pro-plan-sg", "deepseek-chat", "qwen3", "gpt-5.4", "claude-sonnet-4-6", "gemini-2.5-flash"],
    "temperature_profile": "write"
  },
  "runtime": {
    "agent": {
      "max_iterations": 32
    },
    "runner": {
      "tool_timeout_seconds": 90.0
    }
  }
}
```

固定约束：
- 每个 scene manifest 都必须显式配置 `model.default_name`；缺失即视为 manifest 非法，`SceneExecutionResolver` 会直接拒绝当前 scene。
- `model.allowed_names` 必须是非空数组，且 `model.default_name` 必须出现在 `allowed_names` 中。
- `model.temperature_profile` 必须是非空字符串；真正 temperature 真源在 `llm_models.runtime_hints.temperature_profiles[temperature_profile].temperature`。
- `runtime.agent.max_iterations` 为可选正整数；若配置，表示该 scene 的默认 Agent 迭代预算，并覆盖 `run.json` 的全局默认值；只有 CLI / request 级显式 `max_iterations` 可以再覆盖它。
- `runtime.agent.max_consecutive_failed_tool_batches` 为可选正整数；若配置，表示该 scene 的默认连续失败工具批次上限。
- `runtime.runner.tool_timeout_seconds` 为可选正数；若配置，表示该 scene 的默认 Runner 工具调用超时秒数，并覆盖 `run.json.runner_running_config.tool_timeout_seconds`。
- `runtime` 整体可省略；未显式声明时，该 scene 完全回退到 `run.json` 的全局默认值。
- 对 `tool_selection.mode = none` 的无工具 scene，通常不需要声明 `runtime.runner.tool_timeout_seconds`；若也不需要特殊预算，整个 `runtime` 都可以省略。
- CLI 的 `--model-name` 只有在显式传入时才覆盖普通 scene 的 `model.default_name`。
- `write` 命令中的 `--audit-model-name` 只有在显式传入时才覆盖 `decision` / `audit` / `confirm` 的 `model.default_name`。
- CLI 的 `--temperature` 只有在显式传入时才覆盖全部 scene 的 temperature；最终优先级为 `CLI --temperature > llm_models.runtime_hints.temperature_profiles[scene.temperature_profile].temperature`。
- 若 profile 缺失 temperature，则运行直接报错，不再隐式使用顶层 `temperature`。
- `regenerate` / `fix` / `repair` 的 `model.temperature_profile` 当前统一复用 `write`；scene 语义仍由各自的 `scenes/*.md` 契约区分，profile 只负责温度标定。
- 当前包内默认中，`write` / `regenerate` / `fix` / `repair` / `confirm` / `decision` / `infer` 显式声明更高的 scene 预算；`audit` / `overview` 则复用 `run.json` 的全局默认值。

当前默认模型策略：
- 写作链路（`write` / `regenerate` / `fix` / `repair`）默认使用 `mimo-v2-pro-plan`，且 `allowed_names` 预置非 thinking 的写作侧模型；切换默认模型时通常只需要改 `model.default_name`。
- 推理问答链路（`prompt` / `interactive` / `infer` / `decision` / `audit` / `confirm` / `conversation_compaction`）默认使用 `mimo-v2-pro-thinking-plan`，且 `allowed_names` 预置 thinking / 推理侧模型；切换默认模型时通常只需要改 `model.default_name`。
- 需要注意：DeepSeek 官方文档说明 `deepseek-reasoner` 不支持 `temperature` / `top_p`，传入不会报错，但也不会生效；因此审计链路的真实行为主要由模型本身与 prompt 契约决定，而不是 temperature。
- 项目内当前建议温度口径统一为：`mimo-v2-pro = write 0.8 / overview 0.3`、`mimo-v2-pro-thinking = prompt 0.8 / interactive 0.8 / audit 0.4`、`deepseek-thinking = prompt 1.3 / interactive 1.3 / audit 0.8`、`qwen3-thinking = prompt 0.6 / interactive 0.6 / audit 0.2`。
- 对于 `gpt-5.4`、`claude-sonnet-4-6`、`gemini-2.5-flash` 这类官方只给通用口径、未给 scene 明细表的模型，当前默认按“分析低温、交互中温、创作高温”映射：`audit / infer / overview / conversation_compaction = 0.2`，`prompt / interactive / decision = 0.6`，`write = 0.8`；其中 `claude-sonnet-4-6` 的创作档按 Anthropic 文档再抬一档到 `0.9`。

`manifests/*.json` 顶层支持 `tool_selection`：

```json
{
  "tool_selection": {
    "mode": "all"
  }
}
```

可选值固定为：
- `mode=all`：注册当前运行时允许的全部业务工具
- `mode=none`：不注册任何业务工具
- `mode=select`：仅注册 tag 命中 `tool_tags_any` 的工具

当前内置 scene 默认如下：
- `prompt`：单轮问答场景，`model.default_name=mimo-v2-pro-thinking`，按 manifest 注册所需工具。
- `interactive`：交互场景，`model.default_name=mimo-v2-pro-thinking`，`conversation.enabled=true`，按 manifest 注册所需工具。
- `write`：初稿写作场景，`model.default_name=mimo-v2-pro`，允许财报与联网工具。
- `regenerate`：整章重建场景，`model.default_name=mimo-v2-pro`，允许财报与联网工具。
- `repair`：局部修复场景，`model.default_name=mimo-v2-pro`，`tool_selection.mode = none`，不注册任何工具。
- `decision`：研究决策综合场景，`model.default_name=mimo-v2-pro-thinking`，允许财报与联网工具，但其模型覆盖链路归入 `--audit-model-name`。
- `audit`：疑似审计场景，`model.default_name=mimo-v2-pro-thinking`，`tool_selection.mode = none`；它只基于正文与 `证据与出处` 文本输出疑似违规，不承担最终证据复核。
- `confirm`：证据复核场景，`model.default_name=mimo-v2-pro-thinking`，允许 `fins + web` 工具，但只可复核 `证据与出处` 已列出的来源与定位；不得搜索新证据、不得扩展研究。
- `wechat`：微信交互场景，`model.default_name=mimo-v2-pro-thinking`，`conversation.enabled=true`，工具集合与 `interactive` 一致，但输出约束更窄。

`mode=select` 示例：

```json
{
  "tool_selection": {
    "mode": "select",
    "tool_tags_any": ["fins", "doc"]
  }
}
```

固定约束：
- `tool_selection` 是 scene 候选工具集合的真源。
- scene 未声明的工具不会被注册。
- `toolset_registrars.json` 只负责把 toolset 名称映射到 registrar import path，不参与启用决策。
- 当某个 toolset 已经被 scene manifest 与执行期收窄结果明确启用，但安装清单里缺少对应 registrar 时，scene preparation 会直接失败。
- `base/tools.md` 中的 `<when_tag>` / `<when_tool>` 只基于“真实已注册工具集合”渲染。
- `TOOLS` fragment 只表示“是否展示工具说明片段”，不再负责筛选工具。
- `type=TOOLS` 的 fragment 禁止配置 `tool_filters`；配置即视为 manifest 非法。

当前内置 scene 默认值：
- `write`：`mode=select`，默认启用 `fins`、`web`，不启用 `utils`、`doc`
- `regenerate`：`mode=select`，默认启用 `fins`、`web`，不启用 `utils`、`doc`
- `prompt`：`mode=select`，默认启用 `fins`、`web`、`ingestion`，不启用 `utils`、`doc`
- `interactive`：`mode=select`，默认启用 `fins`、`web`、`ingestion`，不启用 `utils`、`doc`
- `audit`：`mode=none`

### 6.2 task 级 prompt 资产

- `tasks/write_chapter.md`
- `tasks/write_chapter.contract.yaml`
- `tasks/repair_chapter.md`
- `tasks/repair_chapter.contract.yaml`
- `tasks/regenerate_chapter.md`
- `tasks/regenerate_chapter.contract.yaml`
- `tasks/fix_placeholders.md`
- `tasks/fix_placeholders.contract.yaml`
- `tasks/fill_overview.md`
- `tasks/fill_overview.contract.yaml`
- `tasks/audit_facts_tone_json.md`
- `tasks/audit_facts_tone_json.contract.yaml`

这类 prompt 主要服务于写作流水线和章节级任务，不直接等同于 scene system prompt。

当前约定：
- `tasks/*.md` 负责写给模型看的任务说明。
- `tasks/*.contract.yaml` 负责显式定义该 task prompt 的输入字段、字段类型与必填约束。
- 写作流水线不会再把整包输入塞进黑盒 `input_json`；而是按 sidecar contract 渲染显式字段。

字段类型当前固定为：
- `scalar`
- `markdown_block`
- `list_block`
- `mapping_block`
- `json_block`

其中：
- `markdown_block` 会保留 Markdown 结构，并渲染为 fenced `markdown` 代码块。
- `list_block` 会渲染为 fenced `json` 代码块中的 JSON 数组。
- `mapping_block` 会渲染为 fenced `json` 代码块中的 JSON 对象。
- `json_block` 会渲染为 fenced `json` 代码块。

这套渲染的目标不是“让 prompt 看起来机器化”，而是：
- 长文本按 Markdown 原样呈现，降低模型读取骨架的成本。
- 结构化规则按 JSON 呈现，降低模型读取 contract/rule 的歧义。
- 共享 `write / repair / audit` prompt 只放通用动作规则，不承载章节专用写法。
- `write_chapter` 的最小职责是：按骨架写、先完成最小判断链、不编造不越界、让关键断言与关键数字在“证据与出处”闭环、为关键数字选择最能直接支撑该句的具体锚点、在 statement/xbrl 场景下显式写出 `Financial Statement` / `XBRL Facts` 锚点、避免把 `search_document` 的粗糙命中直接写成最终 evidence line、直接输出最终正文。
- `write_chapter` / `regenerate_chapter` / `fill_overview` / `write_research_decision` 会额外显式接收 `report_goal`、`audience_profile` 与 `chapter_goal`：分别定义整份报告的总目标、全局读者画像和当前章节要回答的总问题；它们是独立输入层，不并入 `CHAPTER_CONTRACT`。
- `write_chapter` / `regenerate_chapter` 现在还会额外显式接收 `company_facets_summary`。它不是内部字段名，而是面向模型的“公司业务类型与关键约束”输入块；写作前，系统会先确保当前 ticker 已有公司级归因结果，再用它裁剪本章真正喂给模型的 `preferred_lens` 与 `ITEM_RULE`。
- `write_chapter` / `regenerate_chapter` / `fill_overview` / `write_research_decision` 当前统一要求：若 skeleton 中某个 bullet 只是 reader-facing 的字段标签，输出必须严格采用两行：第一行只保留 bullet 标签本身，第二行紧接着写回答正文，二者之间不留空行；字段标签行不得追加回答内容、解释文字或额外标点，禁止写成 `- 标签：回答`、`- 标签 回答` 或 `- 标签：` 后再换行补内容。相邻 bullet 之间保留空行。
- `repair_chapter` 的最小职责是：只修命中的违规、先修实质再修样式、非最小判断链内容优先删、输出稳定 patch；删除内容时必须删除完整 bullet / 完整段落，并清理空 bullet、残句和多余空段。
- `audit_facts_tone_json` 的最小职责是：初始阶段做整章扫描并尽量一次列全问题；repair 后默认只围绕修复合同命中的 excerpt / slot 与局部上下文做局部复审，再按规则标记的 severity 与类别统一计算疑似 `pass/class`。它只发现事实/证据问题、结构/越界问题和少数高价值风格问题，不承担最终证据裁决。
- `confirm_evidence_violations` 的最小职责是：只复核疑似 `E1/E2` 是否属实；若原始证据支持 claim，则取消 `E` 违规；若只是锚点过粗，或同一 filing 内已有正确 section 但当前 evidence line 漏列了该锚点，则优先返回结构化 `anchor_fix`，让代码直接修 evidence line，不删正文信息。`S7` 只保留为兜底：当 confirm 未返回可执行的结构化锚点、或旧产物仍只有自然语言 `rewrite_hint` 时，才继续作为低优先级提示存在。复核时应优先按 evidence line 的类型选择工具：`Financial Statement` → `get_financial_statement`，`XBRL Facts` → `query_xbrl_facts`，section path → 文档定位工具。
- 对 `confirm` 返回的 `confirmed_missing`，写作流水线现在会把对应 claim 收口为 `delete_claim` 处置模式，而不是继续把“删还是弱化”留给 repair prompt 自行判断。
- `repair_chapter` 输出的是结构化 patch plan。当前 patch 至少包含 `target_excerpt`、`replacement`、`reason`，且 `patches` 至少要有 1 条。当目标片段是完整 bullet / 单行 / 段落时，应显式提供 `target_kind`（`bullet|line|paragraph|substring`），并按 task 中定义的粒度语义选择最稳定命中单元；当目标片段可能重复时，应额外提供 `target_section_heading`，必要时再提供 `occurrence_index`，避免 patch 命中歧义。
- 写作流水线默认走 `repair_chapter` 局部 patch；只有结构性失败（如缺主节、缺证据小节、骨架错位）时，才退回 `regenerate_chapter` 做整章重建。

### 6.3 写作模板内的章节 contract

写作模板章节内部允许声明隐藏规则，但规则不通过 `prompts/tasks/*.contract.yaml` 维护，而是直接写在章节模板本身中。

当前固定格式：
- `REPORT_GOAL`：全文总目标。固定为单个文本块，用于喂给写作类 task prompt，不属于章节 contract。
- `CHAPTER_GOAL`：章节总目标。固定为单个文本块，用于告诉模型“本章回答什么”，不替代 `must_answer`。
- `CHAPTER_CONTRACT`：章节级写作边界，固定字段为
  - `narrative_mode`
  - `must_answer`
  - `must_not_cover`
  - `required_output_items`
  - `preferred_lens`
- `CHAPTER_CONTRACT` 只用于定义章节任务，不承载二级操作规则。
- `REPORT_GOAL / AUDIENCE_PROFILE / CHAPTER_GOAL` 与 `CHAPTER_CONTRACT` 分层明确：前三者分别负责“为什么写 / 写给谁 / 本章回答什么”，后者负责“必须覆盖什么 / 不能写什么 / 最小判断链是什么”。
- `AUDIENCE_PROFILE` 当前默认服务具备基本买方训练的读者：熟悉常见财务指标、资本配置与研究流程，更偏好关键指标、时间序列与可比较数据，而不是基础概念教学或泛泛结论句。
- `COMPANY_FACET_CATALOG`：公司级“主业务类型 / 关键约束”候选词表。它写在模板前导区，由写作流水线读取后喂给 `infer` scene；固定候选的维护责任在模板，不在代码常量里。
- 第 `06` 章“财务表现与资本配置”当前对财务类 `ITEM_RULE` 采取更低触发门槛：只要相关指标已披露或可稳定推导，且能帮助读者更快理解财务质量、财务约束或资本分配平衡，就应优先写出，不必等到它足以单独改变整章结论。
- 第 `06` 章中的“后续最该继续跟踪的财务变量”当前按 watchlist 处理：它默认回答“后续该继续盯什么、为什么它最可能改变研究动作”，而不是把该变量写成当前已成立事实；若暂无稳定披露支撑具体数值或现状判断，可只写变量名称及其跟踪意义。
- 默认不新增字段；只有当新增字段能显著降低模型决策歧义，且不能被骨架或现有字段替代时，才允许引入。
- “优先但非必须”的内容，不新增 contract 字段，而通过章节骨架中的顺序，以及局部隐藏规则表达。
- 审计 prompt 会显式接收 `chapter_contract`，其中 `must_not_cover` 用于检查章节越界，不再只靠正文猜测边界；但为完成本章主问题所必需的客观产业链、技术路线、生态或商业角色描述，不应被机械判成越界。
- `证据与出处` 的职责是提供可追溯的来源识别与定位，不承担摘要、结论复述或原文摘录；若正文需要关键数字或比例，应优先让 evidence line 变得更细、更对位，而不是在证据行中增加摘要字段。
- 派生指标与更强结论的通用证据强度规则由 `base/fact_rules.md` 单点承载：派生指标必须能回溯到分子、分母与计算关系；身份绑定、用途分配、主次归因、因果判断或具体竞争对手指认，只有在证据直接支持该更强结论时才可写。
- 当正文数字来自 `get_financial_statement` 时，应使用 `Financial Statement:{statement_type} | Period:{period_label} | Rows:{row_labels}` 格式表达锚点；当数字来自 `query_xbrl_facts` 时，应使用 `XBRL Facts | Concepts:{concepts} | Period:{period_label}` 格式表达锚点，而不是伪造 section heading。
- 送审前程序审计只负责稳定可机械判定的问题：骨架结构不匹配、内容过短、缺少“证据与出处”小节；这些问题不再交给 LLM 审计。
- 审计 `E1/E2` 的核心检查对象是“断言与 evidence line 是否可对应、是否可追溯”，而不是要求 evidence line 再重复正文数字本身。
- 当 `E1/E2` 的当前输入里已经存在足以稳定推出正确数值、正确口径或正确表述的同源证据时，`rewrite_hint` 应优先给出“修正为……”；只有在无法稳定确定替代值时，才建议删除、降级或补证。
- 对在当前输入中被明确表达为后续跟踪对象、待验证问题、敏感性变量或需持续观察信号的内容，审计默认不按 `E1/E2` 处理；只有当它被写成当前事实、当前状态判断或具体 unsupported 数值时，才进入 `E1/E2` 检查。
- 证据条目的“格式可改进”问题不再作为阻断性证据规则处理，而是归入低优先级风格提示；只要来源与定位稳定可追溯，就不应仅因其不是某一种理想标题样式而导致审计失败。
- 对网页资料，只要 evidence line 已提供页面标题、发布/访问日期与可追溯 URL，默认不触发阻断性的 `E3`；若只是缺少更细定位，应降为低优先级提示，而不是否掉正文。
- 当 `audit` 只能得到疑似 `E1/E2` 时，由后续 `confirm` 环节使用工具复核已引用证据，而不是让 `audit` 自己兼做第二次研究。
- 写作流水线会在送审前对 `证据与出处` 做确定性预处理，只清理可机械判定的摘要型尾缀与明显格式噪音；这一步不改变正文事实语义，也不替代审计。
- 审计中的 `C2` 以及 `S4/S5/S6/S7` 只保留为低优先级提示，不再单独阻断章节通过；它们用于提醒写作边界与事实/分析区分，但不应压倒更有价值的买方表述。
- `S4/S5/S6` 的目标是防止把分析、前瞻或公司预期写成当前事实；它们不应再通过要求句首添加 `【分析】`、`【前瞻】`、`【前瞻（原文）】` 这类标签来修复正文。
- 第 `06` 章“财务表现与资本配置”当前默认要求先给关键财务数据与近年走势，再落到结果可信度、资产负债表约束与资本去向判断；不再只写抽象财务结论句。
- `ITEM_RULE`：条目级条件写作规则。固定字段为：
  - `mode`：`conditional | optional`
  - `item`：需要按条件补充的输出项名称
  - `when`：触发条件说明
- 第一阶段写作裁剪里，`preferred_lens` 与 `ITEM_RULE` 都支持 facet 条件：
  - `preferred_lens` 现在采用有序对象列表，而不是旧的映射块。
  - 每条 lens 固定字段为：
    - `lens`
    - `priority`：`core | supporting`
    - `facets_any`
  - `ITEM_RULE` 可额外声明：
    - `facets_any`
- 写作流水线不会把模板中的 `preferred_lens` / `ITEM_RULE` 全量直接喂给模型，而是会先按当前 ticker 的公司级归因结果做一轮过滤，再把过滤后的结果注入 `章节合同` 与 `条件写作规则`。
- 公司级归因结果当前分两层：
  - 主业务类型，例如 `平台互联网`、`半导体设备/制造`、`保险`
  - 关键约束，例如 `监管敏感`、`出口限制敏感`、`高资本开支`
- facet 归因是章节写作的 gate：
  - 任何章节写作前，只要 manifest 中缺 `company_facets`，就会自动先推理一次。
  - 默认复用已有结果；只有显式 `write --infer` 才会强制重跑。
- `ITEM_RULE` 必须贴在具体标题附近，并绑定到最近的上一个标题；它只服务写作类场景（`write / regenerate`），不进入 `audit / repair`。
- `ITEM_RULE` 负责：这类行业或这类公司才需要补充写什么。先判断 `when` 是否成立；不成立时整项不输出。
  - `mode=conditional`：成立时应优先显化，默认优先把最有判断价值的 `1–2` 条写成可见条件小节或显眼条件条目，不要把所有成立的条件项都悄悄并入泛化 bullet。
  - `mode=optional`：成立时只补入最有判断价值、最能改变本章判断路径的少数内容；其余可省略，不必为了“覆盖规则”而机械罗列。
  `ITEM_RULE` 应优先承载两层差异：
  - 行业共性判断变量：这个行业在这一章通常最该多写什么。
  - 特殊结构变量：这家公司有哪些不属于行业常规、但会明显改变本章判断的特殊结构。
  `ITEM_RULE` 的成功标准不是“偶尔多写一点”，而是当条件成立时，优先写出真正有判断价值的行业特定内容，让不同业态在同一章节下写出明显不同的重点。
  - 当前默认实现里，模板中的 `ITEM_RULE.when` 已统一收口为“只要有稳定披露且有判断价值就写”。
  - 也就是说，`when` 不再承担高门槛筛选职责；真正的收敛点是“是否有稳定披露”与“是否确有判断价值”。
  - 若一条行业共性 `ITEM_RULE` 和一条特殊结构 `ITEM_RULE` 都明显成立，优先各保留一个最有判断价值的点，而不是只写通用行业内容。
  - 若某个特殊结构变量比行业共性变量更能解释本章判断，它不应只停留在详细情况里，而应被提升到 `required_output_items` / `结论要点` 对应的主判断。
  - 对关键判断章节，尤其第 1 / 3 / 5 章，以及同类的第 2 / 4 / 6 / 7 / 8 / 9 章，仅靠 `ITEM_RULE` 不够；主骨架也应显式预留“哪个变量最能解释这章”的标题位或输出位，让行业共性变量与特殊结构变量都能进入主判断，而不是只在详细情况里补充。
- `ITEM_RULE` 不是独立一级章节规划语言。不要在正文骨架里额外写出 `条件项与可选项` 这类总结性标题，但满足条件的 `item` 本身可以成为可见小节。
- 第 `01 / 02 / 03 / 04 / 05 / 06 / 07 / 08 / 09 / 10` 章当前都已按“业务类型路由 + 横切约束路由”重写 `preferred_lens`；其中第 `02 / 03 / 05 / 06` 章同时更系统地把关键差异规则提升为 `conditional`，而更细的补充数字、术语释义或会计口径解释继续留在 `optional`。
- 第一章中的收入 segments 与重要地理分布，在证据已精确到对应 statement、table、note 或 section，且它们能直接帮助识别“公司真正卖什么、靠什么赚钱、主要面对哪些市场”时，应作为被鼓励的 `ITEM_RULE` 输出；客户集中度、品牌/事实标准等其余高风险信息仍保持更高门槛。
- 第一章默认不输出未来投入分配、资本配置预期或路线图式资源倾斜；这些信息应留给后续更合适的章节。
- 第一章里“产业链位置与商业角色”只应用于定义公司处在价值链哪一段、上下游是谁、卖的是组件/平台/服务/基础设施/撮合能力中的哪一类；默认不展开市场份额、竞争强弱、技术路线图、领导地位、平台愿景或稳定经常性收入。
- 当正文 claim 来自 section-path 工具时，evidence line 必须对齐实际承载该句的最窄 section；若 claim 只在父级 heading 中成立，不得用相邻子节代替。
- `search_document` 命中只可作为检索线索，不可直接作为最终 evidence line；最终 evidence line 必须落到可稳定复核的 section path、`Financial Statement` 或 `XBRL Facts` 格式。`confirm` 仅可在同一已引用 filing / item 内把 `search_document` 作为辅助定位线索，不得把其命中本身当作证据。
- `confirm` 仍只复核已引用证据；若 cited section path 只是选窄了、而同一 filing/同一 item 的直接父级 heading 已可支持 claim，可判为 `supported_but_anchor_too_coarse`，仅修 evidence line。
- 单次、且有助于区分事实、公司原文或公司预期的引用前缀（如“公司披露”）不应触发 `S3`；`S3` 只针对重复、机械、无必要的引用套话。
- evidence line 必须是纯定位信息；不得写“内容包含/涉及/说明了/显示了”等摘要尾巴。网页资料默认只写页面标题、日期与 URL；`Rows` 只能使用工具真实返回的行标签，不得自拟概括性短语。
- `repair` 默认不允许删除仍在支撑正文 claim 的证据行；若要处理证据区，只能替换成支撑力更强或至少不更弱的 locator。
- `repair` 对 `S4/S5/S6` 默认优先做自然改写，让读者能区分“事实 / 分析 / 前瞻 / 公司披露”；不要用句首标签机械修复。
- 章节阶段产物采用前缀式命名，便于按文件名排序阅读：`initial_*`、`repair_N_*`、`regenerate_N_*`。其中疑似审计、证据复核与合并后审计分开落盘为 `*_audit_suspect.json`、`*_confirm.json`、`*_audit.json`。

职责边界：
- `skeleton`：去掉 HTML 注释后的章节骨架，只负责输出结构约束。
- 章节原文中的 `CHAPTER_CONTRACT` / `ITEM_RULE`：只负责生成结构化写作规则。
- task prompt sidecar contract：只负责定义 task prompt 的输入接口，不承载章节领域知识。
- 运行时喂给模型时，章节合同会被聚合成一个 JSON 对象，而不是分散成多个列表块。
- 章节骨架的唯一标准是：是否帮助买方在更短时间内形成更强判断；默认章节保留 `结论要点 / 详细情况 / 证据与出处`，但第0章“投资要点概览”作为封面页例外。

## 7. 推荐修改方式

### 7.1 只想跑通系统

通常只需要：
1. 复制 `dayu/config/* -> workspace/config/`
2. 配置 `MIMO_API_KEY`
3. 不改其它默认值

### 7.2 想切换模型

改：
- `workspace/config/llm_models.json`

并在 CLI 中指定：

```bash
python -m dayu.cli prompt "总结苹果风险" --ticker AAPL --model-name mimo-v2-flash --temperature 0.2
```

### 7.3 想改变 Agent 行为

改：
- `workspace/config/run.json`

例如：
- 增大 `max_iterations`
- 调整 `tool_timeout_seconds`
- 打开 `tool_trace_config.enabled`

补充：
- 若只想提高某个 scene 的预算，优先改对应 `prompts/manifests/*.json` 的 `runtime.agent.max_iterations`；若只想放宽某个 scene 的工具调用超时，改 `runtime.runner.tool_timeout_seconds`，不要再在业务代码里写死覆盖值。

### 7.4 想调整提示词

改：
- `workspace/config/prompts/`

建议：
- 场景行为改 `scenes/*.md`
- 全局规范改 `base/*.md`
- 不要直接改包内默认 prompt，优先在 workspace 中覆盖

## 8. 维护原则

- 本文档只描述当前存在的配置文件与字段。
- 新增、删除、重命名配置字段后，应同步更新本文件。
- 若某项配置同时存在包内默认值与 workspace 覆盖值，应在代码和文档中保持优先级口径一致。
