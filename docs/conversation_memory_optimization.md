# `conversation_memory` 优化建议

面向 `dayu/host/conversation_memory.py` 与 `dayu/config/llm_models.json` / `dayu/config/run.json` 的多轮会话记忆子系统。仅对 `conversation.enabled = true` 的 scene 生效，当前即 `interactive / wechat / prompt_mt` 三个 scene；write pipeline 全程走单轮路径，不受此文档任何建议影响。

## 1. 背景与边界

### 1.1 算法事实

当前送模消息由 `DefaultConversationMemoryManager.build_messages` 组装，分层策略如下：

- `working memory`：最近 raw turns，预算：
  `working_budget = clamp(max_context_tokens * ratio_w, floor_w, cap_w)`
  进一步受 `working_memory_max_turns` 轮数上限约束。
- `episodic memory`：结构化 episode summary，预算：
  `episodic_budget = clamp(max_context_tokens * ratio_e, floor_e, cap_e)`
- `compaction` 触发条件：
  `len(uncompressed) > compaction_trigger_turn_count`
  或 `uncompressed_tokens > working_budget * compaction_trigger_token_ratio`。
  压缩时保留尾部 `compaction_tail_preserve_turns` 轮不压。

默认值（`run.json.conversation_memory.default`）：
`ratio_w = 0.08, floor_w = 1500, cap_w = 12000,
ratio_e = 0.02, floor_e = 2000, cap_e = 12000,
max_turns = 6, trigger_turn_count = 8, trigger_token_ratio = 1.5, tail_preserve = 4`。

模型级覆盖通过 `llm_models.json[model].runtime_hints.conversation_memory` 字段级叠加。

### 1.2 当前配置档位一致性

本轮已完成的对齐：

- **1M 档（13 个模型）**：`working_memory_token_budget_cap = 80000`，episodic 走 default (2000 / 12000)。
- **256K 档（3 个模型）**：`working_memory_token_budget_cap = 32000`，`episodic_floor = cap = 6000`。

跨模型一致性已满足：用户在多轮 scene 内切换模型，memory 行为仅随客观上下文档位变化，不再受历史遗留差异影响。

### 1.3 本次不是"最优"，只是"现阶段稳定 baseline"

以下四点是静态分析层面可观察到的结构性问题，属于继续优化的切入点。

---

## 2. 已识别的结构性问题（按证据同源原则）

### 2.1 `compaction_trigger_token_ratio * working_budget` 是半死代码

- 典型设置 `max_turns = 6`，平均每轮 ~5K tokens，上限约 30K tokens。
- 1M 档 `working_budget * 1.5 = 120000`，30K 永远触不到。
- 实际生效的几乎总是 `len(uncompressed) > 8` 这条轮数分支。

token 阈值不是被触发了才会动作的防御线，而是与 working cap 耦合后永远被压在阈下，等于**把 compaction 退化成纯轮数触发器**。

### 2.2 `working_memory_token_budget_ratio` 在 1M 档是死代码

- `1_048_576 * 0.08 = 83_886`，恒大于 `cap = 80000`，最终恒为 cap。
- ratio 字段保留的"窗口缩小时自动回退"弹性在实际取值域内没发生。

这说明当前实际起控制作用的是 cap 绝对值，ratio 只是语义噪音。

### 2.3 working / episodic 各自独立预算池，没有总池

- `build_messages` 里 working raw turns 与 episodic summaries 分别从各自预算切片取，彼此不感知。
- 理论上两池之和可能超出"实际留给 memory 的总量"，尤其 1M 档默认 80000 + 12000 = 92000。
- 调一个池不会释放另一个池空间，调参必须两个池同步考虑，这是此前复核"策略解释不完整"感觉的根源。

业界 `MemGPT / LangGraph` 方向是**单总池 + 层间消费优先级**，dayu 当前选择是"双独立池"，代价已暴露。

### 2.4 `working_memory_max_turns = 6` 全局硬限

- 全局生效，不随档位变化。
- 1M 档下 6 轮 ~30K，只用 cap(80000) 的约 37%，cap 本身在实际路径上用不满。
- 决定回放量的主约束是 `max_turns` 而不是 `cap`，配置语义与实际行为错位。

---

## 3. 优化方案（按收益 / 风险比排序）

### 3.1 Phase 1：compaction 阈值改"占窗口百分比制"

**目标**：让 token 阈值语义与 "距离爆窗口还有多远" 直接对齐，消除 2.1 的半死代码。

**改动点**：

- `ConversationMemorySettings` 新增 `compaction_trigger_context_ratio: float`，默认 `0.75`。
- `_select_compaction_candidate` 的 token 分支改为：
  `uncompressed_tokens + working_tokens_estimated > max_context_tokens * compaction_trigger_context_ratio`
- 废弃 `compaction_trigger_token_ratio` 或降级为 legacy fallback（若外部 workspace 已有自定义值）。

**收益**：

- 阈值语义清晰、跨档位自动伸缩。
- 不再受 working_cap 调整的传递影响。
- 消除半死代码，减轻未来 reviewer 理解成本。

**风险 / 影响面**：

- `_select_compaction_candidate` 行为变更，需补测试。
- `options.py` 的 `_build_conversation_memory_settings` / `_validate_conversation_memory_settings` 需相应扩展。
- `config/README.md §5.7` 需更新字段列表与示意。
- `run.json` 默认值需补 `compaction_trigger_context_ratio`。

**测试覆盖建议**：

- 新增 1M / 256K 两档下，uncompressed 逐步增长直至触发阈值的单测。
- 原 `trigger_token_ratio` 的测试迁移到新字段。

---

### 3.2 Phase 2：合并为单总池

**目标**：对齐业界 `MemGPT / LangGraph` 主流做法，消除 2.3 的双池独立问题。

**改动点**：

- 新增 `memory_total_token_budget_ratio: float`，默认 `0.20`；废弃 `working_*` 与 `episodic_*` 两组 ratio/floor/cap 字段。
- `build_messages` 先按 working 消费规则填充，溢出部分才给 episodic summary 列表。
- 模型档位覆盖从 2 个字段压缩为 1 个。

**收益**：

- 配置面从 6 个字段压到 1 个，档位表更清爽。
- 两池争抢空间的设计歧义消失，README 能用单一原则讲清。

**风险 / 影响面**：

- 较大：影响 `build_messages`、`_build_memory_block`、`_select_compaction_candidate`。
- schema 变更属于硬兼容断点；按项目约束"按全新 schema 起库处理、不保留兼容读取"的规则，此 Phase 必须一次切断干净。
- 测试、README、所有 workspace 样例配置都需同步更新。

**前置条件**：

- Phase 1 已落地，且稳定运行一段时间。
- 有生产 session 长度分布数据，能验证 0.20 默认值是否合理。

---

### 3.3 Phase 3：`max_turns` 处理

两种可选方向：

**方向 A：分档化**

- 移入 `runtime_hints.conversation_memory.working_memory_max_turns`。
- 1M 档建议 10~12，256K 档保持 6~8。

**方向 B：彻底移除**

- 让 token 预算作为唯一回放约束。
- 对齐 MemGPT / LangGraph 主流做法。

**推荐**：B。当前全局 6 导致 cap 在 1M 档永远用不满，已是"约束错位"的症状。

**风险 / 影响面**：

- 小：仅改 `DefaultWorkingMemoryPolicy.select_turns`。
- 测试仅需调整 working policy 相关用例。

---

### 3.4 Phase 0.5（可选小修）：档位原则写进 README

**目标**：把当前"同档同配"的原则固化为规则，防止未来个案特例重新污染。

**改动点**：`dayu/config/README.md §5.7` 开头补：

1. 前置声明：本节仅作用于 `conversation.enabled = true` 的 scene（当前为 `interactive / wechat / prompt_mt`）。write pipeline 不走 memory manager，本节字段对其无效。
2. 档位一致性原则：`runtime_hints.conversation_memory` 的配置以**跨模型一致性**为第一优先级，单模型调参必须整档一起动，不允许同档内个体特例。

**收益**：

- 消除读者"write pipeline 也被 memory 配置影响"的误解。
- 明确规则后，未来新增模型或调参都有硬标尺。

**风险**：极低，仅文档改动。

---

## 4. 实施顺序建议

| 阶段 | 动作 | 前置 | 风险 |
|---|---|---|---|
| 0 | 本次已完成的档位对齐（deepseek / qwen3:30b） | — | 已落地 |
| 0.5 | README 档位原则前置声明 | — | 低 |
| 1 | compaction 阈值改百分比制 | — | 中 |
| 2 | 单总池合并 | Phase 1 稳定 + 生产数据 | 较大 |
| 3 | `max_turns` 去除或分档 | Phase 2 已落地 | 小 |

---

## 5. 生产数据需求（决定 Phase 2 / 3 最优值的前置）

以下数据缺一项就无法把 Phase 2 / 3 的默认值从"估算"升级为"实测"：

- `interactive / wechat / prompt_mt` 三个 scene 各自的 session 长度分布（turn 数 p50 / p90 / p99）。
- 每轮平均消息 token 量（含工具调用）。
- compaction 实际触发率（按 turn 触发 vs 按 token 触发）。
- 触发后的摘要质量回归（是否丢 pinned_state / open_questions）。

在没有这些数据前，Phase 1 是相对安全的改造（语义清理为主，数值可以保守取 `0.75`）；Phase 2 / 3 应先收数据再动。

---

## 6. 业界参考（知识性总结，非实时核对）

| 系统 | 分层 | compaction 触发 | 预算度量 |
|---|---|---|---|
| Claude Code | 单层滚动 + auto-compact | 占 context window 百分比（常见 80%~90%） | 相对百分比 |
| MemGPT / Letta | main / recall / archival 三层 + tool-callable memory | main context 满时 LLM 主动搬迁 | 每层绝对值 |
| LangGraph `ConversationSummaryBufferMemory` | buffer + running summary | buffer 超 token 即 summary | 单层 token 阈值 |
| AutoGen | 每 agent rolling + summary | turn 或 token 阈值 | 单层 |
| dayu（当前） | working raw + episodic structured | `turns > N` 或 `tokens > working * 1.5` | 双层独立 ratio/floor/cap |

dayu 在"双层结构 + 异步后台 compaction + revision 乐观并发"这三项工程质量上高于主流开源实现；在"阈值语义 + 单池 vs 双池"两项上略落后。Phase 1 / 2 即朝着前者收敛。

---

## 7. 不打算做的事

- **不回滚 1M 档 working_cap 到 20000~32000**：会破坏"跨模型一致性"这条第一优先级原则。若真要下调，必须整档统一动。
- **不再给单个模型特例化 memory 配置**：档位原则一旦写进 README，应作硬规则。
- **不在没有生产数据前随意提高 episodic cap**：当前 12000 是否过薄属于"理论可能"，但提高前需数据支撑。
