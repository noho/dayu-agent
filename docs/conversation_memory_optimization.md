## 背景

`dayu/host/conversation_memory.py` 的多轮记忆 baseline 已跑通，跨模型档位对齐完成（1M 档 13 个模型、256K 档 3 个模型）。静态分析暴露 4 个结构性问题，本 issue 一次性按全新 schema 重构，**硬切换、不留兼容**。范围限定 `conversation.enabled = true` 的 scene（`interactive / wechat / prompt_mt`）。

## 已识别的结构性问题

### 2.1 compaction token 阈值是半死代码
- `compaction_trigger_token_ratio * working_budget` 在 1M 档 = 120K，实际 session 30K 永远触不到
- compaction 退化成纯轮数触发器（`len(uncompressed) > 8`）

### 2.2 `working_memory_token_budget_ratio` 在 1M 档是死代码
- `1M * 0.08 = 83886 > cap = 80000`，ratio 恒被 cap 截断
- ratio 字段实际不起作用，只是语义噪音

### 2.3 working / episodic 双独立预算池，没有总池
- 两池各自独立切片，互不感知，理论和可达 92K
- 调一个池不会释放另一个池空间，调参必须同步考虑

### 2.4 `working_memory_max_turns = 6` 全局硬限
- 1M 档下 6 轮 ~30K，只用 cap(80000) 的 ~37%，cap 用不满
- 决定回放量的主约束是 `max_turns` 而不是 `cap`，配置语义与实际行为错位

---

## 设计原则（财报 agent 第一性原理）

财报会话不变量与通用代码 agent 不同，方案直接从这些不变量长出来，**不糅合各家招式**：

| 不变量 | 设计含义 |
|---|---|
| **目标稳定**：用户开会话是为搞清某公司某财务问题，目标收敛 | `pinned_state` 是会话灵魂，必须永远在、不参与池竞争 |
| **工具结果即事实**：财报数字 / 章节定位 / XBRL facts 不能被 LLM 二次摘要丢精度 | 项目已只存 `ConversationToolUseSummary.result_summary`，无需额外 prune；compaction 阶段 episode summary 通过 `confirmed_facts` 字段保留 |
| **追问连续性是刚需**：最常见是"上一轮说 X，再展开/换口径" | 最近 N 轮 raw turn 强制保留、**不计入** token 预算 — `recent_turns_floor` 语义从"上限"反转为"下限保底" |
| **跨轮一致性 > 上下文丰富度** | episode summary `confirmed_facts` + `pinned_state` 是反幻觉核心 |
| **memory 应克制**（项目自身工程信条）："长上下文优先留给财报材料、检索结果和当前章节上下文，memory 只建议小步上调" | 1M 档总池 cap 设到 60K（不向 200K 走），把窗口让给财报材料 |

**非目标**（明确不做）：
- 不引入 OpenCode `prune_tool_outputs`（项目已 prune，重复造轮子）
- 不引入 OpenCode `reserved_token_buffer`（system_prompt + tool schema 是常量，调小 ratio/cap 即可）
- 不引入 Codex `tokensUsed/tokensRemaining` 显式可观测（trace_infrastructure 已覆盖）
- 不向 1M 档总池 200K 扩张（违反"memory 克制"信条）

---

## 概念澄清

### `pinned_state` 是什么 + 怎么产生

**结构定义**（`dayu/host/conversation_store.py:229-237`），4 字段：

| 字段 | 类型 | 财报场景含义 | 例子 |
|---|---|---|---|
| `current_goal` | `str` | 当前主任务 | "分析贵州茅台 2024 H1 营收增长结构" |
| `confirmed_subjects` | `tuple[str, ...]` | 已确认对象（公司 / 报告期 / 报告类型） | `("贵州茅台 600519.SH", "2024 半年报")` |
| `user_constraints` | `tuple[str, ...]` | 用户口径约束 | `("用 IFRS 口径", "数字以百万元计")` |
| `open_questions` | `tuple[str, ...]` | 未决问题 | `("毛利率下滑原因待解释",)` |

**怎么产生**（已有链路，本次不动）：

```
1. 用户发起对话 → 初始 pinned_state = ConversationPinnedState() 全空

2. 对话累积 N 轮 raw turn → 不动 pinned_state

3. compaction 触发（按 compaction_trigger_context_ratio）
   ↓
   ConversationCompactionCoordinator 调度 LLM 跑 conversation_compaction scene
   （prompts/scenes/conversation_compaction.md，无工具、严格 JSON）
   ↓
   _build_user_payload 把 pinned_state、近 episodes、待压缩 turns 拼 JSON 喂给 LLM
   ↓
   LLM 输出：
     - episode_summary（append 到 transcript.episodes）
     - pinned_state_patch（增量修改 pinned_state）

4. ConversationPinnedStatePatch.apply_to → 字段级合并
   - patch 字段为 None 表示"本次不动"，不是"清空"
   - 因此 pinned_state 从 N 次 compaction 中单调演进
```

要点：
- pinned_state **不是规则提取**，是 LLM 在 compaction 时顺便产出的结构化抽取
- 渲染由 `_render_pinned_state_block` 输出 `[Conversation Memory]` 系统块开头
- token 量典型 < 200，永远全量渲染、不参与池竞争

### "最近 N 轮 raw turn"是上限还是下限？

**这是相比旧设计的关键反转**：

| 旧设计 `working_memory_max_turns = 6` | 新设计 `recent_turns_floor = 2` |
|---|---|
| **上限**："最多放 6 轮" | **下限**："至少保 2 轮" |
| 回放轮数 = `min(6, budget 允许的轮数)` | 回放轮数 = `max(2, budget 允许的轮数)` |
| 1M 档 cap=80K 算下来可放 ~16 轮，被 6 卡死 → cap 永远用不满 | 没有任何上限，budget 允许 20 轮就放 20 轮 |
| 6 轮被 cap 80K 包住，6 是真主约束（§2.4 原文） | 2 轮是反退化保底，绝大多数情况下被 budget 自然超过 |

简言之：**旧字段是天花板，新字段是地板，方向相反**。如果 budget 充足，可能回放 15 轮，N=2 完全不限制它。它只在极端情况兜底（单轮 user_text 巨长 > budget 时不让追问断链，走 `_build_minimum_preserved_turn_view`）。

---

## 最终方案

### 两层结构

```
[Conversation Memory]
├── pinned_state                          ← 永远全量、不计入 token 池
└── 历史单总池（budget = clamp(window * ratio, floor, cap)）
    ├── 最近 N 轮 raw turn                ← recent_turns_floor 强制保留，不计入 budget
    │   （单轮溢出由 _build_minimum_preserved_turn_view 兜底）
    ├── 更老 raw turn（按预算从新到老回放）
    └── episode summaries（按剩余预算从新到老填充）
```

### 配置面（8 字段，旧字段全删）

```json
"conversation_memory": {
  "default": {
    "memory_token_budget_ratio": 0.10,
    "memory_token_budget_floor": 4000,
    "memory_token_budget_cap": 60000,
    "recent_turns_floor": 2,
    "compaction_trigger_context_ratio": 0.70,
    "compaction_tail_preserve_turns": 4,
    "compaction_context_episode_window": 2,
    "compaction_scene_name": "conversation_compaction"
  }
}
```

### 字段说明

`conversation_memory` 与模型 `max_context_tokens` **仍然强相关**，ratio / 触发阈值都是相对窗口的百分比，不是绝对值。差别在：旧设计 ratio 被 cap 截成死代码（§2.2），新设计接受现实，cap 是真主约束、ratio 在小窗口模型生效。

| # | 字段 | 默认 | 与 `max_context_tokens` 关系 | 作用 | 调参信号 |
|---|---|---|---|---|---|
| 1 | `memory_token_budget_ratio` | `0.10` | **池大小 = window * ratio**（再被 floor/cap 截断） | 控制总池占模型窗口百分比 | 通常不动；按当前默认（floor=4000、cap=60000）在 ~40K–600K 窗口范围内真实生效，超出区间被 floor/cap 咬合 |
| 2 | `memory_token_budget_floor` | `4000` | 与 window 无关 | 总池下限保底 | 短窗口模型一轮都放不下 → 上调 |
| 3 | `memory_token_budget_cap` | `60000` | 与 window 无关 | 总池上限封顶。1M 档算出 100K 被 cap 截到 60K | 追问频繁忘上一轮 → 小步上调（每次 +8K） |
| 4 | `recent_turns_floor` | `2` | 无关 | 最近 N 轮强制保留（不计入 budget），反退化下限 | 单轮极长导致追问断链 → 调到 3 |
| 5 | `compaction_trigger_context_ratio` | `0.70` | **触发阈值 = window * ratio** | `system + pinned + actual_episodic_in_prompt + uncompressed_raw_turns + 当前 user_text > window * 0.70` 时压缩；`actual_episodic_in_prompt` 复用 `_build_memory_block` 的真实裁切结果，与渲染口径一致 | 频繁触发拖慢响应 → 0.80；从不触发 → 0.60 |
| 6 | `compaction_tail_preserve_turns` | `4` | 无关 | 压缩时保留最近 4 轮 raw 不压 | 现状不变 |
| 7 | `compaction_context_episode_window` | `2` | 无关 | 生成新 episode summary 时喂给 LLM 多少个最近 episode 作邻近上下文 | 现状不变 |
| 8 | `compaction_scene_name` | `"conversation_compaction"` | 无关 | 压缩用专用 scene（无工具、严格 JSON） | 不动 |

字段 1 + 5 与 window 强耦合，共同保证 1M / 256K / 8K 档自适应，不用为每档手写绝对阈值。

### 跨档配置策略：default 一份打天下

**不分档配置。** 各档实际 budget 自然落位：

| 模型窗口 | `window * 0.10` | clamp(floor=4000, cap=60000) | 实际 budget |
|---|---|---|---|
| 1M | 100K | 截到 cap | **60K** |
| 256K | 25.6K | 在区间内 | **25.6K** |
| 128K | 12.8K | 在区间内 | 12.8K |
| 32K | 3.2K | 兜到 floor | 4K |

理由：
1. 1M 档自然被 cap 截到 60K（default cap 就是为它设的）
2. 256K 档自然落到 25.6K（比 1M 档少 22%，与"窗口越大 memory 越小份额"信条一致）
3. 财报 agent interactive 不服务 < 128K 模型
4. 极致表达"同档同配"——连档都不分

**两个迁移点**：

1. **`dayu/config/llm_models.json`**：删除全部 19 处 `runtime_hints.conversation_memory` 块（不再需要，default 接管）
2. **`dayu/cli/commands/init.py`**：删除 `_build_conversation_memory_overrides`（行 78-103）整个函数，及其在 `init.py:1056 / 1187` 两处调用。`dayu-cli init` 添加 ollama / OpenAI 兼容模型时不再写入 `conversation_memory` 覆盖到 `workspace/config/llm_models.json`，按 default 走

`workspace/config/llm_models.json` 用户已生成的旧 entry **不需要迁移**：按项目硬约束"全新 schema 起库处理"，旧 workspace 由 `workspace_migrations` 在 `dayu-cli init` 流程处理；本次只保证新 init 不再写覆盖、官方 default 干净。

### 移除字段（硬切换）

- `working_memory_max_turns`（→ `recent_turns_floor` 语义反转）
- `working_memory_token_budget_ratio` / `_floor` / `_cap`（→ `memory_token_budget_*` 单总池）
- `episodic_memory_token_budget_ratio` / `_floor` / `_cap`（→ 同上）
- `compaction_trigger_turn_count`（轮数触发器去除，token 占比唯一约束）
- `compaction_trigger_token_ratio`（→ `compaction_trigger_context_ratio`）

### 触发与消费算法

**Compaction 触发**：
```
window_used = system_prompt + pinned_state
            + actual_episodic_in_prompt
            + uncompressed_raw_turns + current_user_text
should_compact = window_used > max_context_tokens * compaction_trigger_context_ratio
```

`actual_episodic_in_prompt` 复用 `_build_memory_block` 的真实裁切结果，**与 prompt 渲染口径完全一致**，避免触发器与渲染器漂移。`schedule_compaction`（持久化后路径）入参不接 `user_text`：用户消息已写入 transcript，再传会与 `uncompressed_raw_turns` 双计；`prepare_transcript`（持久化前路径）保留 `user_text` 形参。两条路径语义分离。

**总池消费（`build_messages` 新流程）**：
```python
budget = clamp(window * ratio, floor, cap)

# 1. pinned_state 独立路径，不动 budget
# 2. 最近 N 轮 raw turn 强制保留（不计 budget）
forced = raw_tail[-recent_turns_floor:]
# 3. 更老 raw turn 按 budget 从新到老回放
extra_pool = budget
for turn in reversed(raw_tail[:-recent_turns_floor]):
    if estimate(turn) <= extra_pool: 加入；扣 extra_pool
    else: break
# 4. episode summaries 用剩余 budget 从新到老填充
for episode in reversed(transcript.episodes):
    if estimate(episode) <= extra_pool: 加入；扣 extra_pool
    else: break
```

`recent_turns_floor` 不计入 budget 的理由：它是反退化硬保底。即使 budget=0、cap=0、配置全错，也保最近 N 轮在不让追问彻底断链。极端单轮溢出由既有 `_build_minimum_preserved_turn_view` 路径处理（user_text 全保、assistant 降级裁剪）。

### 4 个结构性问题对照

| Issue 问题 | 解 |
|---|---|
| 2.1 token 阈值半死 | 改占模型窗口百分比 0.70，跨档位自动伸缩 |
| 2.2 ratio 死代码 | 单总池 cap 主导（接受现实），ratio 仅在小窗口模型生效 |
| 2.3 双独立池 | 合并单总池；pinned_state 单独不参与竞争 |
| 2.4 max_turns 全局硬限 | 反转为 `recent_turns_floor`，从"上限"改"下限"（保追问连续性） |

---

## 风险与说明

- **1M 档 cap 设到 60000**：基于"memory 克制"信条的工程判断；现有 1M 档用户若依赖更大 working 池追问，可小步上调；项目信条明确"长上下文留给财报材料"，方向一致。
- **不再等生产数据**：参数化配置可调，先落地机制；后续按生产观察小步调参。
- **schema 硬切换**：按项目硬约束"全新 schema 起库处理"，旧库迁移作为 `workspace_migrations` 插件进入 `dayu-cli init` 流程。

---

> 业界参考（仅作设计校核，不照搬）：Claude Code（百分比触发 auto-compact）、OpenAI Codex（单池 tokenBudget）、OpenCode（auto/prune/reserved 三件套）、MemGPT/Letta（分层 + 工具可调）。最终方案不是糅合，而是从财报 agent 不变量长出来。
