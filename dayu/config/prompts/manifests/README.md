# Scene Manifests

本目录存放各 scene 的运行时清单（manifest）。每个 `*.json` 描述一个 scene 的模型、工具选择、运行时上限、片段组装顺序和上下文槽位。

## 字段约定

- `scene`：scene 唯一标识，与 prompt 模板目录、scene executor 入口一一对应。
- `model.default_name` / `allowed_names` / `temperature_profile`：模型默认值与白名单、温度档位。
- `runtime.agent.max_iterations`：单次 Agent 执行允许的最大迭代轮数（每轮一次模型调用 + 可选工具调用）。
- `runtime.runner.tool_timeout_seconds`：单次工具调用超时时间。
- `tool_selection.mode`：`select` 表示按 `tool_tags_any` 选工具；`none` 表示禁用工具。
- `fragments`：prompt 拼装片段顺序与必填性。
- `context_slots`：执行前需要填充的上下文槽位。

## `max_iterations` 对照表

| Scene | max_iterations | 任务性质 | 设定理由 |
|---|---|---|---|
| `infer` | 12 | 公司业务类型与关键约束的受控分类 | 9 选 1~3 + 4 选 1~3 的有限分类，3~5 次 fins 调用足够取证；上限留出冗余 |
| `overview` | 12 | 第 0 章封面页生成 | 信息源固定（公司基本盘 + 最新年报封面），无需漫游 |
| `decision` | 12 | 研究决策综合 | 基于已有判断做收口，不需要边走边想 |
| `fix` | 12 | 占位符补强 | 局部修补，作用域窄 |
| `audit` | 16 | 章节合规检查 + 关键数据点核对 | read_section ×2~3 + xbrl/statement ×1~2，留出冗余 |
| `repair` | 16 | 局部修复 | read_section ×2~4 + xbrl ×1~2 |
| `confirm` | 20 | 证据可回溯性逐条确认 | 章节里 5~10 条 claim × (search 0.5 + read 1)，最容易吃 budget |
| `regenerate` | 24 | 整章重建 | 比 repair 范围更大，需要重新走完一章的取证 + 写作 |
| `write` | 32 | 章节正文生成 | 广泛取证 + 写正文，预算最大 |
| `wechat` | 16 | 微信交互式财报分析 | 响应延迟敏感；复杂问题应靠多轮追问拆，不在单 turn 榨干 |
| `interactive` | 20 | 命令行交互式财报分析 | 用户可追问，单 turn 不必榨干预算 |
| `prompt` | 24 | 单轮财报问答 | 单轮要"答完"，预算更宽 |
| `prompt_mt` | 24 | 多轮财报分析 | 单轮口径同 prompt |
| `conversation_compaction` | — | 多轮会话阶段摘要压缩 | 无工具，不进入 agent 迭代循环 |

## 调参原则

- **分类 / 判断 / 检查类任务**：上限 ≤ 12。任务性质要求"信息够就收口"，给得太大反而让模型在思考里反复回炉。
- **生成 / 写作类任务**：24~32。需要边取证边写，预算紧张会导致段落被砍。
- **chat 场景**：瓶颈是延迟，不是上限。单 turn 上限存在的目的是**强制模型早点回话**，复杂问题靠多 turn 拆解。
- **超出上限的处理由调用者决定**：写作流水线的 `infer` 失败会回退到未裁剪写作；`prompt`/`interactive` 由用户决定是否追问；不要在 manifest 里堆"以防万一"的 budget。
- **不要为单一 case 整体抬高上限**。如果某个 scene 在某些场景下规律性触顶，先排查是否陷入工具循环（reasoning 已得到结论但仍在调工具），而不是加预算掩盖。

## 修改流程

1. 改 `runtime.agent.max_iterations` 后，必须更新本表对应行。
2. 调整 scene 的工具范围（`tool_selection`）或职责（`description`）时，同步检查上限是否仍合理。
3. 新增 scene 时，在表中增加一行，并写明任务性质与设定理由。
