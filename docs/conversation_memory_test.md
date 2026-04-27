# Interactive Scene 多轮会话记忆实测 Prompt 集

## 目标

实测验证 issue #48 优化后的两层记忆结构在真实财报对话中的表现。重点观察四件事：

1. **pinned_state 是否单调演进**（公司/期间/口径不漂移）。
2. **compaction 何时触发**（约在 `0.70 * max_context_tokens` 处）。
3. **追问连续性**（最近 N 轮 raw turn 强制保留的反退化保底是否生效）。
4. **confirmed_facts 是否在 compaction 后仍可被引用**（跨轮反幻觉）。

启动方式：

```bash
source .venv/bin/activate
python -m dayu.cli interactive --scene interactive
```

每组测试结束后，建议结合 `output/tool_call_traces/` 与日志中 `compaction` 关键字交叉确认。

---

## 测试组 A：pinned_state 演进与抗漂移

**目的**：确认 `current_goal` / `confirmed_subjects` / `user_constraints` 在多轮中只被增量补丁修改，不被覆盖丢失。

观察项：
- 第 5 轮回答里是否还知道"贵州茅台 600519.SH"是当前主体。
- 第 6 轮换公司时，pinned_state 是被替换还是被混淆。

```text
1. 我想看看贵州茅台 2024 年半年报的营收增长结构。
2. 用百万元口径展示，按产品系列拆分。
3. 茅台酒和系列酒分别同比增长多少？
4. 毛利率有变化吗？
5. 把刚才提到的产品系列对应的销量也一起列出来。
6. 切换到五粮液 2024 半年报，做同样口径的产品系列拆分。
7. 回到茅台，刚才你给的茅台酒系列酒同比增速我再确认一遍。
```

**预期**：第 7 轮"回到茅台"时仍能拿出与第 3 轮一致的同比增速，不能给出新数字。第 5 轮已是触发 compaction 的临界点，第 7 轮压缩后的 episode summary 应记录"茅台 2024H1 营收口径已确认 / 用户要求百万元"。

---

## 测试组 B：追问连续性（最近 N 轮 raw 强制保留）

**目的**：验证 `recent_turns_floor=2` 在 budget 紧张时仍能保住"上一轮 + 上上轮"完整文本。

观察项：
- 第 3 轮和第 5 轮的"那这个"指代是否能正确解析为上一轮的具体表述。

```text
1. 拉一下宁德时代 2024 年报里现金流量表的关键数据。
2. 经营性现金流净额是多少？
3. 这个数和净利润比，差异在哪个项目最大？
4. 投资活动的支出主要花在什么上？
5. 那这部分支出和扩产计划匹配吗？
6. 如果按 IFRS 口径重看一次，关键数据有差别吗？
```

**预期**：第 3、5 轮代词解析无错。第 6 轮触发 `user_constraints` 增加"IFRS 口径"，pinned_state 从此带这条约束。

---

## 测试组 C：单轮极长输入的 minimum_preserve 兜底

**目的**：测试单轮 user_text 远超预算时，`_build_minimum_preserved_turn_view` 是否兜底保留 user_text、降级 assistant。

观察项：
- 第 2 轮粘贴长披露文本后，第 3 轮的追问是否仍能基于第 2 轮的内容连贯回答。

```text
1. 我准备分析比亚迪 2024 半年报的毛利率结构变化。
2. （粘贴一份 8000-15000 字的官方披露原文片段，比如 MD&A 中关于毛利率影响因素的整段描述）
   基于以上原文，给我提炼影响毛利率的三个最重要因素，按重要性排序。
3. 第二个因素能再展开讲讲吗？
4. 这三个因素和你之前给我的拆分口径一致吗？
5. 把这次讨论的毛利率结论和测试组 A 里茅台的毛利率对比一下。
```

**预期**：第 3 轮"第二个因素"的指代能正确还原。第 5 轮可以发现两家公司毛利率口径已经在不同 pinned_state 中分别保存（如果是新开 session）或被 confirmed_facts 同时记录。

---

## 测试组 D：compaction 触发与 confirmed_facts 跨轮一致性

**目的**：验证 compaction 触发后，关键事实通过 episode_summary.confirmed_facts 跨轮保留，新轮不重复检索。

观察项：
- 在第 8-10 轮会观察到日志中 `准备同步压缩 transcript` / `检测到 compaction 候选`；
- 第 12 轮再问"刚才确认过的数据"时，模型应基于 episode summary 而非重新调用工具。

```text
1. 帮我看看招商银行 2024 半年报的息差数据。
2. 净息差是多少？
3. 同比变化幅度多少？
4. 生息资产收益率分项给我列一下。
5. 计息负债成本率呢？
6. 资产端的零售贷款占比变化怎么样？
7. 负债端定期存款占比呢？
8. 把这些数据按"资产 / 负债 / 息差"分三组重新组织一下。
9. 哪一组对净息差下行贡献最大？
10. 给个一句话结论。
11. 我换个问题：招行的不良率怎么样？
12. 回到刚才息差讨论，净息差的具体数值再确认一次。
13. 这次确认的数和第 2 轮一致吗？
```

**预期**：
- 第 8-10 轮时上下文用量逼近 `0.70 * max_context_tokens`，触发 compaction。
- 第 12 轮回答应直接基于 episode summary 的 confirmed_facts，**不再重新工具调用**。
- 第 13 轮答案应与第 2 轮完全一致——这是反幻觉核心校验点。

---

## 测试组 E：长会话稳定性（连续 20+ 轮）

**目的**：观察 budget 长期稳定性，验证 episode 数量增长后 budget 路径是否仍能正常裁剪。

操作：选定一家公司（推荐美的集团 000333.SZ 或宁德时代 300750.SZ），围绕其 2024 半年报，按"营收 → 毛利 → 费用 → 利润 → 资产 → 负债 → 现金流 → 估值 → 同行对比"的展开顺序，每个主题问 2-3 个具体问题。目标 25-30 轮。

观察项：
- 每隔 5-7 轮触发一次 compaction，episode 数稳定增长。
- pinned_state 中 `confirmed_subjects` 不重复添加同一公司，`user_constraints` 不丢失。
- 第 25 轮提问"我们这次对话定下了哪些口径约束？"时，模型应能列全 user_constraints。

---

## 验证清单（每组测试结束后核对）

- [ ] `output/tool_call_traces/` 中查到本次 session 的 trace。
- [ ] 日志中至少出现一次 `准备同步压缩 transcript` 或 `开始后台 compaction`。
- [ ] 用 sqlite3 打开 `.dayu/host/dayu_host.db`，查 `conversation_transcripts` 表中本 session 的 `pinned_state` JSON，确认 `current_goal` / `confirmed_subjects` / `user_constraints` 不为空且单调演进。
- [ ] 测试组 D 第 13 轮答案与第 2 轮严格一致（数值不漂）。
- [ ] 测试组 A 第 7 轮答案与第 3 轮严格一致。

## 调参信号判读

实测中如果观察到以下现象，对应调整方向：

| 现象 | 调整 |
|---|---|
| 追问频繁忘上一轮（最近 1-2 轮内容丢失） | `memory_token_budget_cap` 小步上调（每次 +8K），或确认 `recent_turns_floor` 仍为 2 |
| compaction 频繁触发拖慢响应 | `compaction_trigger_context_ratio` 0.70 → 0.80 |
| 当前问题被旧内容淹没 | `compaction_trigger_context_ratio` 调到 0.60，提早压缩旧轮 |
| confirmed_facts 数据跨轮不一致 | 检查 `prompts/scenes/conversation_compaction.md` 抽取约束，而非动 budget |
| 单轮极长输入后追问断链 | 检查 minimum_preserve 路径，必要时把 `recent_turns_floor` 升到 3（仅在极端用例） |
