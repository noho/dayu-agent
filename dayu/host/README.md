# Host 开发手册

> 面向对象：参与 dayu 开源开发的贡献者。
>
> 本文只写 **设计意图、稳定契约、机制与状态机**；具体实现类名/字段名随代码演进，以 `dayu/host/` 源码为准。

---

## 1. Host 的定位

在分层架构 `UI -> Service -> Host -> Agent` 中，Host 居中：

- **对上**（Service / UI）暴露"会话 + 运行 + 回复投递"的稳定门面，屏蔽 Agent 引擎细节与并发/持久化细节。
- **对下**（Agent / Engine）提供受托执行环境：准备输入、下发 run、收集事件、落库、兜底清理。
- **不做**：不做业务决策（不解释财报、不挑 prompt）、不拼装 UI 回复、不直接与 LLM 交互。

一句话：**Host 是"多会话、多租户、可中断、可恢复、可治理"的运行时底座**。Agent 是"这一次运行里干什么"，Host 是"所有运行合在一起怎么跑、怎么停、怎么捡回来"。

---

## 2. 九项能力地图

Host 对外承担九项稳定能力：

| 能力 | 稳定契约（对上） | 内部机制关键词 |
| --- | --- | --- |
| Session 管理 | 按 `SessionSource` 创建/确认/关闭/列举会话，活性屏障保护写入 | 会话活性门 + 批量级联清理 |
| Run 生命周期 | 创建 / 查询 / 取消 / 订阅事件 / 终态落库 | 7 态状态机 + 终态守门 |
| 并发治理 | 按 lane 申请/释放 permit，多 lane 原子批量申请 | ConcurrencyGovernor + lane 合并顺序 |
| 事件发布 | 订阅 run/session 事件流（流式 + 终态） | 事件总线 + 订阅者解耦 |
| Timeout 控制 | 注册 run 截止时间，到点触发 cancel | Deadline watcher + 取消桥 |
| Cancel 控制 | 双层语义：取消意图 vs 终态落库 | CancellationToken + CancellationBridge |
| Resume | 按 pending turn lease 重发用户轮，受 max_attempts 门控 | CAS lease + attempt_count 门控 |
| 多轮会话托管 | Pinned State + 单总池（最近 N 轮 raw + 老 raw + episodic summary）两层记忆 | 同步裁剪 + 后台 compaction + 乐观锁 |
| Reply outbox | 对外回复的"至少一次 + 幂等 + 失败可重试" | 5 态状态机 + delivery_key 幂等 |
| Agent replay | 在上一次执行的完整对话历史末尾追加一条 user 消息再跑一次 | 进程内 replay stash + opaque `ReplayHandle` + engine 层 `_tools_disabled` 复用 |

**稳定契约 vs 当前默认实现**：

- **稳定契约**：能力语义、状态机、事件类型、错误分类（如 `SessionClosedError`、`PendingTurnResumeConflictError`）、配置键名。下游代码只应依赖这一层。
- **当前默认实现**：SQLite 持久化、内存事件总线、in-process 并发治理、threading 执行器。随版本演进可替换。

---

## 3. 公共入口与装配

Host 的公共导出极窄：

- `Host`：门面类，集中暴露能力 3；
- `HostExecutorProtocol`：执行器协议，被 Host 调用，由 Service/UI 装配具体实现；
- `ResolvedHostConfig` + `resolve_host_config(...)`：Host 启动配置的规范化入口。
- `HostedRunSpec` / `ExecutionHostPolicy`：跨层稳定宿主执行契约；Service 只可
  通过其中的 ``business_concurrency_lane`` 与 ``concurrency_acquire_policy``
  声明业务意图，不能依赖 Host 内部 governor 细节。

配置层约定（由 `resolve_host_config` 规范化，拒绝旧顶层键）：

```
host_config:
  store: { path: ".dayu/host/dayu_host.db" }          # Host 存储位置
  lane:  { default: 1, writer: 2, ... }               # UI 级 lane 覆盖
  pending_turn_resume:    { max_attempts: 3 }         # resume 尝试次数上限
  pending_turn_retention: { retention_hours: 168 }    # UI 未询问窗口的兜底删除阈值（7 天）
  cancellation_bridge:                                 # 跨进程取消桥配置
    poll_interval_seconds: 0.5                         # 轮询 SQLite run 状态的间隔
    failure_grace_period_seconds: 5.0                  # 连续失败容忍窗口；超出后 bridge 停止
```

lane 合并顺序（后者覆盖前者）：

1. Host 内置 `DEFAULT_LANE_CONFIG`（仅 Host 自治 lane）
2. Service 启动期注入的业务 lane 默认值
3. `run.json.host_config.lane`
4. UI/CLI 启动期传入的显式 lane 覆盖

这一顺序保证"Host 不感知业务 lane 语义、上层能覆盖底层但底层能兜底所有 lane 定义"。

---

## 4. Session 状态机

```
           create / ensure
    ┌──────────────────────►  ACTIVE  ◄──────── touch (刷新 last_activity_at)
    │                           │
    │                       cancel_session
    │                           ▼
 （创建）                     CLOSED  ───────► 不再接受任何写入
                                        （pending turn / reply outbox / run 新建）
```

- `SessionState` 四态：`ACTIVE` / `CLEARING` / `CLEARING_FAILED` / `CLOSED`（详见 §10.10）。
- `SessionSource` 标识来源（CLI / WEB / WECHAT / GUI / API / INTERNAL），语义上是"谁是这条会话的主使者"。
- **活性屏障**：所有依赖 session 的写入路径（pending turn 写、reply outbox 写、`run_registry.register_run`）统一经过 `_session_barrier.ensure_session_active`——"会话必须处于 ACTIVE"检查；违反即抛 `SessionWriteBlockedError` 子类（`SessionClosedError` / `SessionClearingError` / `SessionClearingFailedError`）。这把 session 从 TOCTOU 竞态里抽出来，由持久层做一致性保证。
- **cancel_session 顺序（稳定契约）**：
  1. 先把 session 置 CLOSED（关门）；
  2. 再批量 cancel 关联的 active run（不再接新活）；
  3. 再幂等清扫同 session_id 下的 pending turn 与 reply outbox（关门后残留物兜底）。

顺序倒置会出现"关门前有新 run 进来"或"清扫后仍能写入"的并发漏洞；该顺序是契约而非实现细节。

---

## 5. Run 状态机

```
       CREATED
          │  enqueue
          ▼
       QUEUED
          │  start
          ▼
       RUNNING
          │
  ┌───────┼─────────────────────────────┐
  ▼       ▼             ▼               ▼
SUCCEEDED FAILED    CANCELLED       UNSETTLED
                    (user/timeout)  (owner-process gone)
```

**关键区分**：

- `FAILED` 是"业务失败"：LLM 报错、工具异常、约束违反——由正常代码路径落库。
- `UNSETTLED` 是"orphan 吸收态"：拥有该 run 的进程没了（kill -9 / OOM / 掉电）。Host 启动时 `cleanup_orphan_runs` 用 `owner_pid` 匹配把这些"没人管的运行中"落到 `UNSETTLED`，避免永远 RUNNING。
- `CANCELLED` 携带 `RunCancelReason`：`USER_CANCELLED` 与 `TIMEOUT` 必须显式区分，决定 pending turn 是否保留（见 §6）。

**取消的两层语义**（稳定契约）：

- **取消意图**：`CancellationToken` 被设置；运行中的 Agent 会在下一个 checkpoint 看到并主动退出。
- **取消终态**：`Run.state = CANCELLED` 被落库。两者时间点可分离——intent 可以立即设置，但终态由实际退出路径写入。

`CancellationBridge` 桥接引擎回调与 Host 取消链；`RunDeadlineWatcher` 在 timeout 到点时设置 intent，然后依赖同一路径走到终态。

**Bridge 失败降级**：底层 `RunRegistry.get_run` 持续抛出非预期异常时，bridge 会按
`failure_grace_period_seconds / poll_interval_seconds` 推导连续失败阈值（至少为 1），
超过阈值后通过 `Log.error` 告知并停止轮询线程，避免在系统性异常下空转。一旦成功
查询一次，失败计数立即清零。两个参数均可在 `host_config.cancellation_bridge` 配置，
默认 `0.5s` / `5.0s`。

**终态守门**：Run 一旦进入 `TERMINAL_STATES`（SUCCEEDED/FAILED/CANCELLED/UNSETTLED）不再接受任何状态转移，也不再触发事件。转移表在 `dayu/contracts/run.py` 中集中。

**`RunRecord.metadata` 字段契约（稳定）**：

- `metadata` 的类型是 **`ExecutionDeliveryContext`**（强类型结构体），**不是** 自由 dict、不是业务参数袋。
- 承载字段仅限"把一次 run 的回复送回正确的外部通道"所需的投递坐标：`delivery_channel` / `delivery_target` / `delivery_thread_id` / `delivery_group_id` / `interactive_key` / `chat_key`。
- 业务参数（模型选择、工具开关、prompt 变量等）走 `ExecutionOptions` / scene preparer，禁止塞进 metadata。这是"Host 不感知业务语义"的硬边界。

---

## 6. Pending turn 状态机（用户轮的"未交付给 Agent"暂存）

Pending turn 记录"用户已经把一轮输入递给了 Host，但 Agent 还没跑完"的中间状态。它是 **resume** 能力的底座。

### 6.1 状态定义

主状态（与 Agent 生命周期对齐）：

- `ACCEPTED_BY_HOST` — Host 已收到用户输入，尚未构造 run；
- `PREPARED_BY_HOST` — 已准备好 run 输入（prompt / 上下文 / 资源），等待排队；
- `SENT_TO_LLM` — 已送入 Agent/LLM 执行中。

正交 lease：

- `RESUMING` — **原子 lease**，表示"某个 resumer 当前正在基于此 pending turn 重发"。lease 期间 `pre_resume_state` 记录 acquire 前的源状态，释放时按需回写。

### 6.2 转换图

```
   新建
    │
    ▼
ACCEPTED_BY_HOST ─► PREPARED_BY_HOST ─► SENT_TO_LLM
                                              │
       ▲                 ▲                    │
       │ release_lease   │ release_lease      │
       │ (restore        │ (restore           │
       │  pre_resume)    │  pre_resume)       │
       │                 │                    │
       └──acquire_resume_lease (CAS)──────────┘
                     │
                     ▼
                  RESUMING ─► delete on success / over-limit
                          └► rebind_source_run_id_for_resume (新 run 接手)
                          └► record_resume_failure / release_resume_lease（回退到 pre_resume_state）
```

### 6.3 Resume acquire 契约（CAS，事务级原子）

一次 `acquire_resume_lease` 必须原子满足：

1. 当前状态 ∈ **acquirable set** = `{ACCEPTED_BY_HOST, PREPARED_BY_HOST, SENT_TO_LLM}`；
2. `attempt_count < max_attempts`（默认 3）；
3. 记录未被其它 resumer 持有。

恢复快照按当前 schema 严格解析：`AcceptedAgentTurnSnapshot.host_policy` 与
`PreparedAgentTurnSnapshot` 都必须携带完整的 `concurrency_acquire_policy`。
缺字段的旧快照会被判为损坏记录，而不是再猜测默认等待语义。

三种失败分别以 `PendingTurnResumeConflictError` 的不同 reason 抛出（冲突 vs 超限 vs 记录缺失 vs 不可恢复），上层据此决定"重试 / 跳过 / 转告用户"。

**超限即删除**（稳定契约）：在同一事务内发现 `attempt_count >= max_attempts`，直接删除该 pending turn，避免进入半永久残留。

### 6.4 Pending turn cleanup 三分支（启动恢复 + 周期性兜底）

`cleanup_stale_pending_turns` 严格按以下分支顺序：

- **分支 A — RESUMING 过期 lease 回退**：`state == RESUMING` 且 `updated_at` 超过 10 分钟 → 释放 lease，按 `pre_resume_state` 回写。保护"resumer 进程中断但 lease 未释放"场景。
- **分支 B — source_run 终态联动**：`source_run` 已是终态时，按 `should_delete_pending_turn_after_terminal_run` 真值表判定：
  - `run is None` → 删除；
  - `state ∈ {FAILED, UNSETTLED}` 且 `resumable=True` → **保留**（等 resume）；
  - `state == CANCELLED` 且 `resumable=True` 且 `reason=TIMEOUT` → **保留**（timeout 属于可恢复）；
  - 其它 → 删除。
- **分支 C — 超保留期兜底删除**：`state ∈ {ACCEPTED_BY_HOST, PREPARED_BY_HOST}` 且 `updated_at` 超过 `retention_hours`（默认 168h=7 天）且 source_run 已终态 → 删除。该分支是 Host 对**分支 A / B 都走不到的长尾记录**的终结契约，保证 pending turn 生命周期自闭环，避免出现"Host 持有但永远不会主动释放"的状态。正常回到会话的路径由自动 resume 覆盖，不依赖 UI 接力。
- **活跃 source_run 严格保留**：任何分支均不得删除"source_run 还在 ACTIVE" 的 pending turn。

分支顺序是契约：A 先于 B 先于 C，避免误清 RESUMING 或误删仍在 active 的记录。

### 6.5 自动 resume 与长尾兜底的分工

Pending turn 的"回到会话"动作不由 UI 触发，也不由分支 C 触发——分支 C 只负责终结记录。Host 与 UI 通道共同构成两条正交防线：

**正常路径 = 自动 resume**。各 UI 通道在**能自然触发的时机**自动走 `acquire_resume_lease → 起 run → release_lease`：

- CLI `interactive` 进入 REPL 前，启动 hook 扫描当前 session 的可恢复 pending turn 并直接续上；
- WeChat daemon 启动时 `_resume_pending_turns()` 全量恢复，之后每条入消息前再做一次 session 级恢复（`fail_fast=True`）；
- Web 在前端显式触发 resume 端点时恢复。

自动 resume 由 Host 侧 `acquire_resume_lease` 的三重门护住：`attempt_count < max_attempts`（默认 3）、RESUMING lease 10 分钟过期回退（分支 A）、`resumable=True` 的 scene 才允许。这几条约束保证"能自动 resume 的都会被自动 resume，失败次数有限、并发安全"。

**长尾 = 自动 resume 永远跑不到**。分支 C 定位为此：

- CLI 用户换了 workspace / 换了话题 / 再也不跑 `interactive`——启动 hook 不会再触发；
- Web 用户关 tab 不再打开；
- WeChat 用户永久沉默——daemon 的两个自动 resume 触发点（启动 / 入消息）都跑不到。

这些记录会停在 `ACCEPTED_BY_HOST` / `PREPARED_BY_HOST`：不是 RESUMING（分支 A 不管）、source_run 终态但 `resumable=True`（分支 B 判为保留，等 resume）。分支 C 的 168h 兜底删就是为这层兜底而存在。

**为什么不在长尾上再做 UI 询问**。这三种"自动 resume 永远跑不到"的情况，本质上是用户已经用脚投票放弃了这轮对话：

- CLI 换 workspace / 换话题：意图很清楚，不想要那个结果；继续追问"要不要重发"反而打扰。
- Web 关 tab 不回来：用户已离场。
- WeChat 沉默：同上。

因此 Host 层不提供"长尾 UI 询问窗口"这种接力机制；分支 C 是**静默终结**，不是"UI 未接力时的兜底"。UI 通道只需保持正常路径的自动 resume；不要在长尾上追加打扰用户的交互。

---

## 7. Reply outbox 状态机（对外回复的"至少一次 + 幂等"）

Reply outbox 存放 Host 需要向 UI/外部通道投递的回复。它把"回复生成"与"回复投递"解耦，使得 UI 重连、进程崩溃、下游接口抖动都可以不丢消息。

### 7.1 状态定义

- `PENDING_DELIVERY` — 待投递；
- `DELIVERY_IN_PROGRESS` — 某 worker 已 claim，正在投递；
- `DELIVERED` — 投递成功（**吸收态**）；
- `FAILED_RETRYABLE` — 本次投递失败但可重入（可再次 claim）；
- `FAILED_TERMINAL` — 永久失败（**吸收态**，幂等 mark_failed 可重复）。

### 7.2 转换图

```
             submit (INSERT OR IGNORE by delivery_key)
                        │
                        ▼
                PENDING_DELIVERY ◄──────────────┐
                        │ claim (CAS)           │
                        ▼                       │
                DELIVERY_IN_PROGRESS ───────────┤
                     │    │                     │ 15min stale 回退
           mark_    │    │ mark_failed         │ (同时打标 STALE_IN_PROGRESS)
        delivered   │    │  retryable=True     │
                   ▼    ▼                     │
              DELIVERED  FAILED_RETRYABLE ────┘
             （吸收态）         │
                               │ mark_failed retryable=False
                               ▼
                         FAILED_TERMINAL
                           （吸收态）
```

### 7.3 关键不变量（稳定契约）

- **claim 谓词**：`claim` 的 CAS 条件是 `state ∈ {PENDING_DELIVERY, FAILED_RETRYABLE}`；两者共享"可再次取出投递"的语义。
- **mark_delivered 谓词**：CAS 条件是 `state == DELIVERY_IN_PROGRESS`；若记录已是 `DELIVERED`，幂等返回，不是错误——这是"同一条回复被重复确认"的正常场景。
- **DELIVERED 拒绝失败转移**：一旦 DELIVERED，`mark_failed` 直接报错；业务层不得反悔。
- **FAILED_TERMINAL 幂等**：重复 mark_failed(retryable=False) 不抛错。
- **delivery_key 幂等**：`submit` 用 `INSERT OR IGNORE` + 同 key 的 payload 一致性检查保证"同一 delivery_key 只入库一次且 payload 未被偷偷改掉"。这是对上游重复提交的幂等防线。
- **Stale in-progress 兜底**：超过 15 分钟仍停在 `DELIVERY_IN_PROGRESS` 的记录，被 `cleanup_stale_reply_outbox_deliveries` 回退到 `FAILED_RETRYABLE` 并打上 `STALE_IN_PROGRESS_ERROR_MESSAGE`，供后续 claim 重入。
- **claim/ack 与 worker 身份解耦**：当前实现不绑定 owner；任何合法 worker 均可推进状态机。lease/owner-token 是可选演进方向，不属于当前契约。

### 7.4 UI worker 契约延伸（非 Host 内部）

Reply outbox 的状态机由 Host 拥有，但"谁把 `PENDING_DELIVERY` 拉出来、如何打给外部通道、失败如何分类"由 UI 层的 worker 决定。以下是 UI worker 必须遵守的契约延伸点（具体实现与重试数值见各 UI 包 README）：

- **投递路径唯一**：UI 必须经由统一的回复投递服务（`ReplyDeliveryService`）走 outbox，禁止绕过 outbox 直接向通道写。
- **DELIVERED 语义**：只有下游通道真正确认收下，才能打 `mark_delivered`；SSE 断连、HTTP 超时均**不**等于 DELIVERED。
- **失败分类**：通道的业务级永久错误（如 IM 返回 ret != 0、HTTP 4xx、缺失投递目标）应落到 `FAILED_TERMINAL`；网络抖动/下游 5xx 走 `FAILED_RETRYABLE`。
- **启动恢复协作**：UI worker 启动时不能自行"清库"，必须依赖 Host 的 `cleanup_stale_reply_outbox_deliveries` 把 stale in-progress 回退后再 claim。
- **Resume 入口统一**：UI 侧触发 pending turn 重发必须走 Host 的 resume 门（acquire lease），禁止自行构造等价 run。

这些延伸契约由 Host 的 outbox/pending turn 语义"自然推出"，但归属在 UI 层文档（WeChat、Web、CLI 各自 README），不是 Host 内部细节。

---

## 8. 并发治理

Host 内建 `ConcurrencyGovernor`，按 lane 限制同时运行的 run 数。

- **lane 合并顺序**见 §3。
- **`acquire_many`**：一次申请多条 lane 的 permit，要么全部拿到、要么一个都不拿（事务性）。用于避免"拿到 A、拿不到 B 然后半开状态"造成的局部死锁。
- **stale permit 回收**：启动恢复阶段会扫描 ``owner_pid`` 已死亡的 permit 残留并直接回收；运行期如果某个 run 在等待 permit 时长期阻塞，governor 也会按节流频率主动回收 dead-PID stale permit，避免把“别的进程刚崩掉”的残留永久留给上层无限等待。
- **等待可取消**：Host 在阻塞等待 permit 时会绑定当前 run 的 `CancellationToken`；`cancel_run` / `cancel_session` 一旦落库，等待中的 acquire 也必须尽快退出并把 run 收敛到 `CANCELLED`，不能继续卡在 governor 轮询里。

Host **不**感知业务 lane 的意义（例如"writer"、"retrieval"）；它只按 lane 名字做计数限流。业务层自己决定分 lane 的粒度。

**Host 自治 lane `llm_api`（稳定契约）**：

- `llm_api` 是 Host 内置的自治 lane，用于限制同一进程内对 LLM API 的**并发调用数**（与业务 lane 正交）。
- **Service 代码禁止**显式写"llm_api"字面量或在 lane 覆盖里指定 `llm_api`；lane 名由 Host 侧常量统一拥有。
- **自动叠加**：agent-stream 执行路径由 `DefaultHostExecutor` **自动**把 `llm_api` lane 叠加到 `acquire_many` 的申请集合中；直接 operation 路径（不经 agent-stream）不自动叠加。
- **等待策略由稳定契约声明**：Service 只能通过 `HostedRunSpec.concurrency_acquire_policy`
  或 `ExecutionHostPolicy.concurrency_acquire_policy` 声明 `host_default` /
  `timeout` / `unbounded` 三种等待意图；Host 再把该意图翻译成 governor
  的实际 `timeout` 参数。写作链路当前统一声明 `unbounded`，其它路径保持
  `host_default`。
- 这一设计让业务方只需声明业务 lane；"避免打爆外部 API"的自治防线由 Host 自动接管，不依赖上层正确配置。

---

## 9. 事件发布

Host 暴露的事件只有两类：

- **Run 流式事件**：来源于 Agent/Engine，被 Host 透明转发给订阅者。Host 不做内容理解，只负责路由 + 终态封口。
- **Session 生命周期事件**：create / close / cancel，用于 UI 做会话列表刷新。

设计约束：

- 订阅接口不暴露内部事件总线类型；上层仅依赖"事件类型枚举 + payload 契约"。
- 终态事件保证"在状态落库之后发出"，订阅者看到的事件顺序与持久化顺序一致。

---

## 10. 多轮会话托管（两层记忆）

多轮会话的上下文由**两层结构**组成：`pinned_state` 独立保留、不参与 token 池竞争；其余历史共享一个**单总池**，按 budget 从新到老回放最近 raw turn 与 episode 摘要。

| 层 | 语义 | 是否计入 budget | 典型内容 |
| --- | --- | --- | --- |
| Pinned State | 会话级反幻觉锚点，写入后增量更新 | 否（独立路径） | `current_goal` / `confirmed_subjects` / `user_constraints` / `open_questions`，典型 < 200 tokens |
| 单总池 | 历史回放预算池 | 是 | 最近 N 轮 raw turn（强制保留下限）+ 更老 raw turn（按预算回放）+ episode summaries（剩余预算填充） |
| Raw Transcript | 原始事件流（独立物理存储） | — | 审计/回放用的完整日志，永不删除 |

**预算公式**：
```
budget = clamp(max_context_tokens * memory_token_budget_ratio,
               memory_token_budget_floor,
               memory_token_budget_cap)
```

`recent_turns_floor` 是反退化下限——即使 budget 算下来为 0，也强制保留最近 N 轮 raw turn 不让追问彻底断链；它**不计入 budget**。极端单轮溢出由 `_build_minimum_preserved_turn_view` 兜底（user_text 全保、assistant 降级裁剪）。

### 10.1 Raw Transcript 的分区策略

Raw transcript 是一个按时间递增的 turn 列表，通过会话级字段 **`compacted_turn_count`** 把列表划分成两个区：

- **已压缩区**：下标 `< compacted_turn_count` 的 turn，语义上"已被摘要进 episode summaries"，**不再参与**后续单总池消费（防止 episode 摘要与原文双发，制造冗余与矛盾）。
- **未压缩尾区**：下标 `>= compacted_turn_count` 的 turn，是单总池中 raw turn 部分的候选池；同时也是下一次 compaction 的输入来源。

`compacted_turn_count` 只由 compaction 成功写入时**单调推进**，杜绝"压缩后再回放原文"。

### 10.2 单总池消费顺序

单总池消费按**从新到老**优先级，对未压缩尾区与 episode summaries 联合裁切：

1. **最近 N 轮 raw turn 强制保留**（`raw_tail[-recent_turns_floor:]`）：不计入 budget，单轮极长走 minimum_preserve 兜底，不丢追问连续性。
2. **Episode summaries 按 budget 从新到老填充**：每条估算 token，能装下就加入并扣减 budget，装不下就 break。
3. **更老 raw turn 用剩余 budget 从新到老回放**：每轮估算 token，能装下就加入并扣减剩余 budget，装不下就 break（更早历史已被 compaction 压成 episode 摘要）。

> 顺序设计意图：episode summaries 是"老 turn 的结构化摘要"，单位 token 信息密度显著高于 raw turn 长尾；先把预算分给 episodes 能在压缩后稳定保住历史覆盖，再用剩余预算尽可能回放原文细节。

**最近一轮整轮超预算时的降级顺序**（稳定契约，保证"最新用户意图永远不丢"）：

1. 保留 user_text；
2. 保留完整的 assistant_final；
3. 丢弃 tool 调用/结果摘要；
4. 仍超预算，则对 assistant_final 做末尾截断，并附显式截断标记（如 `...<truncated>`）以免下游把不完整答案当完整答案。

降级顺序是契约：任何实现不得颠倒"先丢工具后截最终答复"的优先级。

**单轮溢出阈值**：`max_context_tokens / max(2, actual_forced_count + 1)`。除数取**当前实际 forced 轮数**而非 `recent_turns_floor` 配置值——避免新会话/刚 compaction 后只剩 1-2 轮 raw turn 时，配置高 floor 把单轮上限错误压低。

### 10.3 Compaction 触发策略

触发判定**仅按"占模型窗口百分比"单维度**（轮数触发器已废）：

```
window_used = system_prompt + pinned_state
            + actual_episodic_in_prompt
            + uncompressed_raw_turns + current_user_text
should_compact = window_used > max_context_tokens * compaction_trigger_context_ratio
```

`actual_episodic_in_prompt` **复用 `_build_memory_block` 的真实裁切结果**，与渲染 prompt 的口径完全一致，避免触发器与渲染器漂移导致 thrashing 或漏触发。

同时**始终保留 `compaction_tail_preserve_turns` 轮不参与压缩**，保护当前对话的连续性与用户体感。

### 10.4 Compaction 的输入/输出语义

**输入**（给 LLM 的压缩上下文）语义上包括：

- 固定的压缩任务指令；
- 当前 pinned_state（让 LLM 知道会话主线与稳定偏好）；
- `compaction_context_episode_window` 个最近已有 episode 摘要（维持"摘要风格的延续感"，避免每轮重写）；
- 本次待压缩的 turns（来自未压缩区但保留尾部以外的那一段）。

**输出**两件事：

- **`episode_summary`**：对本段对话的结构化摘要，追加进 episode summaries；包含 `confirmed_facts` 字段——财报场景反幻觉刚需，永远完整保留不参与 token 截断。
- **`pinned_state_patch`**：对 pinned_state 的**增量补丁**。合并语义（`apply_to`）是"只覆盖明确给出的字段、缺省字段沿用旧值"——保证 LLM 每次不需要重述整个 pinned_state，也不会因为漏输出某字段而把旧值清空。

"先得到结果、再按乐观锁写回"是语义分离点：得到 patch 不等于已落库。

### 10.5 消息组装的四段固定顺序

每轮发给 LLM 的消息列表按以下**固定顺序**拼接，顺序是契约：

1. **System Prompt**：角色与任务描述；
2. **Conversation Memory 段**：pinned_state + episode 摘要，统一以一条 system 级消息承载，给模型"这是背景而不是对话"的信号；
3. **Working Memory 段**：§10.2 选中的若干轮 raw turn，按 user/assistant 原始交替回放；
4. **当前轮 user message**：最新用户输入。

这一顺序保证：背景信息在对话之前、历史对话保真回放、当前意图在最末尾（减少 LLM 忽略当前指令的概率）。

### 10.6 同步与后台 Compaction 的调度策略

Compaction 有两条触发路径，职责分离：

- **同步 compaction**（`prepare_transcript`）：**在本轮开始前**，如果未压缩区已越过阈值（消息列表若不压缩就会超预算），同步执行压缩，确保本轮能立即开跑。这条路径**显式接收 `user_text`** 作为入参——本轮用户消息尚未持久化，必须显式传入参与 `window_used` 估算。
- **后台 compaction**（`schedule_compaction`）：**在上一轮结束后**，越过阈值则异步调度。这条路径**不接收 `user_text`** 入参——调用时机在 `persist_turn` 之后，用户消息已写入 `transcript.turns` 末位，自然出现在 `uncompressed_raw_turns` 中，再传 `user_text` 会双计。

两条路径共享同一套输入/输出语义（§10.4），差异只在调度时机与是否显式传 `user_text`。

**乐观锁并发冲突策略**：compaction 的写回以会话级 **revision** 作为乐观锁——读入时记录 revision，写回前比对；若不一致（说明期间已有其它 compactor/会话写入），**直接丢弃本次摘要结果，不覆盖**。这与"后来者胜"相反：在"继续跑"与"保持一致"之间选择一致。丢弃的代价是下次重算，可接受。

### 10.7 层间关系的稳定不变量

- **Pinned 永不被 compaction 挤掉**，只由显式 API 或 compaction 的 `pinned_state_patch` 改写；其它层都可以在预算压力下被重塑。
- **Episode 只能追加**，不支持回改；若 episode 本身需要再浓缩（episode-of-episodes），属于演进方向，不在当前契约内。
- **`confirmed_facts` 永远完整保留**，不参与 token 截断（财报场景反幻觉核心依赖）。
- **Raw Transcript 永不被 compaction 物理删除**，只通过 `compacted_turn_count` 标记为"已摘要"；审计/回放总能拿到全量原文。

### 10.8 默认配置与跨档自适应

`default` 一份打天下，不分档配置；通过 `clamp(window * ratio, floor, cap)` 自动伸缩：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `memory_token_budget_ratio` | `0.10` | 池大小占模型窗口比例；按当前 floor/cap 在 ~40K–600K 窗口范围内真实生效 |
| `memory_token_budget_floor` | `4000` | 总池下限保底（短窗口模型用） |
| `memory_token_budget_cap` | `60000` | 总池上限封顶（1M 档被截到此值） |
| `recent_turns_floor` | `2` | 最近 N 轮强制保留下限，反退化兜底 |
| `compaction_trigger_context_ratio` | `0.70` | 占模型窗口百分比触发阈值 |
| `compaction_tail_preserve_turns` | `4` | 压缩时保留尾部不压的轮数 |
| `compaction_context_episode_window` | `2` | 给 LLM 的近邻 episode 数 |
| `compaction_scene_name` | `"conversation_compaction"` | 压缩用专用 scene |

**各档实际 budget**：

| 模型窗口 | `window * 0.10` | clamp 后 budget | 触发阈值 |
| --- | --- | --- | --- |
| 1M | 100K | **60K**（cap 截断） | 700K |
| 256K | 25.6K | **25.6K**（区间内） | 179K |
| 128K | 12.8K | **12.8K**（区间内） | 89.6K |
| 32K | 3.2K | **4K**（floor 兜底） | 22.4K |

#### 10.8.1 128K 档使用注意事项

128K 档自适应**功能上完全成立**，但相比 256K/1M 档对"材料 headroom"更敏感：

- **Compaction 阈值 89.6K**，扣除 system_prompt + pinned + 12.8K 总池后，留给"当前轮工具结果 + 财报材料"的有效空间约 **38K**——比 256K 档的 154K 小一个量级。
- 若单轮工具一次取出多块 XBRL fact + 长财报段落 > 38K，会出现"每轮都触发 compaction"的抖动现象。
- **调参方向**（按抖动现象选）：
  - 抖动严重：`memory_token_budget_cap` 下调（如 8000）把池让给材料；或 `compaction_trigger_context_ratio` 上调（如 0.80）延后触发。
  - 追问频繁忘上一轮：保持 cap，调高 `recent_turns_floor` 到 3。
- **单轮溢出阈值 ≈ 128K / 3 ≈ 42K**（floor=2、actual_forced=2 时），财报追问中 user_text 几百字 + assistant 摘要几千字远低于此值，正常对话不会触发 minimum_preserve。
- **不需要在 `llm_models.json` 写 `runtime_hints.conversation_memory` 覆盖**——`default` 已自适应；如确需档位特化，建议先按上述抖动信号调参，验证默认行为不够再考虑覆盖。
- **当前 `dayu/config/llm_models.json` 仅注册 1M / 256K 档**，128K 档需通过 `dayu-cli init` 添加自定义模型进入；上线前建议跑一次 `interactive` 8+ 轮实测，观察 trace 里的 budget/compaction 日志确认行为符合预期。

具体的 token 预算数值、触发阈值倍数定义在 `dayu/config/run.json`、`dayu/contracts/execution_options.py::ConversationMemorySettings` 与 `dayu/host/conversation_memory.py`，三处默认值通过单元测试 `test_default_conversation_memory_settings_match_runtime_default` 保持一致。详细字段说明见 [config/README.md §5.8](../config/README.md)。

### 10.9 会话存储（ConversationSessionArchive）

会话级真源被收敛到一个聚合根 `ConversationSessionArchive`，定义在 `dayu/host/conversation_session_archive.py`，物理落盘在 `<workspace>/<CONVERSATION_STORE_RELATIVE_DIR>/<session_id>.json`。聚合根包含两个**逻辑分离、物理同写**的子视图：

| 子视图 | 内容 | 谁能读 | 谁能写 |
| --- | --- | --- | --- |
| `runtime_transcript` | 运行态真源（`ConversationTurnRecord` 列表 + `compacted_turn_count` + memory layers） | 送模/memory/compaction/resume/prepared snapshot | `with_next_turn`（新轮）/ `with_runtime_transcript`（compaction 写回） |
| `history_archive` | 仅供历史展示的扁平副本（`ConversationHistoryTurnRecord`，含 `assistant_reasoning`） | 历史展示链路 | 仅 `with_next_turn`（新轮同步推进） |

**单聚合 / 单 revision / 单文件原子提交**：`runtime_transcript` 与 `history_archive` 共享同一 `archive.revision`，落盘走 `FileConversationSessionArchiveStore.save(archive, expected_revision=...)`：取文件锁 → 校验 revision → 临时文件 → fsync → replace → fsync 父目录。任一失败 → 整个聚合写失败、磁盘旧版本完整保留。**不**做 sidecar 双写、不做两阶段提交。

**首轮装配走 `load_or_create`，不用 `save(expected_revision=None)`**：`scene_preparer.prepare` 在 session 第一次进来时通过 `archive_store.load_or_create(session_id)` 拿到真源——该方法在文件锁保护下做 load-or-create，先到的胜出、后到的拿到 live archive。这条路径**严禁**回退到"先 `load` 看是否存在再 `save(create_empty, expected_revision=None)`"——`save` 只在 `expected_revision is not None` 时做冲突校验，TOCTOU 窗口里的并发 prepare 会把另一进程已经写好的 live archive 直接覆盖成空文件。

**`from_dict` fail-closed**：`ConversationSessionArchive.from_dict` 对 `history_archive` 字段缺失或非对象**直接抛 `ValueError`**，不静默降级为 `create_empty`。理由：`history_archive` 是聚合根的物理同写组成，缺失只可能是数据损坏或迁移不完整；若降级为空历史，下一次 `persist_turn` / compaction 写回会把这份空历史持久化覆盖，永久抹掉已有 `assistant_reasoning`。损坏数据必须走 migration / repair 路径修复，不通过运行态读取被静默吞掉。

**结构边界（硬约束）**：

- `runtime_transcript` / `ConversationTurnRecord` 字段集**不含** `assistant_reasoning`。
- `history_archive` 内容**不进入** prepared snapshot / `resume_source_json` / `restore_prepared_execution` / `to_messages` / compaction 输入；它纯粹是展示侧字段。
- 模块结构隔离由 `tests/application/test_history_archive_isolation.py` 反射式回归——`dayu/host/conversation_memory.py` / `pending_turn_store.py` / `prepared_turn.py` / `dayu/contracts/agent_execution.py` / `agent_execution_serialization.py` / `agent_types.py` 六处源码不得出现 `assistant_reasoning` / `ConversationHistoryTurnRecord` / `ConversationHistoryArchive` 三个 token。

**历史读 read model（`#116`）**：`Host.list_conversation_session_turn_excerpts(session_id, *, limit)` 是历史展示的**唯一**对外读入口，返回 `ConversationSessionTurnExcerpt(user_text, assistant_text, reasoning_text, created_at)`，按时间从旧到新排列。读源**只**是 `archive.history_archive.turns`——禁止从 `runtime_transcript` 投影"近似历史"。`reasoning_text` 映射自 `assistant_reasoning`，无 reasoning 的轮次为空字符串；archive 缺失 / JSON 损坏 / schema 非法（含旧 transcript 未迁移） / `limit <= 0` 一律降级返回空列表（损坏路径同时打 warning）。`Host.get_conversation_session_digest` 共享同一 fail-soft 加载入口（`_safe_load_archive_for_read`），异常一律按"无 archive"语义处理。**写路径不受此降级影响**，仍走 `archive_store.load` 的 fail-closed 行为，避免静默覆盖损坏数据。`reasoning_text` 命名上刻意区别于内部 `assistant_reasoning`，提示上层这是展示视图，绝不流回送模 / resume / memory / compaction。回归用例见 `tests/application/test_host_list_conversation_excerpts.py`。

**reasoning 的采集与落盘**（`#118` 关键链路）：

1. Engine 层流式产出 `EventType.REASONING_DELTA`。
2. Executor 在 `agent_input.session_state` 存在时调 `session_state.record_reasoning_delta(text)`，否则静默丢弃。
3. `ConversationSessionState` 把 chunk 累计在内部 `_reasoning_buffer`。
4. 在 `persist_turn(...)` 时把 `"".join(_reasoning_buffer)` 一次性投影到 `ConversationHistoryTurnRecord.assistant_reasoning`，与新 `ConversationTurnRecord` 一同走 `with_next_turn` 推进聚合根，再一次原子落盘；落盘成功才清空 buffer，失败重试不丢内容。
5. `ConversationTurnPersistenceProtocol.persist_turn` 签名**不变**——展示字段不污染契约。

**resume：输入只信 prepared snapshot；输出走 reconcile-on-write**

- **输入**：`restore_prepared_execution` 仍只从 `PreparedConversationSessionSnapshot.transcript` 重建运行态——prepare 与 resume 之间外部状态变化（清空、compaction、文件丢失）**不影响** resume 决策。
- 重建 `current_archive` 时构造**临时 placeholder archive**：`runtime_transcript = snapshot.transcript`、`history_archive = create_empty(session_id)`、`revision = ""`。`revision == ""` 是路径分叉标记位。
- **输出**：`persist_turn` 路径分叉
  - `current_archive.revision == ""` → 走 `archive_store.append_turn(session_id, *, turn_record, history_record)`：取文件锁 → load live → live 缺失抛 `ConversationArchiveMissingError` → 在 live 上 `with_next_turn` → 原子写。**不**预设"自动 create_empty"，缺失时显式报错（清空 vs pending turn 并发处置策略由 `#117` 决定）。
  - 否则（正常路径）→ `with_next_turn` + `save(... expected_revision=current_archive.revision)`。

**compaction：两层 stale-check（业务层 + 物理层）**

- **业务层**：比较 `live.runtime_transcript.revision` 与 `compaction_input_transcript.revision`，不一致 → 丢弃结果（依靠下一轮 turn 触发 schedule_compaction 自愈）。这一层**不能**被偷换成"只看 archive 乐观锁"。
- **物理层**：`archive_store.save(next_archive, expected_revision=live.revision)`。
- compaction 任何路径**不修改** `history_archive`，仅通过 `with_runtime_transcript` 替换运行态子视图。

**旧 schema 迁移**：`dayu-cli init` 走 `dayu/cli/workspace_migrations/conversation_archive_init.py`：识别"顶层缺 `runtime_transcript` 但含 `turns`"的旧 transcript 文件，原地包成 archive；`history_archive.turns` 全量从旧 turns 投影（`assistant_reasoning=""`）。损坏文件 warning 跳过、新 schema 文件 no-op。`FileConversationSessionArchiveStore.load` 遇到旧 schema 直接抛 `RuntimeError` 提示运行迁移，**不**做兼容读取。

### 10.10 清空会话历史（`#117`）

`Host.clear_session_history(session_id)` 在不关闭 session 的前提下原子清空"五真源"——`history_archive.turns` / `runtime_transcript.turns` + memory / `pending_turn_store` / `reply_outbox_store` / executor replay stash——清完后 session 仍 `ACTIVE`，下一轮可继续；`#116` 历史读返回 `[]`。

**`SessionState` 状态机扩展**：`{ACTIVE, CLEARING, CLEARING_FAILED, CLOSED}`。`is_session_active` 收紧为"仅 ACTIVE"，三类写入路径——`pending_turn_store` / `reply_outbox_store` 写入、以及 **`run_registry.register_run`**——统一经 `_session_barrier.ensure_session_active` 检查：`CLEARING` 抛 `SessionClearingError`、`CLEARING_FAILED` 抛 `SessionClearingFailedError`、`CLOSED` 抛 `SessionClosedError`。三者共享基类 **`SessionWriteBlockedError(RuntimeError)`**，便于上层用单条 `except` 统一吸收，同时 `type()` / `isinstance()` 仍可区分子类用于 observability。

**屏障吸收链路**：executor 的 `_register_accepted_pending_turn` / `_register_prepared_pending_turn` 在 scene prepare 与 cancel 时间窗内的迟到登记统一 `except SessionWriteBlockedError`，降级为 no-op（日志携带 `barrier=type(exc).__name__` 区分子类）；Host 的 `resume_pending_turn_stream` 在 lease acquire / lease 回退两处也统一 `except SessionWriteBlockedError`，对外收敛为 `ValueError("...session 已不再接受写入...")`，`__cause__` 保留原始屏障异常便于诊断。这条链路是"清空 / 关停期间防止 executor 迟到写入产生孤儿数据"的最终防线，不依赖 archive 文件锁。

**写入顺序（强约束）**：

1. `SessionRegistry.begin_clearing` — `ACTIVE → CLEARING`；非 `ACTIVE` 直接 `ConversationClearRejectedError`。
2. 锁内拒绝预检：active run / pending turn / reply outbox 任一非空 → `ConversationClearRejectedError`，**不**自动 cancel。
3. `archive_store.save(empty, expected_revision=live.revision)`；`with_runtime_transcript` 与 `history_archive.create_empty` 共用同一 revision。`save` 抛"revision 冲突" → `ConversationClearStaleError`（场景 b：compaction 写回竞速）。**这是清空成功的唯一切点**。
4. archive 写成功后，对 pending turn / reply outbox / replay stash 做幂等 `delete_by_session_id` / `discard_replay_state_for_session`；任一失败进入有界 retry（3 次、固定 50ms 间隔）。
5. retry 仍未收敛 → `mark_clearing_failed`（`CLEARING → CLEARING_FAILED`，**持久锁定**），抛 `ConversationClearPartiallyAppliedError`，错误体携带 `residual_sources`；session 写入面持续锁定直至外部修复路径（reopen / cancel）触发，不在 `#117` 范围。
6. 全部收敛 → `end_clearing`（`CLEARING → ACTIVE`，释放屏障）。

**失败语义（分层契约）**：

| 时间窗 | 失败类型 | 五真源状态 | session 状态 |
| --- | --- | --- | --- |
| archive 写之前 | `ConversationClearRejectedError` / `ConversationClearStaleError` | 全部不变 | `CLEARING → ACTIVE`（屏障释放） |
| archive 写之后、其他真源未收敛 | `ConversationClearPartiallyAppliedError` | archive 已空，其余可能残留 | `CLEARING → CLEARING_FAILED`（持久锁定） |

**调用方契约**：`ConversationClearPartiallyAppliedError` **不可重试**——清空内部已尽力 retry，再调一次会被 `CLEARING_FAILED` 屏障拒；上层应升级观测告警 / 人工介入。

回归用例见 `tests/application/test_host_clear_session_history.py`、`tests/application/test_session_registry.py::TestClearingBarrier`、`tests/application/test_session_barrier.py`、`tests/application/test_run_registry.py::TestRegisterRunSessionBarrier`。

---

## 11. 场景准备（scene preparation）

Host 内部的 `scene_preparer` 把"这一次运行需要哪些材料"集中起来：prompt 资产、模型配置、工具 schema、执行选项。它是 Host 内部 public 模块（Service/UI 可见但不应绕过 Host 直接调用 Agent）。

设计理由：让 Agent 启动参数在 Host 边界内完成规范化，避免 UI/Service 直接把"未完成的半成品输入"下发到 Engine。这也是所有 run 之所以能被 Host 重建（resume）的前提——材料来源稳定，所以同一 pending turn 可以重新构造等价的 run 输入。

---

## 11b. Agent replay 能力

Service 在收到一次 Agent 输出但发现"格式脏 / 空文本"等纯解析失败时，需要在原对话历史末尾追加一条提示再跑一次，避免重发 prompt 把已有取证全部丢弃。Host 暴露的 replay 能力就服务于这一刚需，约束如下：

- **接口**：`Host.run_agent_and_wait_replayable(contract) -> (AppResult, ReplayHandle)` 与 `Host.replay_agent_and_wait(handle, contract) -> (AppResult, ReplayHandle)`，同步暴露在 `HostExecutorProtocol` 上。普通 `run_agent_and_wait` 行为不变，向后兼容。Service 层通过 `Host` facade 调用，禁止访问 `_executor` 私有字段。
- **handle 不透明**：`ReplayHandle` 仅承载 `handle_id` 字符串，**不**暴露 `AgentMessage` 等 engine 数据结构；完整对话历史由 Host 在内部 stash 中持有。
- **生命周期内存约束**：handle 仅在颁发它的 Host 实例进程内有效。Host 重启、`cancel_session` 调用，或同一 handle 已被 `replay_agent_and_wait` 一次性消费，都会让句柄失效，再次使用抛 `RuntimeError`。
- **跨 session 拒绝**：replay contract 的 `host_policy.session_key` 必须与原 handle 的 session 一致。
- **追加策略**：Host 取出原 messages，把 `contract.message_inputs.user_message` 包成 user message 追加到末尾，再走 `AsyncAgent.run_messages`；replay 路径基于上一次的 `AgentInput` 通过 `build_async_agent` **重建** `AsyncAgent`，并把 runner 绑定到本次 `_start_run` 颁发的新 `CancellationToken`——复用上一次的 `AsyncAgent` 会让取消桥 / deadline watcher 形同虚设，因此不可复用。
- **取消语义**：replay 期间若已被请求取消（用户主动 / deadline timeout），即使模型流自然结束也会把 run 收敛为 `CANCELLED` 并向 caller 抛 `CancelledError`，不再返回 `AppResult`，避免上层把"中途取消"误读为"成功的脏数据"。
- **可选禁用工具**：`contract.message_inputs.replay_disable_tools=True` 时透传到 engine 的 `run_messages(disable_tools=True)`，强制本次只能输出文本。该路径与 engine 的 `force_answer` 共享同一 `_tools_disabled` 上下文管理器，不会污染后续运行的 runner 工具状态。
- **不重试失败 replay**：Host 不在 replay 内部对脏数据做二次 replay；新 handle 仍然返回，但具体重试策略归属 service 层。
- **不污染 AppResult**：`AppResult` 不增加任何字段，跨层只暴露 handle，避免上层误以为可以离线重放或自己重建历史。

---

## 12. 启动恢复契约

Host 启动时按固定顺序执行：

1. **`cleanup_orphan_runs`** — 把上轮宿主进程遗留的 RUNNING run 吸收到 `UNSETTLED`（按 owner_pid 精确匹配）；
2. **`cleanup_stale_permits`** — 回收 ``owner_pid`` 已死亡的 lane permit；
3. **`cleanup_stale_reply_outbox_deliveries`** — 15 分钟 in-progress 回退到 `FAILED_RETRYABLE`；
4. **`cleanup_stale_pending_turns`** — 执行 §6.4 的三分支（A → B → C）。

顺序是契约：先把"谁还活着"定死（run 吸收），再依此判定其它子系统里谁可以动（permit / outbox / pending turn）。

运行期另有 `cancel_run_and_settle(run_id)`：由 `dayu.process_lifecycle` 协调器在 SIGINT/SIGTERM/SIGHUP/atexit 钩子里调用，把指定 active run 同步推到 `CANCELLED` 终态、清理关联 pending turn，并通知默认 executor 释放该 run 占用的 `(bridge, deadline_watcher, permits)`，避免 sync CLI Ctrl+C 后 run 卡 `running`、pending turn 卡 `prepared_by_host` 阻塞下一轮 prompt，同时避免 permit 泄漏与守护线程残留。

> **资源释放契约**：`KeyboardInterrupt` 同步打断 `asyncio.run()` 时，executor 内 `_finish_run` 所在 `finally` 没有机会执行。`DefaultHostExecutor` 内置以 `run_id` 为键的资源注册表，`_finish_run`（异步终态）与 `release_resources_for_run`（SIGINT 同步路径）通过 atomic-pop 二选一释放，保证至多一次真实释放，资源对象自身已幂等的 `stop()` / governor.release 不会被双重触发。

### 12.1 进程级优雅退出契约

`dayu.process_lifecycle.ProcessShutdownCoordinator` 是 sync CLI、async daemon、atexit 钩子共用的真源，对外只暴露一个动作 `settle_active_runs(*, trigger)`：先取 observer 通过 `register_active_run` 登记的 run，再合并 `Host.list_active_run_ids_for_current_owner()` 兜底扫描当前 owner_pid 持有的活跃 run（覆盖 `fins` 直接同步执行、interactive/prompt 首事件前窗口等未登记路径），去重后逐个调 `Host.cancel_run_and_settle(run_id)`（同步 cancel + mark_cancelled + cleanup_stale_pending_turns，全部幂等）。该原子操作让退出路径从"协作式取消 + 强收敛"两步幂等补丁简化为一次同步收敛，同时保留 owner 级兜底能力。

| 触发源 | 覆盖端 | 入口 |
|---|---|---|
| SIGINT / SIGTERM / SIGHUP / atexit | sync CLI（`dayu interactive` / `dayu prompt` / `dayu write` / `dayu fins` / `dayu download` / `dayu conv`） | `register_process_shutdown_hook(coordinator, *, interactive)` |
| SIGINT / SIGTERM | async daemon（`dayu wechat run`） | `install_async_signal_handlers` |
| SIGKILL / 断电 | 不可拦截 | 由下次启动 `cleanup_orphan_runs` 收敛（稳定例外） |

`register_process_shutdown_hook` 由 `_prepare_cli_host_dependencies` 在装配阶段一次性注册，按命令类型参数化 SIGINT 行为：

- `interactive=True`：SIGINT 触发 `settle_active_runs` 后抛 `KeyboardInterrupt`（不还原 `SIG_DFL`，REPL 多次 Ctrl+C 都能命中自定义 handler，第二次只取消新 run）。
- `interactive=False`：SIGINT 触发 `settle_active_runs` 后抛 `SystemExit(EXIT_CODE_SIGINT)` 直接退出，不依赖业务层 try/except 兜底。
- SIGTERM/SIGHUP：一律 settle + 还原 `SIG_DFL` + `SystemExit(map_signal_to_exit_code(name))`。
- atexit：`settle_active_runs(trigger="atexit")` 兜底覆盖未走信号路径的退出。

退出码由 `dayu.process_lifecycle.exit_codes` 统一：`EXIT_CODE_SIGINT=130`、`EXIT_CODE_SIGTERM=0`，禁止散落魔法数。

---

## 13. 扩展点

稳定扩展入口：

- **`HostExecutorProtocol`**：替换执行器实现（例如改成远程执行或异步 worker 池）；
- **存储协议**：pending turn / reply outbox / run / session / conversation 均以 Protocol 暴露，可整体替换为非 SQLite 后端；
- **事件订阅者**：UI 通道、日志管道、审计都通过订阅事件接入，不改 Host 核心；
- **lane 注入**：Service 层可按业务注入 lane 默认值，无需触碰 Host 代码。

可预见的演进方向（不属于当前契约，仅备忘）：

- Reply outbox claim 加 lease/owner-token 做强隔离；
- Pending turn 的多租户/分片；
- Agent lineage（sub-agent 父子关系）的一等公民化；
- 持久化外置（PostgreSQL / 远端 KV）。

---

## 14. 作为 Host 的上游：UI / Service 使用指南

本节面向**只消费 Host** 的 UI / Service 开发者，把 Host 当黑盒，回答"我要调什么、按什么顺序调、必须处理哪些错误、不能越界做什么"。Host 内部如何实现（状态机、存储、并发）在前几节，本节不重复。

### 14.1 一次典型请求的生命周期（调用序）

一次"用户输入 → 外部通道看到回复"的路径，UI / Service 共同经过以下阶段，每阶段的拥有者固定：

1. **UI**：收到用户输入，构造 `Request DTO` + `ExecutionDeliveryContext`（投递坐标）。
2. **UI → Service**：调用对应 Service 入口；不绕过 Service 直接访问 Host。
3. **Service**：解释业务语义，决定 scene / prompt contributions / execution options，生成 `ExecutionContract`。
4. **Service → Host**（下述 §14.2 的稳定接口）：
   - 解析或创建 session（活性屏障在此层把关）；
   - 提交 pending turn（承接用户输入，取得 pending_turn_id）；
   - 下发 run（Host 内部走 scene preparation → executor → engine）；
   - 订阅事件并把 Host 事件映射到 `AppEvent` 返回给 UI。
5. **Host → UI**（经 reply outbox）：Agent 产生的回复经 `ReplyDeliveryService` 进入 outbox；UI 的 worker claim / deliver / ack。
6. **取消 / 超时 / resume**：Service 调 Host 的 cancel / resume 接口；Host 内部自行维护语义。

**顺序契约**：session → pending turn → run，**不可**省略 pending turn 直接起 run——否则 resume 与 cleanup 所需的因果链会断。

### 14.2 Service 应该调的 Host 稳定接口

按能力分组，Service/UI 只应依赖以下入口（具体方法名以 `dayu/host/__init__.py` 与 protocol 文件为准）：

| 能力 | Service 典型调用 | 消费者注意事项 |
| --- | --- | --- |
| Session | `create_session / ensure_session / get_session / list_sessions / cancel_session / touch_session` | 所有写入路径的活性屏障由 Host 兜底，上游只需处理 `SessionClosedError` |
| Pending turn | `submit_pending_turn / get_pending_turn / list_pending_turns / acquire_resume_lease / release_resume_lease` | Resume 必须先 acquire 再 release；异常必走 release 分支 |
| Run | `start_run / get_run / cancel_run / subscribe_events` | 订阅事件必须在 start 之前完成，避免丢首个事件 |
| Agent run | `run_agent_and_wait / run_agent_and_wait_replayable / replay_agent_and_wait` | replay 路径仅服务于"上一次输出脏数据"兜底，handle 仅本 Host 实例内存有效，session 关闭即失效 |
| Reply outbox | `submit_reply / claim_next_pending_reply / mark_reply_delivered / mark_reply_failed` | UI worker 专属；Service 只写入，不 claim |
| Governor | 不直接调用 | lane 配置在启动期注入，运行期由 Host 自治 |
| Admin / cleanup | 不直接调用 | 启动恢复与周期性清理由 Host 自行编排 |

**禁止**（硬边界）：
- 不得从 Host 内部模块直接 import（例如 `from dayu.host.pending_turn_store import ...`）；只从 `dayu.host` 顶层导入。
- 不得绕过 outbox 向通道直接投递回复。
- 不得自行构造"等价 run"做重发，必须走 resume 接口。
- 不得在 `RunRecord.metadata` 里塞业务参数（见 §5 字段契约）；业务参数走 `ExecutionContract`。
- 不得在 `lane` 覆盖里出现 Host 自治 lane 名字（见 §8）。

### 14.3 Session 解析策略（Service 侧规则）

Service 对 "session_id 从哪来、能不能新建" 有三条稳定策略，归口在 Service 的 `SessionResolutionPolicy`；Host 只提供底层 create/ensure/get 原语：

- **AUTO**：无 session_id → 新建；有且存在 → 复用并 touch；有但不存在 → 报错或新建（由 Service 策略决定）。
- **REQUIRE_EXISTING**：必须给 session_id 且必须已存在；否则拒绝。用于"明确接续既有会话"的请求。
- **ENSURE_DETERMINISTIC**：按确定性 id（例如 `{source}:{external_key}`）ensure；用于 WeChat/Web 这种由外部 id 推导 session 的场景。

策略本身属于 Service 层，Host 只按"按 id 找 / 按 id 确保"两条原语配合。

### 14.4 必须处理的 Host 错误（上游契约）

Host 抛出的错误分两类：**业务可恢复** vs **上游编程错**。前者必须显式处理，后者是 bug。

**业务可恢复（必须 catch）**：

- `SessionClosedError` — 目标 session 已 CLOSED；UI 应提示用户"会话已结束"。
- `PendingTurnResumeConflictError` — 按 reason 分流：
  - `attempt_exhausted` → 告知用户"已超最大重试"，不再 resume；
  - `lease_conflict` → 有其它 resumer 在跑，通常等一下或跳过；
  - `not_resumable` / `record_missing` → 对应"状态不合法"与"记录已不存在"，UI 按"静默丢弃 / 提示"处理。
- Run 超时 / 取消抛出的事件在事件流里以 `ERROR` / `CANCELLED` 呈现；UI 按事件处理，**不要**等异常。

**上游编程错（不该 catch）**：

- 向 CLOSED session 写 pending turn、往 DELIVERED outbox 上再 mark_failed、用非法 lane 名——都是契约违反，应让其冒泡成 bug。

### 14.5 事件消费规则

Host 的事件订阅面是 Service → UI 流的中继点：

- **订阅必须在 `start_run` 之前**完成；否则会漏掉首事件。
- **终态事件保证在落库之后发出**（见 §9）；UI 看到 `DONE/ERROR/CANCELLED` 后再查 Run 状态一定一致。
- **SSE/WebSocket 断连 ≠ 事件丢失**：断连只是订阅端断开；重新订阅可重放（当前实现不保留历史事件，但 run 已落库状态 + outbox 仍在，是语义等价的恢复点）。
- **UI 不要自行推断"中间状态"**：只消费 Host 发出的事件类型，不做"没收到 DONE 就当失败"这种推断，容易与 UNSETTLED 吸收态打架。

### 14.6 取消、超时、Resume 的上游语义

- **取消**：Service 调 `cancel_run(run_id, reason=USER_CANCELLED)`；Host 异步推进到 CANCELLED 终态。**不要**自己维护 run 生命周期。
- **Session 级关停**：调 `cancel_session`；Host 内部按 §4 三步顺序执行，无需 Service 配合清理。
- **超时**：Service 不自己看表；在提交 run 时把 deadline 作为执行选项传下去，由 Host 的 deadline watcher 负责。
- **Resume**：总是走 `acquire_resume_lease → 基于 pending turn 重新起 run → release_resume_lease`；**acquire 失败按 §14.4 分流**。

### 14.7 启动期装配约束

UI 在 composition root 装配 Service 时需要：

- **一个** `Host` 实例在进程内共享（多 Service 共用同一 Host 是契约，避免两把清理/governor 互相踩）。
- 通过 `resolve_host_config(...)` 规范化配置后再构造 Host；禁止绕过规范化直接塞字典。
- Host 构造完成后**必须**调用 Host 提供的启动恢复入口（`recover_host_startup_state` 等）；恢复未跑就开始接请求，相当于在未知脏数据上继续跑。

这些装配动作在 `dayu/services/startup_preparation.py` 里已有一次收敛，UI 直接复用即可，不必重新发明。

### 14.8 测试接缝建议

Service 单测写"如何调 Host"时：

- **Mock 只针对 Host 的 Protocol**（`SessionOperationsProtocol`、`PendingTurnStoreProtocol`、`ReplyDeliveryProtocol` 等），不 mock 内部实现类；Protocol 是稳定契约，实现会变。
- **不要**在测试里去 patch Host 内部的状态机或事务方法；那是 Host 自己的单元测试职责。
- E2E 层用真实 `Host` + in-memory / tmp SQLite；这是唯一能验证"调用序是否正确"的层面。

---

## 15. 阅读代码的建议顺序

1. `dayu/host/__init__.py` — 看清楚对外只导出什么；
2. `dayu/contracts/` 下 `run.py` / `session.py` / `reply_outbox.py` — 把状态机枚举与合法转移先吃透；
3. `dayu/host/host.py` — 门面与清理入口；
4. `dayu/host/pending_turn_store.py` — resume CAS 的关键；
5. `dayu/host/reply_outbox_store.py` — delivery 幂等与 stale 回退；
6. `dayu/host/executor.py` / `host_execution.py` — 执行路径与 `should_delete_pending_turn_after_terminal_run` 真值表；
7. `dayu/host/concurrency.py` — lane 合并与 `acquire_many`；
8. `dayu/host/conversation_*.py` — 两层记忆与 compaction；
9. `dayu/host/startup_preparation.py` — 配置规范化与默认值；
10. `dayu/host/host_cleanup.py` — Host 包对装配期一次性会话残留清理的稳定 API（`purge_sessions_from_host_db`），由 UI 装配点在构造新 Host 之前 / 拆 service 时按 `host_db_path + session_ids` 直接调用。

最后，任何与本文叙述冲突之处以代码为准；发现不一致请修正 README，而不是反过来。
