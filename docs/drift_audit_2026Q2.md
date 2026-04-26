# docs/drift_audit_2026Q2.md — README / CHANGELOG 漂移审计

**审计范围**：`v0.1.2`（tag `43338ad`, 2026-04-20）发布后至 HEAD（2026-04-25）之间的全部提交，涵盖 PR #41、#43、#47、#49、#50、#51、#53、#54 及散落提交 `1455f42`（docling Windows 非 ASCII）、`6cac7f9`（docling Windows 后端排序 + chcp）。

**目标文件**：`CHANGELOG.md`、`README.md`、`dayu/README.md`、`dayu/host/README.md`、`dayu/engine/README.md`、`dayu/fins/README.md`、`dayu/config/README.md`、`tests/README.md`。

**审计人**：Marketing Agent（Run `run-mofwm9tl`）

**审计日期**：2026-04-26

---

## 一、执行摘要

共发现 **14 条** 漂移条目，其中：

| 严重级别 | 数量 | 描述 |
|---------|------|------|
| **HIGH** | 4 | 对用户造成直接误导（版本号错误、功能描述过时） |
| **MEDIUM** | 7 | CHANGELOG 遗漏用户可见的功能/平台/修复项 |
| **LOW** | 3 | 术语不一致、内部改进遗漏 |

最关键问题：
1. `README.md` 在线安装示例引用 `v0.1.3`，但 `pyproject.toml` 仍为 `0.1.2`，无此 release。
2. `README.md` 第 0 节和 `dayu/README.md` 第 0 节均称"Web UI 目前仍只有 FastAPI 骨架"，但 PR #49/#50 已实现完整 Streamlit UI，同一个 README 的第 2.2 节也已反映该实现。
3. `CHANGELOG.md [Unreleased]` 重复了 `[0.1.2]` 的离线安装包条目（条目内容和措辞完全相同）。
4. `CHANGELOG.md [Unreleased]` 缺少 PR #49（Streamlit Web UI）、PR #50（macOS x64 离线包）、PR #51（Agent 执行进度事件）、PR #54（streamlit 收敛到 `[web]` extras）及两个 Windows 关键修复。

---

## 二、漂移明细

### F-001 ⚠️ HIGH — README 在线安装版本号超前

| 字段 | 内容 |
|------|------|
| 关联提交/PR | `2893993` (README patch, #40, 2026-04-24)；`pyproject.toml` 未更新 |
| 文件路径 + 行 | `README.md` 第 62、80、81、88、89 行 |
| 当前文本 | `pip install https://github.com/noho/dayu-agent/releases/download/v0.1.3/dayu_agent-0.1.3-py3-none-any.whl` 及 `dayu-agent-0.1.3-macos-arm64-offline.tar.gz` 等 |
| 期望文本 | 版本号应与 `pyproject.toml` 的 `version = "0.1.2"` 一致，或更新 pyproject 到 `0.1.3` 并发布对应 Release |
| 问题描述 | `pyproject.toml` 当前 version 为 `0.1.2`，但 README 在线安装命令和离线安装命令均已写死 `v0.1.3`，对应 Release 不存在，用户按文档安装将 404。 |
| 严重级别 | **HIGH** |

---

### F-002 ⚠️ HIGH — README 第 0 节和 dayu/README.md 第 0 节将 Streamlit Web UI 描述为"仅骨架"

| 字段 | 内容 |
|------|------|
| 关联提交/PR | PR #49 `c846dff`（streamlit 实现 web 服务基础框架）；PR #50 `1194f1d`（macOS x64 docling）；PR #54 `a35d194`（streamlit 收敛到 web extras） |
| 文件路径 + 行 | `README.md` 第 31 行；`dayu/README.md` 第 65 行 |
| 当前文本（README.md 第 31 行） | `**Web UI 目前仍只有 FastAPI 骨架**。` |
| 当前文本（dayu/README.md 第 65 行） | `GUI 尚未实现；Web UI 目前仍只有 FastAPI 骨架。` |
| 期望文本 | 应更新为反映已有 Streamlit UI 的现状，例如："**Web UI 已有 Streamlit 入口（`dayu-web`），可完成交互问答和报告生成；FastAPI 仅用于程序化调用。**" |
| 问题描述 | PR #49 实现了完整的 Streamlit Web 入口，包含自选股管理、财报下载、交互分析和报告生成。同一个 `README.md` 的第 2.2 节（Web 入口）已正确描述 Streamlit，但第 0 节的贡献指引仍在引导"Web UI 仅骨架"方向。`dayu/README.md` 第 0 节同样过时。 |
| 严重级别 | **HIGH** |

---

### F-003 ⚠️ HIGH — CHANGELOG.md [Unreleased] 缺少 PR #49（Streamlit Web UI）

| 字段 | 内容 |
|------|------|
| 关联提交/PR | PR #49 `c846dff`（2026-04-25）；PR #54 `a35d194`（2026-04-25） |
| 文件路径 + 行 | `CHANGELOG.md` `[Unreleased]` 章节 `### 新增` 块（第 15–25 行） |
| 当前文本 | `[Unreleased] ### 新增` 中无 Streamlit Web UI 相关条目 |
| 期望文本 | 应在 `### 新增` 中增加：`- `dayu-web` 启用 Streamlit Web UI，提供交互问答、财报下载、报告生成与自选股管理；需安装 `[web]` extras（`pip install "...[web]"`）。` |
| 问题描述 | Streamlit Web UI 是本周期最重要的用户可见新功能，但在 `[Unreleased]` 中完全缺失。PR #54 还将 streamlit 收敛到可选 `[web]` extras，影响用户安装方式，同样未记录。 |
| 严重级别 | **HIGH** |

---

### F-004 ⚠️ MEDIUM — CHANGELOG.md [Unreleased] 离线安装包条目与 [0.1.2] 完全重复

| 字段 | 内容 |
|------|------|
| 关联提交/PR | 任务描述已明确指出；`[Unreleased]` 行 17 vs `[0.1.2]` 行 43 |
| 文件路径 + 行 | `CHANGELOG.md` 第 17 行（`[Unreleased]`）和第 43 行（`[0.1.2]`） |
| 当前文本（两处相同） | `- 提供离线安装包，覆盖 \`macOS ARM64\`、\`Linux x64\`、\`Windows x64\` 三个平台。` |
| 期望文本 | `[Unreleased]` 中的该条目应更新为：`- 离线安装包新增 \`macOS x64\` 平台支持，现覆盖 macOS ARM64、macOS x64、Linux x64、Windows x64 四个平台。`（原 [0.1.2] 条目保持不变） |
| 问题描述 | (1) 完全重复是 CHANGELOG 维护错误，任务描述中已明确标记。(2) 更重要的是，PR #50 (`1194f1d`, `mig/macos x64 docling`) 增加了 macOS x64 离线包支持，`[Unreleased]` 条目仅写 3 个平台，而 README 第 1.1.2 节已正确列出 4 个平台（含"Mac Intel芯片"），造成 CHANGELOG 与 README 不一致。 |
| 严重级别 | **MEDIUM** |

---

### F-005 ⚠️ MEDIUM — CHANGELOG.md [Unreleased] 缺少 macOS x64 平台条目（PR #50）

| 字段 | 内容 |
|------|------|
| 关联提交/PR | PR #50 `1194f1d`（mig/macos x64 docling，2026-04-25） |
| 文件路径 + 行 | `CHANGELOG.md` `[Unreleased]` 章节（第 9–36 行）；对比 `README.md` 第 74 行 |
| 当前文本（CHANGELOG） | `[Unreleased]` 中无 macOS x64 相关条目 |
| 当前文本（README） | `- Mac Intel芯片：\`dayu-agent-<version>-macos-x64-offline.tar.gz\`` |
| 期望文本（CHANGELOG） | 应在 `### 新增` 中明确说明新增 macOS x64 离线包（与 F-004 合并处理）|
| 严重级别 | **MEDIUM** |

---

### F-006 ⚠️ MEDIUM — CHANGELOG.md [Unreleased] 缺少 PR #51（Agent 执行进度事件 iteration_start）

| 字段 | 内容 |
|------|------|
| 关联提交/PR | PR #51 `174eafd`（Feat/agent execution progress，2026-04-25） |
| 文件路径 + 行 | `CHANGELOG.md` `[Unreleased]` 章节；`dayu/engine/README.md` 第 216 行 |
| 当前文本（CHANGELOG） | `[Unreleased]` 中无 iteration_start 事件相关条目 |
| 当前文本（engine README 第 216 行） | `- \`iteration_start\` 由 \`AsyncAgent\` 在每轮迭代开始时产出，携带 \`{iteration, run_id}\`；Host 透传为 \`AppEventType.ITERATION_START\`，供 UI 层展示"第 N 轮思考..."` |
| 期望文本（CHANGELOG） | 应在 `### 新增` 中增加：`- CLI / Web UI 新增 Agent 执行进度展示，可在终端或 Web 页面看到"第 N 轮思考..."进度提示。` |
| 问题描述 | `iteration_start` 事件是用户可感知的体验改善（进度可见性），在 `dayu/engine/README.md` 中已有描述，但未进入 CHANGELOG。 |
| 严重级别 | **MEDIUM** |

---

### F-007 ⚠️ MEDIUM — CHANGELOG.md [Unreleased] 缺少 Windows 上传 bug 修复（1455f42）

| 字段 | 内容 |
|------|------|
| 关联提交/PR | `1455f42`（fix(docling): Windows 非 ASCII 路径上传 bug + 配套 flaky 测试修复，2026-04-25） |
| 文件路径 + 行 | `CHANGELOG.md` `[Unreleased]` 的 `### 修复` 块（第 33–35 行） |
| 当前文本 | `- 若干缺陷修复。` |
| 期望文本 | 应明确说明：`- 修复 Windows 环境下上传含非 ASCII 字符路径的财报时报错的问题（docling 路径编码错误）。` |
| 问题描述 | Windows 用户在上传非 ASCII 路径（如含中文目录）的财报时会碰到此 bug。"若干缺陷修复"过于模糊，该修复对 Windows 用户有明确价值。 |
| 严重级别 | **MEDIUM** |

---

### F-008 ⚠️ MEDIUM — CHANGELOG.md [Unreleased] 缺少 Windows docling 后端排序修复（6cac7f9）

| 字段 | 内容 |
|------|------|
| 关联提交/PR | `6cac7f9`（fix(windows): docling 后端排序 + chcp 65001 中文 REM 规避，2026-04-25） |
| 文件路径 + 行 | `CHANGELOG.md` `[Unreleased]` 的 `### 修复` 块（第 33–35 行） |
| 当前文本 | `- 若干缺陷修复。` |
| 期望文本 | 应说明：`- 修复 Windows 离线安装脚本中因 chcp 65001 导致中文注释乱码及 docling 后端排序不稳定的问题。` |
| 问题描述 | `6cac7f9` 修复了 Windows PowerShell 中文环境运行 `install.cmd` 脚本时的乱码问题，以及 docling 后端选择顺序不稳定问题。对 Windows 用户有直接影响。 |
| 严重级别 | **MEDIUM** |

---

### F-009 ⚠️ MEDIUM — CHANGELOG.md [Unreleased] 缺少 [web] extras 安装说明（PR #54）

| 字段 | 内容 |
|------|------|
| 关联提交/PR | PR #54 `a35d194`（chore(deps): 把 streamlit 收敛到 web extras 并补齐受控锁定，2026-04-25） |
| 文件路径 + 行 | `CHANGELOG.md` `[Unreleased]` 章节；`README.md` 第 107 行 |
| 当前文本（CHANGELOG） | 无相关条目 |
| 当前文本（README 第 107 行） | `` `web` extras 启用 `dayu-web`（streamlit）入口；不需要 Web UI 时可从 extras 列表中省略 `` |
| 期望文本（CHANGELOG） | 应在 `### 变更` 中增加：`- `dayu-web`（Streamlit 入口）已从默认依赖中分离，移至可选 \`[web]\` extras；如需 Web UI，安装时应加 \`[web]\` extras。` |
| 问题描述 | 依赖分拆影响已有用户的安装和升级命令，是需要注意的 breaking-adjacent 变更，应在 CHANGELOG 中明确说明。README 已有说明，但 CHANGELOG 漏记。 |
| 严重级别 | **MEDIUM** |

---

### F-010 ⚠️ MEDIUM — CHANGELOG.md [Unreleased] 对写作优化（PR #53）描述过于模糊

| 字段 | 内容 |
|------|------|
| 关联提交/PR | PR #53 `77a61cb`（opt/optimize write pipeline，2026-04-25） |
| 文件路径 + 行 | `CHANGELOG.md` `[Unreleased]` `### 新增` 第 25 行 |
| 当前文本 | `- 优化写作提高成功率。` |
| 期望文本 | 例如：`- 优化写作流水线：改进 repair/audit/confirm 重试策略与 scene 执行分层，提升长篇报告整体写成率和章节质量。` |
| 问题描述 | PR #53 标题为 "opt/optimize write pipeline"，是一个涉及多处写作流水线性能和成功率的优化。当前 CHANGELOG 描述"优化写作提高成功率"过于笼统，无法让用户理解具体改进点。`dayu/engine/README.md` 中 `test_write_pipeline.py` 相关说明体现了多项新语义约束。 |
| 严重级别 | **MEDIUM** |

---

### F-011 ⚠️ LOW — README.md 第 0 节内部自相矛盾（Web UI 描述）

| 字段 | 内容 |
|------|------|
| 关联提交/PR | PR #49 `c846dff`（2026-04-25） |
| 文件路径 + 行 | `README.md` 第 31 行（第 0 节）vs 第 279–296 行（第 2.2 节） |
| 当前文本（第 31 行） | `**Web UI 目前仍只有 FastAPI 骨架**。` |
| 当前文本（第 279–295 行） | 详细描述 `dayu-web` Streamlit 入口、启动命令、URL 等 |
| 期望文本（第 31 行） | 移除该项，或改为："**WeChat UI 仅支持文本消息首版，还可添加更多好玩的功能**（贡献方向）。**Web UI 有 Streamlit 入口，欢迎补充写作进度实时展示等 UI 能力。**" |
| 问题描述 | 同一文档中，第 0 节（贡献指引）和第 2.2 节（Web 入口说明）对 Web UI 的描述矛盾。这与 F-002 是同一根因，但此条专注于 README 内部一致性。 |
| 严重级别 | **LOW** |

---

### F-012 ⚠️ LOW — CHANGELOG.md [Unreleased] 模型名称术语不一致（qwen3.6-plus vs 配置键 qwen-plus）

| 字段 | 内容 |
|------|------|
| 关联提交/PR | PR #47 `8ffa53f`（config/upgrade mimo chatgpt version，2026-04-24） |
| 文件路径 + 行 | `CHANGELOG.md` `[Unreleased]` 第 31 行；`dayu/config/README.md` 第 219–222 行 |
| 当前文本（CHANGELOG 第 31 行） | `` `qwen` 模型更新到qwen3.6-plus。 `` |
| 当前文本（config/README.md 第 219 行） | `- \`qwen-plus\`` 和 `- \`qwen-plus-thinking\`` 和 `- \`qwen3:30b-thinking\`` |
| 期望文本（CHANGELOG） | 应明确是 API 底层模型 ID 变更，而配置键 `qwen-plus` 保持不变，例如：`` `qwen-plus` 系列底层模型升级至 qwen3 Plus（qwen3.6-plus），配置键名不变。 `` |
| 问题描述 | CHANGELOG 用 `qwen3.6-plus` 指代模型，但用户在配置文件（`llm_models.json`）和 `config/README.md` 中看到的配置键是 `qwen-plus`，容易产生混淆。 |
| 严重级别 | **LOW** |

---

### F-013 ⚠️ LOW — CHANGELOG.md [Unreleased] 缺少 PR #41（清理 max output tokens）

| 字段 | 内容 |
|------|------|
| 关联提交/PR | PR #41 `24a706f`（refactory/clean max output tokens，2026-04-24） |
| 文件路径 + 行 | `CHANGELOG.md` `[Unreleased]` |
| 当前文本 | 无相关条目 |
| 期望文本 | 可合入 `### 变更`：`- 移除模型配置中的历史遗留 max_output_tokens 字段，清理运行时冗余配置。` |
| 问题描述 | 虽为内部重构，但涉及配置格式变更（max_output_tokens 移除），有轻微向后影响，应记录在 CHANGELOG 中以便追溯。 |
| 严重级别 | **LOW** |

---

### F-014 ⚠️ LOW — CHANGELOG.md [Unreleased] 缺少 PR #43（离线构建 --wheel-cache-dir 支持）

| 字段 | 内容 |
|------|------|
| 关联提交/PR | PR #43 `c74f8be`（feat: build_offline_bundle 支持 --wheel-cache-dir，2026-04-24） |
| 文件路径 + 行 | `CHANGELOG.md` `[Unreleased]` |
| 当前文本 | 无相关条目 |
| 期望文本 | 可合入 `### 新增`：`- 离线包构建脚本（build_offline_bundle）新增 \`--wheel-cache-dir\` 参数，支持指定 wheel 缓存目录，加速重复构建。` |
| 问题描述 | 对参与离线包构建的开发者有实际价值，虽是开发侧工具，但属于 release-related 的基础设施改善，应在 CHANGELOG 中记录。 |
| 严重级别 | **LOW** |

---

## 三、受影响文件汇总

| 文件 | 漂移条目 | 建议动作 |
|------|---------|---------|
| `CHANGELOG.md` | F-003, F-004, F-005, F-006, F-007, F-008, F-009, F-010, F-012, F-013, F-014 | 补全 [Unreleased] 条目；修复重复条目；更新离线平台描述 |
| `README.md` | F-001, F-002, F-011 | 修复版本号引用；更新第 0 节 Web UI 描述 |
| `dayu/README.md` | F-002 | 更新第 0 节 Web UI 描述 |
| `dayu/host/README.md` | 无漂移（已包含 agent replay，Host 九项能力等最新内容） | — |
| `dayu/engine/README.md` | 无漂移（iteration_start 事件已记录）| — |
| `dayu/fins/README.md` | 无漂移 | — |
| `dayu/config/README.md` | 无漂移（模型键名与实现一致）| — |
| `tests/README.md` | 无漂移（已包含最新测试边界说明） | — |
| `pyproject.toml` | F-001（关联） | 版本号需与 README 一致（发布 v0.1.3 或回退 README） |

---

## 四、优先修复建议（供 CEO 拆分 fix 任务参考）

**Cluster A（CHANGELOG 补全，单次提交可完成）**

- F-003 + F-009：补充 Streamlit Web UI + [web] extras 条目
- F-004 + F-005：修复重复条目 + 补充 macOS x64 平台
- F-006：补充 Agent 执行进度条目
- F-007 + F-008：补充两个 Windows 修复条目
- F-010：扩写写作优化描述
- F-012 + F-013 + F-014：低优先级 CHANGELOG 清理

**Cluster B（README 内容修正，影响用户阅读体验）**

- F-001：解决版本号不一致（依赖 v0.1.3 release 或回退 README）
- F-002 + F-011：更新 Web UI 描述（README.md 第 0 节 + dayu/README.md 第 0 节）

---

*此文档仅记录漂移发现，不修改任何被审计文件。*
