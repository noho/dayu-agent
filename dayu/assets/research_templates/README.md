# Dayu 研究模板库

<!-- DAYU_RESEARCH_TEMPLATE schema_version=1 name=index -->

这个目录保存可复用的买方研究模板。它们不是最终报告，而是把财报证据、行业变量、反证条件和后续监控统一到同一套研究框架中。

## 模板清单

- `common`：通用深挖研究模板。
- `consumer`：消费、品牌、零售、本地生活。
- `cyclical`：周期、资源、航运、化工、工程机械。
- `technology`：科技、平台、软件、硬件、AI 基础设施。
- `financial`：银行、保险、券商、资管等金融公司。

## 推荐工作流

每个模板都覆盖研究问题、经营拆解、证据、监控、否决项、结论、估值与预期差、催化剂与时间轴、管理层与资本配置、组合决策与风险预算。行业模板中的决策问题按商业模式定制，不应直接用短期利润、单一倍数、管理层叙事或主观信心替代可验证证据、情景分析和仓位纪律。

1. 用 `dayu-cli research-template list` 查看模板。
2. 用 `dayu-cli research-template show consumer` 快速预览。
3. 用 `dayu-cli write --ticker <ticker> --research-template consumer` 安全组合通用与消费行业模板并进入正式写作流水线。
4. 如需手工定制，再用 `dayu-cli research-template compose consumer --base ./workspace` 生成组合文件，并通过 `--template <path>` 显式使用。

`--research-template` 不会静默覆盖已被修改的组合文件；发生漂移时应先审阅差异，再显式运行 `research-template compose <name> --overwrite`。

`research-template materialize` 会同时生成可编辑 research workbook 和初始 Markdown progress report。更新 workbook 后，应重新生成 progress report，使 bundle 的研究进度视图与证据状态保持一致。

同一次 materialize 还会生成已校验的 dry-run monitoring plan。计划中的数据源保持未绑定，且禁止自动执行；完成 source binding 审核后需要显式重建计划。

materialize 还会写入 monitoring、workbook、progress report 三类状态快照，供本地 UI 或组合看板直接读取；这些快照与其他 workspace 工件处于同一异常回滚边界。

研究更新后可运行 `research-template refresh-workspace --bundle <bundle.json>` 预览派生工件刷新，再加 `--write` 一次性重建 progress report、dry-run plan、guide 和状态快照。该命令不会掩盖损坏的 workbook 或源 manifest 漂移。

需要自动选择时，直接运行 `dayu-cli write --ticker <ticker> --research-template auto`。若 `draft/<ticker>/manifest.json` 尚不存在，CLI 会先用官方模板执行一次 infer-only 阶段，成功写入公司 facet 后再选择行业模板并继续正式写作；已有 manifest 时不会重复 bootstrap。`--research-template auto --infer` 仍只归因不写作，可用于显式刷新 facet。
