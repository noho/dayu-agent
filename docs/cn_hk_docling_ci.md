# 提高 CN/HK Docling CI 分数（执行版）

## 任务目标
- 对当前工作区内**已存在且可被 `dayu.fins.storage` 扫描识别**的 A 股 / 港股文档全集做 CN/HK Docling CI 评估。
- 评分对象固定为 `SourceKind.FILING` 与 `SourceKind.MATERIAL`。
- 评分输入固定为 `process --ci` 产出的现有 `tool_snapshot_*`。
- scorer 固定为 `python -m dayu.fins.score_docling_ci`。
- 优先级固定：**annual → semiannual → quarterly → material**。
- 每轮只处理**一个问题簇**：同一 `report_kind`、同一扣分模式、同一真源。

这不是 SEC CI。禁止使用 `dayu.fins.score_sec_ci`，禁止把 CN/HK 文档映射成 `10-K/10-Q/8-K`，禁止评估 SEC Item 覆盖率。

## 成功标准
- 消除可修复的 completeness failure 与 hard gate failure。
- **最终成功判定只看全量 `baseline` vs 全量 `final`。**
- 以 Step 1 的全量 CN/HK Docling CI 基线为起点，经过 N 轮优化后：
  - `final.overall_avg > baseline.overall_avg`
  - `final.overall_hard_gate_failures <= baseline.overall_hard_gate_failures`
  - `final.overall_completeness_failure_count <= baseline.overall_completeness_failure_count`
- 目标问题簇的改善不能导致其它 `report_kind`、`source_kind` 或 `market` 明显恶化。
- 局部 ticker / 局部 document / 单轮 iter 的改善只是内环信号，不构成最终成功判定。
- 不修改评分标准刷分。
- 不把 CN/HK 业务规则塞进 engine。

## 已知接口
- `process`：`python -m dayu.cli process --ticker {ticker} --ci --overwrite [--document-id ...]`
  - 只能按 ticker 执行。
  - `--document-id` 可重复传入，也支持单个参数中用逗号分隔多个 ID。
  - 传入 `--document-id` 时，只处理指定文档，不清空同 ticker 下其它 processed 结果。
- `score`：`python -m dayu.fins.score_docling_ci --base workspace --tickers {tickers}`
  - 支持 `--output-json` / `--output-md`。
  - 支持 `--source-kind {all,filing,material}`。
  - 支持 `--report-kind {all,annual,semiannual,quarterly,material}`。
  - 支持 `--min-doc-pass` / `--min-doc-warn` / `--min-batch-avg` / `--min-batch-p10`，优化流程中禁止靠这些参数刷分。
  - 后续分析以 **JSON 报告** 为主，stdout 只作日志。
- 已有脚本：
  - 直接使用 `utils/llm_ci_process.py` 做最小增量 process。
  - CN/HK 文档全集必须先由 storage 扫描 active `filing` / `material` 得到，再把明确的 `--tickers` 或 `--documents-json` 传给 `utils/llm_ci_process.py`。
  - 直接使用 `utils/llm_docling_ci_score.py` 做 CN/HK Docling score；未传 `--tickers` 时它会通过 storage 扫描 ticker 名可归一为 CN/HK、且存在 active source 的 ticker。
  - 不要把 CN/HK scoring 塞进 `utils/llm_ci_score.py`；该脚本是 SEC form runner。

## 必须遵守
1. 遵循 `AGENTS.md`。
2. 不修改财报工具 schema。
3. 除非`dayu.fins.score_docling_ci` 本身有问题，不修改 `dayu.fins.score_docling_ci` 来刷出 CN/HK 分数。
4. 不使用 SEC Item / Part / form 覆盖率判断 CN/HK。
5. CI 扣分后必须核查同源证据，区分“原文如此”“Docling 抽取问题”“fins 增强问题”“snapshot 导出问题”“scorer/profile 问题”。
6. 禁止硬编码公司特例；中文/繁中文同义词、report kind profile 与结构规则允许。
7. 禁止通过修改 scorer 阈值、权重、hard gate 规则提分；除非真源归类明确为 `scorer / profile 问题`。
8. engine 不能反向依赖 fins；engine 中不能包含 CN/HK 财报业务语义。
9. **先判定问题真源，再改代码。** 每个问题必须先归类到以下之一：
   - `engine 通用 Docling 抽取层`
   - `fins Docling 财报增强层`
   - `pipeline / snapshot 导出问题`
   - `scorer / profile 问题`
10. `DoclingProcessor` 只放通用 Docling 能力。除非证据证明真源在 engine 通用层，否则不要改它。
11. CN/HK 财报语义增强放 fins 层；评分规则放 scorer/profile；pipeline/snapshot 只负责同源快照导出。
12. 文档、processed、blob、snapshot 的读取和定位必须走 `dayu.fins.storage` 仓储协议 / 实现，禁止手拼 `workspace/portfolio/...` 路径。
13. process 很耗时；除首次补齐 baseline 和最终人工总验外，**禁止每轮全量 process**。
14. 内环验证采用最小增量；`pyright`、全量 `pytest tests/ -x`、README 对齐、全量 final 回归留到最后人工统一检查。

## 执行规则
- **JSON 优先**：机器分析只读 `score_docling_ci` JSON 报告。
- **最小增量 process**：
  - 优先使用 `process --document-id ...` 精确重跑受影响文档。
  - 只要进入 ticker 级批量 process，**必须并发 27 个任务**，以减少总耗时。
  - 直接使用 `utils/llm_ci_process.py`：
    - 支持 `--base`、`--tickers`、`--documents-json`、`--tag`、`--max-documents-per-job`
    - `--documents-json` 为数组，每项至少包含：`ticker`、`document_id`
    - 若传 `--documents-json`，按 ticker 聚合后调用 `process --document-id ...`
    - 日志写 `workspace/tmp/process_logs/{tag}/`
    - 汇总写 `workspace/tmp/process_runs/{tag}.json`
    - 未传 `--tickers` 且未传 `--documents-json` 时，它会扫描全部 available ticker；CN/HK CI 禁止依赖这个默认扫描作为文档全集真源
  - 只有缺 processed / 缺快照 / 快照损坏 / baseline 补齐时，才做较大范围 process。
- **最小增量 score**：
  - 优先只重跑受影响 ticker 子集。
  - 优先只重跑受影响 `source_kind` 与 `report_kind`。
  - 直接使用 `utils/llm_docling_ci_score.py`；不要在执行 prompt 时重新拼装一套 score runner。
- **同源证据闭环**：问题结论必须同时由 scorer 扣分、snapshot/Docling JSON/原文证据、当前代码行为支持。

## 输出目录规范

CN/HK Docling score 输出固定放在：

```text
workspace/tmp/docling_ci_score/{tag}/
```

单次全量 score 输出：

```text
workspace/tmp/docling_ci_score/{tag}/score.json
workspace/tmp/docling_ci_score/{tag}/score.md
workspace/tmp/docling_ci_score/{tag}/score.txt
```

按 `source_kind + report_kind` 拆分输出：

```text
workspace/tmp/docling_ci_score/{tag}/by_kind/score_filing_annual.json
workspace/tmp/docling_ci_score/{tag}/by_kind/score_filing_annual.md
workspace/tmp/docling_ci_score/{tag}/by_kind/score_filing_annual.txt
workspace/tmp/docling_ci_score/{tag}/by_kind/score_filing_semiannual.json
workspace/tmp/docling_ci_score/{tag}/by_kind/score_filing_quarterly.json
workspace/tmp/docling_ci_score/{tag}/by_kind/score_material_material.json
```

score runner 额外生成：

```text
workspace/tmp/docling_ci_score/{tag}/summary.json
workspace/tmp/docling_ci_score/{tag}/overall_summary.json
```

process runner 生成：

```text
workspace/tmp/process_runs/{tag}.json
workspace/tmp/process_logs/{tag}/
```

`summary.json` 至少包含每个 `source_kind + report_kind` 的：

- `avg`
- `p10`
- `hard_gate_failures`
- `completeness_failure_count`
- `document_count`
- `expected_document_count`
- `score_return_code`
- `output_json`
- `output_md`
- `output_txt`

`overall_summary.json` 至少包含：

- `overall_avg`
- `overall_p10`
- `overall_hard_gate_failures`
- `overall_completeness_failure_count`
- `overall_document_count`
- `overall_expected_document_count`
- `source_kinds_included`
- `report_kinds_included`
- `markets_included`
- `score_return_code`

如果需要 by market 指标，从 `score.json.documents[].market` 聚合。不要要求 scorer 新增未实现的 `--market` 参数。

## 流程

### Step 1：全量 CN/HK Docling CI 基线
- 创建报告：`workspace/cn_hk_docling_ci_report_mmdd_HHMM.md`。
- 本轮不新增 ticker，不执行下载。
- 用 `CompanyMetaRepositoryProtocol.scan_company_meta_inventory()` 扫描当前工作区已有公司清单；跳过并记录隐藏目录、缺失 `meta.json`、非法目录。
- 通过 `dayu.fins.storage` 公开仓储接口，扫描**本轮应纳入 CN/HK Docling CI 的文档全集**，作为 baseline 真源清单。
- 文档全集只包含 active source：
  - `SourceKind.FILING`
  - `SourceKind.MATERIAL`
  - `is_deleted != True`
  - `ingest_complete != False`
- 观察维度固定：
  - `source_kind`: `filing`, `material`
  - `report_kind`: `annual`, `semiannual`, `quarterly`, `material`
  - `market`: 从 scorer JSON 的 `documents[].market` 聚合
- 使用 `utils/llm_ci_process.py` 做缺口补齐 process 时，必须显式传 `--documents-json` 或 CN/HK ticker 子集；不要无参运行。
- 直接使用 `utils/llm_docling_ci_score.py` 做 score；不要临时改 `utils/llm_ci_score.py`。

#### Step 1A：baseline_probe
- 先对现有快照跑一次 `baseline_probe`，不要先全量 process。
- 直接命令：

```bash
python utils/llm_docling_ci_score.py \
  --base workspace \
  --tickers {tickers} \
  --tag baseline_probe \
  --source-kinds all \
  --report-kinds all
```

- `baseline_probe` 只用于找缺口，不是正式基线。
- 找出以下缺口：
  - active source 缺 processed
  - 缺 `tool_snapshot_meta.json`
  - `tool_snapshot_meta.json` 损坏或字段不一致
  - 非 meta `tool_snapshot_*` 缺失或损坏
  - `search_query_count` 与 `search_document.calls` 数量不一致
  - 应纳入文档全集但没有出现在 `documents` 中
  - 出现在 `completeness_failures` 中的文档

#### Step 1B：只补齐缺口
- 生成：

```text
workspace/tmp/docling_ci_score/baseline_probe/missing_documents.json
```

- `missing_documents.json` 必须是数组，每项至少包含：

```json
{"ticker": "000001", "document_id": "example_document_id"}
```

- 只对缺口文档执行：

```bash
python utils/llm_ci_process.py \
  --base workspace \
  --documents-json workspace/tmp/docling_ci_score/baseline_probe/missing_documents.json \
  --tag baseline_fill
```

- 默认不要对全部 ticker 执行无条件 `process --ci --overwrite`。

#### Step 1C：正式 baseline
- 缺口补齐后，跑正式全量 baseline：

```bash
python utils/llm_docling_ci_score.py \
  --base workspace \
  --tickers {tickers} \
  --tag baseline
```

- 只有这次 `baseline` 输出，才是正式全量 CI 基线。
- 在报告中记录：
  - overall 的 `avg` / `p10` / `hard_gate_failures` / `completeness_failure_count`
  - 每个 `report_kind` 的 `avg` / `p10` / `hard_gate_failures`
  - 每个 `source_kind` 的 `avg` / `p10` / `hard_gate_failures`
  - 每个 `market` 的 `avg` / `p10` / `hard_gate_failures`
  - 仍无法评分的文档及原因

### Step 2：N 轮优化
- 优先级固定：annual → semiannual → quarterly → material。
- 每轮只处理一个问题簇：同一 `report_kind`、同一扣分模式、同一真源。
- 每轮开始前必须先完成问题归类：
  - `engine 通用 Docling 抽取层`
  - `fins Docling 财报增强层`
  - `pipeline / snapshot 导出问题`
  - `scorer / profile 问题`
- 分析主输入：
  - `workspace/tmp/docling_ci_score/baseline/score.json`
  - 对应 `workspace/tmp/docling_ci_score/baseline/by_kind/*.json`
  - 必要时辅助读取对应 `.txt`
  - 必须通过 `dayu.fins.storage` 核查 snapshot / Docling JSON / 原文
- 当前 `report_kind` 内优先级：
  1. completeness failure
  2. hard gate failure
  3. 高频扣分（同一维度、同一原因、至少 3 个 ticker 或 3 个文档受影响）
  4. annual / semiannual 中三大报表、附注、管理层讨论、审计意见、页面定位相关问题
  5. 孤立低分问题后置
- 每个候选问题必须记录：
  - `source_kind`
  - `report_kind`
  - `market`
  - `ticker`
  - `document_id`
  - 扣分维度
  - scorer 详情字段
  - 原文 / snapshot 证据
  - 真源归类
  - 是否可修复
  - 受影响文档集与 ticker 集
- 修复规则：
  - 真源在 `engine 通用 Docling 抽取层` → 只修 engine 通用 Docling 能力，不能写 CN/HK 财报业务规则
  - 真源在 `fins Docling 财报增强层` → 修 fins CN/HK 财报语义增强
  - 真源在 `pipeline / snapshot 导出问题` → 修 source_kind / report_kind / meta / query pack / snapshot 写出真源，别改 scorer 规则
  - 真源在 `scorer / profile 问题` → 修 scorer/profile，并重新建立可比 baseline；不能把 scorer 修正前后的分数当作 processor 优化收益
- 每处代码修改必须同步新增 / 更新针对该真源的测试，但只跑最小必要测试集。
- 每轮修复后只做内环轻量验证，不做最终成功判定：
  - 只重处理本轮受影响文档，按 ticker 聚合后使用 `process --document-id ...`
  - 只重跑受影响 `source_kind + report_kind` 的 score
  - 若可能影响相邻 `report_kind`，只做小子集 smoke score
- iter process：

```bash
python utils/llm_ci_process.py \
  --base workspace \
  --documents-json workspace/tmp/docling_ci_score/iter_{round_id}/affected_documents.json \
  --tag iter_{round_id}_process
```

- iter score：

```bash
python utils/llm_docling_ci_score.py \
  --base workspace \
  --tickers {affected_tickers} \
  --tag iter_{round_id} \
  --source-kinds {filing|material} \
  --report-kinds {annual|semiannual|quarterly|material}
```

- 本轮比较只用于判断方向是否有效，至少看：
  - 目标文档 completeness / hard gate 是否减少
  - 目标文档对应维度分数是否改善
  - 受影响子集的 `avg` / `p10` / `hard_gate_failures` 是否恶化
  - 相邻 `report_kind` 小子集是否明显恶化
- 若无收益或出现恶化，停止叠补丁，回到问题分析重新判根因。
- 在报告中持续记录每轮问题簇、真源归类、修复位置、增量验证结论、仍未修复的问题与原因。

### Step 3：全量 CN/HK Docling CI final
- 当 N 轮优化结束后，执行一次全量评分，并产出：

```text
workspace/tmp/docling_ci_score/final/score.json
workspace/tmp/docling_ci_score/final/score.md
workspace/tmp/docling_ci_score/final/score.txt
workspace/tmp/docling_ci_score/final/summary.json
workspace/tmp/docling_ci_score/final/overall_summary.json
```

- final score：

```bash
python utils/llm_docling_ci_score.py \
  --base workspace \
  --tickers {tickers} \
  --tag final
```

- 最终成功判定只按全量口径：
  - `final.overall_avg > baseline.overall_avg`
  - `final.overall_hard_gate_failures <= baseline.overall_hard_gate_failures`
  - `final.overall_completeness_failure_count <= baseline.overall_completeness_failure_count`
- `summary.json` 中各 `report_kind` / `source_kind` / `market` 指标用于诊断收益来源和剩余短板，不是最终唯一成功定义。
- 在最终报告中必须包含：
  - Step 1 baseline 摘要
  - Step 2 每轮优化记录
  - Step 3 全量 final 对比
  - 收益来源
  - 剩余短板
  - 被判定为“原文如此”的问题
  - 最终人工检查命令

## 最终人工检查命令（写入报告，不自动执行）
- `python utils/llm_ci_process.py --base workspace --tickers {tickers} --tag final_process`
- `python utils/llm_docling_ci_score.py --base workspace --tickers {tickers} --tag final`
- `pytest tests/fins/test_score_docling_ci.py -q`
- `pytest tests/fins/test_llm_ci_scripts.py -q`
- pyright
- README 对齐检查
- 若修改 engine / fins processor / pipeline / snapshot exporter，追加对应受影响测试

## 必须避免的低效路径
- 不要每轮先对全量 ticker 执行 `process --ci --overwrite`。
- 不要每轮都重跑所有 `report_kind` 的全量 score。
- 不要主要依赖 stdout 文本解析。
- 不要手拼 `workspace/portfolio/...` 查文档或 snapshot。
- 不要把共享根因修成多个文件里的平行补丁。
- 不要因为某个局部样本分数更高就改 scorer 阈值。
- 不要直接改 `DoclingProcessor`，除非证据证明真源在 engine 通用 Docling 抽取层。
- 不要把 CN/HK 业务语义塞进 engine。
- 不要把普通 material 的无财务表误判成财报表格缺失；只有业绩类 material 进入财务表要求。

## 已核实 runner 边界
- `utils/llm_ci_process.py` 只负责 process 子进程调度、日志与汇总；CN/HK 文档全集必须由当前执行 Agent 通过 storage 扫描得到，不能依赖该脚本的默认 available ticker 扫描。
- `utils/llm_docling_ci_score.py` 只负责调用 `python -m dayu.fins.score_docling_ci`，并写入 `workspace/tmp/docling_ci_score/{tag}/`。
- `utils/llm_docling_ci_score.py` 未传 `--tickers` 时会通过 `CompanyMetaRepository` 与 `SourceDocumentRepository` 扫描目录名 / ticker 名可归一为 CN/HK、且存在 active `filing/material` 的 ticker。
- `utils/llm_docling_ci_score.py` 固定生成：
  - `score.json`
  - `score.md`
  - `score.txt`
  - `by_kind/*`
  - `summary.json`
  - `overall_summary.json`
- `utils/llm_docling_ci_score.py` 的参数固定为：
  - `--base`
  - `--tickers`
  - `--tag`
  - `--source-kinds`
  - `--report-kinds`
- 禁止执行 Agent 新建临时 score runner。
- 禁止执行 Agent 修改 `utils/llm_ci_score.py` 来跑 CN/HK。

## 预期产出
1. `workspace/cn_hk_docling_ci_report_mmdd_HHMM.md`
2. `workspace/tmp/docling_ci_score/baseline/*`
3. 每轮 `workspace/tmp/docling_ci_score/iter_{round_id}/*`
4. `workspace/tmp/docling_ci_score/final/*`
5. 对应代码修复与测试

`utils/llm_docling_ci_score.py` 是已知执行脚本，不是本 prompt 的产出；只有 runner 本身被证明有 bug 时，才按普通代码修复流程修改它。
