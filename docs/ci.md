# 提高 CI 分数（执行版）

## 任务目标
- 对当前工作区内**已存在且可被 `dayu.fins.storage` 扫描识别**的 SEC 公司全集做 CI 评估。
- 找出并修复**提取真问题**，提高 CI 分数。
- 优先级固定：**20-F → 10-K → 10-Q → 6-K → 其它 form**。
- 每轮只处理**一个问题簇**：同一 form、同一扣分模式、同一真源。

## 成功标准
- 消除可修复的 GateFail。
- **最终成功判定只看每类form的全量 filings 的 Step 1 基线 vs N 轮优化后的全量结果。**
- 以 Step 1 的每类form的的全量 CI 基线为起点，经过 N 轮优化后，最终**每类form的全量 filings** 的 CI 分数都要高于基线。
- form 内局部样本 / 目标问题簇的改善，只是内环过程信号，不构成最终成功判定。
- 导出超时问题被定位并修复。
- D5 caption 填充率有实质提升。
- 不修改评分标准，不引入明显 regression。

## 已知接口
- `process`：`python -m dayu.cli process --ticker {ticker} --ci --overwrite [--document-id ...]`
  - 只能按 ticker 执行，但可重复 `--document-id` 缩小到指定文档。
- `score`：`python -m dayu.fins.score_sec_ci --form {form} --base workspace --tickers {tickers}`
  - 每次只评一个 form。
  - 支持 `--output-json` / `--output-md`。
  - 后续分析以 **JSON 报告** 为主，stdout 只作日志。
- 已有脚本：
  - 直接使用 `utils/llm_ci_process.py`
  - 直接使用 `utils/llm_ci_score.py`
  - **不要在每次执行 prompt 时重新生成这两个脚本**；只有当脚本本身需要修复或增强时才修改。

## 必须遵守
1. 遵循 `AGENTS.md`。
2. 不修改财报工具 schema。
3. CI 扣分后必须核查原文，区分“原文如此”和“提取问题”。
4. 禁止硬编码公司特例；SEC 固定结构规则与自适应文本/结构规则允许。
5. 禁止通过修改评分标准提分。
6. engine 不能反向依赖 fins；engine 中不能包含金融逻辑。
7. 跨 processor 的重复修复必须抽到共享真源；禁止平行复制补丁。
8. 同一 form 的主路由 processor 与 fallback processor 之间不能互相 import。
9. **先判定问题真源，再改代码。** 每个问题必须先归类到以下之一：
   - `engine 通用提取层`
   - `fins 共享表单逻辑`
   - `单 form 专属 processor`
   - `pipeline / snapshot 导出问题`
10. **禁止用 fallback / 路由切换绕过问题**，除非根因本身就是路由错误。
11. 文档、processed、blob 的读取和定位必须走 `dayu.fins.storage` 仓储协议 / 实现，禁止手拼 `workspace/portfolio/...` 路径。
12. process 很耗时；除首次补齐基线和最终人工总验外，**禁止每轮全量 process**。
13. 内环验证采用最小增量；`pyright`、全量 `pytest tests/ -x`、README 对齐、全 7 form 全量回归留到最后人工统一检查。

## 执行规则
- **JSON 优先**：机器分析只读 `score_sec_ci` JSON 报告。
- **最小增量 process**：
  - 优先使用 `process --document-id ...` 精确重跑受影响文档。
  - 只要进入 ticker 级批量 process，**必须并发 27 个任务**，以减少总耗时。
  - 只有缺快照 / 快照损坏 / 需要首次补齐样本时，才做 ticker 级批量 process
- **最小增量 score**：只重跑受影响 form 和受影响 ticker 子集。
- **同源证据闭环**：问题结论必须同时由 CI 扣分、原文核查、当前代码行为支持。

## 流程

### Step 1：全量 CI 基线
- 创建报告：`workspace/ci_optimization_report_mmdd_HHMM.md`。
- 本轮不新增 ticker，不执行下载。
- 用 `CompanyMetaRepositoryProtocol.scan_company_meta_inventory()` 扫描当前工作区已有公司清单；跳过并记录隐藏目录、缺失 `meta.json`、非法目录。
- 通过 `dayu.fins.storage` 公开仓储接口，扫描**本轮应纳入 CI 的文档全集**，作为 baseline 真源清单。
- 直接使用 `utils/llm_ci_process.py`：
  - 支持 `--tickers`、`--documents-json`、`--tag`
  - `--documents-json` 为数组，每项至少包含：`ticker`、`document_id`
  - 若传 `--documents-json`，按 ticker 聚合后，对每个 ticker 调用一次 `process --document-id ...`
  - 若未传 `--documents-json`，按 ticker 调 `process`
  - ticker 级并发固定为 `ProcessPoolExecutor(max_workers=27)`，以减少总耗时
  - 单个子任务超时 300 秒，日志写 `workspace/tmp/process_logs/`
  - 汇总写 `workspace/tmp/process_runs/{tag}.json`
- 直接使用 `utils/llm_ci_score.py`：
  - 支持 `--forms`、`--tickers`、`--tag`
  - 默认 forms：`10-K,10-Q,20-F,6-K,8-K,SC 13G,DEF 14A`
  - 对每个 form 调用 `python -m dayu.fins.score_sec_ci`
  - 显式输出：
    - `workspace/tmp/ci_score/{tag}/score_{slug}.json`
    - `workspace/tmp/ci_score/{tag}/score_{slug}.md`
    - `workspace/tmp/ci_score/{tag}/score_{slug}.txt`
  - 额外生成 `workspace/tmp/ci_score/{tag}/summary.json`，至少包含每个 form 的：`avg`、`p10`、`hard_gate_failures`、`document_count`
  - 额外生成 `workspace/tmp/ci_score/{tag}/overall_summary.json`，至少包含：`overall_avg`、`overall_p10`、`overall_hard_gate_failures`、`overall_document_count`、`forms_included`
- 先对现有快照跑一次 `--tag baseline_probe`，找出与文档全集相比的缺口：
  - 应纳入 CI、但未出现在 score JSON 中的文档
  - 因缺失快照无法评分的文档
  - 因快照损坏 / 缺文件被跳过的文档
- 只对这些缺口文档执行 `python utils/llm_ci_process.py ...`；默认不要对 600+ ticker 做无条件全量 process。
- 补齐后，对全量 ticker、全量 7 个 form 跑一次 `python utils/llm_ci_score.py --tag baseline`。
- 只有这次 `baseline` 输出，才是正式全量 CI 基线。
- 在报告中记录：
  - 每个 form 的 `avg` / `p10` / `hard_gate_failures`
  - `overall_summary.json` 中的 `overall_avg` / `overall_p10` / `overall_hard_gate_failures`
  - 仍无法评分的文档及原因

### Step 2：N 轮优化
- form 顺序固定：20-F → 10-K → 10-Q → 6-K → 其它。
- 每轮只处理一个问题簇：同一 form、同一扣分模式、同一真源。
- 每轮开始前必须先完成问题归类：
  - `engine 通用提取层`
  - `fins 共享表单逻辑`
  - `单 form 专属 processor`
  - `pipeline / snapshot 导出问题`
- 分析主输入：`workspace/tmp/ci_score/baseline/score_{slug}.json`；必要时辅助读取对应 `.txt`，并通过 `dayu.fins.storage` 核查原文。
- 当前 form 内优先级：
  1. GateFail
  2. 高频扣分（同一维度、至少 3 个 ticker 受影响）
  3. 孤立低分问题后置
- 每个候选问题必须记录：`form_type`、`ticker`、`document_id`、扣分维度、原因分类、真源归类、是否可修复、受影响文档集与 ticker 集。
- 修复规则：
  - 真源在 `engine 通用提取层` → 直接修 engine 共享真源
  - 真源在 `fins 共享表单逻辑` → 修共享模块
  - 真源在 `单 form 专属 processor` → 只改该 form 专属逻辑
  - 真源在 `pipeline / snapshot 导出问题` → 修导出真源，别改 scorer 规则
- 禁止默认用 fallback / 路由切换绕过问题，除非根因本身就是路由错误。
- 每处修改同步新增 / 更新针对该真源的测试，但只跑最小必要测试集。
- 每轮修复后只做内环轻量验证，不做最终成功判定：
  - 只重处理本轮受影响文档，按 ticker 聚合后使用 `process --document-id ...`
  - 若本轮涉及多个 ticker，同时并发运行，**并发数固定 27**
  - 只重跑受影响 form 的 score：`python utils/llm_ci_score.py --forms ... --tickers ... --tag iter_{round_id}`
  - 本轮比较只用于判断当前方向是否有效，至少看：
    - 目标文档 GateFail 是否消失或减少
    - 目标文档对应维度分数是否改善
    - 受影响 form 在本轮 ticker 子集上的 `avg` / `p10` / `hard_gate_failures` 是否恶化
  - 若无收益或出现恶化，停止叠补丁，回到问题分析重新判根因。
- 在报告中持续记录每轮问题簇、真源归类、修复位置、增量验证结论、仍未修复的问题与原因。

### Step 3：全量 CI 满足成功标准
- 当 N 轮优化结束后，执行一次全量评分，并产出 `workspace/tmp/ci_score/final/overall_summary.json`。
- 最终成功判定只按全量口径：
  - `final.overall_avg > baseline.overall_avg`
  - `final.overall_hard_gate_failures` 不高于基线，且可修复项应尽可能减少
- `summary.json` 中各 form 指标用于诊断收益来源和剩余短板，不是最终唯一成功定义。
- 在最终报告中必须包含：
  - Step 1 基线摘要
  - Step 2 每轮优化记录
  - Step 3 全量最终对比
  - 被判定为“原文如此”的问题
  - 最终人工检查命令

## 最终人工检查命令（写入报告，不自动执行）
- `python utils/llm_ci_process.py`
- `python utils/llm_ci_score.py --tag final`
- `pytest tests/ -x`
- pyright
- README 对齐检查

## 必须避免的低效路径
- 不要每轮先对全量 ticker 执行 `process --ci --overwrite`
- 不要每轮都重跑 7 个 form 的全量 score
- 不要主要依赖 stdout 文本解析
- 不要把共享根因修成多个 form 文件里的平行补丁
- 不要因为 fallback 分数更高就直接切路由

## 预期产出
1. `workspace/ci_optimization_report_mmdd_HHMM.md`
2. `utils/llm_ci_process.py`
3. `utils/llm_ci_score.py`
4. `workspace/tmp/ci_score/baseline/*`
5. 每轮 `workspace/tmp/ci_score/iter_{round_id}/*`
6. 对应代码修复与测试
