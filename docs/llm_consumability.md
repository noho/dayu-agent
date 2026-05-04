
## LLM 可喂性（LLM Consumability）框架

### 核心定义

> **LLM 可喂性** = LLM agent 通过 tool 调用获取的文档数据，能否**准确、高效、无歧义**地支撑金融分析任务。

评估的不是「文档是否完整」，而是「agent 是否能用」。

本框架先定义跨市场共享的抽象维度，再由不同 profile 把抽象维度落到具体市场、文档类型、结构信号与阈值。SEC profile 使用 SEC form / Item 规则；CN/HK Docling profile 使用 A 股、港股财报与材料类文档的中文/繁中文结构信号。二者同源于 LLM 可喂性目标，但不能共享同一套 Item 覆盖率和关键章节阈值。

### 通用评估维度

| 维度 | 跨市场抽象 | profile 落地点 |
|------|------------|----------------|
| 结构可导航性 | agent 能否从章节列表快速定位应读区域 | SEC 使用 form / Item / Part；CN/HK 使用 report kind、目录层级、中文/繁中文章节标题 |
| 文本可读性 | `read_section` 返回文本是否可直接理解 | 按语言、OCR、编码、页眉页脚噪声设定检测规则 |
| 表格可消费性 | `list_tables/get_table` 是否支持数值推理 | 按表格标题、表头、行列质量、财务表语义、合并/母公司维度设定规则 |
| 搜索可用性 | `search_document` 是否能稳定返回证据 | 按 query pack、market、report kind 分层设定召回与证据质量阈值 |
| 一致性与可追溯性 | section/table/page refs 是否能跨工具互相解引用 | 统一检查悬挂 ref、跨工具 ref 集、section/table/page 关联 |
| 噪声与完整性 | 处理产物是否缺失关键内容或混入干扰文本 | 按文档类型设定空章节、目录污染、截断、边界溢出、编码噪声规则 |
| 语义可寻址性 | agent 能否按业务语义而非字符串猜测路由 | SEC 使用 Item→topic；CN/HK 使用财报章节语义、报表类型、治理/股东/风险等标签 |

### Profile 分层原则

- **SEC profile**：覆盖 10-K、10-Q、20-F、6-K、8-K、SC 13G、DEF 14A。评分入口为当前已实现的 `dayu.fins.score_sec_ci`，规则真源是 SEC form / Item / Part 与事件类表单关键词。
- **CN/HK Docling profile**：覆盖 A 股、港股 filing/material 的 Docling JSON 工具快照。CN/HK 没有 `Item 7`、`Item 8` 等 SEC 法定结构，不评估 SEC Item 覆盖率；应按年报、半年报、季报、材料类文档分别设定关键章节、表格、搜索和页面定位规则。
- **快照真源**：所有 profile 都以 `dayu-cli process --ci` 产出的 `tool_snapshot_*` 为评分输入；文档、processed、blob、snapshot 的发现与读取必须走 `dayu.fins.storage` 仓储协议和实现。

---

## SEC Profile

### 适用表单与结构差异

| 维度 | 10-K（年报） | 10-Q（季报） | 20-F（外国发行人年报） |
|------|-------------|-------------|------------------|
| 适用主体 | 美国本土上市公司 | 美国本土上市公司 | 在美上市的外国私人发行人（FPI） |
| 标准 Item 数 | 23 个（Part I–IV） | 11 个（Part I–II） | 30 个（Item 1–19 + 4A + 16A–16J） |
| 关键 Item | Item 1, 1A, 7, 7A, 8, 15 | Part I–Item 1, Part I–Item 2, Part II–Item 1A | Item 3, 4, 5, 8, 11, 18 |
| 财务报表 | 经审计，完整附注（US GAAP） | Condensed / Unaudited | 经审计，IFRS 或 US GAAP |
| 典型篇幅 | 80–300 页 | 30–80 页 | 100–400+ 页 |
| Part 前缀必要性 | 信息性（Item 编号全局唯一） | **必须**（Part I/II Item 编号重叠） | 信息性（Item 编号全局唯一） |

> **为什么 Part 前缀在 10-Q 更重要？** 10-Q 的 Part I Item 2（MD&A）与 Part II Item 2（Unregistered Sales）含义完全不同。若标题中缺少 Part 前缀，agent 无法区分两者，直接导致分析错误。

> **20-F 与 10-K 的关系**：20-F 是 SEC 为外国私人发行人（Foreign Private Issuer, FPI）设计的年报表单，等价于本土公司的 10-K。Item 编号体系完全不同（Item 1–19），但内容对应关系明确：Item 3 ≈ Risk Factors，Item 5 ≈ MD&A，Item 18 ≈ Financial Statements。多数 FPI 采用 IFRS 而非 US GAAP，财务报表可能仅有表格提取而无 XBRL 结构化数据。

<details>
<summary><b>20-F Item 完整结构（SEC Form 20-F）</b></summary>

| Item | 含义 | 10-K 对应 | 备注 |
|------|------|----------|------|
| **Part I** | | | |
| Item 1 | Identity of Directors, Senior Management | Item 10 | 公司董事和高管信息 |
| Item 2 | Offer Statistics and Expected Timetable | — | 上市公司通常为 N/A |
| Item 3 | **Key Information / Risk Factors** | **Item 1A** | 含 3A–3D 子项，3D = Risk Factors |
| Item 4 | **Information on the Company** | **Item 1** | 业务描述、组织架构、资产 |
| Item 4A | Unresolved Staff Comments | Item 1B | SEC 审核意见 |
| **Part II** | | | |
| Item 5 | **Operating and Financial Review** | **Item 7 (MD&A)** | 经营分析与财务回顾 |
| Item 6 | Directors, Senior Management, Employees | Item 10/11 | 管理层与薪酬 |
| Item 7 | Major Shareholders and Related Party | Item 12/13 | 大股东与关联交易 |
| Item 8 | **Financial Information** | **Item 8** | 包含分配政策和法律诉讼 |
| Item 9 | The Offer and Listing | — | 上市信息，已上市公司通常极短 |
| Item 10 | Additional Information | Item 14 | 章程、合同、税务信息 |
| Item 11 | **Quant & Qual Market Risk Disclosures** | **Item 7A** | 市场风险定量定性披露 |
| Item 12 | Description of Securities Other Than Equity | — | 非权益证券描述 |
| **Part III** | | | |
| Item 13 | Defaults, Dividend Arrearages | Item 9 | 通常为 "None" |
| Item 14 | Material Modifications to Rights of Securities | — | 通常为 "None" |
| Item 15 | Controls and Procedures | Item 9A | 内控与披露控制 |
| Item 16 | [Reserved] | Item 6 | 保留项 |
| Item 16A–16J | Corporate Governance Disclosures | — | 审计委员会、行为准则等 10 个子项 |
| **Part IV** | | | |
| Item 17 | Financial Statements (alternative) | — | Item 18 的替代选项，极少使用 |
| Item 18 | **Financial Statements** | **Item 8/15** | IFRS 或 US GAAP 审计报表 |
| Item 19 | Exhibits | Item 15/16 | 附件索引 |

</details>

---

### 事件/治理/持股类表单

区别于年报/季报的标准 Item 结构，以下四类表单具有不同的章节组织方式：

| 维度 | 6-K（外国发行人临时报告） | 8-K（重大事件报告） | SC 13G（大额持股披露） | DEF 14A（代理声明书） |
|------|------|------|------|------|
| 适用主体 | 外国私人发行人（FPI） | 美国本土上市公司 | 实益持有人/投资机构 | 上市公司全体股东 |
| 标准章节数 | 无固定结构 | 26+ 可选 Item（每次仅触发 1–3 个） | 10 个（Item 1–10） | 无固定 Item 结构 |
| 章节组织方式 | 主题式标题 | **Item X.XX** 双级小数编号 | **Item N** 简单编号 | **Proposal No.** 议案编号 + 主题标题 |
| 关键内容 | 季度财务结果、IFRS 调和 | 披露事件细节和附件 | 持股比例和身份信息 | 高管薪酬、董事选举、股权结构 |
| 典型篇幅 | 5–20 页 | 2–10 页 | 2–10 页 | 40–100+ 页 |
| XBRL 财务报表 | ❌ | ❌ | ❌ | ❌ |
| Part 前缀必要性 | 不适用 | 不适用 | 不适用 | 不适用 |

> **6-K 与 8-K 的关系**：6-K 是 FPI 版本的 8-K + 10-Q 混合体。FPI 不提交 8-K，而是通过 6-K 报告重大事件和季度业绩。经 6-K 筛选规则过滤后，保留的 6-K 主要为季度/半年度业绩发布（RESULTS_RELEASE）和 IFRS 调和（IFRS_RECON）。

> **8-K 的事件驱动特性**：每份 8-K 仅包含被触发的 Item（通常 1–3 个），不是所有 Item 都出现。典型组合：Item 2.02（业绩发布）+ Item 9.01（附件），或 Item 5.02（高管变动）。

> **SC 13G 的模板式结构**：SC 13G 具有 SEC 规定的完全固定结构（Item 1–10），每份都相同。内容高度格式化，多数 Item 为简短填充式字段。SC 13G/A（修订版）与 SC 13G 使用相同的评估参数。

> **DEF 14A 的议案导向结构**：DEF 14A 以股东大会投票议案（Proposal）为核心组织结构，辅以主题性章节（如 Executive Compensation、Security Ownership）。无标准 Item 编号，使用关键词匹配识别关键章节。

<details>
<summary><b>8-K Item 完整结构（SEC Form 8-K）</b></summary>

| Item | 含义 | 说明 |
|------|------|------|
| **Section 1 — Registrant's Business and Operations** | | |
| Item 1.01 | Entry into a Material Definitive Agreement | 签署重大协议 |
| Item 1.02 | Termination of a Material Definitive Agreement | 终止重大协议 |
| Item 1.03 | Bankruptcy or Receivership | 破产或接管 |
| Item 1.04 | Mine Safety — Reporting of Shutdowns | 矿安全 |
| Item 1.05 | Material Cybersecurity Incidents | 重大网络安全事件 |
| **Section 2 — Financial Information** | | |
| Item 2.01 | Completion of Acquisition or Disposition of Assets | 资产收购/处置完成 |
| Item 2.02 | **Results of Operations and Financial Condition** | 经营结果与财务状况 |
| Item 2.03 | Creation of a Direct Financial Obligation | 产生直接财务义务 |
| Item 2.04 | Triggering Events That Accelerate/Increase Obligation | 触发加速/增加义务 |
| Item 2.05 | Costs Associated with Exit or Disposal Activities | 退出/处置成本 |
| Item 2.06 | Material Impairments | 重大减值 |
| **Section 3 — Securities and Trading Markets** | | |
| Item 3.01 | Notice of Delisting or Failure to Satisfy Continued Listing Rule | 退市通知 |
| Item 3.02 | Unregistered Sales of Equity Securities | 未注册股权销售 |
| Item 3.03 | Material Modification to Rights of Security Holders | 权利重大变更 |
| **Section 4 — Matters Related to Accountants** | | |
| Item 4.01 | Changes in Registrant's Certifying Accountant | 更换审计师 |
| Item 4.02 | Non-Reliance on Previously Issued Financial Statements | 不再依赖已发布财报 |
| **Section 5 — Corporate Governance and Management** | | |
| Item 5.01 | Changes in Control of Registrant | 控制权变更 |
| Item 5.02 | **Departure/Election of Directors or Officers** | 董事/高管变动 |
| Item 5.03 | Amendments to Articles of Incorporation or Bylaws | 章程修订 |
| Item 5.04 | Temporary Suspension of Trading Under EBP | EBP 交易暂停 |
| Item 5.05 | Amendments to the Code of Ethics | 道德准则修订 |
| Item 5.06 | Change in Shell Company Status | 壳公司状态变更 |
| Item 5.07 | Submission of Matters to a Vote of Security Holders | 提交股东投票 |
| Item 5.08 | Shareholder Nominations | 股东提名 |
| **Section 6 — Asset-Backed Securities** | | |
| Item 6.01 | ABS Informational and Computational Material | 资产支持证券信息与计算材料 |
| Item 6.02 | Change of Servicer or Trustee | 资产支持证券服务机构/受托人变更 |
| Item 6.03 | Change in Credit Enhancement or Other External Support | 信用增级或外部支持变更 |
| Item 6.04 | Failure to Make a Required Distribution | 未按要求进行分配 |
| Item 6.05 | Securities Act Updating Disclosure | 《证券法》更新披露 |
| Item 6.06 | Static Pool | 静态资产池披露 |
| **Section 7 — Regulation FD** | | |
| Item 7.01 | Regulation FD Disclosure | FD 监管披露 |
| **Section 8 — Other Events** | | |
| Item 8.01 | Other Events | 其他事件 |
| **Section 9 — Financial Statements and Exhibits** | | |
| Item 9.01 | **Financial Statements and Exhibits** | 财务报表和附件 |

</details>

<details>
<summary><b>SC 13G Item 完整结构（SEC Schedule 13G）</b></summary>

| Item | 含义 | 说明 |
|------|------|------|
| Item 1 | Issuer Name and Ticker | 发行人名称与代码 |
| Item 2 | **Identity and Background of Filing Person** | 申报人身份信息 |
| Item 3 | Source and Amount of Funds | 资金来源与金额 |
| Item 4 | **Ownership Details** | 持股比例与投票权详情 |
| Item 5 | Ownership of 5% or Less | 低于 5% 持股声明 |
| Item 6 | Ownership Attributable to Another Person | 代持声明 |
| Item 7 | Identification and Classification of Subsidiary | 子公司信息 |
| Item 8 | Identification and Classification of Members | 成员信息 |
| Item 9 | Notice of Dissolution of Group | 解散声明 |
| Item 10 | Certification | 认证声明 |

</details>

<details>
<summary><b>DEF 14A 关键章节与关键词匹配规则</b></summary>

DEF 14A 不使用标准 Item 编号，通过以下关键词匹配识别关键章节：

| 标准化标签 | 匹配关键词（不区分大小写，子串匹配） | 分析用途 |
|------|------|------|
| Executive Compensation | "executive compensation", "compensation discussion" | 高管薪酬分析 |
| Directors | "election of director", "directors" | 董事会构成分析 |
| Security Ownership | "security ownership", "beneficial ownership" | 股权集中度分析 |

</details>

<details>
<summary><b>6-K 关键章节与关键词匹配规则</b></summary>

6-K 不使用标准 Item 编号，经筛选后保留的 6-K 通过以下关键词匹配识别关键章节：

| 标准化标签 | 匹配关键词（不区分大小写，子串匹配） | 分析用途 |
|------|------|------|
| Financial Results | "financial results", "key highlights", "results of operations", "financial and business" | 季度业绩分析 |
| Safe Harbor | "safe harbor", "forward-looking" | 法律免责声明 |

</details>

---

### 一、结构可导航性（Structure Navigability）

**目标**：agent 调用 `get_document_sections` 后，能快速定位到需要阅读的章节。

| 指标 | 定义 | 优秀标准 |
|------|------|----------|
| **Item 覆盖率** | 标准 Item 中被识别为独立 section 的比例 | 见下表 |
| **标题可辨识度** | section title 是否包含可唯一定位的结构化标识 | 见下方按表单标准 |
| **SIGNATURE 存在性** | 是否识别出 SIGNATURE 章节 | 建议存在（当前脚本未计分） |
| **章节粒度合理性** | 单 section 字符数是否在合理区间 | 见下表 |
| **无重叠** | section 之间内容是否有大量重复 | 重叠率 < 5% |

**年报/季报表单差异参数**：

| 参数 | 10-K | 10-Q | 20-F |
|------|------|------|------|
| 总 Item 数 | 23 | 11 | 30 |
| 覆盖率优秀线 | ≥ 18/23（~78%） | ≥ 8/11（~73%） | ≥ 22/30（~73%） |
| 合理 section_count | 15–35 | 8–20 | 20–300¹ |
| 章节大小合理区间 | 2K–50K chars | 1K–30K chars | 2K–50K chars |

**事件/治理/持股类表单差异参数**：

| 参数 | 6-K | 8-K | SC 13G | DEF 14A |
|------|-----|-----|--------|--------|
| 结构类型 | 主题式（关键词匹配） | Item X.XX 双级编号 | Item N 简单编号 | 议案式 + 主题标题（关键词匹配） |
| 合理 section_count | 3–15 | 2–8 | 5–12 | 8–25 |
| 章节大小合理区间 | 100–20K chars | 50–10K chars | 50–5K chars | 500–50K chars |

> 6-K 和 DEF 14A 无标准 Item 编号，不评估 Item 覆盖率和排序。8-K 的 Item 为事件驱动（每次仅出现 1–3 个），Item 排序评分有效但覆盖率不评估。

> ¹ 20-F Risk Factors（Item 3）展开为 HTML 标题子节时（如 FUTU=163 节、TCOM=274 节），section_count 天然偏大。A3 高题放宽至 300 避免误做惩罚。

**标题可辨识度优秀线（按表单）**：
- 10-Q：100% 标题含 `Part X - Item N`（必须保留 Part 前缀）
- 10-K / 20-F / SC 13G / 8-K：100% 标题含标准 Item 编号（如 `Item 7` / `Item 2.02`）
- 6-K / DEF 14A：100% 关键章节标题可命中关键词映射标签

**为什么重要**：agent 收到 section 列表后，根据标题决定读哪些章节。标题不清 → 读错章节 → 回答偏差。单 section 过大 → 超 context window 或注意力稀释。

---

### 二、文本可读性（Text Readability）

**目标**：agent 调用 `read_section` 后，得到的文本可以直接理解，无需额外清洗。

| 指标 | 定义 | 优秀标准 |
|------|------|----------|
| **语言纯净度** | 英文文档中不含中文/乱码等异语言污染 | 0 次 |
| **页码噪音** | 内容中是否混入页眉页脚/页码（如 `F-23`） | 0 次 |
| **编码完整性** | 无 mojibake、`\xa0`、`â€™` 等编码残留 | 0 次 |
| **表格占位符可追溯** | `[[tbl_xxxx]]` 占位符是否可通过 `get_table` 解引用 | 100% 可解引用 |
| **空白规范性** | 无连续空行、无异常缩进、无断行乱码 | 格式自然 |

**为什么重要**：LLM 对噪音文本的理解力急剧下降。一个 `F-23` 页码可能被解读为财务数据。

> 本维度与表单类型无关，评估处理质量本身，10-K / 10-Q / 20-F 阈值完全一致。

---

### 三、表格可消费性（Table Consumability）

**目标**：agent 调用 `get_table` 后，表格数据可以直接用于数值推理。

| 指标 | 定义 | 优秀标准 |
|------|------|----------|
| **格式友好度** | markdown 格式 vs records+pipe 列名 | 100% markdown |
| **NaN 清洁度** | 数据中 NaN/null 出现次数 | 0 |
| **列名可解读性** | 列名是否为有意义的文本（非 `0`, `1`, `Unnamed`） | 0 默认列名 |
| **管道符列名** | MultiIndex 扁平化导致的 `col_a\|col_b` 列名 | 0 |
| **幽灵列** | 全空列的数量 | 0 |
| **Financial 标注准确性** | `is_financial=True` 的表格是否确实是财务表格 | 精确率 >80% |
| **表格上下文** | 表格是否有 caption 或所属 section 信息 | 有 section_ref |
| **微型表格比例** | 行数 ≤1 或列数 ≤1 的表格占比 | < 10% |

**为什么重要**：LLM 做数值推理（如计算同比增长率）时，表格格式直接决定能否正确提取数字。`revenue|2024|2023` 这种管道符列名会导致 LLM 解析错误。

> 本维度与表单类型无关。10-Q 的 condensed 财务报表在表格格式上与 10-K 无本质差异。

---

### 四、检索有效性（Search Effectiveness）

**目标**：agent 调用 `search_document` 后，能稳定拿到可引用的证据，并具备足够检索效率。

| 指标 | 定义 | 优秀标准 |
|------|------|----------|
| **C1 覆盖率（9）** | `coverage_rate_weighted = sum(hit*weight)/sum(weight)` | 进入 9 分档 |
| **C2 证据质量（4）** | 命中 query 中，`evidence.context` 非空、长度 `[120,1200]`、`section_ref` 有效的比例 | `>=0.90` |
| **C3 检索效率（2）** | `efficiency_rate = exact_hit_queries / all_hit_queries`（基于 diagnostics.strategy_hit_counts） | `>=0.70` |

**为什么重要**：agent 先搜索再精读是常见模式。搜索失效 = agent 无法找到信息 → 回答「未找到相关信息」。

> 本维度逻辑复用，但阈值按 `query_pack/form` 分层配置，不再使用统一命中率阈值。

#### C1 覆盖率分层阈值（调试期固定）

| query_pack | 适用表单 | t5 | t7 | t9 |
|------|------|------|------|------|
| annual_quarter_core40 | 10-K / 10-Q / 20-F / FY / H1 | 0.55 | 0.70 | 0.85 |
| event_pack | 8-K / 6-K | 0.25 | 0.40 | 0.55 |
| governance_pack | DEF 14A | 0.50 | 0.65 | 0.80 |
| ownership_pack | SC 13* | 0.65 | 0.80 | 0.90 |

---

### 五、信息完整性（Information Completeness）

**目标**：原始文档中的关键信息不丢失。

| 指标 | 定义 | 优秀标准 |
|------|------|----------|
| **财务报表可达性** | 三大报表（IS/BS/CF）能否通过 XBRL 获取 | 100% |
| **关键 Item 覆盖** | 表单关键 Item 是否存在且内容非空 | 100% |
| **Cover Page 信息** | 封面页是否包含公司名、CIK、fiscal year 等 | 包含核心信息 |
| **表格无丢失** | 原文中有意义的表格是否都被提取 | 无明显遗漏 |

**关键 Item 与内容充足性阈值**：

10-K：

| Item | 含义 | 最低长度 | 评分权重 |
|------|------|---------|---------|
| Item 1A | Risk Factors | ≥ 5,000 chars | 8 |
| Item 7 | MD&A | ≥ 5,000 chars | 8 |
| Item 8 | Financial Statements | ≥ 10,000 chars | 10 |
| Item 7A | Quant & Qual Market Risk | ≥ 500 chars | 4 |

10-Q：

| Item | 含义 | 最低长度 | 评分权重 |
|------|------|---------|---------|
| Part I – Item 1 | Financial Statements（condensed） | ≥ 5,000 chars | 10 |
| Part I – Item 2 | MD&A | ≥ 3,000 chars | 12 |
| Part II – Item 1A | Risk Factor Updates | ≥ 2,000 chars | 8 |

20-F：

| Item | 含义 | 10-K 对应 | 最低长度 | 评分权重 |
|------|------|---------|---------|----------|
| Item 3 | Key Information / Risk Factors | Item 1A | ≥ 5,000 chars | 8 |
| Item 5 | Operating and Financial Review | Item 7 (MD&A) | ≥ 5,000 chars | 8 |
| Item 18 | Financial Statements | Item 8 | ≥ 10,000 chars | 10 |
| Item 11 | Quant & Qual Disclosures About Market Risk | Item 7A | ≥ 500 chars | 4 |

6-K（章节以关键词匹配识别）：

| 章节关键词 | 含义 | 最低长度 | 评分权重 |
|------|------|---------|----------|
| Financial Results / Key Highlights | 季度财务结果 | ≥ 300 chars | 20 |
| Safe Harbor / Forward-Looking | 安全港声明 | ≥ 100 chars | 10 |

8-K：

> 8-K 为事件驱动型表单，每次仅包含被触发的 1–3 个 Item，内容长度不具有可比性。内容维度默认满分（30/30）。

SC 13G：

| Item | 含义 | 最低长度 | 评分权重 |
|------|------|---------|----------|
| Item 2 | 申报人身份信息（Identity and Background） | ≥ 50 chars | 15 |
| Item 4 | 持股比例详情（Ownership Details） | ≥ 50 chars | 15 |

DEF 14A（章节以关键词匹配识别）：

| 章节关键词 | 含义 | 最低长度 | 评分权重 |
|------|------|---------|----------|
| Executive Compensation | 高管薪酬 | ≥ 2,000 chars | 12 |
| Directors / Election of Director | 董事信息 | ≥ 500 chars | 10 |
| Security Ownership / Beneficial Ownership | 股权结构 | ≥ 300 chars | 8 |

> 所有表单的内容权重总计均为 **30 分**。其中 8-K 因事件驱动特性不设内容阈值，默认满分。
>
> **SEC 交叉引用豁免（仅限特定 Item）**：为避免短文本误判，内容长度阈值的交叉引用豁免仅适用于法律上常见的少数场景：10-K 的 Item 8、10-Q 的 Part II–Item 1A、20-F 的 Item 18。其他关键 Item 不适用交叉引用豁免。

---

### SEC CI 110 分制评分框架（`score_sec_ci.py`）

#### 评分维度（总分 110）

| 维度 | 权重 | 说明 |
|------|------|------|
| A. 结构完整性 | 25 | Item 顺序（8）+ 关键 Item 存在（10）+ section_count 合理（7） |
| B. 内容充足性 | 30 | 各关键 Item 达到最低长度阈值 |
| C. 检索可用性 | 15 | 覆盖率（9）+ 证据质量（4）+ 检索效率（2） |
| D. 一致性与数据质量 | 15 | 表格 ref 可追溯（3）+ ref 一致（3）+ 财报完整性（4）+ 表格数据质量（3）+ caption 填充率（2） |
| E. 噪声与完整性 | 15 | 空/近空 section（3）+ mojibake（3）+ 最大 section（3）+ 截断检测（3）+ 边界溢出（3） |
| S. 语义可寻址性 | 10 | Level-1 非特殊 section 的 topic 填充率；仅 10-K / 10-Q / 20-F，其他表单默认满分 |

#### 硬门禁（Hard Gate）

以下任一条件命中即 CI 直接失败：

| 规则 | 10-K | 10-Q | 20-F |
|------|------|------|------|
| 缺失关键 Item | Item 1A / 7 / 8 | Part I–Item 1 / Part I–Item 2 | Item 3 / 5 / 18 |
| ToC 污染 | Item 7 / Item 8 内容疑似目录 | Part I–Item 1 / Part I–Item 2 内容疑似目录 | Item 5 / Item 18 内容疑似目录 |
| 超大 section | > 300K chars | > 300K chars | > 300K chars |
| 悬挂 table refs | 任何悬挂引用 | 任何悬挂引用 | 任何悬挂引用 |
| ref 不一致 | 跨工具 section refs 不一致 | 跨工具 section refs 不一致 | 跨工具 section refs 不一致 |

**事件/治理/持股类硬门禁**：

| 规则 | 6-K | 8-K | SC 13G | DEF 14A |
|------|-----|-----|--------|--------|
| 缺失关键 Item | — | — | — | — |
| ToC 污染 | — | — | — | — |
| 超大 section | > 300K chars | > 300K chars | > 300K chars | > 300K chars |
| 悬挂 table refs | 任何悬挂引用 | 任何悬挂引用 | 任何悬挂引用 | 任何悬挂引用 |
| ref 不一致 | 跨工具 section refs 不一致 | 跨工具 section refs 不一致 | 跨工具 section refs 不一致 | 跨工具 section refs 不一致 |

> 事件/治理/持股类表单不设 Item 缺失和 ToC 污染硬门禁：这些表单的 Item 结构或为事件驱动（8-K）、或为非标准化（6-K、DEF 14A）、或为极短文档（SC 13G）。

#### CI 阈值

| 层级 | 指标 | 通过 | 警告 |
|------|------|------|------|
| 单文档 | total_score | ≥ 93 | ≥ 83 |
| 批量 | 平均分 | ≥ 93 | — |
| 批量 | P10 分位 | ≥ 86 | — |
| 批量 | 硬门禁失败数 | = 0 | — |

> **阈值调整说明**：总分上限由 100 → 110（新增 S 维度 10 分），阈值按等比调整：$85\% \times 110 \approx 93$，$75\% \times 110 \approx 83$，$78\% \times 110 \approx 86$。portfolio 重新处理后可重跑 `score_sec_ci` 重新校准基线。

#### 使用方式

```bash
# 评分 10-K
python -m dayu.fins.score_sec_ci --form 10-K --tickers AAPL,AMZN,V,TDG,AXON

# 评分 10-Q
python -m dayu.fins.score_sec_ci --form 10-Q --tickers AAPL,AMZN,V,TDG,AXON

# 评分 20-F
python -m dayu.fins.score_sec_ci --form 20-F --tickers TSM,ASML

# 评分 6-K
python -m dayu.fins.score_sec_ci --form 6-K --tickers TCOM

# 评分 8-K
python -m dayu.fins.score_sec_ci --form 8-K --tickers AAPL,AMZN,V,TDG,AXON

# 评分 SC 13G
python -m dayu.fins.score_sec_ci --form "SC 13G" --tickers AMZN

# 评分 DEF 14A
python -m dayu.fins.score_sec_ci --form "DEF 14A" --tickers AAPL,AMZN,V,TDG,AXON

# Makefile 集成
make score-ci                        # 默认 10-K
make score-ci FORM=10-Q              # 指定 10-Q
make score-ci FORM=20-F              # 指定 20-F
```

---

### SEC profile 维度适用性总结

| 维度 | 跨表单复用性 | 需按表单调整的参数 |
|------|------------------|----------|
| 结构可导航性 | 逻辑复用，参数重设 | Item 列表、覆盖率阈值、section_count 区间 |
| 文本可读性 | **100% 复用** | 无 |
| 表格可消费性 | **100% 复用** | 无 |
| 检索有效性 | 逻辑复用，阈值参数化 | query_pack、form_type、C1 分层阈值 |
| 信息完整性 | 逻辑复用，参数重设 | 关键 Item、内容长度阈值、ToC 检测目标、近空白名单 |
| **语义可寻址性** | **仅 10-K / 10-Q / 20-F；其他表单默认满分** | 表单是否具备标准 Item→topic 映射 |

> SEC profile 当前评分维度是 **form-agnostic** 的：它定义的是《agent 能否用好这份 SEC filing》，而不是《这份 filing 是哪个 form》。差异仅在于**哪些 Item/章节是关键的、多长算充足、结构匹配方式（Item 编号/关键词/议案号）**——这些都是 SEC profile 内部的参数调整，不改变 SEC 评估逻辑。对于事件/治理类表单（6-K、8-K、SC 13G、DEF 14A），SEC profile 自适应：无标准 Item 结构的表单使用关键词匹配识别关键章节，事件驱动型表单（8-K）的内容长度检测默认满分，无 XBRL 的表单财报完整性默认满分，语义可寻址性（S）不适用表单默认满分。

---

### 评分维度 D / E / S 详细设计

#### D. 一致性与数据质量（15 分）

| 子项 | 分值 | 评分逻辑 |
|------|------|----------|
| D1: 表格 ref 可追溯 | 3 | `read_section.tables[]` 引用的 ref 全部可在 `list_tables` 中找到，且 `list_tables.section_ref` 全部可在 `get_document_sections` 中找到。有悬挂 → 0 分 |
| D2: 跨工具 ref 一致 | 3 | `get_document_sections` 的 ref 集合 = `read_section` 的 ref 集合。不一致 → 0 分 |
| D3: 财报完整性 | 4 | 五大报表（IS/BS/CF/equity/comprehensive_income）：存在且 xbrl 质量（2 分）+ 平均行数 ≥ 10（1 分）+ 所有报表期数 ≥ 2（1 分） |
| D4: 表格数据质量 | 3 | 非空表格占比（根据 `data.kind` 检查对应格式：markdown/records/raw_text）：≥ 90% → 3 分，≥ 70% → 2 分，否则 0 分 |
| D5: caption 填充率 | 2 | 表格 caption 填充率（HTML `<caption>` 原生提供 + `context_before` 自动推断）：≥ 40% → 2 分，≥ 20% → 1 分，< 20% → 0 分。可通过改进 `text_utils.infer_caption_from_context` 算法持续优化 |

> 对于不含 XBRL 财务报表的表单（6-K、8-K、SC 13G、DEF 14A），D3 财报完整性默认满分（4/4），因为缺少 XBRL 是表单类型特性而非处理质量问题。

#### E. 噪声与完整性（15 分）

| 子项 | 分值 | 评分逻辑 |
|------|------|----------|
| E1: 空/近空 section | 3 | 空 section 每个扣 1 分；近空（1–100 chars 且非 SEC 白名单 Item）每个扣 0.5 分。下限 0 |
| E2: Mojibake | 3 | 全文 mojibake模式匹配命中数：0 → 3 分，否则 0 分 |
| E3: 最大 section 尺寸 | 3 | ≤ 200K → 3 分，≤ 300K → 1 分，> 300K → 0 分 |
| E4: 章节截断检测 | 3 | 检测 section 内容以悬挂介词/连词结尾（to, and, or, the, of, in, for, with, refer, see, including 等）。0 个 → 3 分，≤ 2 个 → 1 分，> 2 个 → 0 分 |
| E5: 边界溢出检测 | 3 | 检测 section 末尾是否包含其他 Part/Item 的标题文本（如 Part I section 末尾出现 "PART II"）。0 个 → 3 分，≤ 2 个 → 1 分，> 2 个 → 0 分 |

> **SEC 白名单 Item**（允许近空）：
> - **10-K**：Item 1B, 1C, 3, 4, 5, 6, 9, 9C — SEC 规定内容可为 “None” 或 “Not applicable”
> - **10-Q**：Part I-Item 3、Part I-Item 4、Part II-Item 1、Part II-Item 1A、Part II-Item 2、Part II-Item 3、Part II-Item 4、Part II-Item 5 — 允许简短声明；Part I-Item 1/2 不在白名单
> - **20-F**：Item 2, 9, 12, 13, 14, 16, 16A–16J, 17 — FPI 年报中大量监管披露项允许 “N/A”
> - **8-K**：Item 9.01 — 附件列表通常极短
> - **6-K**：无白名单 — 6-K 章节数少，每个章节都应有意义内容
> - **SC 13G**：Item 5, 6, 7, 8, 9, 10 — 多数为 “N/A” 或简短法定声明
> - **DEF 14A**：无白名单 — 使用关键词匹配，非关键词章节不进入 Item 评估
#### S. 语义可寻址性（10 分）

**适用范围**：仅限 10-K / 10-Q / 20-F。6-K / 8-K / SC 13G / DEF 14A 默认满分（10/10）。

**为什么 S 只针对年报/季报表单？**标准化 Item → topic 映射只存在于具有固定 Item 结构的表单。对于事件驱动型（8-K）、关键词匹配型（6-K、DEF 14A）、模板式短文档（SC 13G），语义路由已通过其他机制实现，无需 topic 字段。

**评分规则**：

| 分层 | topic 填充率 | 得分 |
|------|------|------|
| S1 | ≥ 90% | 10 |
| S2 | ≥ 70% | 6 |
| S3 | ≥ 50% | 3 |
| S4 | < 50% | 0 |

**给分对象定义**：
- 取 `get_document_sections` 返回的全部 `level=1` section
- 排除 **Cover Page**（封面页，标题精确匹配）和 **SIGNATURE**（`topic="signature"`）
- 计算剩余 section 中 `topic ≠ null` 的比例

**为什么对写作重要**：agent 延读 10-K / 10-Q / 20-F 时，首先按语义标签（`risk_factors`、`mda`、`financial_statements`）路由工具调用。若 `topic` 字段全为 `null`，agent 需逐字匹配标题字符串，容易转到错误章节、漏读关键内容。
---

### SEC 当前评估脚本覆盖情况

| 维度 | 已覆盖 | 未覆盖（待后续迭代） |
|------|--------|---------------------|
| 结构可导航性 | sections、Items 覆盖、排序、section_count | 章节粒度合理性（单 section 大小分布）、内容重叠率 |
| 文本可读性 | mojibake 检测 | 编码残留（`\xa0` 等）、占位符可追溯性 |
| 表格可消费性 | ref 可追溯性、**表格数据质量**、**caption 填充率**（D5） | Financial 精确率、微型表格比例 |
| 检索有效性 | coverage_rate_weighted、evidence_quality_rate、efficiency_rate | query_intent 级别召回（需标注集） |
| 信息完整性 | 关键 Item 存在 + 长度、**财报深度**（NEW）、**截断检测**（NEW）、**边界溢出**（NEW） | Cover Page 信息完整性 |
| **语义可寻址性** | **topic 覆盖率检测（NEW）；仅 10-K / 10-Q / 20-F** | XBRL 事实细粒度（query_xbrl_facts 每文档仅 1 调用，样本量不足） |

---

## CN/HK Docling Profile

### 适用对象与评分输入

CN/HK Docling profile 适用于 A 股与港股的 `filing`、`material` 文档，评分输入是 CN/HK pipeline 通过 `process --ci` 导出的 Docling JSON 工具快照。源 PDF 到 Docling JSON 的转换属于 pipeline 处理链路，scorer 不直接读取 PDF 或手拼工作区路径。

该 scorer 入口尚未实现，本文只定义 Phase 1 可直接落地的 scorer 边界：

- 新增独立入口，例如 `dayu.fins.score_docling_ci`；不把 CN/HK 规则硬塞进 `dayu.fins.score_sec_ci`。
- 支持 `SourceKind.FILING` 与 `SourceKind.MATERIAL`，发现样本、读取 source/processed/blob/snapshot 一律走 `dayu.fins.storage`。
- 读取现有 `tool_snapshot_*`：`get_document_sections`、`read_section`、`list_tables`、`get_table`、`get_page_content`、`get_financial_statement`、`search_document`、`query_xbrl_facts`。CN/HK Docling 首版评分主要依赖 section/table/page/search 快照，不把 XBRL 可用性作为 hard gate。
- 输出 JSON/MD 报告时可复用 SEC scorer 的报告形态，但 profile、扣分项、hard gate 必须独立。

### Report Kind

CN/HK 不按 SEC form 评分，首版按 report kind 分层：

| report kind | 适用文档 | 结构预期 |
|-------------|----------|----------|
| 年报 | A 股年度报告、港股年报 | 完整经营讨论、公司治理、股东信息、审计意见、三大报表、附注 |
| 半年报 | A 股半年度报告、港股中期报告 | 经营讨论、主要财务数据、三大报表、附注或简明附注、重大事项 |
| 季报 | A 股一季报、三季报、港股季度业绩或季度报告 | 主要财务数据、经营简述、三大报表或核心财务表、风险/重大事项提示 |
| 材料 | 公告、通函、业绩快报、业绩预告、投资者材料等 | 主题标题、事件正文、影响说明、风险提示或董事会/监管声明 |

report kind 由 source meta 中的现有字段与 pipeline 已推导的文档类型共同确定。scorer 只能把它作为 profile 选择信号，不能把 CN/HK 文档强行映射成 `10-K/10-Q/8-K`。

### 结构可导航性指标

| 指标 | 年报 | 半年报 | 季报 | 材料 |
|------|------|--------|------|------|
| section_count 合理区间 | 20–220 | 12–140 | 6–90 | 1–50 |
| 标题可辨识度 | 中文/繁中文一级、二级标题可读，非目录行堆叠 | 同年报 | 允许更短，但标题需能定位财务与经营内容 | 标题需体现公告或材料主题 |
| 目录层级 | 目录页可存在，但正文 section 不应被目录替代 | 同年报 | 记录目录污染，首版不强制 gate | 通常不要求目录 |
| 关键章节召回 | 公司简介、主营业务、管理层讨论与分析或董事会报告、主要会计数据和财务指标、公司治理、股东信息、重大事项/风险提示、审计意见 | 主营业务或经营情况、管理层讨论、主要财务数据、重大事项/风险提示 | 主要财务数据、经营情况或管理层讨论、风险/重大事项提示 | 事件主题、正文、影响、风险提示 |
| 财务结构召回 | 资产负债表、利润表、现金流量表、合并/母公司报表、附注 | 三大报表、合并/母公司或简明报表、附注或简明附注 | 三大报表或核心财务表 | 仅对业绩类材料要求财务摘要 |
| 父子层级 | 子章节 parent_ref 应反映目录层级 | 同年报 | 记录为主 | 记录为主 |

关键章节采用中文/繁中文同义词匹配与标题层级共同判定，例如“管理层讨论与分析”“管理层讨论及分析”“董事会报告”“主要会计数据和财务指标”“主要財務資料”“綜合損益表”“合併資產負債表”。禁止把这些信号包装成 SEC Item 变体。

### 文本可读性指标

| 指标 | 定义 | 首版判定 |
|------|------|----------|
| CJK 文本完整性 | 中文/繁中文正文不应被 OCR 空格、逐字断行、乱码切碎 | 年报/半年报 gate；季报/材料记录并扣分 |
| 编码清洁度 | 无 mojibake、异常替换字符、HTML 实体残留 | 进入 hard gate 候选 |
| 页眉页脚噪声 | 公司简称、页码、报告名不应在正文高频重复并干扰段落 | 首版记录并扣分，不 gate |
| 目录污染 | 关键章节 `read_section` 只返回目录页或目录片段 | 年报/半年报 gate |
| 空/近空 section | 非法定短声明章节不应为空或近空 | 全 report kind 扣分；材料类按篇幅放宽 |
| 中繁混合可读性 | 港股繁中文、A 股简中文可混排，但不能出现编码污染 | 记录语言分布，编码污染扣分 |

### 表格可消费性指标

| 指标 | 定义 | 首版判定 |
|------|------|----------|
| 财务表召回 | `list_tables` 能识别三大报表和主要财务指标表 | 年报/半年报 gate；季报对核心财务表 gate |
| 表格标题与上下文 | snapshot 暴露的 caption、within_section/section_ref、page_no、headers 至少一项能说明表格含义；`context_before` 当前仅作为 processor 内部信号，不作为首版 snapshot 必填字段 | 扣分；低填充率进入问题簇 |
| 合并/母公司维度 | 年报、半年报应区分合并报表与母公司报表标题或上下文 | 首版记录并扣分，不 gate |
| 表头可读性 | 列名不是默认数字、空列、`Unnamed`、不可解释碎片 | 扣分 |
| 数值可读性 | 金额单位、期间列、本期/上期、同比等列能保留 | 扣分 |
| 空表/微型表 | 行数或列数过低的表格占比不应过高 | 记录并扣分，不 gate |
| NaN/null 清洁度 | `get_table` 返回不应含大量 NaN/null 或结构化空洞 | 扣分 |

CN/HK 财务表首版关键词至少覆盖：资产负债表、利润表、现金流量表、合并资产负债表、母公司资产负债表、合并利润表、母公司利润表、合并现金流量表、母公司现金流量表、綜合財務狀況表、綜合損益表、綜合現金流量表、主要会计数据和财务指标、主要財務資料。

### 搜索可用性指标

| 指标 | 定义 | 首版判定 |
|------|------|----------|
| query pack 覆盖率 | CN/HK `annual_quarter_core40`、event/governance/ownership pack 的加权命中率 | 按 market/report kind 分层阈值 |
| 证据质量 | `evidence.context` 非空、长度适中、带有效 `section_ref` | gate 候选；低于阈值扣分 |
| 搜索效率 | exact/title/section 命中占比，不应主要依赖低质量全文兜底 | 扣分 |
| 简繁适配 | HK profile 应命中繁中文关键词，CN profile 应命中简中文关键词；同义词可辅助 | 记录无结果 query 与语言分布 |
| 关键主题召回 | 主营业务、营业收入、净利润、现金流、主要股东、关联交易、风险提示、审计意见等可搜索 | 年报/半年报扣分重点 |

首版阈值建议：

| report kind | C1 覆盖率 t5 | C1 覆盖率 t7 | C1 覆盖率 t9 |
|-------------|--------------|--------------|--------------|
| 年报/半年报 | 0.55 | 0.70 | 0.85 |
| 季报 | 0.45 | 0.60 | 0.75 |
| 材料 | 0.25 | 0.40 | 0.55 |

### 一致性与可追溯性指标

| 指标 | 定义 | 首版判定 |
|------|------|----------|
| section ref 一致 | `get_document_sections` 返回 refs 与 `read_section` 可读 refs 一致 | hard gate |
| table ref 可追溯 | `read_section.content` 中的 `[[t_xxxx]]` 占位符、`list_tables.table_ref`、`get_table` 可互相解引用 | hard gate |
| section/table 归属 | 财务表应有有效 `within_section`/`section_ref`、页码或标题上下文 | 扣分 |
| snapshot 元信息一致 | `tool_snapshot_meta` 的 `market`、`source_kind`、`document_type`、`search_query_pack_name`、`search_query_pack_version`、`search_query_count`、`search_queries` 与样本一致 | hard gate |
| 同源读取 | scorer 发现和读取文档、snapshot 不绕过仓储 | 实现验收项 |

### Docling 页面定位指标

| 指标 | 定义 | 首版判定 |
|------|------|----------|
| section page_range | 关键章节应有 page_range，且页码为正整数 | 年报/半年报扣分重点 |
| table page_no | 财务表应有 page_no | 财务表缺失 page_no 扣分 |
| page_content 可复核 | `get_page_content(page_no)` 能返回对应 section/table 或文本预览 | hard gate 候选 |
| 页码边界 | page_range 不应倒置、越界或全部为空 | hard gate 候选 |

Docling 页面定位是 CN/HK profile 的独立维度，因为 PDF 转 Docling 后，人工复核和后续归因高度依赖页码。SEC HTML/XBRL profile 不应照搬此维度的权重。

### 初步 Hard Gate 建议

| 规则 | 年报 | 半年报 | 季报 | 材料 |
|------|------|--------|------|------|
| 关键章节完全缺失 | 缺三大报表任一、缺管理层讨论/董事会报告、缺附注或审计意见 | 缺三大报表任一、缺经营讨论、缺附注/简明附注 | 缺核心财务表或主要财务数据 | 缺标题或正文 |
| 目录污染 | 关键章节正文疑似仅目录 | 关键章节正文疑似仅目录 | 记录并扣分 | 不适用 |
| 超大 section | 单 section > 300K chars | 同年报 | 同年报 | 同年报 |
| 悬挂 refs | 任何悬挂 section/table ref | 同年报 | 同年报 | 同年报 |
| 页面定位不可复核 | 关键章节和财务表均无有效页码 | 同年报 | 财务表无有效页码 | 记录并扣分 |
| snapshot 元信息不一致 | `market`、`source_kind`、`document_type`、`search_query_pack_name`、`search_query_pack_version`、`search_query_count`、`search_queries` 缺失或冲突 | 同年报 | 同年报 | 同年报 |

首版不建议 gate 的指标：

- 合并/母公司报表区分不完整：先记录并扣分，避免把标题表达差异误判成不可用。
- 页眉页脚高频噪声：先记录噪声模式，形成问题簇后再决定阈值。
- 空表/微型表比例：先观察 Docling 对不同 PDF 的分布，避免惩罚真实披露中的小表。
- 简繁同义词互召回：先记录 no-hit query 与 market 语言分布，避免用错误语言假设惩罚港股双语文档。
- 材料类文档的财务表覆盖：只有业绩类材料进入财务表要求，普通公告不 gate。

### 架构边界

- `dayu.engine.processors.docling_processor.DoclingProcessor` 只放通用 Docling 抽取能力：线性 items、sections、tables、`read_section`、`read_table`、search、page_content。
- `dayu.fins.processors.fins_docling_processor.FinsDoclingProcessor` 与 fins 处理层放 CN/HK 财报语义增强，例如金融表格识别、中文/繁中文章节语义、三大报表定位。
- scorer/profile 放评分规则、阈值、hard gate、report kind 选择；CN/HK scorer 不反向污染 engine，也不改 SEC scorer 的法定 Item 规则。
- pipeline/snapshot 只负责同源快照导出，继续复用 `process --ci` 与 `tool_snapshot_*`，不在导出阶段内嵌评分规则。
- 所有文档、processed、blob、snapshot 读取必须走 `dayu.fins.storage`，禁止手拼 `workspace/portfolio/...` 路径。
