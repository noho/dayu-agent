# 大愚 Agent — 用户手册

`大愚 Agent` 是每个投资者的助理分析师。  
- `大愚 Agent` 是一个面向买方财报分析场景的 Agent 系统，但它不是简单的功能组合，`大愚 Agent` 让AI读财报的方式从丢给它整份财报“大海捞针”变成“按图索骥”，让数据有置信度，让投资结论、投资报告可审计、可追踪。  
- `大愚 Agent` 还具备完整的“宿主强约束下的 LLM in the loop 的能力”，基础架构能力上已经对齐 OpenClaw ，后续会加上现在 OpenClaw 能做的事情。

当前你可以用它完成四类工作：
- 财报数据管线：美股财报下载、美股 / A 股 / 港股财报上传。
- 投研问答：下载、上传财报后，执行 `prompt` 单次提问、`interactive` 多轮提问、或通过微信向`大愚 Agent` 提问。
- 买方分析报告写作：下载、上传财报后，执行 `write` 写作。
- 结果渲染：把 Markdown 报告渲染为 HTML / PDF / Word。

本文档面向读者：
- 最终使用者。

如果你要参与开发，而不是只使用系统：
- 总览开发手册：[dayu/README.md](dayu/README.md)
- Host 手册：[dayu/host/README.md](dayu/host/README.md)
- Engine 手册：[dayu/engine/README.md](dayu/engine/README.md)
- Fins 手册：[dayu/fins/README.md](dayu/fins/README.md)
- 配置手册：[dayu/config/README.md](dayu/config/README.md)
- 贡献指南：[CONTRIBUTING.md](CONTRIBUTING.md)

## 0. 如果你想参与项目
- 定性分析模板 读起来机械感还很强，还没写出差异化：
  - 同一章节里，不同行业公司写出明显不同的判断路径。
  - 同一行业里，不同公司写出公司自己的特殊结构变量。
- 位于 Engine 的 web tools 现在的对抗challenge能力很弱，很多网站无法访问。
- 位于 Fins 的港股、A股财报下载功能尚未实现。
- **GUI 尚未实现**；
- **Web UI 目前仍只有 FastAPI 骨架**。
- **WeChat UI 仅支持文本消息首版，还可添加更多好玩的功能**。
- 财报电话会议记录音频转录文字后信息提取（起码要区分信息来自提问还是回答）尚未实现。
- 财报presentation信息提取尚未实现。
- 欢迎围绕以下方向提交 issue 或 PR：
  - 普通文件（非财报文件）信息提取还需要优化。
  - 优化 Fins 里的港股/A股/美股财报信息提取。
  - Anthropic 原生 API 支持。
  - Durable memory / Retrieval layer（ Memory只实现了working memory 和 episode summary ）。
  - FMP 工具（调研工作已做，见 [docs/fmp_integration_research.md](docs/fmp_integration_research.md) ）尚未实现。
  - 更多LLM 工具。

## 1. 五分钟快速开始

### 1.1 安装

安装使用 `大愚 Agent` 前需安装Python 3.11。

#### 1.1.1 在线安装

如果你当前机器可以联网，可以直接安装通过 `pip` 命令安装：

命令格式：

```bash
pip install https://github.com/noho/dayu-agent/releases/download/<version>/dayu_agent-<version>-py3-none-any.whl
```

示例（替换为最新版本号）：

```bash
pip install https://github.com/noho/dayu-agent/releases/download/v0.1.3/dayu_agent-0.1.3-py3-none-any.whl
```

这种方式最轻，但安装耗时和成功率会受网络、平台和上游依赖发布状态影响。

#### 1.1.2 离线安装

从 [Releases](https://github.com/noho/dayu-agent/releases) 页面下载对应平台的离线安装包：

- Mac ARM芯片：`dayu-agent-<version>-macos-arm64-offline.tar.gz`
- Mac Intel芯片：`dayu-agent-<version>-macos-x64-offline.tar.gz`
- Linux：`dayu-agent-<version>-linux-x64-offline.tar.gz`
- Windows：`dayu-agent-<version>-windows-x64-offline.zip`


macOS / Linux 示例：

```bash
tar -xzf dayu-agent-0.1.3-macos-arm64-offline.tar.gz
cd dayu-agent-0.1.3-macos-arm64-offline
./install.sh
```

Windows PowerShell 示例：

```powershell
Expand-Archive .\dayu-agent-0.1.3-windows-x64-offline.zip -DestinationPath .
cd .\dayu-agent-0.1.3-windows-x64-offline
.\install.cmd
```
#### 1.1.3 clone 源代码安装

源代码clone到本地后，运行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,dev,browser]" -c constraints/lock-macos-arm64-py311.txt
```

说明：

- macOS Intel 开发环境改用 `constraints/lock-macos-x64-py311.txt`
- Linux 开发环境改用 `constraints/lock-linux-x64-py311.txt`
- Windows 开发环境改用 `constraints/lock-windows-x64-py311.txt`

#### 1.1.4 安装额外依赖

安装完成后，还需要执行一次：  

```bash
playwright install chromium
```

如需渲染 PDF，可选安装 `pandoc`：

- macOS：`brew install pandoc`
- Ubuntu / Debian：`sudo apt-get install pandoc`
- Windows：`choco install pandoc` 或从 [pandoc 官网](https://pandoc.org/installing.html) 下载安装

### 1.2 验证安装

安装完成后，先确认命令入口可用：

```bash
dayu-cli --help
dayu-wechat --help
dayu-render --help
```

### 1.3 初始化工作区与配置

安装后运行一次 `init`，交互式完成配置复制、模型供应商选择和 API Key 设置：

```bash
dayu-cli init
```

`init` 会依次执行：

1. 复制包内默认配置到 `./workspace/config/` ，复制包内默认写作模板到 `./workspace/assets/` 。
2. 让你选择初始化模型方案（Mimo Token Plan / Mimo Token Plan SG / Mimo Pro / DeepSeek Pro / DeepSeek Flash / OpenAI / Anthropic / Gemini / 通义千问 / 自定义 OpenAI 兼容 API）。
3. 输入对应 API Key 并永久写入环境变量。
4. 可选配置联网检索 API Key（TAVILY / SERPER / FMP）
5. 自动检测 HuggingFace 官方 Hub 连通性：不可达时默认启用镜像加速，可达时默认跳过。可选配置 `HF_TOKEN` 提升下载稳定性。


可选参数：

```bash
dayu-cli init --base ./my_workspace    # 指定工作区目录（默认 ./workspace）
dayu-cli init --reset                  # 删除 .dayu / config / assets 后重新初始化
dayu-cli init --overwrite              # 覆盖已有配置
```

API Key 申请地址：
- MIMO_PLAN_API_KEY / MIMO_PLAN_SG_API_KEY / MIMO_API_KEY：https://platform.xiaomimimo.com/#/console/api-keys
- DEEPSEEK_API_KEY：https://platform.deepseek.com/api_keys
- FMP_API_KEY：https://site.financialmodelingprep.com/developer/docs/dashboard
- TAVILY_API_KEY：https://app.tavily.com/home
- SERPER_API_KEY：https://serper.dev/

说明：
- 默认推荐 Mimo Token Plan（mimo-v2.5-pro-plan），性价比最优。（注： MIMO_PLAN_API_KEY / MIMO_API_KEY 是两个不同的KEY，不能混用）。
- 海外用户选Mimo Token Plan SG。
- 如需接入 OpenRouter 等聚合服务，可在 `init` 中选择“自定义 OpenAI 兼容 API”，填写 `CUSTOM_OPENAI_API_KEY`、Base URL 与模型 ID。
- `--reset` 确认后会删除 `workspace/.dayu/`、`workspace/config/`、`workspace/assets/`，再按首次初始化流程重建；它比 `--overwrite` 更彻底，会一并清空运行时状态。
- 联网搜索默认可走 `auto`，若配置了 Tavily / Serper，会优先使用对应 provider。
- 若运行环境需要访问 `localhost`、私网 IP 或内网域名，可在 `workspace/config/run.json` 的 `web_tools_config.allow_private_network_url` 中显式打开内网访问开关。
- 修改默认模型请参考 [8. 模型配置](#model-config)。

工作区最重要的目录：

```text
workspace/
├── config/           # 运行时配置（覆盖包内默认配置）
├── assets/           # 定性分析模板（覆盖包内默认模板）
├── .dayu/            # 系统隐藏工作目录（batch 暂存、备份恢复等）
├── portfolio/        # 每个 ticker 的财报与材料
├── draft/            # write 输出目录
└── output/           # tool trace 等辅助输出
```

说明：`workspace/.dayu/` 由系统自动维护，当前会承载财报仓储的 batch 暂存与 crash recovery 备份；不需要手动创建或清理。如果运行有异常全部删除也没有影响。  

### 1.4 跑通第一条命令

推荐先下载一份财报：

```bash
dayu-cli download --ticker AAPL
```

下载完成后，再跑一条单次 prompt：

```bash
dayu-cli prompt "总结最新财报的主要风险" --ticker AAPL
```

如果你已经通过 `download`、`upload_filing` 或 `upload_filings_from` 导入过 AAPL 的财报，也可以直接提问；命令会自动检测本地财报并挂载财报工具后返回结果。

如果你希望先不指定 `ticker`，也可以这样写：

```bash
dayu-cli prompt "总结苹果最新财报的主要风险"
```

> 也可在微信对话或`interactive`里发送"下载苹果财报"进行下载。


## 2. 系统入口总览

### 2.1 CLI 入口

统一入口：

```bash
dayu-cli <subcommand> [参数]
```

直接执行 `dayu-cli` 会打印总帮助和全部子命令简介；需要查看某个子命令的完整参数时，继续使用 `dayu-cli <subcommand> --help`。

当前支持的主命令：

| 命令 | 用途 |
|------|------|
| `prompt` | 单次问答 |
| `interactive` | 交互式终端对话 |
| `write` | 自动逐章写作报告；传 `--summary` 时打印上次写作结果摘要 |
| `download` | 下载 filings |
| `upload_filing` | 上传单份财报 |
| `upload_filings_from` | 扫描目录并生成批量上传脚本 |
| `upload_material` | 上传补充材料 |
| `process` | 全量预处理（最终用户可无视） |
| `process_filing` | 预处理单份 filing（最终用户可无视） |
| `process_material` | 预处理单份 material（最终用户可无视） |
| `conv` | 管理带 label 的可恢复 CLI 对话（最终用户可无视） |
| `sessions` | 列出或关闭宿主会话（最终用户可无视） |
| `runs` | 列出运行记录（最终用户可无视） |
| `cancel` | 取消运行中的 run（最终用户可无视） |
| `host` | 宿主维护（清理孤儿运行/查看状态，最终用户可无视） |
> 注：预处理命令仅供开发使用，最终用户可忽略。

共享参数：

| 参数 | 适用命令 | 说明 |
|------|------|------|
| `--base` | 全部主命令 | 工作区根目录，默认 `./workspace` |
| `--config` | 全部主命令 | 配置目录，默认 `workspace/config` |
| `--ticker` | `prompt` `write` | 股票代码；传入后会把该 `ticker` 作为当前研究对象 |
| `--log-level` | 全部主命令 | 直接指定日志级别，可选 `debug`、`verbose`、`info`、`warn`、`error` |
| `--debug` | 全部主命令 | 把日志级别设为 `DEBUG` |
| `--verbose` | 全部主命令 | 把日志级别设为 `VERBOSE` |
| `--info` | 全部主命令 | 把日志级别设为 `INFO` |
| `--quiet` | 全部主命令 | 把日志级别设为 `ERROR` |
| `--model-name` | `prompt` `interactive` `write` | 指定模型配置名称 |
| `--temperature` | `prompt` `interactive` `write` | 覆盖模型 temperature |
| `--label` | `prompt` `interactive` | 把当前对话绑定到可恢复 label；`prompt` 会进入 labeled multi-turn，对应 scene 为 `prompt_mt` |
| `--new-session` | `interactive` | 不续接上一次 interactive 多轮会话，改为从头开始一个新会话 |
| `--web-provider` | `prompt` `interactive` `write` | 指定联网检索 provider，如 `auto`、`tavily`、`serper`、`duckduckgo` |
| `--enable-tool-trace` | `prompt` `interactive` `write` | 开启工具调用追踪，覆盖 `run.json` 中的 trace 配置 |
| `--tool-trace-dir` | `prompt` `interactive` `write` | 指定 trace 输出目录，覆盖 `run.json` 中的 trace 配置 |
| `--thinking` / `--no-thinking` | `prompt` `interactive` | 控制是否在终端回显模型思考过程 |

说明：
- `--log-level`、`--debug`、`--verbose`、`--info`、`--quiet` 是同一组日志参数，使用其一即可。
- `prompt`、`interactive`、`write` 还支持更多 Agent 运行参数，例如 `--tool-timeout-seconds`、`--max-iterations`、`--doc-limits-json`、`--fins-limits-json`；需要时可用 `dayu-cli <subcommand> --help` 查看完整列表。
- 宿主管理命令同样支持 `--base` / `--config` / 日志参数；例如 `dayu-cli host --base ./workspace status`、`dayu-cli sessions --base ./workspace --source cli --scene interactive`、`dayu-cli conv --base ./workspace list`。
- `interactive` 默认会续接本地绑定的同一个多轮会话；如果上一次回答还没完整回显到终端，重启 CLI 会先把那次回答补完，再进入新的输入循环。

### 2.2 WeChat 入口

统一入口：

```bash
dayu-wechat <command> [参数]
```

直接执行 `dayu-wechat` 会打印总帮助和全部命令简介；需要查看某个命令的完整参数时，继续使用 `dayu-wechat <command> --help`。

当前支持的命令：

| 命令 | 用途 |
|------|------|
| `login` | 扫码建立微信登录态 |
| `run` | 以前台方式运行微信问答 daemon |
| `service install` | 安装后台托管配置 |
| `service start` | 启动后台托管 |
| `service restart` | 重启后台托管 |
| `service stop` | 停止后台托管 |
| `service status` | 查看后台托管状态 |
| `service list` | 列出当前 workspace 下已安装的后台托管实例 |
| `service uninstall` | 删除后台托管配置 |

常用参数：

| 参数 | 适用命令 | 说明 |
|------|------|------|
| `--base` | 全部命令 | 工作区根目录，默认 `./workspace` |
| `--config` | `login` `run` `service install` | 配置目录，默认 `workspace/config` |
| `--label` | 全部命令 | WeChat 实例标签，默认 `default`；状态目录映射到 `<base>/.dayu/wechat-<label>` |
| `--relogin` | `login` | 忽略已有登录态，强制重新扫码 |
| `--qrcode-timeout-sec` | `login` | 扫码登录超时秒数 |
| `--model-name` | `run` `service install` | 指定模型配置名称 |
| `--temperature` | `run` `service install` | 覆盖模型 temperature |
| `--web-provider` | `run` `service install` | 指定联网检索 provider |
| `--debug-sse` | `run` `service install` | 开启 SSE 高频调试日志 |
| `--debug-tool-delta` | `run` `service install` | 开启工具调用参数增量日志 |
| `--debug-sse-sample-rate` | `run` `service install` | 设置 SSE 调试日志采样率 |
| `--debug-sse-throttle-sec` | `run` `service install` | 设置 SSE 调试日志节流窗口 |
| `--tool-timeout-seconds` | `run` `service install` | 覆盖工具超时 |
| `--max-iterations` | `run` `service install` | 覆盖 Agent 最大迭代次数 |
| `--fallback-mode` | `run` `service install` | 覆盖超限处理模式 |
| `--fallback-prompt` | `run` `service install` | 覆盖超限补充提示 |
| `--max-consecutive-failed-tool-batches` | `run` `service install` | 覆盖连续失败工具批次上限 |
| `--max-duplicate-tool-calls` | `run` `service install` | 覆盖重复工具调用连续上限 |
| `--duplicate-tool-hint-prompt` | `run` `service install` | 覆盖重复工具调用提示词 |
| `--enable-tool-trace` | `run` `service install` | 开启工具调用追踪 |
| `--tool-trace-dir` | `run` `service install` | 指定 trace 输出目录 |
| `--doc-limits-json` | `run` `service install` | 覆盖文档工具 limits |
| `--fins-limits-json` | `run` `service install` | 覆盖财报工具 limits |
| `--typing-interval-sec` | `run` `service install` | 控制 typing 提示发送间隔 |
| `--delivery-max-attempts` | `run` `service install` | 控制微信 reply delivery 的最大发送次数 |

说明：
- `login` 用于建立或刷新登录态。
- `run` 用于在当前终端以前台方式运行。
- 同一个 `--label` 对应同一个 `state_dir`；当前实现会对 `state_dir` 加 daemon 单实例锁，避免前台 `run` 和后台 service 或两个前台进程并发运行导致重复补发。
- `service install/start/stop/status/list/uninstall` 用于以后台服务的形式运行。
- Windows 目前不支持 `service` 相关命令；在 Windows 上可继续使用 `login` 和 `run`。

## 3. 最常用工作流

### 3.1 财报下载：`download`

命令用途：
下载美股财报到本地工作区，供后续问答、对话和写作复用。
**目前还不支持港股A股财报下载**

参数 / 说明：

| 参数 | 说明 |
|------|------|
| `--ticker` | 必填，股票代码，传入半角逗号分隔的股票代码标识有多地上市 |
| `--forms` | 可选，指定表单类型，如 `10K`、`10Q`、`DEF14A` |
| `--start` | 可选，开始日期，支持 `YYYY`、`YYYY-MM`、`YYYY-MM-DD` |
| `--end` | 可选，结束日期，支持 `YYYY`、`YYYY-MM`、`YYYY-MM-DD` |
| `--overwrite` | 可选，覆盖本地已存在结果 |
| `--rebuild` | 可选，只基于本地已下载文件重建 `meta/manifest`，不重新下载 |
| `--infer` | 可选，使用 FMP 推断跨市场 alias；成功时与 SEC alias 合并，失败时回退到 `--ticker` CSV 中的显式 alias |
| `--base` | 可选，工作区根目录，默认 `./workspace` |
| `--config` | 可选，配置目录，默认 `workspace/config` |

命令示例：

```bash
dayu-cli download --ticker AAPL
```

常见命令示例：

```bash
dayu-cli download --ticker AAPL --forms 10K 10Q --start 2024 --end 2025
dayu-cli download --ticker AAPL --forms 10K
dayu-cli download --ticker AAPL --rebuild
dayu-cli download --ticker BABA,9988,9988.HK --infer
```

命令说明：
- `download` 会根据 `ticker` 自动路由到对应市场。
- `download`、`upload_filing`、`upload_material`、`upload_filings_from` 的 `--ticker` 支持 CSV（半角逗号分隔）；CSV 中**每个 token 都会走真源归一化**（如 `9988.HK`→`9988`）后再整体去重。首个归一化结果作为 canonical ticker，其余作为显式 alias 写入 meta，便于工具后续用任意跨市场变形命中同一公司。
- `--ticker` 支持 `0700.HK` / `HK.00700` / `600519.SH` / `sh600519` / `AAPL.US` 等常见变形，内部统一归一化到裸码（港 4 位补零、沪深 6 位、美股原字母）。公司名仍可作为 ticker 传入，由仓储 alias 查表兜底。
- 显式传 `--infer` 时，CLI 会把 `--ticker` 里的显式 alias 与 FMP infer 结果合并；`download` 场景下 pipeline 还会继续与 SEC 返回的 alias 合并。
- 使用 `--infer` 功能需要申请FMP_API_KEY。
- 首次写入时会自动创建 `workspace/portfolio/{ticker}` 下的源文档目录，不要求你预先手动建好 `filings/`。
- `prompt`、`interactive` 在 `filings/` 缺失时不会直接退出；CLI 会提示当前无本地财报，并继续执行问答。
- SEC 下载必须串行执行，不要并发跑多个 `download` 命令。
- **也可在interactive / wechat中发送`下载xx公司财报`进行下载**

### 3.2 上传本地文件

命令用途：  
上传本地下载好的财报。（**A股/港股财报目前只能上传不支持下载**）  
把你已经准备好的补充材料纳入工作区，适合手动整理 PDF、电话会纪要、演示材料等场景。

参数 / 说明：

| 命令 | 关键参数 | 说明 |
|------|------|------|
| `upload_filing` | `--ticker` `--files` `--fiscal-year` `--fiscal-period` | 上传单份财报；`--action` 可选，默认按 document_id 自动判定 |
| `upload_filings_from` | `--ticker` `--from` | 扫描目录并生成批量上传脚本 |
| `upload_material` | `--ticker` `--forms` `--material-name` `--files` | 上传补充材料；`--action` 可选，`--fiscal-year/--fiscal-period` 可选并参与稳定 document_id 生成 |

命令示例：

```bash
dayu-cli upload_filing \
  --ticker 0300 \
  --files ./tmp/美的2025Q1.pdf \
  --fiscal-year 2025 \
  --fiscal-period Q1 \
  --company-id 000333 \
  --company-name 美的集团
```

常见命令示例：

```bash
dayu-cli upload_filings_from \
  --ticker 0300 \
  --from ./workspace/source

dayu-cli upload_filing \
  --ticker BABA,9988 \
  --files ./tmp/alibaba_2025_q1.pdf \
  --fiscal-year 2025 \
  --fiscal-period Q1 \
  --company-id 1577552 \
  --infer

dayu-cli upload_material \
  --ticker AAPL \
  --forms EARNINGS_CALL \
  --material-name deck \
  --files ./tmp/deck.pdf
```

命令说明：
- `upload_filing` 和 `upload_material` 的 `--action` 现在都可省略；省略时会先按稳定 `document_id` 查工作区：不存在则 `create`，存在则 `update`，若原始上传文件指纹未变化则会在 Docling convert 前直接 `skip`。自动判定只覆盖 `create/update`；若要删除，必须显式传 `--action delete`。
- `upload_filing` 适合单份补录；每个 `ticker` 第一次上传财报时才需要 `--company-id` 和 `--company-name`。若显式传 `--infer`，则在工作区缺少公司级 `meta.json` 时可省略 `--company-name`，由 FMP 推断后补齐；若同时传了 `--company-name`，则以你显式传入的值为准；若 infer 失败且仍缺 `--company-name`，命令会直接失败。
- `upload_material` 的稳定 `document_id` 默认由 `form_type + material_name` 生成；若显式提供 `--fiscal-year/--fiscal-period`，它们也会参与 ID 生成。material 场景下 `document_id` 与 `internal_document_id` 恒等；显式传 `--document-id/--internal-document-id` 时，必须与这套稳定规则一致。
- `upload_filings_from` 不直接上传文件，而是先生成一份适配当前运行平台的可执行脚本；macOS / Linux 默认生成 `.sh`，Windows 默认生成 `.cmd`。
- `upload_filings_from` 未传 `--output` 时，默认把脚本写到 `--base` 指向的 workspace 根目录，文件名为 `upload_filings_{ticker}.sh` / `.cmd`。
- `upload_filings_from --infer` 只会在脚本生成阶段调用一次 FMP，并把“显式 CSV alias + infer alias”的合并结果，以及最终公司名直接 bake 到脚本正文；脚本头部的重生成命令仍会保留原始 `--ticker` 输入和 `--infer`。
- 使用 `--infer` 功能需要申请FMP_API_KEY。
- 生成脚本头部会附带一条注释形式的 `python -m dayu.cli upload_filings_from ...` 重跑命令；脚本正文里的批量上传命令也统一使用 `python -m dayu.cli`，这样在源码工作区里执行时不会依赖外部 `dayu-cli` entrypoint。
- `upload_filings_from` 默认不会在脚本正文里写死 `--action`，这样每条命令都会在执行时按当前工作区状态自动判定 `create/update/skip`；只有你显式传了 `--action`，生成脚本才会固定动作。
- 生成脚本中的每条上传命令都会透传脚本调用时的额外参数；macOS / Linux 使用 `"$@"`，Windows 使用 `%*`，因此可直接执行 `./upload_filings_xxx.sh --overwrite` 之类的批量覆盖调用。
- `upload_filing --overwrite` 和 `upload_material --overwrite` 会先重置当前 `document_id` 的源文档存储，再完整重建该文档；不会像 SEC download 的 ticker 级 overwrite 那样清空同 ticker 下的其他文档。
- `upload_filing` 和 `upload_material` 在首次实际写入时会自动创建 `workspace/portfolio/{ticker}` 下的源文档目录；`upload_filings_from` 只生成批量上传脚本，不直接写入源文档。

### 3.3 单次问答：`prompt`

命令用途：
执行一次性提问，适合快速验证观点、提炼结论或补充某个具体问题。

参数 / 说明：

| 参数 | 说明 |
|------|------|
| `prompt` | 必填，单次执行的问题文本 |
| `--ticker` | 可选，指定研究对象 |
| `--label` | 可选，把本次提问绑定到可恢复 conversation；首次创建时 scene 为 `prompt_mt` |
| `--model-name` | 可选，指定模型配置 |
| `--temperature` | 可选，覆盖模型 temperature |
| `--thinking` / `--no-thinking` | 可选，控制是否回显模型思考过程 |
| `--debug` / `--verbose` | 可选，仅调整日志级别，不改变会话行为 |

命令示例：

```bash
dayu-cli prompt "总结苹果最新财报中的主要风险"
```

常见命令示例：

```bash
dayu-cli prompt "总结最新财报中的主要风险" --ticker AAPL
dayu-cli prompt "总结苹果最新财报中的主要风险" --thinking
dayu-cli prompt --label apple "先总结苹果最新财报中的主要风险"
dayu-cli prompt "总结苹果最新财报中的主要风险" --model-name mimo-v2.5-pro
dayu-cli prompt "总结苹果最新财报中的主要风险" --debug
```

命令说明：
- 使用之前请先下载/上传财报。
- 两种写法都可以：要么在问题里直接写公司名或股票代码，要么用 `--ticker` 明确指定研究对象；一般不需要两边重复写。
- 不带 `--label` 时，`prompt` 保持 one-shot，不承诺后续恢复；带 `--label` 时，本次提问会挂到该 label 对应的可恢复 conversation 上，后续可继续用 `prompt --label <label>` 或 `interactive --label <label>` 接着问。
- 带 `--label` 的 prompt 在本轮拿到最终回答前会独占该 label；如果另一个进程此时也尝试复用同一个 label，CLI 会直接报错并提示等待当前对话结束，或改用新的 `--label`。
- 默认不回显模型思考过程；如需在终端查看，显式传 `--thinking`。

### 3.4 交互式对话：`interactive`

命令用途：
启动一个终端会话，多轮会话连续追问，适合逐步拆解问题。

参数 / 说明：

| 参数 | 说明 |
|------|------|
| `--model-name` | 可选，指定模型配置 |
| `--temperature` | 可选，覆盖模型 temperature |
| `--thinking` / `--no-thinking` | 可选，控制是否回显模型思考过程 |
| `--label` | 可选，恢复或创建指定 label 的可复用 conversation；首次创建时 scene 为 `interactive` |
| `--new-session` | 可选，不续接上一次多轮会话，改为从头开始一个新会话 |
| `--debug` / `--verbose` | 可选，仅调整日志级别，不改变会话行为 |

命令示例：

```bash
dayu-cli interactive
```

常见命令示例：

```bash
dayu-cli interactive --model-name mimo-v2.5-pro
dayu-cli interactive --temperature 0.2
dayu-cli interactive --thinking
dayu-cli interactive --label apple
dayu-cli interactive --new-session
dayu-cli sessions --source cli --scene interactive
dayu-cli conv status --label apple
dayu-cli interactive --verbose
```

命令说明：
- 使用之前请先下载/上传财报。
- `interactive` 默认每次进入都会续接同一个多轮会话，适合连续追问。
- `interactive` 会把当前会话绑定保存在 `<workspace>/.dayu/interactive/state.json`，重新启动时默认续接上一次会话历史。
- 如果你想从头开始一轮新的对话，显式传 `--new-session`；它会丢弃本地保存的旧会话绑定，改为新开一个会话。
- 如果你想显式复用某条长期对话，使用 `--label`。同一个 label 可在 `prompt --label` 与 `interactive --label` 之间互通；第一次通过 `prompt --label` 创建的会话底层 scene 为 `prompt_mt`，第一次通过 `interactive --label` 创建的会话底层 scene 为 `interactive`，之后恢复时沿用首次创建时的 scene。
- 带 `--label` 的 CLI 启动时，会明确提示当前是“新创建标签”还是“恢复标签”；`prompt --label` 在回答末尾还会再次打印标签提示框，方便你后续继续复用同一个 label。
- 同一个 label 在任意时刻只能被一个 CLI 进程占用：`interactive --label` 会在整个 REPL 生命周期内持有该 label，直到双 `Ctrl+D` 完整退出；`prompt --label` 会在本轮返回最终回答前持有该 label。若命中占用中的 label，CLI 会提示你等待当前对话结束后重试，或改用新的 `--label`。
- 如果你在 workspace 本地覆写了 `prompt_mt` 或其他带 label 会命中的 scene manifest，必须保留 `conversation.enabled=true`；否则 CLI 会直接拒绝执行该 labeled conversation。
- 如果某个 label 对应的底层 session 已经被 `dayu-cli sessions close` 关闭，下次再用同名 `--label` 时，CLI 会先提示“旧对话已关闭”，再按全新对话重新创建该 label；如果你是通过 `conv remove --label` 主动释放 label，则下次直接按普通新建处理，不额外提示。
- 如果你想查看底层 Host session，可用 `dayu-cli sessions --source cli --scene interactive` 或 `dayu-cli sessions --source cli --scene prompt_mt`；如果你想查看或释放 label 到会话的映射，使用 `dayu-cli conv list`、`dayu-cli conv list --all`、`dayu-cli conv status --label <label>` 与 `dayu-cli conv remove --label <label>`。其中 `conv list` 默认只展示 active 的 labeled conversation，`conv list --all` 额外包含已关闭对话；若某个 label 的 registry record 已漂移到不存在的 Host session，CLI 会先自动清理再继续执行，不再展示 `missing`。`conv remove --label` 会先关闭底层 session（若仍存在），再释放该 label；之后同名 `--label` 会从全新对话开始。若你通过 `sessions close` 关闭了某个带 label 的底层 session，下次同名 `--label` 会在提示后创建新的会话，而不是恢复旧 transcript。若你需要诊断底层 `session_id`，请用 `conv status --label <label>` 或直接查看 `sessions`。
- 默认不回显模型思考过程；如需在终端查看，显式传 `--thinking`。

### 3.5 微信对话 daemon：

命令用途：
以ClawBot的形式运行微信问答通道。

参数 / 说明：

| 命令 | 关键参数 | 说明 |
|------|------|------|
| `login` | `--label` `--relogin` `--qrcode-timeout-sec` | 建立或刷新登录态 |
| `run` | `--model-name` `--temperature` `--web-provider` `--debug-sse` `--fallback-mode` `--enable-tool-trace` | 在当前终端以前台方式运行 |
| `service install` | `--label` `--model-name` `--temperature` `--web-provider` `--debug-sse` `--fallback-mode` `--enable-tool-trace` | 安装后台服务 |
| `service start` | `--label` | 启动后台服务 |
| `service restart` | `--label` | 重启后台服务 |
| `service stop` | `--label` | 停止后台服务 |
| `service status` | `--label` | 查看后台服务状态 |
| `service list` | 无 | 列出当前 workspace 下已安装的后台服务实例 |
| `service uninstall` | `--label` | 删除后台服务 |

说明：
- 使用之前请先下载/上传财报。
- 首次使用时，一般先执行 `login`，再执行 `run` 或 `service install`。
- `run` 直接在命令行窗口前台运行。
- `service` 适合长期后台运行。
- `service` 相关命令目前支持 macOS 和 Linux；Windows 暂未支持。
- 用不同的 `--label`，并分别由不同扫码主体执行 `login`，可以多开。默认实例标签是 `default`，对应状态目录 `workspace/.dayu/wechat-default`。一个完整例子如下：

```bash
# 实例 A：扫码主体 A 登录，安装并启动 service
dayu-wechat login --label a
dayu-wechat service install --label a --model-name mimo-v2.5-pro-thinking
dayu-wechat service start --label a

# 实例 B：扫码主体 B 登录，安装并启动 service
dayu-wechat login --label b
dayu-wechat service install --label b --model-name deepseek-v4-flash-thinking
dayu-wechat service start --label b

# 列出当前 workspace 下已安装的实例
dayu-wechat service list
```

- 多开后，`start` / `restart` / `stop` / `status` / `uninstall` 都要继续带对应实例的同一个 `--label`，这样命中的才是同一个后台 service；忘记有哪些实例时可直接执行 `dayu-wechat service list`。

命令示例：

```bash
dayu-wechat login
dayu-wechat run
```

常见命令示例：

```bash
dayu-wechat login --relogin
dayu-wechat run --model-name mimo-v2.5-pro-thinking --temperature 0.4
dayu-wechat run --enable-tool-trace
dayu-wechat service install
dayu-wechat service start
dayu-wechat service restart
dayu-wechat service stop
dayu-wechat service status
dayu-wechat service list
dayu-wechat service uninstall
```

命令说明：
- 同一微信会话里的连续追问会自动延续上下文，适合做多轮分析。
- 当前版本主要支持文本问答；更适合问财报、公司、行业和研究相关问题。
- 首次使用时先执行 `dayu-wechat login`；命令会打印并尝试打开登录二维码链接，用手机微信扫码确认即可。若你在管理多实例，统一用 `--label` 指定实例标签。
- `dayu-wechat run` 依赖本地已有登录态；若登录态失效，重新执行同一个 `--label` 的 `login` 即可。
- 同一个 `--label` 的前台 `run` 和后台 service 不能并发运行；新的 daemon 若发现该 `state_dir` 已被占用，会直接拒绝启动。
- macOS / Linux：如果你希望它长期后台运行，先执行 `dayu-wechat service install`，再执行 `dayu-wechat service start`。后续可用 `service restart`、`service stop`、`service status`、`service list`、`service uninstall` 管理。
- `service install` 会把当前 shell 里已设置的关键环境变量快照进后台 service 定义，包括配置文件里 `{{ENV_VAR}}` 占位符引用到的变量，以及少量代码直读变量（如 `SEC_USER_AGENT`、联网检索/FMP API key）。如果你后来改了 API key，需要重新执行一次 `dayu-wechat service install`；若后台 service 已在运行，再执行 `dayu-wechat service restart` 让新配置生效。
- `dayu-wechat service status --label <name>` 会直接打印日志定位信息：macOS 打印 stdout/stderr 文件路径；Linux 打印 `journalctl --user -u <label>.service -f` 查看命令。
- `dayu-wechat service list` 只列出当前 workspace 下已安装的实例，并回显实例标签、状态目录、系统 service label、运行状态和是否已有登录态。
- macOS 下默认日志分流语义是：stdout 文件保留全量运行日志，stderr 文件额外记录真正错误与异常堆栈；因此 ERROR 会同时出现在两边。
- Windows：目前没有后台托管命令，使用方式是先执行 `dayu-wechat login`，再执行 `dayu-wechat run`，需要持续运行时请保持终端窗口开启。
- 若需要重新扫码登录，可重启命令并加上 `--relogin`。

### 3.6 自动写作：`write`

命令用途：
基于模板逐章生成买方分析报告，适合在财报与补充材料准备好后批量写作。

参数 / 说明：

| 参数 | 说明 |
|------|------|
| `--ticker` | 必选，指定研究对象 |
| `--chapter` | 可选，只写指定章节 |
| `--fast` | 可选，只执行写作，不运行 `audit` / `confirm` / `repair` |
| `--force` | 可选，允许1-9章audit失败也能写作第 0 章和第 10 章 |
| `--infer` | 可选，只执行公司级 facet 归因并写回 manifest |
| `--summary` | 可选，只打印上次写作结果摘要，不进入写作 |
| `--resume` / `--no-resume` | 可选，控制是否断点恢复 |
| `--template` | 可选，写作模板路径，默认 `workspace/assets/定性分析模板.md`，回退 `dayu/assets/定性分析模板.md` |
| `--output` | 可选，输出目录，默认 `workspace/draft/{ticker}` |
| `--model-name` | 可选，主写作模型配置 |
| `--audit-model-name` | 可选，审计模型配置 |
| `--debug` / `--verbose` | 可选，仅调整日志级别，不改变会话行为 |

命令示例：

```bash
dayu-cli write --ticker AAPL
```

常见命令示例：

```bash
dayu-cli write --ticker AAPL --chapter "公司做的是什么生意"
dayu-cli write --ticker AAPL --chapter "经营表现与核心驱动" --fast
dayu-cli write --ticker AAPL --infer
dayu-cli write --ticker AAPL --summary
dayu-cli write --ticker AAPL \
  --template ./workspace/assets/定性分析模板.md \
  --output ./workspace/draft/AAPL \
  --enable-tool-trace
```

命令说明：
- 写任何章节前，系统都会先检查当前 `ticker` 的写作 manifest 是否已有“公司级 facets”结果；若缺失，会自动先推理一次，再继续写作。
- 默认会复用 manifest 中已有的“公司级 facets”，不会每次重跑。
- 显式传 `--infer` 时，只会强制重跑一次“公司级 facets”并写回 manifest，随后立即退出，不进入章节写作。
- 按场景使用参数：
  - 第一次完整写报告：直接运行 `dayu-cli write --ticker AAPL`。
  - 上次写到一半中断，想从已有结果继续：直接重新运行同一条命令即可；默认就是 `--resume`。
  - 只想快速出初稿，不想等待审查和修复：加 `--fast`。
  - 上一次是用 `--fast` 跑的，想继续沿用这套“只写正文”的方式：继续加 `--fast` 再运行。
  - 只想重写某一章正文：用 `--chapter "章节名" --fast`。
  - 前面章节还没完全通过检查，但你仍然想先生成第 0 章或第 10 章看结果：加 `--force`。
  - 只想刷新公司级归因，不进入正式写作：用 `--infer`。
- 章节不满意时，推荐这样处理：
  - 正常模式 `write` 写完后，发现第 8 章不满意：先运行 `dayu-cli write --ticker AAPL --chapter "第8章的章节名"` 重写这一章；再运行 `dayu-cli write --ticker AAPL`，把新章节应用到整份报告。
  - `write --fast` 写完后，发现第 8 章不满意，且你还想继续保持“快速草稿”模式：先运行 `dayu-cli write --ticker AAPL --chapter "第8章的章节名" --fast` 重写这一章；再运行 `dayu-cli write --ticker AAPL --fast`，把新章节应用到整份草稿报告。
  - 单独重写某一章时，不会自动重建第 0 章和第 10 章。
  - 重写某一章后，再跑一次全文 `write` 时，系统会重新生成整份报告文件；但默认 `--resume` 会跳过当前模式下已经完成的章节，所以第 0 章和第 10 章如果已完成，通常不会自动重写。
  - 如果你改动了中间章节后，希望第 0 章和第 10 章也反映新的内容，建议依次重跑该中间章节、第 10 章、第 0 章，最后再运行一次全文 `write`。

### 3.7 财报预处理：`process`

命令用途：
把已下载或已上传的财报做结构化预处理，并导出快照。

参数 / 说明：

| 参数 | 说明 |
|------|------|
| `--ticker` | 必填，股票代码 |
| `--document-id` | 可选，仅处理指定文档 ID；可重复传入，也支持单个参数中用逗号分隔多个 ID |
| `--overwrite` | 可选，覆盖已存在结果 |
| `--ci` | 可选，额外导出 `search_document` 与 `query_xbrl_facts` 快照 |
| `--base` | 可选，工作区根目录，默认 `./workspace` |
| `--config` | 可选，配置目录，默认 `workspace/config` |

命令示例：

```bash
dayu-cli process --ticker AAPL --overwrite
```

常见命令示例：

```bash
dayu-cli process --ticker AAPL
dayu-cli process --ticker AAPL --ci
dayu-cli process --ticker AAPL --ci --document-id fil_001 --document-id fil_002
```

命令说明：
- 预处理命令主要供开发和数据准备场景使用，最终用户通常不需要手动执行。
- 快照会写入 `workspace/portfolio/{ticker}/processed`。
- 传入 `--document-id` 时，只会重建这些文档的快照；不会清空同一 ticker 下其它 processed 结果。

## 4. 自动写作详解

`write` 命令怎么用，在前面已经说明清楚。  
这一节更关心另一件事：

- 当你想把报告写成自己满意的样子时，应该怎么理解写作流程
- 理解流程后，应该优先改模板的哪里

### 4.1 写作流程

自动写作的顺序可以先记成一条主线：

1. `infer`
2. 第 1-9 章
3. 第 10 章
4. 第 0 章
5. 来源清单（如果模板中存在）

你可以这样理解这条流程：

- `infer` 先判断这家公司属于什么业务类型、有哪些关键约束。
- 第 1-9 章先把主体分析写出来，这部分是整份报告的事实和判断基础。
- 第 10 章再基于前面各章，回答“是否值得继续深研与待验证问题”。
- 第 0 章最后回填“投资要点概览”，它不是独立开写，而是对前面结果做浓缩。
- 来源清单最后统一整理，不需要你手写。

为什么先理解这个流程很重要：
- 因为模板不是“想到什么就加什么”。
- 某个内容应该放在第 1-9 章、第 10 章，还是第 0 章，取决于它在这条流程里承担什么作用。
- 你先知道系统怎么写，再去改模板，通常会少走很多弯路。

### 4.2 第 10 章、第 0 章和来源清单有什么特殊性

这三个部分和普通章节不一样，改模板时要单独看待。

第 10 章 `是否值得继续深研与待验证问题`：
- 它不是普通分析章，而是“研究决策章”。
- 它更适合回答：现在值不值得继续投入研究资源；如果继续，接下来最该验证什么。
- 这章的重点不是重复前文，而是基于前文做取舍和排序。
- 标题不要改。你可以改内容框架，但不要把它改名成别的标题。

第 0 章 `投资要点概览`：
- 它是整份报告最后给读者看的“快速入口”。
- 这章更适合浓缩结论、核心判断和最重要的不确定性，不适合再铺一遍细节。
- 标题不要改。你可以改里面的小节和表达方式，但不要改一级标题。

来源清单：
- 它是末章，用来汇总整份报告的证据出处。
- 这章主要是为了让你快速检查“这份报告到底引用了什么”，不是拿来承载新的分析。
- 标题也不要改。

一级章节修改时，建议记住这条边界：
- `投资要点概览`、`是否值得继续深研与待验证问题`、`来源清单` 这三个一级标题保留原名。
- 修改 `是否值得继续深研与待验证问题` 标题会使这一章变成普通章节。
- 删除 `来源清单` 报告里就没有 `来源清单` 。
- 除这三个之外，其它一级章节的标题、顺序和数量都可以按你的研究框架调整。

### 4.3 什么时候应该改模板

如果你遇到下面这些情况，优先改模板，而不是先改 prompt：

- 报告整体导向不对，例如总在证明公司“为什么好”
- 某一章总是写偏，例如写成竞争百科、管理层赞歌或财务复述
- 某些信息对你很重要，但模型总是不写
- 某些内容只适合少数公司，但模型总是机械地写出来

你可以把模板理解成几层：

- high level：整篇文章怎么组织
- detail level：每一章内部怎么展开
- 条件项：什么内容只在特定公司、特定证据条件下才写

先想清楚自己不满意的是哪一层，再动模板，会比“到处加规则”更有效。

### 4.4 怎么改模板：先改文章骨架

如果你对整份报告都不满意，先改 high level，也就是整篇文章骨架。

**这里最重要的一点是：**

- 第 1-10 章不是固定编制。
- 你完全可以把普通分析章节改成 3 章、4 章、6 章，或者按你自己的研究框架重排。
- 系统并不要求你必须保留“第 1 章到第 10 章”这种数量和顺序。

你可以把“改文章骨架”理解成：决定整份报告一共要有哪些一级章节，以及这些一级章节按什么顺序出现。

最常见的文章骨架修改有：

- 调整一级章节顺序
- 新增或删除普通一级章节
- 把原来的 1-10 章改成更少或更多章
- 让报告更偏“继续研究 / 暂缓 / 放弃”的筛选框架
- 让报告更偏“重建公司全貌”，而不是“归纳优点”

适合先改文章骨架的场景：

- 报告整体方向不对
- 每一章都在写，但串起来不是你想要的研究路径
- 你希望读者先看到“值不值得继续看”，还是先看到“公司到底是什么”

改文章骨架时，优先问自己三个问题：

1. 这份报告最后要帮我做什么判断？
2. 我希望读者按什么顺序建立判断？
3. 哪些一级章节是真正必要的，哪些只是习惯性保留？

改文章骨架时，一级章节标题可以分成两类看：

- 普通一级章节：可以随便改标题、顺序和数量。你想保留 3 章、4 章、7 章都可以。
- 特殊一级章节：`投资要点概览`、`是否值得继续深研与待验证问题`、`来源清单` 建议保留原名。

其中最需要记住的是：

- `是否值得继续深研与待验证问题` 这个标题如果改掉，就会变成普通章节，不再按“研究决策章”处理。

如果这三个问题没想清楚，直接去改某一章的小节，通常只会把局部修得更复杂，不能真正改善整份报告。

### 4.5 怎么改模板：再改章节骨架

当整篇文章方向没问题，但某一章总是写偏时，再改 detail level，也就是章节骨架。

你可以把“改章节骨架”理解成：一级章节已经定了，现在再决定这一章里面具体怎么展开。

章节骨架通常包括：

- 本章目标
- 固定小节
- 小节标题
- 每个小节想回答的问题

适合先改章节骨架的场景：

- 标题看起来没问题，但模型总是抓错本章主问题
- 你希望模型固定输出某几个关键判断
- 你想删掉长期低价值的小节
- 你想把某一章从“信息罗列”改成“研究判断”

一个简单判断法：

- 如果你不满意的是“整份报告该有哪些章、这些章怎么排”，先改文章骨架。
- 如果你不满意的是“某一章里面该分哪几个小节、每个小节回答什么”，先改章节骨架。

### 4.6 条件项怎么用

条件项适合表达“只对某类公司才有意义”的补充内容。

例如：

- 只有平台公司才适合写网络效应
- 只有跨区域经营的公司才值得写重要 geography
- 只有证据足够精确时，才值得写更细的 segments、份额、客户集中度

经验法则：

- 所有公司都该写的，放进文章骨架或章节骨架
- 只有部分公司才该写的，放进条件项
- 某类行业优先看的判断入口，放进 `preferred_lens`
- 某类行业才值得补充的局部内容，放进 `ITEM_RULE`

`preferred_lens` 和 `ITEM_RULE` 可以这样理解：

- `preferred_lens`：这一章优先从什么视角进入判断
- `ITEM_RULE`：在特定条件下，额外补哪些局部内容

不要把写法技巧、句式要求、修辞规则塞进条件项。条件项更适合管“写不写什么”，不适合管“怎么写得好看”。

### 4.7 实用原则

好的模板，不是让模型“写更多”，而是让模型：

- 更快抓住这章真正要回答的问题
- 更少写偏
- 在证据不够时宁可少写，也不硬写

如果某条模板规则只是让报告更花哨、看起来更专业，但不帮助你更快做“继续研究 / 暂缓 / 放弃”的判断，通常就不值得加。

改模板时，一个很实用的顺序是：

1. 先改文章骨架
2. 再改章节骨架
3. 最后才补条件项

如果报告方向不对，通常先改“结构”和“本章问题”，效果会比一开始就堆很多条件规则更明显。

### 4.8 自动写作会产出什么

自动写作会在输出目录下按章节落盘。常见文件包括：

- `manifest.json`：记录章节状态，以及当前公司的“公司级 facets”等写作上下文
- 每章最终的 `.md`：这是你最该优先看的正文结果
- 对应的 `*_audit.json`：如果你想知道某章为什么没写好，可以看这里
- `run_summary.json`：整次写作的结果摘要

如果你只关心结果，优先看：

- 每章最终的 `.md`
- `dayu-cli write --summary --ticker AAPL` 的摘要输出
- 需要排查问题时，再看对应的 `*_audit.json`

## 5. tool trace 分析

如果你在 `prompt` / `interactive` / `write` 时开启了 trace：

```bash
dayu-cli prompt \
  "总结最新财报风险" \
  --ticker AAPL \
  --enable-tool-trace
```

trace 默认写入 `workspace/output/tool_call_traces`。分析命令：

```bash
python -m utils.analyze_tool_trace \
  --input ./workspace/output/tool_call_traces \
  --ticker AAPL \
  --output ./workspace/trace_analysis_AAPL.md
```

这个脚本会输出：
- 工具级诊断总表（成功率、截断率、载荷大小、错误分布）
- 详细失败签名（例如 URL 拦截、HTTP 状态、超时等），而不只停留在粗粒度 `error_code`
- 围绕“降低模型认知负担”的信号分析
- 截断续读、重复调用、上下文负担、失败与降级诊断
- 各 run / turn 的工具调用链与 trace 完整性检查
- 面向工具设计的优化建议

### 5.1 网页抓取诊断

当你遇到“浏览器能打开，但 `fetch_web_page` / `requests` / Playwright 访问失败”的 URL 时，可以用下面的脚本把同源证据导出成 JSON：

```bash
python -m utils.diagnose_web_access \
  --url "https://investor.pddholdings.com/news-releases/news-release-details/pdd-holdings-announces-fourth-quarter-2025-and-fiscal-year-2025" \
  --output ./workspace/output/web_diagnostics/pdd-ir.json
```

如需用有界面 Chrome 观察实际打开过程：

```bash
python -m utils.diagnose_web_access \
  --url "https://example.com" \
  --headed \
  --channel chrome
```

如需在人工完成验证后导出可复用的浏览器状态：

```bash
python -m utils.diagnose_web_access \
  --url "https://www.reuters.com/..." \
  --headed \
  --channel chrome \
  --manual-wait-seconds 30 \
  --pause-before-snapshot \
  --storage-state-out ./workspace/output/web_diagnostics/storage_states/www.reuters.com.json
```

仓库内也提供了更省事的包装脚本：

```bash
./utils/diag_web.sh "https://www.reuters.com/..."
```

它默认会用有界面 Chrome 打开页面，并等待 30 秒供你人工操作；30 秒到达后会自动继续采样页面状态并保存诊断结果。

它也会按 host 自动读写 `workspace/output/web_diagnostics/storage_states/<host>.json`：
- 第一次运行时，若状态文件不存在，只会导出新的 state，不会报错
- 后续再运行同一 host 时，会自动把已有 state 喂回诊断脚本

脚本会导出：
- 浏览器主文档 request headers 与导航结果
- 当前仓库 `fetch_web_page` 的调用结果，以及工具层 `extra.internal_diagnostics`
- `requests` 将发送的 headers 与实际 GET 结果
- Playwright 观察到的网络请求摘要
- 浏览器页面的文本/HTML 前缀，便于判断 challenge/access gate

若某些受保护站点只能靠人工浏览器通过验证，可把导出的 state 文件放进 `run.json.web_tools_config.playwright_storage_state_dir` 指向的目录；`fetch_web_page` 会按 host 自动复用对应的 `<host>.json`。

### 5.2 批量诊断与 CI

如果你的目标不是“修一个站点”，而是让 CI 持续产出一批可分析的 `web_diagnostics` 原始证据，再由后续分析去定位 `fetch_web_page` 的 root cause，推荐直接使用 `diagnose_web_access` 的批量模式：

```bash
python -m utils.diagnose_web_access \
  --url-file ./utils/web_ci_urls.jsonl \
  --run-label 20260406-sample \
  --storage-state-dir ./workspace/output/web_diagnostics/storage_states
```

仓库里已经附带了一份 sample corpus：`./utils/web_ci_urls.jsonl`。它覆盖常见新闻网站、财经网站和政府/监管组织，并包含中国与海外站点，可直接作为第一轮诊断样本。

如果你直接使用仓库内的包装脚本：

```bash
./utils/diag_web_batch.sh ./utils/web_ci_urls.jsonl
```

它默认也会开启有界面 Chrome，并为每条 URL 等待 30 秒供你人工操作；30 秒到达后会自动继续采样并保存诊断结果。

它会对每个 URL 逐条导出同源诊断 JSON，单条诊断里默认包含三条访问路径：
- 人工/自动浏览器侧的 Playwright 诊断
- `requests` 侧的实际请求结果
- 当前仓库 `fetch_web_page` 的调用结果

如果你需要人工浏览器先完成验证，再采样页面状态并保存可复用 cookie / local storage，可在批量模式下继续加：

```bash
python -m utils.diagnose_web_access \
  --url-file ./utils/web_ci_urls.jsonl \
  --run-label 20260406-manual \
  --headed \
  --channel chrome \
  --manual-wait-seconds 30 \
  --pause-before-snapshot \
  --storage-state-dir ./workspace/output/web_diagnostics/storage_states
```

只有显式传入 `--pause-before-snapshot` 时，脚本才会在等待结束后继续要求你按回车确认。

默认每条 URL 都会在独立子进程里执行，并在执行前删除 `workspace/.dayu/session`，避免进程内 Session、warmup host 和 Playwright 单例污染下一条 URL。这样 CI 侧只需要稳定地产出 `web_diagnostics`，后续分析和优化可以基于这些原始结果继续进行。

URL 文件支持两种格式：

1. JSONL：适合带元数据的长期基准集

```json
{"url":"https://finance.yahoo.com/quote/AAPL/","label":"Yahoo Finance AAPL","region":"foreign","category":"quote"}
{"url":"https://www.reuters.com/markets/companies/AAPL.OQ/","label":"Reuters AAPL","region":"foreign","category":"company-page"}
{"url":"https://www.stcn.com/article/detail/1568835.html","label":"证券时报示例","region":"china","category":"news"}
```

2. TXT：每行一个 URL，适合快速试跑

```text
https://finance.yahoo.com/quote/AAPL/
https://www.reuters.com/markets/companies/AAPL.OQ/
```

常用参数：
- `--storage-state-dir`：按 host 自动读写 storage state 目录
- `--headed`：启用有界面浏览器，便于人工观察和手工通过验证
- `--manual-wait-seconds`：导航后先额外等待多少秒
- `--pause-before-snapshot`：人工确认完成后，再按回车继续采样和保存 state
- `--skip-playwright`：只跑 `requests` 与 `fetch_web_page`
- `--skip-tool-fetch`：只收集浏览器与 `requests` 诊断，不调用 `fetch_web_page`

输出目录默认写到 `workspace/output/web_diagnostics/<run_label>/`，其中包括：
- `corpus.normalized.jsonl`：本轮归一化后的样本集
- `diagnostics/`：每条 URL 的完整诊断 JSON
- `results.jsonl`：从完整诊断提炼出的批量索引，便于后续程序分析
- `summary.json` / `summary.md`：本轮批量诊断汇总

如果你希望我直接参与 CI 优化流程，那么 CI 侧只需要跑完这一步并保留 `workspace/output/web_diagnostics/<run_label>/`，后续我就可以基于这批 `web_diagnostics` 结果做差异分析、找 root cause，并继续优化 `fetch_web_page`。

## 6. 渲染输出

Markdown 报告渲染入口：

```bash
dayu-render <输入文件.md> [输出文件]
```

常见示例：

```bash
dayu-render workspace/draft/AAPL/AAPL_qual_report.md
dayu-render workspace/draft/AAPL/AAPL_qual_report.md report.pdf
dayu-render workspace/draft/AAPL/AAPL_qual_report.md report.html
```

支持格式：
- `.docx`（默认）
- `.html`
- `.pdf`

说明：
- 生成 PDF 需要 `pandoc + Chrome`
- 若 Chrome 不在标准位置，可设置 `PUPPETEER_EXECUTABLE_PATH`
- 渲染器会保留 Markdown 里的普通换行；例如列表项里单独一行的“标签”与下一行正文，在 `.docx` 中会继续换行显示

## 7. 配置文件从哪里改

大多数用户只需要关注这三个位置：

| 文件/目录 | 用途 |
|-----------|------|
| `workspace/config/llm_models.json` | 模型配置、API Key 占位符 |
| `workspace/config/run.json` | Agent 行为、Host 配置、工具超时、trace、budget、limits；CLI / WeChat 的 chat 与 prompt 默认执行参数也在运行时由 Host 基于这里继续收敛 |
| `workspace/config/prompts/` | prompt 资产 |

建议修改方式：
- 想换模型：改 `llm_models.json`
- 想新增自定义模型：先在 `workspace/config/llm_models.json` 里添加模型配置；再把该模型加入对应 scene manifest 的 `workspace/config/prompts/manifests/*.json -> model.allowed_names`，必要时改 `model.default_name`；如果该模型要长期参与 `interactive` 多轮会话，建议同时给该模型补 `runtime_hints.conversation_memory`
- 想调 Agent 行为：改 `run.json`；其中 `conversation_memory` 控制 `interactive` 多轮会话的历史预算与压缩策略
- 想改系统提示词和任务提示词：改 `prompts/`

配置说明请看：
- [dayu/config/README.md](dayu/config/README.md)

<a id="model-config"></a>

## 8. 模型配置

如果你只是临时切换模型，最简单的办法是在命令里直接传 `--model-name`。  
如果你想长期修改默认模型，或者接入一个新的模型，再改配置文件。

最常用的两个位置是：

- `workspace/config/llm_models.json`：定义“有哪些模型可以用”
- `workspace/config/prompts/manifests/*.json`：定义“每个场景默认用哪个模型”

### 8.1 怎么修改默认模型

每个场景都有自己的默认模型，配置在对应 scene manifest 里。

最常改的几个文件：

- `workspace/config/prompts/manifests/prompt.json`
- `workspace/config/prompts/manifests/interactive.json`
- `workspace/config/prompts/manifests/prompt_mt.json`
- `workspace/config/prompts/manifests/write.json`
- `workspace/config/prompts/manifests/audit.json`
- `workspace/config/prompts/manifests/confirm.json`

你主要看这一段：

```json
"model": {
  "default_name": "mimo-v2.5-pro",
  "allowed_names": [
    "mimo-v2.5-pro",
    "mimo-v2.5-pro",
    "deepseek-v4-flash"
  ],
  "temperature_profile": "write"
}
```

改默认模型时，通常只需要两步：

1. 把 `default_name` 改成你想用的模型名
2. 确认这个模型名已经出现在 `allowed_names` 里

例如：

- 想把 `write` 默认模型从 `mimo-v2.5-pro` 改成 `gpt-5.4`，就改 `workspace/config/prompts/manifests/write.json`
- 想把 `interactive` 默认模型改成 `qwen-plus-thinking`，就改 `workspace/config/prompts/manifests/interactive.json`
- 想把 `audit` / `confirm` 默认模型换掉，就分别改 `audit.json` 和 `confirm.json`

一个简单理解：

- `default_name`：这个场景默认会用谁
- `allowed_names`：这个场景允许切换到哪些模型
- `temperature_profile`：这个场景默认使用哪组温度参数，一般不用改

### 8.2 怎么添加新模型

如果现有模型不够用，你可以自己往 `workspace/config/llm_models.json` 里加一个新条目。

最简单的做法是：

1. 先复制一个最接近的现有模型配置
2. 改模型名、接口地址、鉴权头和能力参数
3. 再把这个模型名加入对应 scene manifest 的 `allowed_names`
4. 如果希望它成为默认模型，再修改 `default_name`

例如，你可以复制一段现有配置，改成这样：

```json
"my-model": {
  "runner_type": "openai_compatible",
  "name": "my-model",
  "endpoint_url": "https://api.example.com/v1/chat/completions",
  "model": "my-model",
  "headers": {
    "Authorization": "Bearer {{MY_API_KEY}}",
    "Content-Type": "application/json"
  },
  "timeout": 3600,
  "supports_stream": true,
  "supports_tool_calling": true,
  "max_context_tokens": 128000,
  "runtime_hints": {
    "temperature_profiles": {
      "write": {
        "temperature": 0.8
      },
      "audit": {
        "temperature": 0.2
      }
    }
  }
}
```

加完之后，再去对应 scene manifest，例如 `workspace/config/prompts/manifests/write.json`，把 `"my-model"` 加进 `allowed_names`。如果你想让它直接成为默认模型，再把 `default_name` 改成 `"my-model"`。

### 8.3 模型参数怎么理解

对最终用户来说，下面这些参数最重要：

- `name`：配置名。你在 `--model-name`、`default_name`、`allowed_names` 里写的就是它。
- `endpoint_url`：模型服务地址。
- `model`：真正发给服务商的模型标识。
- `headers`：鉴权和请求头，通常在这里放 API Key 占位符。
- `timeout`：单次请求超时时间，单位秒。
- `supports_stream`：是否支持流式输出。
- `supports_tool_calling`：是否支持工具调用。用于 `prompt`、`interactive`、`write` 的模型通常需要支持。
- `max_context_tokens`：模型可用上下文上限。

`runtime_hints.temperature_profiles` 里最常看到的是 `temperature`：

- `temperature` 越低，输出通常越稳、更保守
- `temperature` 越高，输出通常越发散、更有变化

通常可以这样理解：

- `write`：正文写作时的参数
- `overview`：第 0 章概览时的参数
- `audit`：审计场景的参数
- `decision`：第 10 章研究决策场景的参数
- `prompt`：单轮问答场景的参数
- `interactive`：交互对话场景的参数
- `infer`：公司类型与关键约束判断场景的参数

如果你只是新增一个模型，最稳的办法不是从零设计全部参数，而是复制一个相近模型，再按你的服务商要求做最小改动。

### 8.4 一个推荐顺序

如果你准备调整模型配置，建议按这个顺序来：

1. 先临时用 `--model-name` 试跑
2. 满意后再改 scene manifest 的 `default_name`
3. 如果现有模型都不合适，再去 `llm_models.json` 新增模型

这样做的好处是：

- 先验证效果，再改默认配置
- 不容易一上来改太多，最后不知道是哪一步导致结果变化

## 9. 文档导航

- 用户手册（当前文档）：[README.md](README.md)
- 开发手册总览：[dayu/README.md](dayu/README.md)
- Engine 包开发手册：[dayu/engine/README.md](dayu/engine/README.md)
- Fins 包开发手册：[dayu/fins/README.md](dayu/fins/README.md)
- 配置说明手册：[dayu/config/README.md](dayu/config/README.md)
- 贡献指南：[CONTRIBUTING.md](CONTRIBUTING.md)

## 10. 开源与许可证

本仓库采用 `Apache License 2.0` 开源协议发布。

你在分发或修改本项目时，至少需要注意三件事：

- 保留仓库中的 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)
- 对你修改过的文件添加明确变更说明
- 不要把仓库名称、作者名称或项目商标暗示成对你分发版本的背书

如果你准备贡献代码、文档或测试，请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。
