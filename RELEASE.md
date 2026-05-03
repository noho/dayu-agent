## 安装

```bash
pip install https://github.com/noho/dayu-agent/releases/download/v0.1.4/dayu_agent-0.1.4-py3-none-any.whl
```

## 更新到新版本

```bash
pip install --upgrade https://github.com/noho/dayu-agent/releases/download/v0.1.4/dayu_agent-0.1.4-py3-none-any.whl
```

## 离线安装

从 [Releases](https://github.com/noho/dayu-agent/releases) 页面下载对应平台的离线安装包：

- Mac ARM芯片：`dayu-agent-0.1.4-macos-arm64-offline.tar.gz`
- Mac Intel芯片：`dayu-agent-0.1.4-macos-x64-offline.tar.gz`
- Windows：`dayu-agent-0.1.4-windows-x64-offline.zip`

Linux 用户请使用上面的在线 wheel 安装或源码安装。

### 注意

如果你从 v0.1.3 升级：
- 本次更新后需运行一次 `dayu-cli init` ，已下载/上传的财报不会丢失，已生成的报告不会丢失。

如果你从更早版本升级：
- 本次更新后需运行一次 `dayu-cli init --reset` ，已下载/上传的财报不会丢失，已生成的报告不会丢失。

## 本次更新

**新功能**

- 支持从巨潮下载 A 股财报，并完成 PDF 下载、Docling JSON 导出、断点恢复、元数据提交与本地重建。
- 支持从披露易下载港股财报，覆盖年报、半年报与独立季度业绩公告；缺失的独立季度报告会按 skipped 收口。
- Web UI 支持交互式分析：按 ticker 绑定稳定会话、流式展示回答，并支持清空历史。

**优化**

- 美股、A 股、港股下载使用独立并发 lane，避免不同市场的长下载任务互相占用许可。
- A 股 / 港股下载过程补充更完整的日志、候选过滤、年度/期间去重、覆盖下载和中断恢复能力。
- ticker 归一化继续收敛跨市场写法，保护交易所后缀 alias，并统一美股 class share 分隔符。

**修复与改进**

- 修复 A 股覆盖下载、运行时防护和 A 股 / 港股独立季度处理问题。
- 修复 Web 交互式分析中的重复事件消费、历史加载、清空历史和异常收口问题。

安装后可用命令：
- `dayu-cli init` — 初始化配置
- `dayu-cli` — 财报分析 CLI
- `dayu-web` — Web UI
- `dayu-wechat` — WeChat 服务
- `dayu-render` — 报告渲染

感谢以下贡献者参与本次发布（按 GitHub 用户名排序）：
@noho、@xingren23