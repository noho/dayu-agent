# 贡献指南

本文档面向准备给 `dayu Agent` 提交 issue、文档修订或代码贡献的开发者。

## 先看什么

提交改动前，先阅读这些文档：

- 用户入口：[README.md](README.md)
- 开发总览：[dayu/README.md](dayu/README.md)
- Engine 边界：[dayu/engine/README.md](dayu/engine/README.md)
- Fins 边界：[dayu/fins/README.md](dayu/fins/README.md)
- 测试约定：[tests/README.md](tests/README.md)
- 仓库协作约束：[AGENTS.md](AGENTS.md)

## 提交前原则

- 先从第一性原理说明问题和目标，不要直接沿用历史实现假设。
- root cause 必须与逻辑或数据同源，禁止用间接证据拼结论。
- 严格遵守分层边界：UI、Application、Runtime、Services、Engine、Capability、Fins 的职责不要混写。
- 对财报文档的读写只通过 `FsDocumentRepository` / `DocumentRepository`。
- 禁止把显式参数塞进 `extra_payload`。
- 写作链路优化目标是写出更好的买方分析报告，而不是为了更容易通过 audit。

## 开发流程

1. 先开 issue，或在已有 issue 里明确问题定义、动机、范围和验收标准。
2. 修改代码时同步补测试；测试应该跟着实现边界走，不要让生产代码去兼容旧测试。
3. 涉及文档、命令入口、分层边界变化时，同步更新对应 README，文档以代码为准。
4. 提交 PR 时，说明：
   - 你解决的具体问题
   - root cause 证据来自哪里
   - 为什么改动位置符合模块边界
   - 你跑了哪些测试或验证命令

## 测试与质量

常用命令：

```bash
pytest
ruff check .
mypy dayu
```

如果你修改了财报处理链路，请尽量补充最小可复现样本或夹具，避免只靠人工描述说明行为变化。

## 文档与接口

- 新增或修改 CLI、配置入口、trace/render 工作流时，更新根目录 [README.md](README.md)。
- 修改 Engine 边界、事件流、trace 契约时，更新 [dayu/engine/README.md](dayu/engine/README.md)。
- 修改 Fins capability 定位、对外接口或内部机制时，更新 [dayu/fins/README.md](dayu/fins/README.md)。
- 修改测试分层或新增测试约定时，更新 [tests/README.md](tests/README.md)。

## 许可证

向本仓库提交贡献，即表示你同意你的贡献以 `Apache-2.0` 协议发布。

## 沟通

如果你想参与以下方向，建议直接开 issue 讨论方案，或在已有 issue 下认领：

- 定性分析模板 读起来机械感还很强，还没写出差异化：
  - 同一章节里，不同行业公司写出明显不同的判断路径。
  - 同一行业里，不同公司写出公司自己的特殊结构变量。
- 位于 Engine 的 web tools 现在的对抗challenge能力很弱，很多网站无法访问。
- 位于 Fins 的港股、A股财报下载功能尚未实现。
- GUI 尚未实现；Web UI 目前仍只有 FastAPI 骨架。
- WeChat UI 仅支持文本消息首版，还可添加更多好玩的功能。
- 财报电话会议记录音频转录文字后信息提取（起码要区分信息来自提问还是回答）尚未实现。
- 财报presentation信息提取尚未实现。
- 欢迎围绕以下方向提交 issue 或 PR：
  - 普通文件（非财报文件）信息提取还需要优化。
  - 优化 Fins 里的港股/A股/美股财报信息提取。
  - Anthropic 原生 API 支持。
  - Durable memory / Retrieval layer（ Memory只实现了working memory 和 episode summary ）。
  - FMP 工具（调研工作已做，见 [docs/fmp_integration_research.md](docs/fmp_integration_research.md) ）尚未实现。
  - 更多LLM 工具。
