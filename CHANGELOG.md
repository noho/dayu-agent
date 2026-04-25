# Changelog

本项目的所有重要变更都会记录在这里。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

-- 

## [Unreleased]

### 注意

- 本次更新后需运行一次 `dayu-cli init --reset` ，删除workspace/ 目录下的 .dayu/ config/ assets/ 目录，已下载/上传的财报不会丢失，已生成的报告不会丢失。

### 新增

- 提供离线安装包，覆盖 `macOS ARM64`、`Linux x64`、`Windows x64` 三个平台。
- 支持自定义OpenAI 兼容模型。
- interactive支持历史对话发现和恢复。

### 变更

- 小米 `mimo` 模型更新到2.5 Pro。
- `DeepSeek` 模型更新到V4。

### 修复

- 若干缺陷修复。

-- 

## [0.1.2] - 2026-04-20

### 新增

- 提供离线安装包，覆盖 `macOS ARM64`、`Linux x64`、`Windows x64` 三个平台。

### 变更

- 支持 MiMo Plan 海外环境；已安装用户升级到该版本后，需要执行 `dayu-cli init --overwrite` 刷新初始化配置。

### 修复

- 若干缺陷修复。

-- 

## [0.1.1] - 2026-04-18

### 新增

- 新增安装后初始化命令 `dayu-cli init`，用于生成项目运行所需的初始配置。

-- 

## [0.1.0] - 2026-04-17

首次开源发布。

### 新增

- 发布首个可安装版本，可通过 GitHub Releases 提供的 Python wheel 安装使用。
- 提供 `dayu-cli` 命令，可完成美股 `10-K`、`10-Q`、`20-F` 财报下载，并在已导入财报基础上执行 `prompt` 单次问答、`interactive` 多轮问答和 `write` 报告写作。
- 提供 `dayu-wechat` 文本消息入口，可通过微信发起基础问答。
- 提供 `dayu-render` 命令，可将 Markdown 报告渲染为 `HTML`、`PDF`、`Word`。
- 提供默认配置与模板，支持通过 `workspace/config/` 覆盖运行时配置。

### 已知限制

- A 股、港股财报下载未实现。
- GUI 未实现；Web UI 仅提供骨架能力。
- WeChat 入口仅支持文本消息首版。
- 财报电话会议音频转录后的问答区分未实现。
- 定性分析模板对不同公司的差异化判断路径仍偏机械。

[Unreleased]: https://github.com/noho/dayu-agent/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/noho/dayu-agent/releases/tag/v0.1.2
[0.1.1]: https://github.com/noho/dayu-agent/releases/tag/v0.1.1
[0.1.0]: https://github.com/noho/dayu-agent/releases/tag/v0.1.0
