## `dayu.web` 开发说明

本文档说明 `dayu.web` 下 Web 适配层的当前实现边界，重点说明 Streamlit 模块。

## 1. 模块定位

- `dayu.web` 是 UI 适配层，负责把宿主入口请求转成 `Service` 调用，并把结果渲染给用户。
- 当前 Web 有两条并行入口：
  - Streamlit UI：`dayu/web/streamlit_app.py`（用户交互主入口），为用户提供本地访问的 Web 页面
  - FastAPI：`dayu/web/fastapi_app.py`（HTTP API 入口）,为独立的 Web 服务提供 API 接口。
- 设计基线保持稳定分层：`UI -> Service -> Host -> Agent`。

细分职责如下：

| 入口文件 | 目标对象 | 主要职责 | 不负责 |
| --- | --- | --- | --- |
| `dayu/web/streamlit_app.py` | 人工交互用户（浏览器页面） | 页面级 UI 交互、会话状态管理（`st.session_state`）、页面路由与本地文件预览入口装配 | 对外 HTTP API 契约、路由 schema 设计 |
| `dayu/web/fastapi_app.py` | 程序化调用方（HTTP 客户端/Worker） | API 装配、路由注册、请求/响应契约与后台任务受理边界 | 页面渲染、前端会话态维护、Streamlit 组件状态 |

统一约束：

- 两条入口都遵循 `UI -> Service -> Host -> Agent`，不允许 UI 直接绕过 `Service` 调 `Host` 内部细节。
- `streamlit_app.py` 可以维护 UI 会话状态，但不扩展成通用 API 网关。
- `fastapi_app.py` 负责稳定 API 契约，但不承载 Streamlit 页面行为或页面状态。
- 同一业务能力优先复用同一组 `ServiceProtocol`，保证 CLI / Streamlit / FastAPI 语义一致。

## 2. Streamlit Web 核心功能

- 初始化配置，并配置各个场景的模型
- 配置自选股
- 下载个股财报，目前支持从SEC下载财报
- 交互式分析，目前未保存会话历史，刷新页面将开启新的会话
- 分析报告，支持生成完整的分析报告，查看报告详情(Service未暴露分析过程的详细信息，暂时通过跟踪draft输出的文件跟踪任务状态)，并导出为Markdown、PDF、Html等格式的完成报告

