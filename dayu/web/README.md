## `dayu.web` 开发说明

本文档说明 `dayu.web` 的当前实现边界，重点覆盖 Streamlit 模块的目录职责、聊天链路与稳定约束。

## 1. 模块定位与入口

- `dayu.web` 是 UI 适配层：把页面/HTTP 请求收口为 `ServiceProtocol` 调用，再把结果渲染或序列化返回。
- 稳定分层保持为 `UI -> Service -> Host -> Agent`，UI 不直接触碰 Host 内部实现细节。
- 当前有两条 Web 入口：
  - Streamlit UI：`dayu/web/streamlit_app.py`
  - FastAPI API：`dayu/web/fastapi_app.py`

| 入口 | 目标用户 | 负责 | 不负责 |
| --- | --- | --- | --- |
| `dayu/web/streamlit_app.py` | 人工交互用户（浏览器） | 页面渲染、`st.session_state`、Service 注入、Tab 组合 | HTTP API 契约与路由 schema |
| `dayu/web/fastapi_app.py` | 程序化调用方（HTTP 客户端/Worker） | API 装配、路由注册、请求/响应契约、后台任务受理 | Streamlit 页面状态与组件交互 |

## 2. Streamlit 模块结构

`streamlit_app.py` 的职责边界：

- 解析 workspace（优先 `DAYU_WORKSPACE`，否则 `<cwd>/workspace`）。
- 只在启动阶段装配一次 Host runtime，并在 `st.session_state` 缓存 Service 实例。
- 调用 `render_sidebar()` 与 `render_stock_detail_page()`，不把业务逻辑塞入入口文件。

`dayu/web/streamlit/` 当前按“组件 + 页面”拆分：

- `components/`：边栏与通用组件（如自选股列表）。
- `pages/main_page.py`：主功能区入口，组合三大 Tab（财报管理、交互式分析、分析报告）。
- `pages/filing_tab.py`：财报下载与处理页面。
- `pages/chat_tab.py`：交互式分析页面编排（渲染、输入、历史加载、清空会话、流式轮询）。
- `pages/report_tab.py`：分析报告页面。


## 3. 开发约束与测试锚点

- Streamlit/FastAPI 都必须通过 `ServiceProtocol` 访问业务能力，不绕过 Service 直连 Host 内部对象。
- 聊天历史读取与清空走 `ChatServiceProtocol` 的公开方法，不在 UI 侧直接读写 archive。
- 聊天会话 ID 语义固定为公司级稳定键（`streamlit-web-{COMPANY_ID}`），避免同公司不同 ticker 写法割裂历史。
- 自选股持久化边界由 `tests/application/test_streamlit_watchlist.py` 守护。
