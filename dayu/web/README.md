## `dayu.web` 开发说明

本文档说明 `dayu.web` 下 Web 适配层的当前实现边界，重点覆盖 Streamlit 路径（`dayu-web`）。

## 1. 模块定位

- `dayu.web` 是 UI 适配层，负责把宿主入口请求转成 `Service` 调用，并把结果渲染给用户。
- 当前 Web 有两条并行入口：
  - Streamlit UI：`dayu/web/streamlit_app.py`（用户交互主入口）
  - FastAPI：`dayu/web/fastapi_app.py`（HTTP API 入口）
- 设计基线保持稳定分层：`UI -> Service -> Host -> Agent`。

## 2. Streamlit 入口与装配主线

Streamlit 路径的组合根是 `dayu/web/streamlit_app.py`，主线职责如下：

1. 解析工作区（优先级：`--workspace` > `DAYU_WORKSPACE` > `./workspace`）。
2. 通过 `dayu/web/streamlit/bootstrap.py` 初始化并缓存服务依赖到 `st.session_state`：
   - `FinsService`
   - `WriteService`
   - `ChatServiceClient`（内含 `ChatService`）
   - `HostAdminService`（供报告任务管理）
   - `FileServerHandle`（本地只读文件服务句柄）
3. `dayu/web/streamlit_app.py` 负责页面入口编排：渲染侧栏并根据是否选中股票路由到欢迎页或详情页。

装配逻辑在 `bootstrap` 中集中管理，Tab 页面不直接创建 `Host`。

## 3. `streamlit_app.py` 与 `fastapi_app.py` 职责划分

两者都属于 Web 适配层，但服务对象和交互形态不同。可先按用途快速区分：

- 用途：`streamlit_app.py` 为用户提供本地访问的 Web 页面。
- 用途：`fastapi_app.py` 为独立的 Web 服务提供 API 接口。

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

## 4. Streamlit 目录职责

- `dayu/web/streamlit_app.py`：Streamlit 应用入口与页面编排（会话初始化、路由、UI 入口）。
- `dayu/web/streamlit/bootstrap.py`：工作区解析、服务装配、文件服务启动。
- `dayu/web/streamlit/pages/main_page.py`：欢迎页与三 Tab（财报管理、交互式分析、分析报告）编排。
- `dayu/web/streamlit/pages/filing_tab.py`：财报下载、下载进度与已下载财报列表展示。
- `dayu/web/streamlit/pages/chat_tab.py`：交互式分析页面渲染与状态管理。
- `dayu/web/streamlit/pages/chat_client.py`：聊天页面的 Service 客户端封装。
- `dayu/web/streamlit/pages/chat_stream_bridge.py`：异步事件流到同步 UI 的桥接。
- `dayu/web/streamlit/pages/report_tab.py`：报告页状态路由与任务操作 UI。
- `dayu/web/streamlit/pages/report_markdown_view.py`：报告 Markdown 目录/锚点/双栏渲染。
- `dayu/web/streamlit/pages/report_export.py`：报告导出（Markdown/HTML/PDF）转换。
- `dayu/web/streamlit/pages/report_manifest.py`：manifest 章节快照解析。
- `dayu/web/streamlit/pages/report_host_sync.py`：报告任务与 Host 运行态同步辅助。
- `dayu/web/streamlit/stream_chat_events.py`：将 `AppEvent` 流折叠为可展示文本和侧边提示。
- `dayu/web/streamlit/file_server.py`：提供受限本地只读文件访问能力（仅财报文件路径）。
- `dayu/web/streamlit/components/sidebar.py`：自选股侧栏与持久化加载。
- `dayu/web/streamlit/components/watchlist_dialog.py`：自选股管理对话框。

## 5. 页面职责与边界

### 5.1 `main_page`

- 未选中股票：展示欢迎信息与操作指引。
- 选中股票：进入三个功能页签并注入 `workspace_root` 与对应服务。

### 5.2 `filing_tab`

- 调用 `FinsService` 提交下载流程并消费下载事件。
- 使用 `fins.storage` 仓储读取已下载文档列表和元数据用于展示。
- 通过 `FileServerHandle` 生成本地可访问文件 URL（新标签打开）。

### 5.3 `chat_tab`

- 组装 `ChatTurnRequest` 并通过 `ChatServiceProtocol.submit_turn()` 发起轮次。
- 将异步 `AppEvent` 流桥接到 Streamlit 同步渲染路径。
- 按股票维度维护 `session_id`、消息历史与输入状态。
- 助手回答在历史区提供“截图版（可复制 Markdown）”折叠区，便于保留 Markdown 结构后复用到截图文案。
- 当后端返回 warning/error 且主文为空时，页面保留当前帧展示错误，不立即 `rerun` 覆盖提示。

### 5.4 `report_tab`

- 使用 `WriteServiceProtocol.run()` 启动报告任务。
- 使用 `HostAdminServiceProtocol` 执行 `list_runs`、`get_run`、`cancel_run` 管理与观测。
- 读取 `workspace/draft/<ticker>/` 下产物展示报告与中间状态。

### 5.5 自选股管理（`sidebar` + `watchlist_dialog`）

- 左侧栏负责展示自选股列表与当前选中标的，管理入口由对话框组件承载。
- 用户可在对话框中新增、编辑、删除自选股；保存后侧栏刷新并保持当前选择一致性。
- 自选股持久化文件固定为 `workspace/streamlit_watchlist.json`，页面刷新后不会丢失。

## 6. 会话状态约定（`st.session_state`）

全局初始化键（入口层）：

- `initialized`
- `workspace_root`
- `fins_service`
- `write_service`
- `chat_service_client`
- `host_admin_service`
- `file_server_handle`

页面级状态（按需）：

- 侧栏：`selected_ticker`、`watchlist_needs_refresh`
- 财报下载：`active_downloads`、下载设置相关键
- 聊天：按 `ticker` 派生的 `messages` / `session_id` / 输入框键
- 报告：`active_write_tasks`、`write_task_settings`

## 7. 数据与事件流

```mermaid
flowchart LR
    streamlitApp[streamlit_app.py] -->|"初始化/注入"| pageTabs[main_page_and_tabs]
    pageTabs -->|"submit_turn submit run"| serviceProtocols[ServiceProtocols]
    serviceProtocols --> hostLayer[Host]
    hostLayer -->|"events_and_run_state"| serviceProtocols
    serviceProtocols -->|"streamed_events"| pageTabs
    pageTabs --> workspaceData[workspace_files]
```

说明：

- Tab 页面只消费 `Service` 协议，不直接操作 `Host` 实现对象。
- `Host` 的运行态观测与取消通过 `HostAdminService` 暴露，不向 UI 泄漏 Host 内部细节。

## 8. 路径与文件约定

- 自选股文件：`workspace/streamlit_watchlist.json`
- 报告目录：`workspace/draft/<ticker>/`
  - 常见产物：`run_summary.json`、`<ticker>_qual_report.md`、`manifest.json`、`chapters/`
- 财报文件目录：`workspace/portfolio/<ticker>/filings/<document_id>/...`
- 本地文件服务仅允许访问受限财报路径，不开放任意工作区文件访问。

## 9. 开发约束（Web 侧）

- 页面层保持 `UI -> Service` 调用，不新增 `UI -> Host` 反向穿透。
- 业务写路径遵循既有仓储与服务边界，UI 读取文件仅用于展示。
- 新增页面功能时，优先复用 `ServiceProtocol`，避免在 UI 内拼装跨层流程。
- 状态键新增时应遵循“全局键最少、页面键按 ticker 隔离”的原则，避免串会话。

## 10. 运行方式

- CLI 入口：`dayu-web`
- 模块入口：`python -m dayu.web`
- 直接运行：`streamlit run dayu/web/streamlit_app.py`
- 工作区可通过 `--workspace` 或 `DAYU_WORKSPACE` 指定。
