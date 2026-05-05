# MinerU 云 API 集成 — 变更说明

> PR: feat/mineru-v4.1
> 基线: v0.1.4 (noho/dayu-agent)
> 日期: 2026-05-06

## 一、变更概述

为 dayu-agent 集成 MinerU 云 API v4 作为 PDF 解析后端。当 Docling 解析失败（港股复杂 PDF 超时/崩溃）时，自动 fallback 到 MinerU 云 API。

**核心收益**：快手 2025 年报（374 页 / 9.9MB 港股 PDF）解析耗时从 Docling CPU >10min（未完成）降低到 MinerU 云 API 78.7s。

## 二、文件变更清单

### 新增文件（5 个模块 + 2 个测试）

| 文件 | 行数 | 职责 |
|------|------|------|
| `dayu/document_protocol.py` | 362 | 统一中间格式 `ConvertedDocument`（frozen dataclass）+ bbox 版本探测 |
| `dayu/mineru_runtime.py` | 899 | MinerU 云 API v4 集成：COS 上传 → 并发提交 → 并发轮询 → zip 下载解析 → 五层回退链 |
| `dayu/quota_tracker.py` | 182 | 配额跟踪器：本地计数器 + 文件持久化 + 每日 5000 页上限 |
| `dayu/cos_helper.py` | 167 | 腾讯云 COS 上传/删除/URL 提取辅助 |
| `dayu/config/pdf_backend.py` | 112 | MinerU 配置模块（环境变量 → 配置值） |
| `tests/test_mineru_basic.py` | 306 | 30 个基础测试（document_protocol + quota_tracker） |
| `tests/test_mineru_runtime.py` | 458 | 26 个运行时测试（回退链 + 并发轮询 + zip 解析） |

### 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `dayu/fins/docling_export.py` | +86 -5 | 新增 `convert_pdf_bytes_with_fallback`：Docling 优先，失败自动 fallback 到 MinerU |
| `dayu/fins/pipelines/docling_upload_service.py` | +22 -22 | 接入 fallback 函数，Docling 失败时存 `_mineru.json`（而非挂掉） |
| `pyproject.toml` | +2 | 新增依赖：`pikepdf`, `cos-python-sdk-v5`（httpx 已存在） |
| 6 个已有测试文件 | +70 -34 | 适配新接口签名（mock 返回 tuple、Playwright fake 模块注入） |

## 三、架构设计

### 3.1 五层回退链

```
parse_pdf_bytes_with_mineru(pdf_bytes, filename)
  ├─ 层1: MinerU 云 API 单次（≤200 页）
  ├─ 层2: MinerU 云 API 分批（>200 页，page_ranges 服务端分页）
  ├─ 层3: MinerU 本地 CLI（TODO）
  ├─ 层4: MinerU 本地 Python API（TODO）
  └─ 层5: Docling（终极兜底）
```

### 3.2 云 API 调用流程

```
pdf_bytes → upload_pdf_to_cos(pdf_bytes) → 公开 URL
                                              ↓
         check_and_consume(total_pages) → 配额扣减
                                              ↓
         POST /api/v4/extract/task × N → 并发提交（page_ranges 服务端分页）
                                              ↓
         GET /api/v4/extract/task/{id} × N → 并发轮询（指数退避 + jitter）
                                              ↓
         GET full_zip_url × N → 下载 zip 解析（full.md + content_list_v2.json）
                                              ↓
         _merge_chunk_results → ConvertedDocument
```

### 3.3 Upload Pipeline Fallback

```
DoclingUploadService.execute_upload()
  → _convert_bytes_with_fallback(raw_data, stream_name)
    → try: Docling → _docling.json
    → except: MinerU → _mineru.json
```

### 3.4 配额检查时机（v4.1 修正）

配额检查放在 COS 上传成功之后、API 提交之前，避免 COS 上传失败时配额被白扣。

### 3.5 统一中间格式

```python
@dataclass(frozen=True)
class ConvertedDocument:
    backend: DocumentBackend  # MINERU_CLOUD | MINERU_LOCAL | DOCLING
    sections: tuple[DocumentSection, ...]
    tables: tuple[DocumentTable, ...]
    images: tuple[DocumentImage, ...]
    raw_markdown: str
    metadata: dict[str, str]
```

## 四、关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| PDF 拆分 | 服务端 page_ranges | API 原生支持，不需要客户端 pikepdf 拆分 |
| 文件中转 | 腾讯云 COS 公开 URL | MinerU API 不支持 multipart，需要 URL |
| 结果下载 | zip 包解析 | v4 API 通过 `full_zip_url` 返回 zip（full.md + content_list_v2.json） |
| 存储格式 | `_mineru.json` | 不模拟 Docling dict（逆向工程 SDK 太脆弱），独立格式 |
| 配额持久化 | JSON 文件 + atexit | 跨 CLI 调用共享，进程退出时写入 |
| PDF 拆分库 | pikepdf（MIT） | PyMuPDF 是 GPLv3，避免开源合规风险 |
| 并发模型 | asyncio.gather | 自动 cancel 剩余任务，比 abort_event 更可靠 |

## 五、否决记录

| 提议 | 否决理由 |
|------|----------|
| 预签名 URL | Bucket 保持 public-read，临时文件有生命周期兜底 |
| COS client 单例 | CosS3Client 非线程安全 |
| zip 下载/解析拆分 | CLI 工具 YAGNI |
| 配额文件锁 | CLI 单进程，write_text 原子操作 |
| BlockType class | frozenset 常量足够 |
| 双文件存储（.md + .json） | 冗余、来源混淆、维护成本高 |
| 模拟 Docling dict 格式 | 逆向工程 SDK 内部结构，SDK 升级即 break |
| 按文档类型前端路由 | MinerU 对所有类型效果好，配额充足 |

## 六、环境变量

```bash
# MinerU 云 API
DAYU_MINERU_TOKEN=<token>          # 从 mineru.net 控制台获取
DAYU_MINERU_API_BASE=https://mineru.net
DAYU_MINERU_CHUNK_SIZE=200         # 每批最多页数
DAYU_MINERU_LANG=ch                # 文档语言

# 腾讯云 COS（MinerU 文件中转）
DAYU_COS_SECRET_ID=<secret_id>
DAYU_COS_SECRET_KEY=<secret_key>
DAYU_COS_BUCKET=<bucket>
DAYU_COS_REGION=ap-chengdu
```

## 七、测试结果

| 指标 | 值 |
|------|-----|
| 单元测试 | 4281 通过 / 1 失败（serper 网络超时，预存） |
| pyright | 0 errors, 0 warnings |
| 第三方审查 | P0/P1/P3 全部 pass |
| 快手 2025 年报 | 78.7s, 2398 sections, 202 tables, 112 images, 341KB markdown |
| 配额消耗 | 1496 页 / 5000 页（2 批: 1-200 + 201-374） |

## 八、技术债务

| 项 | 严重性 |
|---|--------|
| 层3/层4（MinerU 本地）未实现 | 中（当前云 API 够用） |
| process pipeline 不消费 `_mineru.json` | 中（阶段 2：需 MineruProcessor） |
| 潜在循环依赖路径（docling_export ↔ mineru_runtime） | 低（当前路径不触发） |
